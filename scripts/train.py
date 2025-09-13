import os

# Force disable XFormers and fast attention BEFORE any other imports
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_ATTENTION"] = "1"
os.environ["UNSLOTH_FORCE_SDPA"] = "1"

import sys, json, math, argparse, warnings
from pathlib import Path
from typing import Dict, Any, List

import torch
import yaml
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from jinja2 import Template
from scripts.utils.model_store import prepare_local_model_dir

warnings.filterwarnings("ignore", category=UserWarning)


# ---------- Config loader (merge base + run) ----------
def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        run_cfg = yaml.safe_load(f)
    base_path = run_cfg.get("include")
    if base_path:
        with open(base_path, "r") as f:
            base_cfg = yaml.safe_load(f)
        return deep_merge(base_cfg, run_cfg)
    return run_cfg


# ---------- VRAM probe & auto batch/accum ----------
def choose_bs_and_accum(
    cfg_train: Dict[str, Any], max_len: int, model_name: str
) -> tuple[int, int]:
    bs = cfg_train.get("batch_size", "auto")
    accum = cfg_train.get("grad_accum", "auto")
    if bs != "auto" and accum != "auto":
        return int(bs), int(accum)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))

    # crude heuristic based on free VRAM and context length per rank
    free = 0
    if torch.cuda.is_available():
        try:
            free = torch.cuda.mem_get_info(local_rank)[0] / (1024**3)  # GB
        except Exception:
            free = 0

    # defaults by context & GPU budget
    # aim effective batch ~8 for stability
    if bs == "auto":
        bs = 1
    if accum == "auto":
        if free >= 20:
            accum = 8
        elif free >= 12:
            accum = 16 if max_len >= 512 else 12
        else:
            accum = 16
        accum = max(1, accum // world_size)

    return int(bs), int(accum)


# ---------- Target modules resolver ----------
def resolve_target_modules(model_name: str, override):
    if override and override != "auto":
        return override
    name = model_name.lower()
    if any(k in name for k in ["llama", "mistral", "qwen", "phi", "gemma"]):
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    # seq2seq (t5) often target feed-forward + attention
    if "t5" in name or "flan" in name:
        return ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]  # safe fallback


# ---------- Dataset (structured chat; Jinja on-the-fly) ----------
class ChatDataset(Dataset):
    def __init__(self, path: str):
        self.rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class Collator:
    def __init__(self, tokenizer, template_path, max_len, model_type):
        self.tok = tokenizer
        self.max_len = max_len
        self.model_type = model_type
        # Load template if it exists, otherwise we'll handle rendered format
        self.template = None
        if template_path and Path(template_path).exists():
            self.template = Template(Path(template_path).read_text())

    def __call__(self, batch: List[Dict[str, Any]]):
        inputs = []
        targets = []

        # Auto-detect data format and handle both processed and rendered formats
        for row in batch:
            if "input" in row and "target" in row:
                # Rendered format: {input: "...", target: "..."}
                inputs.append(row["input"].strip())
                targets.append(row["target"].strip())
            elif "user" in row and "assistant" in row:
                # Processed format: {system: "...", user: "...", assistant: "..."}
                if self.template:
                    # Apply template if available
                    sys_txt = (row.get("system") or "").strip()
                    user_txt = row["user"].strip()
                    asst_txt = row["assistant"].strip()
                    rendered = self.template.render(
                        system=sys_txt, user=user_txt
                    ).strip()
                    inputs.append(rendered)
                    targets.append(asst_txt)
                else:
                    # No template - create simple format
                    sys_txt = (row.get("system") or "").strip()
                    user_txt = row["user"].strip()
                    asst_txt = row["assistant"].strip()
                    if sys_txt:
                        simple_input = (
                            f"System: {sys_txt}\nUser: {user_txt}\nAssistant:"
                        )
                    else:
                        simple_input = f"User: {user_txt}\nAssistant:"
                    inputs.append(simple_input)
                    targets.append(asst_txt)
            else:
                raise ValueError(
                    f"Unsupported data format. Expected either {{input, target}} or {{user, assistant}} format. Got: {list(row.keys())}"
                )

        if self.model_type == "seq2seq":
            model_inputs = self.tok(
                inputs, max_length=self.max_len, truncation=True, padding=True
            )
            with self.tok.as_target_tokenizer():
                labels = self.tok(
                    targets, max_length=self.max_len, truncation=True, padding=True
                )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        else:
            # causal: concatenate input + target with labels masked on the prompt
            # Strategy: tokenize separately, then build labels where prompt tokens = -100
            model_inputs = self.tok(
                inputs,
                max_length=self.max_len,
                truncation=True,
                padding=True,
                add_special_tokens=True,
            )
            target_tok = self.tok(
                targets,
                max_length=self.max_len,
                truncation=True,
                padding=True,
                add_special_tokens=False,
            )

            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            labels = []
            for i in range(len(input_ids)):
                # Get prompt text and target text
                prompt = self.tok.decode(input_ids[i], skip_special_tokens=True)
                target = targets[i]

                # Tokenize the full sequence (prompt + target)
                full_text = prompt + target
                full_tokens = self.tok(
                    full_text,
                    max_length=self.max_len,
                    truncation=True,
                    padding=False,
                    add_special_tokens=True,
                )
                full_ids = full_tokens["input_ids"]

                # Get prompt length to mask it in labels
                prompt_tokens = self.tok(prompt, add_special_tokens=True, padding=False)
                prompt_len = len(prompt_tokens["input_ids"])

                # Create labels: -100 for prompt tokens, actual tokens for target
                labels_seq = [-100] * prompt_len + full_ids[prompt_len:]

                # Ensure labels match the length of input_ids
                if len(labels_seq) > len(full_ids):
                    labels_seq = labels_seq[: len(full_ids)]
                elif len(labels_seq) < len(full_ids):
                    labels_seq.extend([-100] * (len(full_ids) - len(labels_seq)))

                input_ids[i] = full_ids
                attention_mask[i] = [1] * len(full_ids)
                labels.append(labels_seq)

            # pad to same length
            maxL = max(len(x) for x in input_ids)
            for i in range(len(input_ids)):
                pad = maxL - len(input_ids[i])
                if pad > 0:
                    input_ids[i] += [self.tok.pad_token_id] * pad
                    attention_mask[i] += [0] * pad
                    labels[i] += [-100] * pad
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
            }


# ---------- Metric: ROUGE-L (simple) ----------
def compute_metrics(tokenizer):
    def _fn(eval_pred):
        try:
            from evaluate import load as load_metric

            rouge = load_metric("rouge")

            preds, labels = eval_pred
            # replace -100 with pad for decode
            labels = [
                [(tok if tok != -100 else tokenizer.pad_token_id) for tok in seq]
                for seq in labels
            ]
            pred_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
            label_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
            r = rouge.compute(predictions=pred_text, references=label_text)
            return {"rougeL": r["rougeL"]}
        except Exception as e:
            print(f"[train] Warning: Could not compute ROUGE metrics: {e}")
            return {"rougeL": 0.0}

    return _fn


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/config_run.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    torch.manual_seed(cfg.get("seed", 42))

    # Backend stamp and resume check
    outdir = cfg["train"]["output_dir"]
    backend_file = os.path.join(outdir, "backend.json")
    current_backend = cfg["tuning"]["backend"]
    current_dtype = "bf16" if cfg["train"].get("bf16", False) else "fp16"

    if os.path.exists(backend_file):
        # Resume check - ensure backend consistency
        try:
            with open(backend_file, "r") as f:
                saved_info = json.load(f)
            if saved_info.get("backend") != current_backend:
                raise ValueError(
                    f"Backend mismatch! Saved run used '{saved_info.get('backend')}' "
                    f"but current config specifies '{current_backend}'. "
                    f"Use a different output_dir or ensure backend consistency."
                )
            print(
                f"[train] Resuming {saved_info.get('backend')} run with {saved_info.get('dtype')} precision"
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[train] Warning: Could not read backend info: {e}")
    else:
        # New run - create backend stamp
        os.makedirs(outdir, exist_ok=True)
        backend_info = {
            "backend": current_backend,
            "dtype": current_dtype,
            "attn_impl": "sdpa" if current_backend == "unsloth" else "default",
        }
        with open(backend_file, "w") as f:
            json.dump(backend_info, f, indent=2)
        print(
            f"[train] Starting new {current_backend} run with {current_dtype} precision"
        )

    # Precision sanity check
    bf16_enabled = cfg["train"].get("bf16", False)
    fp16_enabled = cfg["train"].get("fp16", False)

    if bf16_enabled and fp16_enabled:
        print("[train] Warning: Both bf16 and fp16 enabled. Disabling fp16.")
        cfg["train"]["fp16"] = False
    elif not bf16_enabled and not fp16_enabled:
        # Default to bf16 on Ada GPUs (RTX 40xx series), fp16 otherwise
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            if "rtx 40" in gpu_name or "ada" in gpu_name:
                cfg["train"]["bf16"] = True
                print("[train] Auto-enabled bf16 for Ada GPU")
            else:
                cfg["train"]["fp16"] = True
                print("[train] Auto-enabled fp16 for non-Ada GPU")
        else:
            cfg["train"]["fp16"] = True
            print("[train] Auto-enabled fp16 (no CUDA detected)")

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    local_model_dir = prepare_local_model_dir(cfg["model"], hf_token=hf_token)

    model_name = local_model_dir  # from now on, load FROM DISK
    trust_remote_code = bool(cfg["model"].get("trust_remote_code", False))
    model_type = cfg["model"]["type"]  # "causal" | "seq2seq"
    max_len = int(cfg["model"].get("max_seq_len", 512))

    device_map = {"": local_rank} if world_size > 1 else "auto"

    # auto bs/accum by VRAM
    bs, accum = choose_bs_and_accum(cfg["train"], max_len, model_name)

    # tokenizer
    tok = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=trust_remote_code
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "<|pad|>"

    # datasets (structured chat)
    paths = cfg["data"]

    # Check if data files exist
    train_path = paths["train_path"]
    val_path = paths["val_path"]
    template_path = cfg["data"]["template_path"]

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    ds_train = ChatDataset(train_path)
    ds_val = ChatDataset(val_path)

    print(
        f"[train] Loaded {len(ds_train)} training samples, {len(ds_val)} validation samples"
    )

    collator = Collator(tok, template_path, max_len, model_type)

    # backend & model load
    backend = cfg["tuning"]["backend"]  # "bnb" | "unsloth"
    mode = cfg["tuning"]["mode"]  # "qlora" | "lora" | "full"

    use_bnb = True
    if backend == "unsloth":
        print(
            "[train] WARNING: Unsloth backend has XFormers compatibility issues on this system."
        )
        print("[train] Automatically falling back to BitsAndBytes for stability.")
        print("[train] Use 'make train-bnb' for optimal performance with BitsAndBytes.")
        use_bnb = True
        # Force backend change for this run
        current_backend = "bnb"

    is_seq2seq = model_type == "seq2seq"
    if is_seq2seq:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    else:
        if mode == "qlora":
            if use_bnb:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                )
                base_model = prepare_model_for_kbit_training(base_model)
            else:
                # Unsloth path (simplified): unsloth handles 4-bit load internally
                from unsloth import FastLanguageModel

                base_model, tok = FastLanguageModel.from_pretrained(
                    model_name, load_in_4bit=True, trust_remote_code=trust_remote_code
                )

                # Force standard attention implementation to avoid XFormers issues
                try:
                    if hasattr(base_model.config, "attn_implementation"):
                        base_model.config.attn_implementation = "sdpa"
                        print(
                            "[train] Forced model to use SDPA attention implementation"
                        )
                    if hasattr(base_model.config, "_attn_implementation"):
                        base_model.config._attn_implementation = "sdpa"
                        print(
                            "[train] Forced model to use SDPA attention implementation (private attr)"
                        )
                except Exception as e:
                    print(f"[train] Warning: Could not force SDPA attention: {e}")
        else:
            # LoRA or Full: fp16/bf16 path
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
            )

    # Apply LoRA if needed
    if mode in ["qlora", "lora"]:
        lcfg = cfg["tuning"]["lora"]
        targets = resolve_target_modules(model_name, lcfg.get("target_modules"))
        lora_cfg = LoraConfig(
            r=int(lcfg.get("r", 32)),
            lora_alpha=int(lcfg.get("alpha", 32)),
            lora_dropout=float(lcfg.get("dropout", 0.05)),
            target_modules=targets,
            bias="none",
            task_type="SEQ_2_SEQ_LM" if is_seq2seq else "CAUSAL_LM",
        )
        base_model = get_peft_model(base_model, lora_cfg)

    # Training args
    outdir = cfg["train"]["output_dir"]
    import transformers
    from transformers.trainer_utils import IntervalStrategy

    # ...
    # Build kwargs adaptively so it works across TF versions
    ta_fields = getattr(transformers.TrainingArguments, "__dataclass_fields__", {})
    has_eval_strategy = "evaluation_strategy" in ta_fields
    has_save_strategy = "save_strategy" in ta_fields

    # Map string -> IntervalStrategy
    def to_interval(name: str) -> IntervalStrategy:
        return (
            IntervalStrategy.STEPS
            if str(name).lower() == "steps"
            else IntervalStrategy.EPOCH
        )

    ta_kwargs = dict(
        output_dir=outdir,
        run_name="sft-play",  # <-- name shown in TB
        logging_dir="outputs/tb",  # <-- single TB root
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=max(1, bs),
        gradient_accumulation_steps=accum,
        num_train_epochs=int(cfg["train"]["epochs"]),
        learning_rate=float(cfg["train"].get("lr", 2e-4)),
        warmup_ratio=float(cfg["train"].get("warmup_ratio", 0.06)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
        fp16=bool(cfg["train"].get("fp16", True)),
        logging_steps=1,
        save_total_limit=int(cfg["train"].get("save_total_limit", 2)),
        load_best_model_at_end=bool(cfg["train"].get("load_best_model_at_end", False)),
        metric_for_best_model=cfg["train"].get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(cfg["train"].get("greater_is_better", False)),
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    # Add eval/save strategies in a version-safe way
    eval_strategy_key = (
        "evaluation_strategy"
        if "evaluation_strategy" in cfg["train"]
        else "eval_strategy"
    )
    eval_strategy_value = cfg["train"].get(eval_strategy_key, "epoch")

    if has_eval_strategy:
        ta_kwargs["evaluation_strategy"] = to_interval(eval_strategy_value)
    else:
        ta_kwargs["eval_strategy"] = to_interval(eval_strategy_value)  # older API

    save_strategy_value = cfg["train"].get("save_strategy", "epoch")
    if has_save_strategy:
        ta_kwargs["save_strategy"] = to_interval(save_strategy_value)
    else:
        ta_kwargs["save_steps"] = int(
            cfg["train"].get("save_steps", 100)
        )  # fallback behavior

    # Only add step args if using steps
    if str(eval_strategy_value).lower() == "steps":
        ta_kwargs["eval_steps"] = int(cfg["train"].get("eval_steps", 50))

    if str(save_strategy_value).lower() == "steps":
        ta_kwargs["save_steps"] = int(cfg["train"].get("save_steps", 100))

    args_tr = transformers.TrainingArguments(**ta_kwargs)

    # Always use our custom collator (it handles both causal & seq2seq)
    data_collator = collator

    # Resume if checkpoint present
    last_ckpt = get_last_checkpoint(outdir) if os.path.isdir(outdir) else None
    if last_ckpt:
        print(f"[train] Resuming from checkpoint: {last_ckpt}")

    trainer = Trainer(
        model=base_model,
        args=args_tr,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        tokenizer=tok,  # fine to keep; warning is harmless
        compute_metrics=compute_metrics(tok),
    )

    trainer.train(resume_from_checkpoint=last_ckpt)

    # Force one eval write so TB has eval scalars even on tiny runs
    trainer.evaluate()  # <-- add this
    trainer.save_state()
    if mode in ["qlora", "lora"]:
        base_model.save_pretrained("adapters/last")
    print("[train] done.")


if __name__ == "__main__":
    main()
