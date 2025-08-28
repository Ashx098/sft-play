import argparse, json, sys, re
from pathlib import Path
from typing import Dict, Any, List

import torch, yaml
from jinja2 import Template
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import PeftModel
from scripts.utils.model_store import prepare_local_model_dir
import os


# ---------- config ----------
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


# ---------- gen ----------
@torch.inference_mode()
def generate(
    model,
    tok,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    model_type: str,
):
    inputs = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tok.model_max_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Set up generation parameters
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tok.eos_token_id,
        pad_token_id=(
            tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ),
    )

    # Allow generation to stop if the model starts a new user turn.
    # This covers ChatML-style templates where <|im_start|>user marks
    # the beginning of the next message.
    if model_type != "seq2seq":
        stop_tokens = []
        for tkn in ["<|im_start|>user", "<|user|>"]:
            try:
                ids = tok.encode(tkn, add_special_tokens=False)
                if len(ids) == 1:
                    stop_tokens.append(ids[0])
            except Exception:
                pass
        if stop_tokens:
            gen_kwargs["eos_token_id"] = [tok.eos_token_id] + stop_tokens

    out_ids = model.generate(**inputs, **gen_kwargs)
    return tok.batch_decode(
        out_ids, skip_special_tokens=False
    )  # Keep special tokens for proper parsing


def load_model_and_tok(cfg, adapters_path: str = None):
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    local_model_dir = prepare_local_model_dir(cfg["model"], hf_token=hf_token)

    model_name = local_model_dir  # from now on, load FROM DISK
    trust_remote_code = bool(cfg["model"].get("trust_remote_code", False))
    model_type = cfg["model"]["type"]  # "causal" | "seq2seq"

    tok = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=trust_remote_code
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "<|pad|>"

    # Ensure chat template is properly set to match training
    template_path = cfg["data"].get("template_path")
    if template_path and Path(template_path).exists():
        try:
            chat_template = Path(template_path).read_text().strip()
            # Set the chat template on the tokenizer to ensure consistency
            if hasattr(tok, "chat_template"):
                tok.chat_template = chat_template
                print(f"[infer] loaded chat template from {template_path}")
        except Exception as e:
            print(f"[infer] warning: failed to load chat template: {e}")

    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )

    # attach adapters if path exists
    if adapters_path and Path(adapters_path).exists():
        try:
            model = PeftModel.from_pretrained(model, adapters_path)
            print(f"[infer] loaded adapters from {adapters_path}")
        except Exception as e:
            print(f"[infer] warning: failed to load adapters: {e}")

    model.eval()
    return model, tok


def render_prompt(tmpl: Template, system_txt: str, user_txt: str) -> str:
    return tmpl.render(system=(system_txt or "").strip(), user=user_txt.strip()).strip()


def clean_generated_text(text: str, model_type: str) -> str:
    """Clean generated text by extracting only the assistant response."""
    if model_type == "seq2seq":
        return text.strip()

    # ChatML-style cleaning: <|im_start|>assistant ... <|im_end|>
    if "<|im_start|>assistant" in text:
        reply = text.split("<|im_start|>assistant", 1)[1]
        if "<|im_end|>" in reply:
            reply = reply.split("<|im_end|>")[0]
        return reply.strip()

    # Legacy cleaning using <|assistant|> markers
    if "<|assistant|>" in text:
        reply = text.split("<|assistant|>")[-1]
        if "<|user|>" in reply:
            reply = reply.split("<|user|>")[0]
        if "</|assistant|>" in reply:
            reply = reply.split("</|assistant|>")[0]
        return reply.strip()

    # Fallback: strip any regenerated system/user sections and common patterns
    text = re.sub(r"<\|system\|>.*?</\|system\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|user\|>.*?</\|user\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"User:.*?Assistant:", "", text, flags=re.DOTALL)
    text = re.sub(r"Human:.*?Assistant:", "", text, flags=re.DOTALL)
    return text.strip()


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/config_run.yaml")
    ap.add_argument(
        "--adapters", default="adapters/last", help="LoRA adapters dir (if used)"
    )
    ap.add_argument("--mode", choices=["interactive", "batch"], default="interactive")
    ap.add_argument("--input_file", help="Batch: file with one user prompt per line")
    ap.add_argument(
        "--output_file",
        default="outputs/preds.txt",
        help="Batch: where to write generations",
    )
    ap.add_argument(
        "--system",
        default="You are a helpful domain assistant.",
        help="System prompt for inference",
    )
    ap.add_argument(
        "--max_new_tokens", type=int, help="Override config.gen.max_new_tokens"
    )
    ap.add_argument("--temperature", type=float, help="Override config.gen.temperature")
    ap.add_argument("--top_p", type=float, help="Override config.gen.top_p")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tmpl = Template(Path(cfg["data"]["template_path"]).read_text())

    # gen params (allow CLI override)
    gcfg = cfg.get("gen", {})
    max_new_tokens = args.max_new_tokens or int(gcfg.get("max_new_tokens", 200))
    temperature = args.temperature or float(gcfg.get("temperature", 0.2))
    top_p = args.top_p or float(gcfg.get("top_p", 0.9))

    model, tok = load_model_and_tok(cfg, adapters_path=args.adapters)

    if args.mode == "interactive":
        print("=== SFT-Play :: Interactive ===")
        print("Type your question. Press Ctrl+C to exit.")
        while True:
            try:
                user = input("\nYou: ").strip()
                if not user:
                    continue
                prompt = render_prompt(tmpl, args.system, user)

                # For causal models, append assistant tag to get cleaner generation
                if cfg["model"]["type"] == "causal" and not prompt.endswith(
                    "<|im_start|>assistant"
                ):
                    prompt += "<|im_start|>assistant"

                pred = generate(
                    model,
                    tok,
                    [prompt],
                    max_new_tokens,
                    temperature,
                    top_p,
                    cfg["model"]["type"],
                )[0]

                # Clean the generated text
                cleaned_pred = clean_generated_text(pred, cfg["model"]["type"])
                print(f"Model: {cleaned_pred}")
            except KeyboardInterrupt:
                print("\nbye!")
                break
    else:
        if not args.input_file:
            print("[infer] batch mode requires --input_file", file=sys.stderr)
            sys.exit(1)
        lines = [
            ln.strip()
            for ln in Path(args.input_file).read_text().splitlines()
            if ln.strip()
        ]
        prompts = [render_prompt(tmpl, args.system, u) for u in lines]

        # For causal models, append assistant tag to get cleaner generation
        if cfg["model"]["type"] == "causal":
            prompts = [
                (
                    p + "<|im_start|>assistant"
                    if not p.endswith("<|im_start|>assistant")
                    else p
                )
                for p in prompts
            ]

        preds = generate(
            model,
            tok,
            prompts,
            max_new_tokens,
            temperature,
            top_p,
            cfg["model"]["type"],
        )

        # Clean all predictions
        cleaned_preds = [clean_generated_text(p, cfg["model"]["type"]) for p in preds]

        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            for p in cleaned_preds:
                f.write(p.strip() + "\n")
        print(f"[infer] wrote {len(preds)} generations -> {args.output_file}")


if __name__ == "__main__":
    main()
