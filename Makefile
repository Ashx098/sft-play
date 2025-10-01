.PHONY: help install process style render train train-with-tb stop-tb tensorboard tb tb-stop tb-clean tb-open train-and-watch eval eval-test eval-val eval-quick eval-full infer infer-batch infer-interactive merge merge-bf16 merge-test check clean setup-dirs download-model print-python

# Default config file
CONFIG ?= configs/run_bnb.yaml

# Style prompt (can be overridden)
STYLE ?= "Answer concisely in 2 lines. No markdown. If unsure, say 'Not sure'."

# Number of processes for distributed training
# Defaults to all visible GPUs (or 1 if none detected)
NPROC ?= $(shell python -c "import torch,os;print(torch.cuda.device_count() or 1)" 2>/dev/null || echo 1)

# ---- TensorBoard config ----
TB_PORT ?= 6006
TB_LOGDIR ?= $(shell realpath outputs/tb)

help:
	@echo "SFT-Play Makefile Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install dependencies (pip or uv)"
	@echo "  setup-dirs    Create necessary directories"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  process       Process raw data to structured chat format"
	@echo "  style         Apply style/system prompts to processed data"
	@echo "  render        Render chat templates to seq2seq format"
	@echo ""
	@echo "Training & Evaluation:"
        @echo "  train         Start training with current config"
        @echo "  train-bnb     Start training with BitsAndBytes backend"
        @echo "  train-unsloth Start training with Unsloth backend"
        @echo "  train-with-tb Start training with TensorBoard monitoring"
        @echo "  train-bnb-tb  Start BitsAndBytes training with TensorBoard"
        @echo "  train-unsloth-tb Start Unsloth training with TensorBoard"
        @echo "  train-multi    Start distributed training with torchrun"
        @echo "  train-accelerate Start distributed training with Accelerate"
        @echo "  eval          Evaluate trained model (validation set)"
	@echo "  eval-test     Evaluate on test set"
	@echo "  eval-val      Evaluate on validation set"
	@echo "  eval-quick    Quick evaluation (200 samples)"
	@echo "  eval-full     Full evaluation (no sample limit)"
	@echo "  infer         Interactive inference (chat mode)"
	@echo "  infer-batch   Batch inference from file"
	@echo "  infer-interactive Interactive inference (explicit)"
	@echo ""
	@echo "Monitoring:"
	@echo "  tensorboard   Start TensorBoard on outputs/tb"
	@echo "  tb-stop       Kill any running TensorBoard"
	@echo "  tb-clean      Remove TB event files"
	@echo "  tb-open       Print exact path & suggest URL"
	@echo ""
	@echo "Model Management:"
	@echo "  merge         Merge LoRA adapters to FP16 model"
	@echo "  merge-bf16    Merge LoRA adapters to BF16 model"
	@echo "  merge-test    Test merged model loading"
	@echo ""
	@echo "Utilities:"
	@echo "  check         Validate project setup before training"
	@echo "  clean         Clean generated files"
	@echo "  full-pipeline Run complete data processing pipeline"
	@echo ""
	@echo "Variables:"
	@echo "  CONFIG=path   Specify config file (default: configs/run_bnb.yaml)"
	@echo "  STYLE=text    Specify style prompt for style command"

print-python:
	@echo "which python: `which python`"
	@python -c "import sys; print(f'sys.executable: {sys.executable}')"

.PHONY: download-model
MODEL?=Qwen/Qwen2.5-3B-Instruct
LOCAL?=models/$(shell echo $(MODEL) | sed 's/[\/:@#]\+/-/g')

download-model:
	@python -c "from huggingface_hub import snapshot_download; import os; \
token=os.getenv('HUGGINGFACE_HUB_TOKEN', None); \
snapshot_download(repo_id='$(MODEL)', local_dir='$(LOCAL)', local_dir_use_symlinks=False, token=token); \
print('Saved to: $(LOCAL)')"

install:
	@echo "Installing dependencies..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using uv for installation..."; \
		uv venv --python 3.10; \
		uv pip install -e .; \
	else \
		echo "Using pip for installation..."; \
		pip install -r requirements.txt; \
	fi

setup-dirs:
	@echo "Creating necessary directories..."
	@mkdir -p data/raw data/processed data/processed_with_style data/rendered
	@mkdir -p outputs adapters
	@touch data/raw/.gitkeep data/processed/.gitkeep data/processed_with_style/.gitkeep data/rendered/.gitkeep
	@touch outputs/.gitkeep adapters/.gitkeep
	@echo "Directories created successfully!"

process:
	@echo "Processing raw data..."
	python scripts/process_data.py --config $(CONFIG)

style:
	@echo "Applying style prompts..."
	@if [ ! -f data/processed/train.jsonl ]; then \
		echo "Error: data/processed/train.jsonl not found. Run 'make process' first."; \
		exit 1; \
	fi
	python scripts/style_prompt.py \
		--config $(CONFIG) \
		--style "$(STYLE)" \
		--in data/processed/train.jsonl \
		--out data/processed_with_style/train.jsonl \
		--mode prepend
	@for split in val test; do \
		if [ -f data/processed/$$split.jsonl ]; then \
			echo "Processing $$split split..."; \
			python scripts/style_prompt.py \
				--config $(CONFIG) \
				--style "$(STYLE)" \
				--in data/processed/$$split.jsonl \
				--out data/processed_with_style/$$split.jsonl \
				--mode prepend; \
		fi; \
	done

render:
	@echo "Rendering chat templates..."
	@if [ ! -f data/processed/train.jsonl ]; then \
		echo "Error: data/processed/train.jsonl not found. Run 'make process' first."; \
		exit 1; \
	fi
	python scripts/render_template.py \
		--config $(CONFIG) \
		--in data/processed/train.jsonl \
		--out data/rendered/train.jsonl
	@for split in val test; do \
		if [ -f data/processed/$$split.jsonl ]; then \
			echo "Rendering $$split split..."; \
			python scripts/render_template.py \
				--config $(CONFIG) \
				--in data/processed/$$split.jsonl \
				--out data/rendered/$$split.jsonl; \
		fi; \
	done

train:
	@echo "Starting training..."
	PYTHONPATH=. python scripts/train.py --config $(CONFIG)

train-bnb:
	@echo "Starting training with BitsAndBytes backend..."
	PYTHONPATH=. python scripts/train.py --config configs/run_bnb.yaml

train-unsloth:
        @echo "Starting training with Unsloth backend (XFormers disabled)..."
        XFORMERS_DISABLED=1 UNSLOTH_DISABLE_FAST_ATTENTION=1 PYTHONPATH=. python scripts/train.py --config configs/run_unsloth.yaml

train-multi:
        @echo "Starting distributed training on $(NPROC) GPU(s)..."
        torchrun --nproc_per_node=$(NPROC) scripts/train.py --config $(CONFIG)

train-accelerate:
        @echo "Starting distributed training with Accelerate on $(NPROC) GPU(s)..."
        accelerate launch --num_processes $(NPROC) scripts/train.py --config $(CONFIG)

## train-with-tb: Train + print how to launch TB
train-with-tb:
	@echo "Starting training…"
	PYTHONPATH=. python scripts/train.py --config $(CONFIG)
	@echo ""
	@echo "✅ Training finished. To view logs:"
        @echo "   make tensorboard TB_PORT=$(TB_PORT)"

train-bnb-tb:
        @echo "Starting BitsAndBytes training with TensorBoard..."
	@mkdir -p outputs/tb
	@pkill -f tensorboard || true
	@nohup tensorboard --logdir $(TB_LOGDIR) --port $(TB_PORT) --host 0.0.0.0 >/dev/null 2>&1 &
	@sleep 2
	@echo "📈 TensorBoard started at http://localhost:$(TB_PORT)"
	PYTHONPATH=. python scripts/train.py --config configs/run_bnb.yaml
	@echo ""
	@echo "✅ Training finished. TensorBoard is still running at:"
        @echo "   http://localhost:$(TB_PORT)"
        @echo "   To stop TensorBoard: make tb-stop"

train-unsloth-tb:
	@echo "Starting Unsloth training with TensorBoard..."
	@mkdir -p outputs/tb
	@pkill -f tensorboard || true
	@nohup tensorboard --logdir $(TB_LOGDIR) --port $(TB_PORT) --host 0.0.0.0 >/dev/null 2>&1 &
        @sleep 2
	@echo "📈 TensorBoard started at http://localhost:$(TB_PORT)"
	XFORMERS_DISABLED=1 UNSLOTH_DISABLE_FAST_ATTENTION=1 PYTHONPATH=. python scripts/train.py --config configs/run_unsloth.yaml
	@echo ""
	@echo "✅ Training finished. TensorBoard is still running at:"
	@echo "   http://localhost:$(TB_PORT)"
	@echo "   To stop TensorBoard: make tb-stop"

## train-and-watch: Start TB (bg) then train
train-and-watch:
	@mkdir -p outputs/tb
	@pkill -f tensorboard || true
	@nohup tensorboard --logdir $(TB_LOGDIR) --port $(TB_PORT) --host 0.0.0.0 >/dev/null 2>&1 &
	@sleep 2
	@echo "📈 TensorBoard at http://localhost:$(TB_PORT)"
	PYTHONPATH=. python scripts/train.py --config $(CONFIG)

## tensorboard: Start TensorBoard on outputs/tb (override TB_PORT=6007 if needed)
tensorboard tb:
	@if [ ! -d "outputs/tb" ]; then mkdir -p outputs/tb; fi
	@echo "👉 Launching TensorBoard at http://localhost:$(TB_PORT) (logdir=$(TB_LOGDIR))"
	@tensorboard --logdir $(TB_LOGDIR) --port $(TB_PORT) --host 0.0.0.0

## tb-stop: Kill any running TensorBoard
tb-stop:
	@pkill -f tensorboard || true
	@echo "✅ Stopped any running TensorBoard"

## tb-clean: Remove TB event files
tb-clean:
	@rm -rf outputs/tb
	@mkdir -p outputs/tb
	@echo "🧹 Cleaned outputs/tb"

## tb-open: Print exact path & suggest URL
tb-open:
	@echo "Logdir: $(TB_LOGDIR)"
	@echo "Visit:  http://localhost:$(TB_PORT)"

eval:
	@echo "Running evaluation on validation set..."
	PYTHONPATH=. python scripts/eval.py --config $(CONFIG) --split val

eval-test:
	@echo "Running evaluation on test set..."
	PYTHONPATH=. python scripts/eval.py --config $(CONFIG) --split test

eval-val:
	@echo "Running evaluation on validation set..."
	PYTHONPATH=. python scripts/eval.py --config $(CONFIG) --split val

eval-quick:
	@echo "Running quick evaluation (200 samples)..."
	PYTHONPATH=. python scripts/eval.py --config $(CONFIG) --split val --limit 200

eval-full:
	@echo "Running full evaluation (no limit)..."
	PYTHONPATH=. python scripts/eval.py --config $(CONFIG) --split val --limit 0

infer:
	@echo "Starting interactive inference..."
	@echo "Type your questions. Press Ctrl+C to exit."
	PYTHONPATH=. python scripts/infer.py --config $(CONFIG) --mode interactive

infer-batch:
	@echo "Running batch inference..."
	@if [ ! -f demo_inputs.txt ]; then \
		echo "Creating demo_inputs.txt..."; \
		echo "What is machine learning?" > demo_inputs.txt; \
		echo "Explain neural networks briefly." >> demo_inputs.txt; \
		echo "What is QLoRA?" >> demo_inputs.txt; \
	fi
	PYTHONPATH=. python scripts/infer.py --config $(CONFIG) --mode batch --input_file demo_inputs.txt --output_file outputs/preds.txt
	@echo "Results saved to outputs/preds.txt"

infer-interactive:
	@echo "Starting interactive inference..."
	@echo "Type your questions. Press Ctrl+C to exit."
	PYTHONPATH=. python scripts/infer.py --config $(CONFIG) --mode interactive

merge:
	@echo "Merging LoRA adapters..."
	PYTHONPATH=. python scripts/merge_lora.py --config $(CONFIG) --adapters adapters/last --out outputs/merged_fp16 --dtype fp16

merge-bf16:
	@echo "Merging LoRA adapters (bfloat16)..."
	PYTHONPATH=. python scripts/merge_lora.py --config $(CONFIG) --adapters adapters/last --out outputs/merged_bf16 --dtype bf16

merge-test:
	@echo "Testing merged model..."
	@if [ ! -d outputs/merged_fp16 ]; then \
		echo "Error: Merged model not found. Run 'make merge' first."; \
		exit 1; \
	fi
	@echo "Testing merged model loading..."
	@python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
		print('Loading merged model...'); \
		m = AutoModelForCausalLM.from_pretrained('outputs/merged_fp16', torch_dtype='auto', device_map='auto'); \
		t = AutoTokenizer.from_pretrained('outputs/merged_fp16', use_fast=True); \
		print('✓ Model loaded successfully!'); \
		print(f'Model type: {m.config.model_type}'); \
		print(f'Pad token: {t.pad_token}'); \
		print('✓ Merged model is ready for deployment!')"

check:
	@echo "🔍 SFT-Play Sanity Check"
	@echo "========================"
	@echo ""
	@echo "Checking project setup..."
	@echo ""
	@# Check config file exists
	@if [ ! -f $(CONFIG) ]; then \
		echo "❌ Config file not found: $(CONFIG)"; \
		exit 1; \
	else \
		echo "✅ Config file found: $(CONFIG)"; \
	fi
	@echo ""
	@# Check processed data exists
	@if [ ! -f data/processed/train.jsonl ]; then \
		echo "❌ Training data not found: data/processed/train.jsonl"; \
		echo "   Run 'make process' or './workflows/quick_start.sh' first"; \
		exit 1; \
	else \
		echo "✅ Training data found: data/processed/train.jsonl"; \
		@wc -l data/processed/train.jsonl | awk '{print "   📊 Training samples: " $$1}'; \
	fi
	@echo ""
	@# Check validation data
	@if [ -f data/processed/val.jsonl ]; then \
		echo "✅ Validation data found: data/processed/val.jsonl"; \
		@wc -l data/processed/val.jsonl | awk '{print "   📊 Validation samples: " $$1}'; \
	else \
		echo "⚠️  Validation data not found: data/processed/val.jsonl"; \
	fi
	@echo ""
	@# Check test data
	@if [ -f data/processed/test.jsonl ]; then \
		echo "✅ Test data found: data/processed/test.jsonl"; \
		@wc -l data/processed/test.jsonl | awk '{print "   📊 Test samples: " $$1}'; \
	else \
		echo "⚠️  Test data not found: data/processed/test.jsonl"; \
	fi
	@echo ""
	@# Check template file
	@if [ -f chat_templates/default.jinja ]; then \
		echo "✅ Chat template found: chat_templates/default.jinja"; \
	else \
		echo "❌ Chat template not found: chat_templates/default.jinja"; \
		exit 1; \
	fi
	@echo ""
	@# Check directories
	@if [ -d outputs ] && [ -d adapters ]; then \
		echo "✅ Output directories ready"; \
	else \
		echo "⚠️  Output directories missing - run 'make setup-dirs'"; \
	fi
	@echo ""
	@echo "🎉 Sanity check completed!"
	@echo ""
	@echo "Next steps:"
	@echo "  • Run 'make train' to start training"
	@echo "  • Run 'make train-with-tb' for training with TensorBoard"
	@echo "  • Run 'make help' to see all available commands"

clean:
	@echo "Cleaning generated files..."
	rm -rf data/processed/* data/processed_with_style/* data/rendered/*
	rm -rf outputs/* adapters/*
	rm -f demo_inputs.txt
	@echo "Clean complete!"

full-pipeline: setup-dirs process style render
	@echo "Full data processing pipeline completed!"
	@echo "Next steps:"
	@echo "  1. Review processed data in data/processed_with_style/"
	@echo "  2. Run 'make check' to validate setup"
	@echo "  3. Run 'make train' to start training"
	@echo "  4. Run 'make eval' to evaluate the model"
