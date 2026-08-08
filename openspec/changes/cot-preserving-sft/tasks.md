## 1. Dataset & Curriculum Formatting

- [x] 1.1 Implement CoT reasoning trace formatting helper in dataset generation pipeline for `qwen3` and `deepseek-r1` model families.
- [x] 1.2 Regenerate curriculum targets to include `<think>...</think>` reasoning traces prior to `<|jit_start|>` triggers.

## 2. Fine-Tuning Trainer & Loss Masking

- [x] 2.1 Add response-only completion loss masking (`DataCollatorForCompletionOnlyLM`) in `training/train_enhanced_curriculum_sft.py`.
- [x] 2.2 Configure prompt template boundaries (`<|im_start|>assistant` / `<|start_header_id|>assistant`) to ensure input prompts receive `-100` label masking.

## 3. CoT-Aware Evaluation & Sweep

- [x] 3.1 Update `benchmarks/finetuned_sweep.py` to support 1024-token thinking budgets for reasoning models before scanning for `<|jit_start|>` triggers.
- [x] 3.2 Execute fine-tuning sweep on reasoning models (`qwen3-0.6b`, `qwen3-1.7b`, `deepseek-r1-qwen-1.5b`) to verify accuracy recovery.

## 4. Verification & Documentation

- [x] 4.1 Execute `benchmarks/plot_sweep_comparison.py` to generate baseline vs. CoT-preserving fine-tuned accuracy charts.
- [x] 4.2 Record comparative metrics in `benchmarks/results/finetuned_sweep.json` and update local/remote GitHub wiki documentation.
