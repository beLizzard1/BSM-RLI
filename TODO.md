# BSM-RLI Project TODO & Immediate Next Steps

---

## 1. 🚀 OpenAI Batches API Frontier Distillation (`gpt-5.6-luna`)

- [ ] **Launch Full 10,000-Item Distillation Batch**:
  - Run the updated distillation script (configured with `max_completion_tokens` and default temperature):
    ```bash
    python3 dataset/distill_batch_api.py --model_name gpt-5.6-luna --num_samples 10000 --max_cost_usd 10.00
    ```
- [ ] **Monitor & Download Batch Output**:
  - Check batch completion status and automatically run C++ host verification on all outputs:
    ```bash
    python3 dataset/distill_batch_api.py --check_batch <batch_id>
    ```
  - Verify saved output file at `dataset/bsm_rli_distilled_batch_10k.json`.

---

## 2. ⚡ GRPO Reinforcement Learning Policy Training (`grpo-kernel-rl`)

- [ ] **Execute GRPO Training Runs**:
  - Run Group Relative Policy Optimization (GRPO) for target edge SLMs (`Qwen3-1.7B` and `DeepSeek-R1-1.5B`):
    ```bash
    python3 training/train_grpo_kernel_rl.py \
      --model_name unsloth/Qwen3-1.7B-bnb-4bit \
      --output_dir models/grpo/qwen3-1.7b-4bit \
      --max_steps 50
    ```
- [ ] **Evaluate & Benchmark GRPO Adapters**:
  - Execute GRPO evaluation sweep:
    ```bash
    python3 benchmarks/grpo_sweep.py
    ```
  - Generate 3-bar comparative plots (Base CoT vs. SFT vs. GRPO RL):
    ```bash
    python3 benchmarks/plot_sweep_comparison.py
    ```
- [ ] **Archive OpenSpec Change**:
  - Mark tasks complete and archive `openspec/changes/grpo-kernel-rl/`.

---

## 3. 📚 Documentation & Distilled Dataset Fine-Tuning

- [ ] **Train SLMs on Distilled Frontier Dataset**:
  - Fine-tune `Llama-3.2-3B` and `Qwen3-1.7B` on `dataset/bsm_rli_distilled_batch_10k.json` once batch downloads complete.
- [ ] **Update README & Wiki Benchmarks**:
  - Push updated GRPO benchmark figures to GitHub repo `main` and remote Wiki `master`.
