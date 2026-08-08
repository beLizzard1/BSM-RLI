## 1. Multi-Objective Reward Function Engine

- [ ] 1.1 Create `training/grpo_reward_function.py` implementing composite rewards ($R_{\text{correctness}}$, $R_{\text{validity}}$, $R_{\text{kernel\_select}}$, $R_{\text{economy}}$).
- [ ] 1.2 Connect `grpo_reward_function.py` to `gpu_microkernels.py` and `bsm_rli_engine.cpp` for fast batched candidate evaluation.

## 2. GRPO Training Pipeline

- [ ] 2.1 Implement `training/train_grpo_kernel_rl.py` leveraging TRL's `GRPOTrainer` with 4-bit Unsloth fast patching.
- [ ] 2.2 Configure group sampling ($G=8$), token economy length penalty, and 4-bit QLoRA target adapters.

## 3. Evaluation & Sweep Verification

- [ ] 3.1 Create `benchmarks/grpo_sweep.py` to evaluate GRPO-trained policy weights on GSM8K subset with host interception.
- [ ] 3.2 Execute GRPO training and evaluation on reasoning models (`Qwen3-1.7B`, `DeepSeek-R1-1.5B`).

## 4. Documentation & GitHub Integration

- [ ] 4.1 Update benchmark plot comparison scripts (`benchmarks/plot_sweep_comparison.py`) to include GRPO vs. SFT vs. Baseline accuracy bars.
- [ ] 4.2 Record results in repository README and GitHub Wiki documentation.
