## Context

BSM-RLI currently relies on Supervised Fine-Tuning (SFT) over synthetic template pairs. While SFT achieves format convergence, it caps accuracy at 24%–46% across 1B–2B reasoning models (Qwen3-1.7B, DeepSeek-R1-1.5B), down from 84%–94% baseline CoT reasoning.

To bridge this gap, we adopt **Group Relative Policy Optimization (GRPO)** (Shao et al., 2024 - DeepSeek-R1 methodology). GRPO replaces teacher-forcing with policy sampling ($G=8$ candidate outputs per prompt). We pass candidates to our C++/CUDA micro-kernel engine, which returns a scalar reward composite.

## Goals / Non-Goals

**Goals:**
- Implement `training/train_grpo_kernel_rl.py` using `trl.GRPOTrainer` and Unsloth fast patching.
- Build a Python/C++ reward evaluator (`training/grpo_reward_function.py`) providing composite rewards:
  - $R_{\text{correctness}}$: $+1.0$ if host C++ execution matches ground truth, else $-1.0$.
  - $R_{\text{validity}}$: $+0.3$ for well-formed `<|jit_start|>` syntax.
  - $R_{\text{kernel\_select}}$: $+0.5$ for direct SIMD kernels over generic C++ loops.
  - $R_{\text{economy}}$: $+0.4 \times (1 - \frac{N_{\text{tokens}}}{N_{\text{max}}})$ for early triggering.
- Demonstrate GRPO policy recovery to >75% GSM8K accuracy while maintaining sub-20 token generation.

**Non-Goals:**
- Replacing the baseline C++ logit interceptor engine (the host runtime remains unchanged).
- Full PPO with critic model (GRPO computes relative advantages across group samples $G=8$ without a separate critic network).

## Decisions

### Decision 1: GRPO over PPO / DPO
- **Rationale**: PPO requires a separate Critic model, doubling VRAM footprint (~12GB VRAM limit on RTX 4070 Ti). DPO requires static pair collection. GRPO computes relative advantage across $G=8$ candidate completions generated in parallel, fitting easily within 12GB VRAM using 4-bit LoRA.

### Decision 2: Batched Host C++ Reward Execution
- **Rationale**: Evaluating 8 candidate completions per prompt across a micro-batch of 4 prompts (32 samples) requires ultra-fast execution. We bind python `gpu_microkernels.py` and C++ `bsm_rli_engine.cpp` directly into `grpo_reward_function.py` for sub-millisecond multi-sample execution.

## Risks / Trade-offs

- **[Risk] Reward Hacking (Generating empty/random triggers)** → *Mitigation*: Strict $R_{\text{correctness}}$ penalty ($-1.0$) whenever execution fails or returns an incorrect answer.
- **[Risk] Collapse of Native Thinking Workspace** → *Mitigation*: Format reward $R_{\text{validity}}$ only activates if `<|jit_start|>` is preceded by a valid `<think>` block for reasoning models.
