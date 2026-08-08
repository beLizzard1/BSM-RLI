## Why

Supervised Fine-Tuning (SFT) teacher-forcing capping accuracy at 24%–46% on reasoning Small Language Models (Qwen3-1.7B, DeepSeek-R1-1.5B) compared to 84%–94% baseline Chain-of-Thought performance. SFT restricts the model's natural reasoning search space and over-fits to synthetic template strings.

Transitioning to Reinforcement Learning via Group Relative Policy Optimization (GRPO) allows reasoning SLMs to generate candidate reasoning paths freely, rewarding policy weights based on host C++/CUDA micro-kernel execution accuracy, format validity, kernel selection directness, and early token offloading.

## What Changes

- **GRPO Policy Trainer**: Implement `training/train_grpo_kernel_rl.py` leveraging TRL's `GRPOTrainer` with multi-candidate sampling ($G=8$).
- **Host C++ Execution Reward Pipeline**: Integrate live C++/CUDA micro-kernel execution into the reward evaluation loop to verify bit-exact correctness ($R_{\text{correctness}}$).
- **Format & Delimiter Parsing Reward**: Evaluate valid `<|jit_start|>` and `<|jit_end|>` trigger syntax ($R_{\text{validity}}$).
- **Optimum Kernel Directness Reward**: Prefer direct specialized SIMD micro-kernels (`SUM_F64`) over verbose multi-line dynamic code (`DYN_CPP`) ($R_{\text{kernel\_select}}$).
- **Token Economy Efficiency Reward**: Reward early micro-kernel triggering within the `<think>` workspace to compress context consumption ($R_{\text{economy}}$).

## Capabilities

### New Capabilities
- `grpo-kernel-rl`: Group Relative Policy Optimization pipeline evaluating live C++/CUDA host micro-kernel execution rewards, optimum kernel selection, and token economy incentives.

### Modified Capabilities
- None.

## Impact

- `training/train_grpo_kernel_rl.py`: New GRPO training entry point.
- `kernels/gpu_microkernels.py` & `src/bsm_rli_engine.cpp`: Exposed reward verification bindings for batched candidate traces.
- `benchmarks/grpo_sweep.py`: New comparative evaluation script for GRPO-trained policy weights.
