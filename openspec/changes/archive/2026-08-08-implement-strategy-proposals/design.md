## Context

To turn BSM-RLI into a production-ready neural-symbolic engine for 1B–8B parameter open-weights models, we must implement three primary technical pillars: (1) Unsloth SFT & GRPO training pipelines, (2) `llama.cpp` C++ inference engine hooks, and (3) SIMD & BigNum math kernel hardening.

## Goals / Non-Goals

**Goals:**
- **Unsloth Fine-Tuning Pipeline**: Provide complete Python scripts (`training/train_unsloth_sft.py` and `training/train_unsloth_grpo.py`) using 4-bit QLoRA fast-patching.
- **GRPO Preference Alignment**: Define multi-objective reward functions prioritizing schema accuracy, exact result match, and token economy.
- **Synthetic Data Generation**: Expand `dataset/generate_synthetic_data.py` to produce structured JSON training datasets (`bsm_rli_sft.json`).
- **llama.cpp Inference Hook**: Provide `include/bsm_rli/llama_cpp_hook.hpp` allowing host C++ inference engines to bind BSM-RLI to token generation loops.
- **SIMD & BigNum Math**: Harden `src/math_kernels.cpp` with SIMD vector reductions and exact decimal math.

**Non-Goals:**
- Multi-node distributed cloud RPC.
- Fine-tuning models > 8B parameters.

## Decisions

### 1. Unsloth QLoRA & GRPO Pipeline
- Use `unsloth` for 4-bit PEFT QLoRA fast patching (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- Register `<|jit_start|>` and `<|jit_end|>` as special tokens in the tokenizer and resize model embeddings.
- Implement 3 GRPO reward functions:
  - `reward_jit_trigger_format`: Rewards valid EBNF trigger formatting (+1.5).
  - `reward_ground_truth_match`: Rewards exact match with ground-truth execution (+2.0).
  - `reward_token_economy`: Rewards concise <30 token trigger calls (+1.0) and penalizes >150 token CoT bloat (-0.5).

### 2. llama.cpp C++ Hook Interface
- `bsm_rli::LlamaCppHook` wraps `llama_sample_token` and `llama_kv_cache_seq_add`.
- Unconstrained during standard sampling; clamps candidate logits to EBNF rules inside JIT regions.

## Risks / Trade-offs

- **[Risk]** Large dataset generation slowing down training setup.  
  *Mitigation*: Pre-generate synthetic dataset to `dataset/bsm_rli_sft.json` for deterministic, fast SFT loading.
- **[Risk]** Reward conflict in GRPO alignment.  
  *Mitigation*: Balance reward weights so token economy does not suppress necessary text context surrounding the kernel invocation.
