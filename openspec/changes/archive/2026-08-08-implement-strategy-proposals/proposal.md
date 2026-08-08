## Why

To transform the initial C++ foundation of BSM-RLI into an operational neural-symbolic engine for 1B–8B open-weights models (Llama-3.1-8B, Qwen-2.5-7B, Llama-3.2-3B), we must execute the strategic proposal pillars: (1) SIMD hardware acceleration & BigNum exact evaluation, (2) Unsloth Supervised Fine-Tuning (SFT) & GRPO preference alignment pipelines, and (3) `llama.cpp` inference engine LogitsProcessor hooks.

## What Changes

- **SIMD & BigNum Math Hardening**: Integrate AVX2 SIMD vector reductions and BigNum exact arithmetic into `src/math_kernels.cpp`.
- **Unsloth Fine-Tuning Pipeline**: Add complete Python scripts (`training/train_unsloth_sft.py` and `training/train_unsloth_grpo.py`) utilizing Unsloth QLoRA fast-patching and custom reward functions (schema precision, ground-truth match, token economy penalty).
- **Synthetic Dataset Expansion**: Expand `dataset/generate_synthetic_data.py` to generate 500+ samples covering math, string operations, graph traversals, and logic grid puzzles.
- **Inference Hook Interface**: Create `include/bsm_rli/llama_cpp_hook.hpp` providing C++ bindings for `llama.cpp` `LogitsProcessor` and KV-cache injection.

## Capabilities

### New Capabilities
- `unsloth-alignment-pipeline`: Training pipeline for 1B-8B models combining Unsloth QLoRA SFT and GRPO preference alignment.
- `llamacpp-inference-hook`: LogitsProcessor and KV-cache result token injection interface for `llama.cpp`.

### Modified Capabilities
- `math-microkernels`: Require BigNum arbitrary precision and SIMD vector reduction implementations.

## Impact
- **Training Suite**: [`training/train_unsloth_sft.py`](file:///home/liz/Projects/BSM-RLI/training/train_unsloth_sft.py), [`training/train_unsloth_grpo.py`](file:///home/liz/Projects/BSM-RLI/training/train_unsloth_grpo.py).
- **Dataset Suite**: [`dataset/generate_synthetic_data.py`](file:///home/liz/Projects/BSM-RLI/dataset/generate_synthetic_data.py).
- **C++ Headers & Source**: [`include/bsm_rli/llama_cpp_hook.hpp`](file:///home/liz/Projects/BSM-RLI/include/bsm_rli/llama_cpp_hook.hpp), [`src/math_kernels.cpp`](file:///home/liz/Projects/BSM-RLI/src/math_kernels.cpp).
