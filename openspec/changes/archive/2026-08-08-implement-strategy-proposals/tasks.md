## 1. Unsloth Fine-Tuning & Preference Alignment Pipeline

- [x] 1.1 Create `training/train_unsloth_sft.py` supporting 1B-8B model loading, PEFT QLoRA fast-patching, and special token registration
- [x] 1.2 Create `training/train_unsloth_grpo.py` defining multi-objective reward functions for schema precision, numerical accuracy, and token economy

## 2. Synthetic Data Expansion

- [x] 2.1 Create `dataset/generate_synthetic_data.py` generating JSON training pairs for arithmetic, character counting, bitwise logic, and Dijkstra graph search
- [x] 2.2 Execute generator to build `dataset/bsm_rli_sft.json`

## 3. llama.cpp Host LogitsProcessor Integration

- [x] 3.1 Create `include/bsm_rli/llama_cpp_hook.hpp` exposing C++ sampling hook and KV-cache result token injection interface

## 4. C++ Math Kernel Hardening

- [x] 4.1 Update `src/math_kernels.cpp` with SIMD vector reductions and BigNum exact decimal math

## 5. Verification & Tests

- [x] 5.1 Run dataset generation script and verify `dataset/bsm_rli_sft.json`
- [x] 5.2 Build C++ library and run CTest suite (`ctest`)
- [x] 5.3 Run Python unittest suite (`PYTHONPATH=bindings/python python3 -m unittest tests/test_bindings.py`)
