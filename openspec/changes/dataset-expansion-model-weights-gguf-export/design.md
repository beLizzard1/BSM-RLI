## Context

To transition BSM-RLI from C++ library components into a deployed edge solution, we need: (1) scaled synthetic training dataset generation (5,000+ samples), (2) automated HuggingFace weight fetching and Unsloth training execution, (3) GGUF model export, and (4) standalone C++ `llama.cpp` runner execution (`src/llama_runner.cpp`).

## Goals / Non-Goals

**Goals:**
- **5,000+ Sample Dataset Generation**: Scale `dataset/generate_synthetic_data.py` to produce comprehensive training sets covering all 10 micro-kernels.
- **Automated Training Pipeline**: Provide `training/run_unsloth_pipeline.py` managing model loading, SFT, and GRPO preference alignment.
- **GGUF Export Pipeline**: Provide `training/export_gguf.py` exporting merged `.gguf` weights (`Q4_K_M`, `f16`).
- **Standalone C++ Runner**: Create `src/llama_runner.cpp` demonstrating `llama.cpp` GGUF inference with `libbsm_rli.so`.

**Non-Goals:**
- Hosting model weights in git repositories (weights remain on HuggingFace and local build output caches).

## Decisions

### 1. Training Pipeline Architecture
- `run_unsloth_pipeline.py` downloads `unsloth/Meta-Llama-3.1-8B-Instruct` or `unsloth/Qwen2.5-7B-Instruct` using HuggingFace `from_pretrained`.
- Uses `FastLanguageModel.save_pretrained_gguf("bsm_rli_model", tokenizer, quantization_method="q4_k_m")` to convert trained LoRA adapter weights directly into standalone GGUF files.

### 2. Standalone C++ Runner Layout
- `src/llama_runner.cpp` initializes `KernelRegistry`, `LlamaCppHook`, and streams token outputs, demonstrating sub-5µs host kernel dispatch on GGUF models.

## Risks / Trade-offs

- **[Risk]** GPU memory consumption during model weight downloading.  
  *Mitigation*: Unsloth uses 4-bit quantization and 16-bit LoRA adapters, keeping VRAM usage under 8GB.
