## Why

To complete the end-to-end model training and on-device deployment cycle for 1B–8B open-weights models, we must scale synthetic dataset generation to 5,000+ samples, download target HuggingFace model weights (`Llama-3.1-8B-Instruct`, `Qwen-2.5-7B`), execute Unsloth fine-tuning, and export GGUF quantized model checkpoints (`bsm-rli-llama-3.1-8b.gguf`) for `llama.cpp` single-tenant edge deployment.

## What Changes

- **Synthetic Dataset Scale-up**: Expand `dataset/generate_synthetic_data.py` to generate 5,000+ instruction-response pairs covering vector math, character frequency scans, bitwise logic, Dijkstra graph search, and SAT solver tasks.
- **Automated Training Pipeline**: Create `training/run_unsloth_pipeline.py` to manage HuggingFace weight downloading, Unsloth QLoRA SFT training, and GRPO preference alignment.
- **GGUF Quantization Export**: Create `training/export_gguf.py` to export 16-bit and 4-bit GGUF model files (`bsm-rli-llama-3.1-8b-Q4_K_M.gguf`).
- **Standalone llama.cpp C++ Runner**: Create `src/llama_runner.cpp` demonstrating end-to-end GGUF model generation with BSM-RLI `libbsm_rli.so` logit masking and host kernel dispatch.

## Capabilities

### New Capabilities
- `gguf-export-and-runner`: Quantized GGUF model export pipeline and standalone C++ `llama.cpp` runner linking `libbsm_rli.so`.

### Modified Capabilities
- `unsloth-alignment-pipeline`: Require automated HuggingFace weight downloading, multi-kernel dataset integration, and GGUF quantization export.

## Impact
- **Training & Export Suite**: [`training/run_unsloth_pipeline.py`](file:///home/liz/Projects/BSM-RLI/training/run_unsloth_pipeline.py), [`training/export_gguf.py`](file:///home/liz/Projects/BSM-RLI/training/export_gguf.py).
- **C++ Standalone Runner**: [`src/llama_runner.cpp`](file:///home/liz/Projects/BSM-RLI/src/llama_runner.cpp).
- **Build System**: [`CMakeLists.txt`](file:///home/liz/Projects/BSM-RLI/CMakeLists.txt).
