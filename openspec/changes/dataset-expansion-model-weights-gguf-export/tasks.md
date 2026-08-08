## 1. Synthetic Dataset Expansion

- [x] 1.1 Scale `dataset/generate_synthetic_data.py` to generate 5,000+ training pairs across all 10 registered micro-kernels
- [x] 1.2 Re-generate dataset file `dataset/bsm_rli_sft.json`

## 2. Automated Training & Weight Fetcher

- [x] 2.1 Create `training/run_unsloth_pipeline.py` managing HuggingFace model weight downloading and Unsloth training execution

## 3. GGUF Quantization & Export Pipeline

- [x] 3.1 Create `training/export_gguf.py` exporting merged LoRA weights to 4-bit/16-bit `.gguf` quantized format

## 4. Standalone C++ llama.cpp Runner

- [x] 4.1 Create `src/llama_runner.cpp` demonstrating C++ GGUF inference linking `libbsm_rli.so`
- [x] 4.2 Update `CMakeLists.txt` to compile `bsm_rli_llama_runner` executable
- [x] 4.3 Build and verify executable in `build/`
