## ADDED Requirements

### Requirement: GGUF Model Export Pipeline
The system SHALL provide a model export utility (`training/export_gguf.py`) that exports fine-tuned Unsloth LoRA model weights to quantized `.gguf` format.

#### Scenario: Exporting GGUF quantized model file
- **WHEN** `export_gguf.py` is executed on trained model weights
- **THEN** it SHALL merge LoRA adapters and produce `bsm_rli_model.Q4_K_M.gguf`

### Requirement: Standalone C++ llama.cpp Runner
The system SHALL provide a standalone C++ runner (`src/llama_runner.cpp`) executing GGUF model token generation with BSM-RLI `libbsm_rli.so` logit masking and host kernel dispatch.

#### Scenario: Running GGUF model with BSM-RLI logit interceptor
- **WHEN** `llama_runner` is executed with a GGUF model file and prompt
- **THEN** it SHALL clamp candidate logits to EBNF rules during JIT regions and dispatch host C++ micro-kernels in < 5 microseconds
