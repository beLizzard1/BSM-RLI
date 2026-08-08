## ADDED Requirements

### Requirement: llama.cpp Host LogitsProcessor Integration
The system SHALL provide a C++ interface (`include/bsm_rli/llama_cpp_hook.hpp`) to bind BSM-RLI region-scoped interceptors directly into `llama.cpp` sampling loops.

#### Scenario: Binding LogitsProcessor to llama.cpp token sampler
- **WHEN** `bsm_rli::LlamaCppHook::sample_and_intercept()` is invoked during token generation
- **THEN** it SHALL monitor sampled tokens for `<|jit_start|>`, apply EBNF logit masks, execute host micro-kernels upon `<|jit_end|>`, and inject ground-truth tokens into the sequence
