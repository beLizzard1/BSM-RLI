## Why

Post-sampling string parsing cannot prevent a model from generating invalid syntax midway through a kernel call. By integrating **`llguidance`** / **`XGrammar`** token-level logit masking into BSM-RLI, candidate token logits are clamped to valid EBNF grammar transitions at every step between `<|jit_start|>` and `<|jit_end|>`, guaranteeing 100% syntactically valid micro-kernel triggers.

## What Changes

- **llguidance / XGrammar FSM Integration**: Implement `include/bsm_rli/llguidance_adapter.hpp` to wrap grammar-based Finite State Machine (FSM) token matchers.
- **Logit Bitmask Compiler**: Generate next-token vocabulary bitmasks (`std::vector<bool>`) zeroing out invalid token IDs during sampling steps inside `<|jit_start|>` regions.
- **Dynamic Grammar Reloading**: Connect `KernelRegistry::generate_ebnf_grammar()` to dynamically compile active registered kernels into the FSM logit mask engine.

## Capabilities

### New Capabilities
- `llguidance-grammar-adapter`: Real-time FSM logit mask clamping engine for `llguidance` / `XGrammar`.

### Modified Capabilities
- `symbolic-logit-interceptor`: Require next-token logit clamping via FSM bitmasks during `<|jit_start|>` regions.

## Impact
- **C++ Headers & Source**: [`include/bsm_rli/llguidance_adapter.hpp`](file:///home/liz/Projects/BSM-RLI/include/bsm_rli/llguidance_adapter.hpp), [`src/ebnf_generator.cpp`](file:///home/liz/Projects/BSM-RLI/src/ebnf_generator.cpp), [`src/interceptor.cpp`](file:///home/liz/Projects/BSM-RLI/src/interceptor.cpp).
- **Inference Integration**: Connects with `llama.cpp` LogitsProcessor and `vLLM` sampling hooks.
