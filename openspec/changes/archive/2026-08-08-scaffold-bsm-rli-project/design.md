## Context

Standard LLM inference engines incur severe latency, token generation cost, and hallucination risks when performing multi-digit math, exact character counting, state tracking, and constraint satisfaction. This is especially severe for **smaller open-weights models ($\le 8\text{B}$ parameters, such as Llama-3.1-8B, Qwen-2.5-7B, or sub-3B models)** deployed on single-tenant edge runtimes (`llama.cpp`, Apple MLX, local workstations).

BSM-RLI turns smaller open-weights models into **hybrid neural-symbolic engines**. The LLM operates as an agile, expressive intent parser, while the bare-metal host runtime acts as its instant, deterministic co-processor (< 5µs zero-IPC execution).

## Goals / Non-Goals

**Goals:**
- **Asymmetric Capability Boosting**: Enable 1B-8B models to achieve symbolic parity with 70B-405B models on math, sorting, graph traversal, and logic puzzles.
- **Single-Tenant Edge & On-Device Optimization**: Target single-tenant local inference runtimes (`llama.cpp`, local C++ host engines) where zero-IPC mid-stream kernel execution incurs zero batch-stalling / warp divergence penalties.
- **Context & Parameter Economy**: Replace multi-hundred token CoT scratchpads with compact 2-token kernel calls, preserving effective context windows and parameter capacity for reasoning.
- Provide a high-performance C++20 project structure for `BSM-RLI`:
  - **AOT Kernel Registry**: Managing pre-compiled VRAM/CPU function pointers and argument extraction.
  - **Math Micro-Kernel Suite**: Arbitrary-precision math (`MATH_EVAL_EXACT`), SIMD vector reductions (`SUM_F64`), linear algebra (`DOT_PRODUCT`), and transcendental functions (`POWER_ROOT`, `TRIG_EVAL`).
  - **Bitwise & Crypto Suite**: Register-level scalar bitwise ops (`BITWISE_OP`), hashing (`HASH_DIGEST`), base64 (`BASE64_CODEC`).
  - **State Machine Suite**: Dijkstra shortest path (`GRAPH_DIJKSTRA`), Union-Find (`UNION_FIND`).
  - **Structural Suite**: Schema validation (`VALIDATE_SCHEMA`), AST diff (`STRUCT_DIFF`).
  - **Constraint Solvers**: Embedded SAT (`SOLVE_SAT`), ILP (`SOLVE_ILP`), and SMT (`SOLVE_SMT`) solvers.
- Provide dynamic EBNF grammar generators for `llguidance` / `XGrammar` / `llama.cpp` integration.

**Non-Goals:**
- Multi-tenant cloud batching optimization (BSM-RLI targets single-tenant on-device / local edge agentic runtimes).
- JIT compilation at runtime (kernels are pre-compiled AOT C++/CUDA primitives).

## Architecture & System Flow

```
┌─────────────────────────────────────────┐
│   Small Model (e.g., Llama-3.1-8B)      │
│   Intent Parsing & Reasoner             │
└───────────────────┬─────────────────────┘
                    │ Emits <|jit_start|>
                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                          Region-Scoped Logit Intercept (BSM-RLI)                          │
├───────────────────────────────────────────────────┬───────────────────────────────────────┤
│        Symbolic Primitive Execution (< 5μs)       │       Formal Constraint Solver        │
│    (SUM_F32, COUNT_CHAR, GRAPH_DIJKSTRA, etc.)    │        (SOLVE_SMT / SOLVE_ILP)        │
└───────────────────┬───────────────────────────────┴───────────────────────────────────────┘
                    │ Injects scalar/solution into KV Cache
                    ▼
┌─────────────────────────────────────────┐
│   Small Model Resumes Generation        │
│   with Guaranteed Symbolic Truth      │
└─────────────────────────────────────────┘
```

## Decisions

### 1. Project Directory Layout & CMake Structure
- `include/bsm_rli/`: Public C++ headers (`registry.hpp`, `math_kernels.hpp`, `crypto_kernels.hpp`, `state_kernels.hpp`, `solver_kernels.hpp`, `interceptor.hpp`).
- `src/`: Core implementation files (`registry.cpp`, `math_kernels.cpp`, `crypto_kernels.cpp`, `state_kernels.cpp`, `solver_kernels.cpp`, `ebnf_generator.cpp`).
- `ebnf/`: EBNF grammar rules.
- `bindings/python/`: Python Pybind11 / CTypes wrapper.
- `tests/`: GoogleTest C++ unit tests and pytest integration tests.
- `CMakeLists.txt`: Root CMake build configuration with C++20 standard, SIMD optimization flags (`-O3 -march=native`), and optional CUDA / OpenMP backends.

### 2. Zero-IPC Argument Marshalling & KV Injection
- Arguments emitted between `<|jit_start|>` and `<|jit_end|>` are parsed using `std::string_view` zero-copy buffer views to maintain host dispatch latency below 5 microseconds. Calculated results are tokenized and written directly into the KV cache tensor.

### 3. Logit Mask State Machine
- The inference engine runs unconstrained during regular text sampling. Upon generating `<|jit_start|>`, the logit mask engine activates an EBNF Finite State Machine (FSM) constraining next-token probabilities strictly to valid kernel invocation grammars.

## Risks / Trade-offs

- **[Risk]** Model fine-tuning trigger precision for small models ($\le 8\text{B}$).  
  *Mitigation*: Synthesize dataset (Unsloth QLoRA SFT + GRPO preference optimization) to train trigger precision for `<|jit_start|>` only when exact computation is needed.
- **[Risk]** Logit masking overhead slowing down overall token generation throughput.  
  *Mitigation*: Logit masks are strictly region-scoped, staying active **only** inside `<|jit_start|>` ... `<|jit_end|>` regions.
