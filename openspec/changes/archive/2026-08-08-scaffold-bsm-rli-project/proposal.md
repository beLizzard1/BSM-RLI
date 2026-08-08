## Why

Standard LLMs suffer from fundamental arithmetic errors, tokenizer sub-word splitting artifacts, and heavy token consumption when simulating step-by-step state tracking or constraint solving via Chain-of-Thought (CoT). Bare-Metal Symbolic Micro-Kernels via Region-Scoped Logit Interception (BSM-RLI) addresses this by intercepting generation at specified `<|jit_start|>` regions with EBNF logit masks, executing pre-compiled host C++/CUDA micro-kernels in < 5µs with zero IPC overhead, and injecting exact ground-truth tokens directly into the LLM context.

## What Changes

- Scaffold the core `BSM-RLI` codebase architecture in C++ (C++20) and Python.
- Establish the **AOT Kernel Registry** for dynamic pointer registration and EBNF grammar generation.
- Implement the **Mathematical Micro-Kernel Suite** (arbitrary precision `MATH_EVAL_EXACT`, SIMD reductions `SUM_F64`/`PRODUCT_F64`, linear algebra `DOT_PRODUCT`, and transcendental functions `POWER_ROOT`/`TRIG_EVAL`).
- Implement the **Bitwise & Cryptographic Suite** (`BITWISE_OP`, `HASH_DIGEST`, `BASE64_CODEC`).
- Implement the **State Machine & Structural Suite** (`GRAPH_DIJKSTRA`, `UNION_FIND`, `VALIDATE_SCHEMA`, `STRUCT_DIFF`).
- Implement the **Constraint Solver Suite** (`SOLVE_SAT`, `SOLVE_ILP`, `SOLVE_SMT` wrappers).
- Provide EBNF logit grammar generators for `llguidance` / `XGrammar` / `llama.cpp` integration.

## Capabilities

### New Capabilities
- `kernel-registry`: C++ registry engine managing AOT function pointers, argument decoding, and zero-copy result buffer serialization.
- `math-microkernels`: SIMD-accelerated, high-precision, vector, statistical, and transcendental math primitives.
- `symbolic-logit-interceptor`: Region-scoped logit mask state machine for detecting `<|jit_start|>` triggers and enforcing EBNF schemas.
- `constraint-solver-kernels`: Embedded SAT, ILP, and SMT constraint solving primitives wrapping C++ solver backends.

### Modified Capabilities
*(None - fresh project repository initialization)*

## Impact
- **C++ Engine**: Produces `libbsm_rli.so` / `libbsm_rli.a` library under `src/` and headers under `include/bsm_rli/`.
- **Python Bindings**: Exposes C++ primitives to Python host runtimes via Pybind11 / CTypes under `bindings/python/`.
- **Inference Integration**: Integrates directly with `llama.cpp` LogitsProcessor and `vLLM` / `SGLang` sampling hooks.
- **Dependencies**: Requires standard C++20 compiler (`g++` / `clang++`), CMake >= 3.22, SIMD intrinsics (`AVX2`/`AVX-512`/`NEON`), and optional CUDA / OpenMP backends.
