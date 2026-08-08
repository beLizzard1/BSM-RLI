## 1. Project Directory Scaffolding & CMake Configuration

- [x] 1.1 Create directory tree (`include/bsm_rli/`, `src/`, `ebnf/`, `bindings/python/`, `tests/`)
- [x] 1.2 Create `CMakeLists.txt` with C++20 standard, SIMD optimization flags, GoogleTest, and Pybind11 integration

## 2. C++ Core Header Definitions

- [x] 2.1 Create `include/bsm_rli/registry.hpp` defining `KernelRegistry`, function pointer signatures, and argument structs
- [x] 2.2 Create `include/bsm_rli/math_kernels.hpp` for exact, SIMD, vector, transcendental, and statistical math signatures
- [x] 2.3 Create `include/bsm_rli/crypto_kernels.hpp` for bitwise logic, hashing, and base64 codecs
- [x] 2.4 Create `include/bsm_rli/state_kernels.hpp` for Dijkstra, Union-Find, and structural schema kernels
- [x] 2.5 Create `include/bsm_rli/solver_kernels.hpp` for SAT, ILP, and SMT constraint solver interfaces
- [x] 2.6 Create `include/bsm_rli/interceptor.hpp` for region-scoped logit state machine and zero-copy memory dispatch

## 3. C++ Core Kernel Implementations

- [x] 3.1 Implement `src/registry.cpp` for thread-safe kernel registration, lookup, and string view argument decoding
- [x] 3.2 Implement `src/math_kernels.cpp` with SIMD vector reductions (`SUM_F64`), arbitrary precision math (`MATH_EVAL_EXACT`), `DOT_PRODUCT`, and `STATS_SUMMARY`
- [x] 3.3 Implement `src/crypto_kernels.cpp` with scalar register bitwise logic (`BITWISE_OP`), SHA256/BLAKE3 (`HASH_DIGEST`), and base64
- [x] 3.4 Implement `src/state_kernels.cpp` for Dijkstra graph traversal (`GRAPH_DIJKSTRA`), Union-Find, and schema validation
- [x] 3.5 Implement `src/solver_kernels.cpp` for SAT/ILP/SMT logic solver wrappers
- [x] 3.6 Implement `src/interceptor.cpp` for parsing JIT tags and handling KV cache token injection callbacks

## 4. EBNF Logit Grammar Engine

- [x] 4.1 Create `ebnf/base_grammar.ebnf` defining `<|jit_start|>` and `<|jit_end|>` trigger tokens and parameter primitives
- [x] 4.2 Implement `src/ebnf_generator.cpp` to dynamically compile registered kernel signatures into `llguidance` / `XGrammar` compatible EBNF rules

## 5. Python Bindings & Host Integration

- [x] 5.1 Implement `bindings/python/bsm_rli_py.cpp` exposing C++ kernel registry and interceptor to Python via Pybind11 / CTypes
- [x] 5.2 Implement `bindings/python/bsm_rli/__init__.py` and `pyproject.toml` package configuration

## 6. Test Suite & Verification

- [x] 6.1 Create C++ unit test suite (`tests/test_registry.cpp`, `tests/test_math.cpp`, `tests/test_crypto.cpp`, `tests/test_solvers.cpp`)
- [x] 6.2 Build C++ binary and run tests via `ctest`
- [x] 6.3 Implement pytest suite (`tests/test_bindings.py`) to verify Python wrapper execution
