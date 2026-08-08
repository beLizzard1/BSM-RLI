## 1. llguidance Adapter Header

- [x] 1.1 Create `include/bsm_rli/llguidance_adapter.hpp` defining `LLGuidanceAdapter` FSM state machine and logit masking interface

## 2. Adapter Implementation & Logit Mask Clamping

- [x] 2.1 Create `src/llguidance_adapter.cpp` implementing vocabulary bitmask generation and `-infinity` logit clamping

## 3. EBNF Generator & Interceptor Integration

- [x] 3.1 Update `src/ebnf_generator.cpp` to output FSM-compatible grammar rules for `llguidance`
- [x] 3.2 Update `include/bsm_rli/interceptor.hpp` and `src/interceptor.cpp` to integrate `LLGuidanceAdapter`

## 4. CMake & Unit Tests

- [x] 4.1 Update `CMakeLists.txt` to include `src/llguidance_adapter.cpp` in `libbsm_rli`
- [x] 4.2 Create `tests/test_llguidance.cpp` testing FSM logit clamping and valid/invalid token filtering
- [x] 4.3 Build and verify GoogleTest suite (`ctest`)
