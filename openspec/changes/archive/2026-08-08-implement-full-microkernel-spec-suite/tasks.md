## 1. Array & Vector Aggregations

- [x] 1.1 Implement `SUM_F32`, `SUM_INT`, `AVG_F32`, `STD_DEV_F32`, `MIN_MAX_F32`, `PRODUCT_F32`, and `PERCENT_DELTA` in `src/math_kernels.cpp`

## 2. Character & String Micro-Primitives

- [x] 2.1 Implement `LEN_CHAR`, `REVERSE_STR`, `SUBSTRING_INDEX`, `CONCAT_STR`, and `CASE_TRANSFORM` in `src/state_kernels.cpp`

## 3. Regular Expression & Pattern Extraction

- [x] 3.1 Create `include/bsm_rli/regex_kernels.hpp` and `src/regex_kernels.cpp` implementing `REGEX_MATCH`, `REGEX_EXTRACT`, `REGEX_REPLACE`, and `SANITIZE_URL`

## 4. Temporal & Calendar Arithmetic

- [x] 4.1 Create `include/bsm_rli/temporal_kernels.hpp` and `src/temporal_kernels.cpp` implementing `DATE_ADD`, `DATE_DIFF`, `DAY_OF_WEEK`, and `TZ_CONVERT`

## 5. Precise Scalar Math & Units

- [x] 5.1 Create `include/bsm_rli/scalar_kernels.hpp` and `src/scalar_kernels.cpp` implementing `EVAL_EXPR`, `UNIT_CONVERT`, and `ROUND_PREC`

## 6. Higher-Order Cognitive & Algorithmic Extensions

- [x] 6.1 Create `include/bsm_rli/algo_kernels.hpp` and `src/algo_kernels.cpp` implementing `MEMOIZED_DP`, `VALIDATE_SCHEMA`, `STRUCT_DIFF`, `SQL_CANONICALIZE`, `SORT_ARRAY`, `SET_OPERATION`, and `TOP_K_RANK`

## 7. Build System & Unit Tests

- [x] 7.1 Update `CMakeLists.txt` to compile all kernel modules into `libbsm_rli`
- [x] 7.2 Create comprehensive unit test suite `tests/test_full_spec_suite.cpp` and verify with `ctest`
