## Context

To fulfill the complete BSM-RLI micro-kernel specification across 6 operational domains, we must implement all 30+ C++ micro-kernel primitives inside `libbsm_rli.so` and register them in `KernelRegistry`.

## Goals / Non-Goals

**Goals:**
- **Array & Vector Aggregations**: `SUM_F32`, `SUM_INT`, `AVG_F32`, `STD_DEV_F32`, `MIN_MAX_F32`, `PRODUCT_F32`, `PERCENT_DELTA`.
- **Character & String Micro-Primitives**: `COUNT_CHAR`, `LEN_CHAR`, `REVERSE_STR`, `SUBSTRING_INDEX`, `CONCAT_STR`, `CASE_TRANSFORM`.
- **Regular Expression & Pattern Extraction**: `REGEX_MATCH`, `REGEX_EXTRACT`, `REGEX_REPLACE`, `SANITIZE_URL`.
- **Temporal & Calendar Arithmetic**: `DATE_ADD`, `DATE_DIFF`, `DAY_OF_WEEK`, `TZ_CONVERT`.
- **Precise Scalar Math & Units**: `EVAL_EXPR`, `UNIT_CONVERT`, `ROUND_PREC`.
- **Higher-Order Cognitive & Algorithmic Extensions**: `GRAPH_DIJKSTRA`, `UNION_FIND`, `MEMOIZED_DP`, `VALIDATE_SCHEMA`, `STRUCT_DIFF`, `SQL_CANONICALIZE`, `BITWISE_OP`, `HASH_DIGEST`, `BASE64_CODEC`, `SORT_ARRAY`, `SET_OPERATION`, `TOP_K_RANK`, `SOLVE_SAT`, `SOLVE_ILP`, `SOLVE_SMT`.

**Non-Goals:**
- Heavy external Python process runtimes. All kernels run in bare-metal C++ with zero-IPC overhead.

## Decisions

### 1. Modular Header & Kernel Structure
- Organise kernels into functional domain headers:
  - `include/bsm_rli/math_kernels.hpp` & `src/math_kernels.cpp`
  - `include/bsm_rli/string_kernels.hpp` & `src/string_kernels.cpp`
  - `include/bsm_rli/regex_kernels.hpp` & `src/regex_kernels.cpp`
  - `include/bsm_rli/temporal_kernels.hpp` & `src/temporal_kernels.cpp`
  - `include/bsm_rli/scalar_kernels.hpp` & `src/scalar_kernels.cpp`
  - `include/bsm_rli/algo_kernels.hpp` & `src/algo_kernels.cpp`

## Risks / Trade-offs

- **[Risk]** Memory footprint overhead of compiling 30+ kernels.  
  *Mitigation*: Pre-compile as static inline functions and lightweight shared library primitives keeping `libbsm_rli.so` under 5MB.
