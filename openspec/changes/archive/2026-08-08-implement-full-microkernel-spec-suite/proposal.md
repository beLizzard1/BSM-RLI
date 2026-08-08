## Why

Small language models ($\le 8\text{B}$) struggle with sub-word BPE tokenization (e.g., character counts), context drift over array reductions, floating-point rounding errors, leap-year date math, and regex backtracking. Implementing the complete BSM-RLI 6-domain specification in bare-metal C++ guarantees sub-5µs execution latency, zero context drift, and 100% deterministic mathematical soundness.

## What Changes

- **Array & Vector Aggregations**: Implement `SUM_F32`, `SUM_INT`, `AVG_F32`, `STD_DEV_F32`, `MIN_MAX_F32`, `PRODUCT_F32`, `PERCENT_DELTA`.
- **Character & String Micro-Primitives**: Implement `COUNT_CHAR`, `LEN_CHAR`, `REVERSE_STR`, `SUBSTRING_INDEX`, `CONCAT_STR`, `CASE_TRANSFORM`.
- **Regular Expression & Pattern Extraction**: Implement `REGEX_MATCH`, `REGEX_EXTRACT`, `REGEX_REPLACE`, `SANITIZE_URL` using $O(N)$ DFA regex matching.
- **Temporal & Calendar Arithmetic**: Implement `DATE_ADD`, `DATE_DIFF`, `DAY_OF_WEEK`, `TZ_CONVERT` with ISO-8601 leap year and DST handling.
- **Precise Scalar Math & Units**: Implement `EVAL_EXPR`, `UNIT_CONVERT`, `ROUND_PREC` for dimensional unit analysis and fixed-precision rounding.
- **Higher-Order Cognitive & Algorithmic Extensions**: Implement `GRAPH_DIJKSTRA`, `UNION_FIND`, `MEMOIZED_DP`, `VALIDATE_SCHEMA`, `STRUCT_DIFF`, `SQL_CANONICALIZE`, `BITWISE_OP`, `HASH_DIGEST`, `BASE64_CODEC`, `SORT_ARRAY`, `SET_OPERATION`, `TOP_K_RANK`, `SOLVE_SAT`, `SOLVE_ILP`, `SOLVE_SMT`.

## Capabilities

### New Capabilities
- `regex-pattern-kernels`: Bounded $O(N)$ regex pattern matching, extraction, replacement, and URL sanitization.
- `temporal-calendar-kernels`: ISO-8601 calendar arithmetic, date diffs, day of week calculation, and timezone conversion.
- `scalar-unit-kernels`: Precision scalar evaluation, dimensional unit conversion, and rounding.
- `algorithmic-higher-order-kernels`: Dynamic programming grids, schema validation, AST structural diffs, SQL normalization, sorting, and set operations.

### Modified Capabilities
- `simd-vector-math-kernels`: Require `SUM_F32`, `SUM_INT`, `AVG_F32`, `STD_DEV_F32`, `MIN_MAX_F32`, `PRODUCT_F32`, `PERCENT_DELTA`.
- `string-frequency-kernels`: Require `LEN_CHAR`, `REVERSE_STR`, `SUBSTRING_INDEX`, `CONCAT_STR`, `CASE_TRANSFORM`.

## Impact
- **C++ Headers & Source**: [`include/bsm_rli/`](file:///home/liz/Projects/BSM-RLI/include/bsm_rli/), [`src/`](file:///home/liz/Projects/BSM-RLI/src/).
- **Build System**: [`CMakeLists.txt`](file:///home/liz/Projects/BSM-RLI/CMakeLists.txt).
