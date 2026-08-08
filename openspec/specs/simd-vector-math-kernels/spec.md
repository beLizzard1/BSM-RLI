## MODIFIED Requirements

### Requirement: Multi-Operand Vector Math Micro-Kernels
The system SHALL provide pre-compiled SIMD-accelerated micro-kernels for 32-bit/64-bit float summation (`SUM_F32`, `SUM_F64`), integer summation (`SUM_INT`), mean (`AVG_F32`), standard deviation (`STD_DEV_F32`), min/max reduction (`MIN_MAX_F32`), product multiplication (`PRODUCT_F32`, `PRODUCT_F64`), dot product (`DOT_PRODUCT`), exact decimal arithmetic (`MATH_EVAL_EXACT`), percentage delta (`PERCENT_DELTA`), and statistical summaries (`STATS_SUMMARY`).

#### Scenario: Vector float summation
- **WHEN** `SUM_F64([10.5, 20.25, 30.0])` is executed
- **THEN** the system SHALL compute the exact IEEE 754 summation and return `60.75` in < 5 microseconds
