## ADDED Requirements

### Requirement: Arbitrary-Precision and SIMD Math Micro-Kernels
The system SHALL provide pre-compiled SIMD vector reductions (`SUM_F64`, `PRODUCT_F64`), high-precision math (`MATH_EVAL_EXACT`), and linear algebra operations (`DOT_PRODUCT`).

#### Scenario: Evaluating arbitrary precision expression
- **WHEN** `MATH_EVAL_EXACT("((1245.89 * 0.15) + 982.10) / 1.05")` is executed
- **THEN** the system SHALL compute the exact decimal result without floating-point rounding errors

#### Scenario: Running single-pass SIMD vector sum
- **WHEN** `SUM_F64([12.5, 45.0, 100.25])` is executed
- **THEN** the system SHALL utilize SIMD intrinsics to perform vector reduction and return `157.75`

### Requirement: Transcendental and Statistical Micro-Kernels
The system SHALL provide microsecond evaluation for transcendental operations (`POWER_ROOT`, `TRIG_EVAL`, `LOG_EVAL`) and statistical summaries (`STATS_SUMMARY`).

#### Scenario: Executing fast statistical summary
- **WHEN** `STATS_SUMMARY([10, 20, 30, 40, 50])` is executed
- **THEN** the system SHALL return mean, median, variance, std_dev, min, and max in a single pass
