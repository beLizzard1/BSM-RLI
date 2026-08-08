## ADDED Requirements

### Requirement: Scalar Math and Unit Conversion Micro-Kernels
The system SHALL provide precise scalar formula evaluation (`EVAL_EXPR`), dimensional unit conversion (`UNIT_CONVERT`), and fixed precision rounding (`ROUND_PREC`).

#### Scenario: Dimensional unit conversion
- **WHEN** `UNIT_CONVERT(100.0, "lbs", "kg")` is executed
- **THEN** the system SHALL compute the exact conversion result (45.3592 kg) in < 5 microseconds
