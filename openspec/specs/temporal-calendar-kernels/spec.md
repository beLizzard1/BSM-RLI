## ADDED Requirements

### Requirement: Temporal & Calendar Arithmetic Micro-Kernels
The system SHALL provide ISO-8601 calendar arithmetic kernels (`DATE_ADD`, `DATE_DIFF`, `DAY_OF_WEEK`, `TZ_CONVERT`) accurately handling leap years, Daylight Saving Time, and timezone offsets.

#### Scenario: Calculating day of week for date
- **WHEN** `DAY_OF_WEEK("2026-08-08")` is executed
- **THEN** the system SHALL compute the exact ISO day of week ("Saturday") in < 5 microseconds
