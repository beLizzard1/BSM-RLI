## ADDED Requirements

### Requirement: Regular Expression & Pattern Extraction Micro-Kernels
The system SHALL provide pre-compiled C++ kernels for deterministic $O(N)$ regular expression matching (`REGEX_MATCH`), extraction (`REGEX_EXTRACT`), replacement (`REGEX_REPLACE`), and URL sanitization (`SANITIZE_URL`).

#### Scenario: Deterministic regex pattern match
- **WHEN** `REGEX_MATCH("user@example.com", "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$")` is executed
- **THEN** the system SHALL evaluate the DFA pattern match in < 5 microseconds and return true
