## MODIFIED Requirements

### Requirement: Character and Byte-level Frequency Scanning
The system SHALL provide C++ micro-kernels for exact UTF-8 character scanning (`COUNT_CHAR`), grapheme length (`LEN_CHAR`), byte reversal (`REVERSE_STR`), substring index search (`SUBSTRING_INDEX`), delimiter string joining (`CONCAT_STR`), and case transformation (`CASE_TRANSFORM`) without sub-word BPE tokenization loss.

#### Scenario: Counting target character in string
- **WHEN** `COUNT_CHAR("strawberry", "r")` is executed
- **THEN** the system SHALL return `3` in < 5 microseconds
