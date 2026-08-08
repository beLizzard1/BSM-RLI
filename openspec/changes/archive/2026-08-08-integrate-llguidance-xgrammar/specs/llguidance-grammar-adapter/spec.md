## ADDED Requirements

### Requirement: FSM Logit Mask Grammar Adapter
The system SHALL provide an FSM logit mask adapter (`LLGuidanceAdapter`) that parses dynamic EBNF grammars and computes token vocabulary bitmasks during sampling.

#### Scenario: Computing logit bitmask for candidate tokens
- **WHEN** `LLGuidanceAdapter::apply_logit_mask()` is called inside a JIT region
- **THEN** it SHALL set logits of invalid candidate tokens to `-infinity` while preserving logits of syntactically valid EBNF transition tokens
