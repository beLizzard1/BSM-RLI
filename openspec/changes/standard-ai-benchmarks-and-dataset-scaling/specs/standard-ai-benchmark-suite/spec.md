## ADDED Requirements

### Requirement: Standard AI Benchmark Evaluation Suite
The system SHALL provide an automated evaluation harness (`benchmarks/standard_ai_benchmarks.py`) evaluating accuracy, TTFT latency, tokens/sec generation throughput, and context compression ratios on GSM8K, HumanEval, and Strawberry benchmark datasets.

#### Scenario: Evaluating GSM8K arithmetic accuracy
- **WHEN** `benchmarks/standard_ai_benchmarks.py` is executed on GSM8K problems
- **THEN** it SHALL compare baseline CoT accuracy vs. BSM-RLI intercepted execution accuracy and export results to Markdown
