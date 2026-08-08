## Why

To rigorously validate BSM-RLI capability boosting on 1B–8B parameter open-weights models, we must evaluate on standard public AI benchmarks (GSM8K grade school math, HumanEval coding/regex, BIG-bench logic puzzles, and Strawberry character counts). Furthermore, scaling the training dataset to 50,000+ samples mixed with general conversational instruction pairs guarantees high micro-kernel dispatch accuracy without degrading conversational fluency.

## What Changes

- **Standard AI Evaluation Suite**: Implement [`benchmarks/standard_ai_benchmarks.py`](file:///home/liz/Projects/BSM-RLI/benchmarks/standard_ai_benchmarks.py) running GSM8K, HumanEval, and Strawberry evaluations comparing baseline LLM inference vs. BSM-RLI intercepted inference.
- **50,000+ Sample Dataset Generation**: Scale `dataset/generate_synthetic_data.py` to produce 50,000 micro-kernel instruction pairs across all 30+ primitives in 6 operational domains.
- **10,000 Rejection Sample Mixing**: Mix general instruction-response pairs (where `<|jit_start|>` must NOT be emitted) to prevent over-triggering.

## Capabilities

### New Capabilities
- `standard-ai-benchmark-suite`: Standard AI evaluation harness for GSM8K, MATH, HumanEval, and Strawberry character counts.
- `dataset-scaling-and-alignment`: 50,000+ synthetic instruction dataset scaling mixed with non-triggering general conversation pairs.

### Modified Capabilities
*(None)*

## Impact
- **Benchmark Suite**: [`benchmarks/standard_ai_benchmarks.py`](file:///home/liz/Projects/BSM-RLI/benchmarks/standard_ai_benchmarks.py).
- **Dataset Generator**: [`dataset/generate_synthetic_data.py`](file:///home/liz/Projects/BSM-RLI/dataset/generate_synthetic_data.py).
