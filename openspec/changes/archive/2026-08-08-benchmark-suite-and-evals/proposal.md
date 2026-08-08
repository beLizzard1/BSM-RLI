## Why

To empirically demonstrate that 1B–8B parameter open-weights models equipped with BSM-RLI achieve **Symbolic Parity** with massive 70B–405B models, we need an automated benchmarking and evaluation suite measuring execution latency (< 5µs kernel dispatch), token compression ratios (~50x token savings vs CoT), and 100% mathematical accuracy.

## What Changes

- **Automated Evaluation Harness**: Implement [`benchmarks/run_evals.py`](file:///home/liz/Projects/BSM-RLI/benchmarks/run_evals.py) evaluating arithmetic, character counting, graph search, and constraint solvers.
- **Token Economy Analyzer**: Implement [`benchmarks/token_economy_eval.py`](file:///home/liz/Projects/BSM-RLI/benchmarks/token_economy_eval.py) comparing generated token counts of 2-token BSM-RLI triggers vs. 300-token CoT scratchpads.
- **Report Generator**: Export benchmark results to JSON and formatted Markdown reports ([`benchmarks/results.md`](file:///home/liz/Projects/BSM-RLI/benchmarks/results.md)).

## Capabilities

### New Capabilities
- `benchmark-eval-harness`: Automated benchmark runner evaluating accuracy, latency, and token compression ratios for BSM-RLI.

### Modified Capabilities
*(None)*

## Impact
- **Benchmark Suite**: [`benchmarks/run_evals.py`](file:///home/liz/Projects/BSM-RLI/benchmarks/run_evals.py), [`benchmarks/token_economy_eval.py`](file:///home/liz/Projects/BSM-RLI/benchmarks/token_economy_eval.py).
