## Context

To validate BSM-RLI performance, we need a benchmark runner (`benchmarks/run_evals.py`) and token economy analyzer (`benchmarks/token_economy_eval.py`).

## Goals / Non-Goals

**Goals:**
- **Accuracy Benchmarking**: Evaluate exact precision across 4 task domains (SIMD/BigNum Math, UTF-8 Character Frequency, Dijkstra Shortest Path, SAT Constraint Solving).
- **Latency & Throughput Measuring**: Track micro-kernel execution times (< 5µs) vs network JSON REST tool calling latency (~500ms).
- **Token Economy Ratio**: Measure generated token compression ratios (~50x reduction).

**Non-Goals:**
- Human-in-the-loop manual evaluation.

## Decisions

### 1. Benchmark Task Suites
- `eval_math()`: Evaluates multi-operand array summation and product reductions.
- `eval_char_count()`: Evaluates UTF-8 character frequency scanning on string targets.
- `eval_solvers()`: Evaluates 0-1 ILP and SAT constraint satisfiability.

## Risks / Trade-offs

- **[Risk]** Variable execution environments affecting latency measurements.  
  *Mitigation*: Use high-resolution C++ / Python nanosecond timers (`std::chrono::high_resolution_clock`, `time.perf_counter_ns()`).
