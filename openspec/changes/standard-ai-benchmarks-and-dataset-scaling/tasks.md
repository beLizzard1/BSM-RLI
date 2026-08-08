## 1. Standard AI Benchmark Harness

- [x] 1.1 Create `benchmarks/standard_ai_benchmarks.py` evaluating GSM8K math accuracy, HumanEval regex extraction, and Strawberry character counts comparing baseline LLM output vs. BSM-RLI intercepted output

## 2. Dataset Generator Scaling (50,000+ Samples)

- [x] 2.1 Update `dataset/generate_synthetic_data.py` to generate 50,000 synthetic micro-kernel instruction pairs across all 30+ primitives in 6 operational domains
- [x] 2.2 Add 10,000 conversational rejection instruction pairs (where `<|jit_start|>` must NOT be emitted) to prevent over-triggering
- [x] 2.3 Re-generate dataset file `dataset/bsm_rli_sft_50k.json`

## 3. Benchmark Execution & Evaluation Report Export

- [x] 3.1 Execute `benchmarks/standard_ai_benchmarks.py` and export report to `benchmarks/standard_eval_results.md`
