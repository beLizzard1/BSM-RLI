## Context

To demonstrate BSM-RLI performance, we need: (1) an evaluation harness evaluating baseline LLM output vs. BSM-RLI intercepted output on public datasets (GSM8K, MATH, HumanEval, Strawberry), and (2) a dataset generator creating 50,000+ synthetic pairs mixed with 10,000 conversational rejection samples.

## Goals / Non-Goals

**Goals:**
- **Standard Benchmark Evaluation**: Implement `benchmarks/standard_ai_benchmarks.py` evaluating GSM8K (arithmetic accuracy), HumanEval (regex pattern matching), and Strawberry (character frequency scans).
- **Dataset Scaling (50,000+ samples)**: Scale `dataset/generate_synthetic_data.py` across 30+ micro-kernels and mix general instruction rejection pairs.

**Non-Goals:**
- Manual human evaluation.

## Decisions

### 1. Benchmark Harness Architecture
- `evaluate_gsm8k()`: Loads GSM8K test split, runs baseline generation vs. BSM-RLI JIT intercepted generation, and measures exact match accuracy.
- `evaluate_strawberry()`: Tests UTF-8 character frequency questions ("How many 'r's in strawberry?").

### 2. Dataset Mixture Ratio
- 80% BSM-RLI micro-kernel pairs (50,000 samples).
- 20% General conversational instruction pairs (10,000 rejection samples).

## Risks / Trade-offs

- **[Risk]** Over-triggering micro-kernels on general conversational prompts.  
  *Mitigation*: Include 10,000 general instruction rejection samples during SFT and GRPO fine-tuning.
