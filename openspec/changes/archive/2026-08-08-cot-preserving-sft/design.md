## Context

Empirical baseline vs. fine-tuning sweeps revealed a critical divergence in SLM behavior under BSM-RLI SFT:
- **Standard SLMs** (SmolLM2-135M / 360M) improve significantly (+18.0pp) under instant JIT trigger emission.
- **Reasoning SLMs** (Qwen3-1.7B, DeepSeek-R1-1.5B) suffer catastrophic accuracy drops under naive SFT (Qwen3-1.7B drops from 94.0% to 2.0%) because forcing an immediate `<|jit_start|>` trigger bypasses their internal `<think>...</think>` cognitive workspace.

This design introduces **CoT-Preserving SFT** to allow reasoning models to retain their internal thinking phase prior to emitting a structured JIT micro-kernel trigger.

```
                  NAIVE SFT (Causes Reasoning Collapse)
  [User Prompt] ────────► <|jit_start|>SUM_F64(...)<|jit_end|> (No thinking workspace!)

              CoT-PRESERVING SFT (Proposed Architecture)
  [User Prompt] ────────► <think>
                          1. Extract operands
                          2. Identify operation SUM_F64
                          </think>
                          <|jit_start|>SUM_F64(221.17, 463.49, 445.91, 465.58)<|jit_end|>
                          └───────────── Host Interception → 1596.15 ────────────┘
```

## Goals / Non-Goals

**Goals:**
- Formulate CoT-preserving curriculum formatting for reasoning model families (`qwen3`, `deepseek-r1`).
- Implement loss-masking collator in `training/train_enhanced_curriculum_sft.py` to mask out user prompts and focus gradient updates on `<think>` traces and `<|jit_start|>` triggers.
- Update `benchmarks/finetuned_sweep.py` evaluation loop to allow `<think>` block generation up to budget before trigger scanning.
- Demonstrate GSM8K accuracy recovery (targeting 90%+) on reasoning models while maintaining 100% bit-exact micro-kernel computation.

**Non-Goals:**
- Modifying C++/CUDA SIMD micro-kernel C++ headers or Python host bindings.
- Changing fine-tuning behavior for non-reasoning micro-SLMs (SmolLM2), which perform best with instant triggers.

## Decisions

### 1. Curriculum Target Formatting for Reasoning Models
Reasoning models will be trained on targets containing a lightweight `<think>` trace before the trigger:
```text
<think>
Given the quantities 221.17, 463.49, 445.91, 465.58, we need to compute their exact sum.
</think>
<|jit_start|>SUM_F64(221.17, 463.49, 445.91, 465.58)<|jit_end|>
```
*Rationale*: This preserves the attention pattern expected by Qwen3 and DeepSeek-R1 architectures.

### 2. Prompt Loss Masking via Completion Collator
We introduce `DataCollatorForCompletionOnlyLM` using template delimiters (`<|im_start|>assistant\n` / `<|start_header_id|>assistant\n`) so that user prompt tokens are masked out (`label = -100`), concentrating loss gradient strictly on response tokens.

### 3. CoT-Aware Evaluation Loop
In `benchmarks/finetuned_sweep.py`, for `is_thinking=True` models, generation allows up to 1024 tokens. The host engine scans the token stream dynamically:
1. Model generates `<think>...</think>`.
2. Model emits `<|jit_start|>`.
3. Host engine halts generation, parses kernel & args, executes CUDA/SIMD code, and splices the ground truth result.

## Risks / Trade-offs

- **[Risk: Token Consumption Increase for Reasoning SLMs]** → Reasoning models will consume ~100–300 tokens for thinking instead of ~15 tokens. *Mitigation*: Still a ~10x reduction vs pure token CoT (~2,000 tokens), with 100% arithmetic precision guaranteed by the host kernel.
- **[Risk: Loose Syntax inside `<think>` block]** → Model might output wrong numbers in `<think>` text. *Mitigation*: BSM-RLI host engine ignores the text inside `<think>` and only parses the final `<|jit_start|>...<|jit_end|>` arguments for execution.
