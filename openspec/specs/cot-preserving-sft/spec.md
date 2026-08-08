# cot-preserving-sft Specification

## Purpose
TBD - created by archiving change cot-preserving-sft. Update Purpose after archive.
## Requirements
### Requirement: CoT-Preserving Curriculum Formatting
The curriculum dataset generator SHALL support formatting reasoning model training pairs with an explicit `<think>...</think>` block prior to emitting `<|jit_start|>` micro-kernel triggers.

#### Scenario: Formatting training targets for reasoning models
- **WHEN** training data is generated for reasoning model families (`qwen3`, `deepseek-r1`)
- **THEN** the target text MUST include a `<think>` trace followed by `<|jit_start|>OP_NAME(args)<|jit_end|>`

### Requirement: Completion Loss Masking in Training
The fine-tuning trainer SHALL mask user prompt tokens during SFT training so that loss gradients are computed exclusively on assistant response tokens (thinking trace and JIT triggers).

#### Scenario: Computing cross-entropy loss during training
- **WHEN** a batch of training sequences is processed by `train_enhanced_curriculum_sft.py`
- **THEN** input prompt tokens MUST have label `-100` and response tokens MUST contribute to loss calculation

### Requirement: CoT-Aware Evaluation Stream Interception
The benchmark evaluation script SHALL allow reasoning models to complete their `<think>` reasoning block before intercepting the `<|jit_start|>` token trigger.

#### Scenario: Running batched evaluation on fine-tuned reasoning model
- **WHEN** `finetuned_sweep.py` evaluates a reasoning model on a benchmark prompt
- **THEN** generation MUST continue through the `<think>` block until `<|jit_start|>` is detected, at which point host C++/CUDA micro-kernel execution MUST occur

