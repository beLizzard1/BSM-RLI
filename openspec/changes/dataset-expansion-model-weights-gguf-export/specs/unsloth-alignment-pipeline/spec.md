## MODIFIED Requirements

### Requirement: Unsloth Supervised Fine-Tuning Script
The system SHALL provide an automated fine-tuning pipeline script (`training/run_unsloth_pipeline.py`) utilizing Unsloth fast-patching to fetch HuggingFace model weights, fine-tune 1B–8B models, and export quantized GGUF checkpoints.

#### Scenario: Running Unsloth SFT setup
- **WHEN** `run_unsloth_pipeline.py` is executed
- **THEN** it SHALL register `<|jit_start|>` and `<|jit_end|>` as special tokens, apply QLoRA PEFT target modules, format training inputs, and export fine-tuned `.gguf` weights

### Requirement: Unsloth GRPO Preference Alignment
The system SHALL provide a GRPO preference alignment script (`training/train_unsloth_grpo.py`) with reward functions enforcing schema precision, numerical correctness, and token economy.

#### Scenario: Evaluating GRPO reward functions
- **WHEN** a completion contains a valid `<|jit_start|>KERNEL(...)<|jit_end|>` tag, matches ground truth, and uses <30 tokens
- **THEN** the GRPO reward functions SHALL compute a cumulative positive reward score
