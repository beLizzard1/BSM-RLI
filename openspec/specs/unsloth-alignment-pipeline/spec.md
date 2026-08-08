## ADDED Requirements

### Requirement: Unsloth Supervised Fine-Tuning Script
The system SHALL provide a fine-tuning script (`training/train_unsloth_sft.py`) utilizing Unsloth fast-patching to fine-tune 1B–8B parameter models with BSM-RLI control tokens.

#### Scenario: Running Unsloth SFT setup
- **WHEN** `train_unsloth_sft.py` is executed
- **THEN** it SHALL register `<|jit_start|>` and `<|jit_end|>` as special tokens, apply QLoRA PEFT target modules, and format training inputs into conversation templates

### Requirement: Unsloth GRPO Preference Alignment
The system SHALL provide a GRPO preference alignment script (`training/train_unsloth_grpo.py`) with reward functions enforcing schema precision, numerical correctness, and token economy.

#### Scenario: Evaluating GRPO reward functions
- **WHEN** a completion contains a valid `<|jit_start|>KERNEL(...)<|jit_end|>` tag, matches ground truth, and uses <30 tokens
- **THEN** the GRPO reward functions SHALL compute a cumulative positive reward score
