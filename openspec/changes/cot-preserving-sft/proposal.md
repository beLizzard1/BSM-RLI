## Why

Our empirical benchmark sweeps revealed that while naive SFT on BSM-RLI micro-kernel triggers boosts ultra-small standard models (SmolLM2-135M: 10% → 28%, SmolLM2-360M: 14% → 32%), it severely degrades reasoning models with native chain-of-thought pathways (Qwen3-1.7B dropped from 94% baseline CoT down to 2% under SFT). 

Naive SFT forces the model to immediately emit `<|jit_start|>` triggers at the start of the output, effectively suppressing its internal `<think>...</think>` reasoning stream. By introducing a CoT-preserving SFT strategy (loss masking + thinking-trace preservation + trigger insertion at the end of thought blocks), reasoning SLMs can retain their high-level problem comprehension while executing bit-exact C++/CUDA micro-kernels for the final computation step.

## What Changes

- **CoT-Preserving Curriculum Generator**: Extend curriculum dataset generation to format reasoning model targets with `<think>...</think>` reasoning traces followed by JIT triggers (`<think>{reasoning_trace}</think>\n<|jit_start|>KERNEL(...)<|jit_end|>`).
- **Data Loss Masking Trainer**: Update `training/train_enhanced_curriculum_sft.py` with custom data collator and loss-masking options (e.g., masking user prompt and thinking trace, computing loss strictly on trigger tokens vs. computing loss on thinking + trigger tokens).
- **CoT-Aware Evaluation Engine**: Update `benchmarks/finetuned_sweep.py` to allow reasoning models to generate their `<think>` block (up to a budget) before scanning for and intercepting the `<|jit_start|>` trigger.
- **Comparative Sweep Validation**: Re-run fine-tuning sweep on reasoning models (`Qwen3-0.6B`, `Qwen3-1.7B`, `DeepSeek-R1-1.5B`) to measure recovery of accuracy (targeting 95%+ with 100% exact kernel calculation).

## Capabilities

### New Capabilities
- `cot-preserving-sft`: CoT-preserving curriculum formatting, loss-masking trainer integration, and CoT-aware JIT trigger interception during evaluation.

### Modified Capabilities
*(None - existing capabilities maintain current API interfaces)*

## Impact

- `training/train_enhanced_curriculum_sft.py`: Added DataCollatorForCompletionOnlyLM / loss masking logic.
- `dataset/bsm_rli_curriculum_75k.json`: Additional curriculum target variations formatted with `<think>` traces for reasoning families.
- `benchmarks/finetuned_sweep.py`: Updated evaluation generation loop to support thinking trace completion followed by trigger interception.
- `models/finetuned/`: Regenerated LoRA weights for Qwen3 and DeepSeek-R1 model families.
