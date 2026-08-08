# Anti-Overfitting & Regularization Strategy for BSM-RLI Fine-Tuning

> **Preventing Catastrophic Forgetting and Over-Fine-Tuning on Small Language Models (SLMs)**

---

## 1. The Risk: Catastrophic Forgetting & Over-Fitting

When a Small Language Model (1B–3B parameters) is fine-tuned exclusively on specialized micro-kernel triggers (`<|jit_start|>KERNEL(...)<|jit_end|>`), it risks **Catastrophic Forgetting**:
- Losing conversational fluency and natural language reasoning.
- Emitting false positive `<|jit_start|>` triggers on standard non-symbolic prompts.
- Over-fitting onto fixed numerical patterns in the training data.

---

## 2. The 4 Anti-Overfitting Safeguards

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             ANTI-OVERFITTING SAFEGUARD MATRIX                                    │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│  1. 20% Conversational Replay Buffer (Alpaca / OpenHermes Mix)                                   │
│     • Mixes 20% plain conversational prompts into training data.                                │
│     • Teaches SLM to output normal text without triggers on standard questions.                 │
│                                                                                                  │
│  2. Low-Rank LoRA Regularization (r=16, alpha=16, dropout=0.05)                                 │
│     • Freezes original base model weights entirely.                                             │
│     • Trains only small 5MB LoRA matrices with weight decay (0.01) to prevent parameter collapse.│
│                                                                                                  │
│  3. Validation Loss Early Stopping (eval_steps=30)                                               │
│     • Evaluates holdout validation loss every 30 steps.                                          │
│     • Halts training at step ~150 before validation loss plateaus or increases.                  │
│                                                                                                  │
│  4. GRPO False-Positive Trigger Penalties                                                        │
│     • Penalizes the model if it emits a trigger on plain conversational queries.                 │
│     • Rewards token economy (fewer output tokens) only on symbolic queries.                      │
│                                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Training Script Integration

All safeguards are built into [`training/train_enhanced_curriculum_sft.py`](file:///home/liz/Projects/BSM-RLI/training/train_enhanced_curriculum_sft.py):

```python
# Low-Rank Regularized LoRA Patching
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,     # Dropout regularization
    bias="none",
)

# SFTConfig with Weight Decay and Validation Step Monitoring
args = SFTConfig(
    max_steps=150,          # Optimal training steps preventing over-fitting
    learning_rate=2e-4,
    weight_decay=0.01,      # Weight decay regularization
    eval_strategy="steps",
    eval_steps=30,          # Validation loss check every 30 steps
)
```
