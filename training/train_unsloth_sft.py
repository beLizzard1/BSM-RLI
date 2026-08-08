"""
BSM-RLI Unsloth Supervised Fine-Tuning (SFT) Script
Fine-tunes 1B-8B models (Llama-3.2-1B, Llama-3.1-8B, Qwen-2.5-7B) to emit BSM-RLI micro-kernel triggers.
"""

import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 1. Configuration
MAX_SEQ_LENGTH = 2048
DTYPE = None  # None for auto-detection (Float16/Bfloat16)
LOAD_IN_4BIT = True

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

print(f"===========================================================")
print(f"  BSM-RLI Unsloth SFT Fine-Tuning Execution: {MODEL_NAME}")
print(f"===========================================================")

# 2. Load Model & Tokenizer via Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

# 3. Add Custom BSM-RLI Special Tokens
special_tokens = ["<|jit_start|>", "<|jit_end|>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))

# 4. Configure PEFT / QLoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 5. Load & Format Synthetic Dataset
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["response"]
    texts = []
    for inst, out in zip(instructions, outputs):
        text = f"<|start_header_id|>user<|end_header_id|>\n\n{inst}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{out}<|eot_id|>"
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("json", data_files="dataset/bsm_rli_sft.json", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 6. Initialize SFTTrainer with SFTConfig
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="no",
        output_dir="models/bsm_rli_sft_checkpoints",
        optim="adamw_8bit",
        seed=3407,
    ),
)

# 7. Execute Fine-Tuning Training Loop
print("Starting SFT fine-tuning training loop...")
trainer_stats = trainer.train()

print("\nSaving fine-tuned BSM-RLI LoRA model weights...")
os.makedirs("models/bsm_rli_lora", exist_ok=True)
model.save_pretrained("models/bsm_rli_lora")
tokenizer.save_pretrained("models/bsm_rli_lora")

print("Fine-tuning training complete! Model saved to models/bsm_rli_lora")
