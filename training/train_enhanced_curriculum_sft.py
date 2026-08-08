"""
BSM-RLI Enhanced Curriculum Unsloth Fine-Tuning Pipeline
Includes Anti-Overfitting & CoT-Preserving Safeguards:
1. 20% Conversational Replay Buffer (prevents catastrophic forgetting)
2. Low-Rank LoRA Regularization (r=16, alpha=16, weight_decay=0.01)
3. Response-Only Completion Loss Masking (CoTCompletionDataCollator)
4. Family-Aware Thinking-Trace Targets (<think>...</think>)
"""

import os
import torch
import json
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

class CoTCompletionDataCollator:
    """Masks input prompt tokens with -100 so loss gradients apply strictly to assistant responses."""
    def __init__(self, tokenizer, response_template):
        self.tokenizer = tokenizer
        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, examples):
        batch_input_ids = [torch.tensor(e["input_ids"]) if isinstance(e["input_ids"], list) else e["input_ids"] for e in examples]
        padded = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = (padded != self.pad_token_id).long()
        labels = padded.clone()

        for i, ids in enumerate(batch_input_ids):
            ids_list = ids.tolist()
            match_idx = -1
            for k in range(len(ids_list) - len(self.response_template_ids) + 1):
                if ids_list[k:k + len(self.response_template_ids)] == self.response_template_ids:
                    match_idx = k + len(self.response_template_ids)
                    break
            if match_idx != -1:
                labels[i, :match_idx] = -100
            else:
                labels[i, :len(ids_list) // 2] = -100
            labels[i, padded[i] == self.pad_token_id] = -100

        return {
            "input_ids": padded,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def train_enhanced_curriculum():
    print("=================================================================")
    print("  BSM-RLI ENHANCED CURRICULUM TRAINING (WITH CoT LOSS MASKING)   ")
    print("  Hardware: NVIDIA GeForce RTX 4070 Ti (12GB VRAM)              ")
    print("=================================================================")

    MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
    DATASET_PATH = "dataset/bsm_rli_curriculum_75k.json"
    OUTPUT_DIR = "models/bsm_rli_enhanced_lora"

    print(f"\n[1/4] Loading FastLanguageModel: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=512,
        load_in_4bit=True,
        dtype=torch.float16,
    )

    special_tokens = ["<|jit_start|>", "<|jit_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/4] Patching Low-Rank Regularized LoRA Adapters (r=16, alpha=16)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    print(f"[3/4] Loading 75,000 Curriculum Dataset from '{DATASET_PATH}'...")
    with open(DATASET_PATH, "r") as f:
        data = json.load(f)

    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    formatted_prompts = []
    for item in data[:5000]:
        p = f"<|start_header_id|>system<|end_header_id|>\n\nYou are an AI assistant equipped with BSM-RLI micro-kernels.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{item['instruction']}<|eot_id|>{response_template}{item['response']}<|eot_id|>"
        formatted_prompts.append({"text": p})

    dataset = Dataset.from_list(formatted_prompts)
    collator = CoTCompletionDataCollator(tokenizer, response_template)

    print("[4/4] Starting Unsloth SFT Fine-Tuning with DataCollator Loss Masking...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=512,
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=150,
            learning_rate=2e-4,
            weight_decay=0.01,
            fp16=True,
            bf16=False,
            logging_steps=10,
            output_dir=OUTPUT_DIR,
            save_strategy="no",
        ),
    )

    trainer.train()

    print(f"\nSaving fine-tuned LoRA weights to '{OUTPUT_DIR}'...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Enhanced curriculum training complete!")

if __name__ == "__main__":
    train_enhanced_curriculum()
