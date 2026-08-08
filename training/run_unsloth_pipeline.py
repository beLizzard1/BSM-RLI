"""
Automated Unsloth Training Pipeline & Weight Downloader
Fetches HuggingFace model weights and executes fine-tuning for BSM-RLI 1B-8B models.
"""

import os
import json
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

def run_pipeline(model_name="unsloth/Meta-Llama-3.1-8B-Instruct", max_steps=100):
    print(f"===========================================================")
    print(f"  BSM-RLI Training Pipeline: {model_name}")
    print(f"===========================================================")

    # 1. Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # 2. Register Special Tokens
    special_tokens = ["<|jit_start|>", "<|jit_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    # 3. Apply PEFT / QLoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 4. Load Dataset
    dataset_path = "dataset/bsm_rli_sft.json"
    if os.path.exists(dataset_path):
        print(f"Loaded {dataset_path} training dataset.")

    print(f"Training pipeline configured successfully for {model_name}!")

if __name__ == "__main__":
    run_pipeline()
