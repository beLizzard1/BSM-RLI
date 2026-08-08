"""
Capture exact verbatim prompt generations live from:
1. Pure unadapted base model (unsloth/Llama-3.2-1B-Instruct)
2. Fine-tuned SFT LoRA model (models/bsm_rli_lora)
"""

import time
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def capture_real_runs():
    MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
    LORA_PATH = "models/bsm_rli_lora"

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    special_tokens = ["<|jit_start|>", "<|jit_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    base_model.resize_token_embeddings(len(tokenizer))

    # Question from GSM8K Item #0
    gsm8k_ds = load_dataset("openai/gsm8k", "main", split="test")
    question = gsm8k_ds[0]["question"]
    ground_truth = gsm8k_ds[0]["answer"]

    print("\n========================================================")
    print("GSM8K ITEM #0 QUESTION:\n", question)
    print("GROUND TRUTH:\n", ground_truth)
    print("========================================================")

    # 1. PURE BASE MODEL GENERATION
    print("\n--- 1. PURE BASE MODEL ---")
    prompt_base = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    inputs_base = tokenizer(prompt_base, return_tensors="pt").to("cuda")
    
    t0 = time.time()
    with torch.no_grad():
        out_base = base_model.generate(**inputs_base, max_new_tokens=128, do_sample=False)
    t1 = time.time()
    
    text_base = tokenizer.decode(out_base[0][inputs_base.input_ids.shape[1]:], skip_special_tokens=True)
    tokens_base = out_base[0].shape[0] - inputs_base.input_ids.shape[1]
    time_base = t1 - t0

    print("Output:\n", text_base)
    print(f"Tokens: {tokens_base} | Time: {time_base:.4f}s")

    # 2. FINE-TUNED SFT LORA GENERATION
    print("\n--- 2. FINE-TUNED SFT LORA MODEL ---")
    ft_model = PeftModel.from_pretrained(base_model, LORA_PATH)
    ft_model.eval()

    prompt_ft = f"<|start_header_id|>system<|end_header_id|>\n\nYou are an AI assistant equipped with BSM-RLI micro-kernels. Emit <|jit_start|>SUM_F64(...)<|jit_end|> triggers for math.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    inputs_ft = tokenizer(prompt_ft, return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.no_grad():
        out_ft = ft_model.generate(**inputs_ft, max_new_tokens=128, do_sample=False)
    t1 = time.time()

    text_ft = tokenizer.decode(out_ft[0][inputs_ft.input_ids.shape[1]:], skip_special_tokens=False)
    tokens_ft = out_ft[0].shape[0] - inputs_ft.input_ids.shape[1]
    time_ft = t1 - t0

    print("Output:\n", text_ft)
    print(f"Tokens: {tokens_ft} | Time: {time_ft:.4f}s")

    captured_data = {
        "question": question,
        "ground_truth": ground_truth,
        "base_model": {
            "output": text_base,
            "tokens": tokens_base,
            "time_sec": round(time_base, 4)
        },
        "sft_lora_model": {
            "output": text_ft,
            "tokens": tokens_ft,
            "time_sec": round(time_ft, 4)
        }
    }

    with open("experiments/captured_real_runs.json", "w") as f:
        json.dump(captured_data, f, indent=2)

    print("\nCaptured real runs saved to experiments/captured_real_runs.json")

if __name__ == "__main__":
    capture_real_runs()
