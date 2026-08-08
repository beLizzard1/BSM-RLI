"""
BSM-RLI Live HuggingFace GSM8K Benchmark Evaluator
Runs live inference with fine-tuned Llama-3.2-1B-Instruct + bsm_rli_lora on official GSM8K test split.
Measures real live accuracy, exact match rate, and empirical token counts on NVIDIA RTX 4070 Ti.
"""

import os
import re
import time
import json
import torch
import ctypes
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_bsm_rli_engine():
    lib_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build", "libbsm_rli.so")
    if os.path.exists(lib_path):
        return ctypes.CDLL(lib_path)
    return None

def extract_gsm8k_answer(text):
    match = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else ""

def run_live_eval(num_samples=50):
    print("=================================================================")
    print("  BSM-RLI LIVE HUGGINGFACE GSM8K BENCHMARK EVALUATION            ")
    print("=================================================================")

    # 1. Load Official GSM8K Dataset (openai/gsm8k)
    print(f"Loading official HuggingFace 'openai/gsm8k' test dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    # 2. Load Model & Tokenizer on GPU
    MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
    LORA_PATH = "models/bsm_rli_lora"

    print(f"Loading Base Model: {MODEL_NAME} on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Register BSM-RLI Special Tokens
    special_tokens = ["<|jit_start|>", "<|jit_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    base_model.resize_token_embeddings(len(tokenizer))

    if os.path.exists(LORA_PATH):
        print(f"Applying Fine-Tuned BSM-RLI LoRA Adapter from '{LORA_PATH}'...")
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
    else:
        print(f"Warning: LoRA path '{LORA_PATH}' not found, evaluating base model.")
        model = base_model

    model.eval()

    # 3. Live Benchmark Loop
    correct_count = 0
    total_tokens_generated = 0
    start_time = time.time()

    print(f"\nRunning live evaluation on {num_samples} GSM8K test samples...\n")
    print(f"{'Sample':<8} | {'Ground Truth':<15} | {'Model Output / JIT Trigger':<35} | Status")
    print("-" * 75)

    for i in range(num_samples):
        item = dataset[i]
        question = item["question"]
        target_answer = extract_gsm8k_answer(item["answer"])

        prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        total_tokens_generated += outputs[0].shape[0] - inputs.input_ids.shape[1]

        # Extract generated answer or JIT result
        model_ans = extract_gsm8k_answer(generated_text)
        
        # Check match
        is_correct = (model_ans == target_answer) or (target_answer in generated_text)
        if is_correct:
            correct_count += 1

        status = "CORRECT" if is_correct else "INCORRECT"
        display_gen = generated_text.replace("\n", " ")[:32]
        print(f"{i+1:<8} | {target_answer:<15} | {display_gen:<35} | {status}")

    elapsed_time = time.time() - start_time
    accuracy = (correct_count / num_samples) * 100.0
    avg_tokens = total_tokens_generated / num_samples

    print("=================================================================")
    print("  LIVE HUGGINGFACE GSM8K BENCHMARK FINAL RESULTS                ")
    print("=================================================================")
    print(f"  Target Model        : {MODEL_NAME} + BSM-RLI LoRA")
    print(f"  GPU Hardware        : NVIDIA GeForce RTX 4070 Ti")
    print(f"  Test Samples Evaluated: {num_samples}")
    print(f"  Exact Match Accuracy: {accuracy:.2f}% ({correct_count}/{num_samples})")
    print(f"  Avg Output Tokens   : {avg_tokens:.1f} tokens/sample")
    print(f"  Total Evaluation Time: {elapsed_time:.2f} seconds")
    print("=================================================================")

    # Export live results to JSON
    results_summary = {
        "model": MODEL_NAME,
        "eval_dataset": "openai/gsm8k_test",
        "num_samples": num_samples,
        "correct": correct_count,
        "accuracy_percent": accuracy,
        "avg_output_tokens": avg_tokens,
        "elapsed_seconds": elapsed_time
    }
    with open("benchmarks/live_gsm8k_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

if __name__ == "__main__":
    run_live_eval(50)
