"""
BSM-RLI Pure Base Model Baseline Evaluator
Runs live inference with unadapted base model (Llama-3.2-1B-Instruct without any LoRA or BSM-RLI)
on official GSM8K test split for experimental comparison plots.
"""

import os
import re
import time
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_gsm8k_answer(text):
    match = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else ""

def run_baseline_eval(num_samples=50):
    print("=================================================================")
    print("  PURE BASE MODEL (UNADAPTED) GSM8K BENCHMARK EVALUATION          ")
    print("=================================================================")

    # 1. Load Official GSM8K Dataset
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    # 2. Load Pure Base Model (No LoRA)
    MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
    print(f"Loading Pure Base Model: {MODEL_NAME} on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()

    # 3. Benchmark Loop
    correct_count = 0
    total_tokens_generated = 0
    start_time = time.time()

    for i in range(num_samples):
        item = dataset[i]
        question = item["question"]
        target_answer = extract_gsm8k_answer(item["answer"])

        prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )

        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        total_tokens_generated += outputs[0].shape[0] - inputs.input_ids.shape[1]

        model_ans = extract_gsm8k_answer(generated_text)
        is_correct = (model_ans == target_answer) or (target_answer in generated_text)
        if is_correct:
            correct_count += 1

    elapsed_time = time.time() - start_time
    accuracy = (correct_count / num_samples) * 100.0
    avg_tokens = total_tokens_generated / num_samples

    print("=================================================================")
    print("  PURE BASE MODEL BENCHMARK FINAL RESULTS                        ")
    print("=================================================================")
    print(f"  Target Model        : {MODEL_NAME} (Unadapted Base)")
    print(f"  Exact Match Accuracy: {accuracy:.2f}% ({correct_count}/{num_samples})")
    print(f"  Avg Output Tokens   : {avg_tokens:.1f} tokens/sample")
    print(f"  Total Evaluation Time: {elapsed_time:.2f} seconds")
    print("=================================================================")

    results_summary = {
        "model": MODEL_NAME,
        "adapter": "None (Pure Base Model)",
        "num_samples": num_samples,
        "correct": correct_count,
        "accuracy_percent": accuracy,
        "avg_output_tokens": avg_tokens,
        "elapsed_seconds": elapsed_time
    }
    with open("benchmarks/pure_base_model_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

if __name__ == "__main__":
    run_baseline_eval(50)
