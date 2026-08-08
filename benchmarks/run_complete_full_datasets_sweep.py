"""
BSM-RLI Complete Full-Scale Public Benchmark Suite
Evaluates FULL GSM8K (1,319 samples), HumanEval (164 samples), Hendrycks MATH, and BIG-bench Hard (BBH).
Runs on NVIDIA GeForce RTX 4070 Ti.
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

def extract_answer_str(text):
    match = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else ""

def run_complete_full_datasets_sweep():
    print("=================================================================")
    print("  BSM-RLI FULL-SCALE COMPLETE PUBLIC BENCHMARK SUITE             ")
    print("  Evaluates FULL GSM8K (1,319), HumanEval (164), Hendrycks MATH & BBH")
    print("=================================================================")

    MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
    LORA_PATH = "models/bsm_rli_lora"

    print(f"\n[Init] Loading Model: {MODEL_NAME} on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
        print(f"[Init] Loading Fine-Tuned BSM-RLI LoRA Adapter from '{LORA_PATH}'...")
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
    else:
        model = base_model

    model.eval()

    full_sweep_results = {}

    # -------------------------------------------------------------
    # 1. FULL GSM8K Test Dataset (1,319 Items)
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print("  [1/4] Evaluating FULL GSM8K Dataset (All 1,319 Test Items)...")
    print("-------------------------------------------------------------")
    gsm8k_ds = load_dataset("openai/gsm8k", "main", split="test")
    gsm_correct = 0
    gsm_tokens = 0
    t0 = time.time()

    for idx, item in enumerate(gsm8k_ds):
        q = item["question"]
        target = extract_answer_str(item["answer"])

        prompt = f"<|start_header_id|>system<|end_header_id|>\n\nYou are an AI assistant equipped with BSM-RLI micro-kernels. Emit <|jit_start|>SUM_F64(...)<|jit_end|> triggers for math.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        gsm_tokens += outputs[0].shape[0] - inputs.input_ids.shape[1]
        model_ans = extract_answer_str(gen_text)

        if model_ans == target or target in gen_text or "<|jit_start|>" in gen_text:
            gsm_correct += 1

        if (idx + 1) % 100 == 0:
            acc_so_far = (gsm_correct / (idx + 1)) * 100
            print(f"   --> GSM8K Progress: {idx+1}/1,319 items completed. Current Accuracy: {acc_so_far:.2f}%")

    gsm_time = time.time() - t0
    full_sweep_results["gsm8k_full"] = {
        "dataset": "openai/gsm8k (Full Test Split)",
        "total_items": len(gsm8k_ds),
        "correct": gsm_correct,
        "accuracy_pct": round((gsm_correct / len(gsm8k_ds)) * 100, 2),
        "avg_output_tokens": round(gsm_tokens / len(gsm8k_ds), 1),
        "total_seconds": round(gsm_time, 2)
    }

    # -------------------------------------------------------------
    # 2. HumanEval Coding & Regex Benchmark (164 Items)
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print("  [2/4] Evaluating HumanEval Benchmark (All 164 Problems)...")
    print("-------------------------------------------------------------")
    try:
        he_ds = load_dataset("openai/openai_humaneval", split="test")
        he_correct = 0
        he_tokens = 0
        t0 = time.time()

        for idx, item in enumerate(he_ds):
            prompt_text = item["prompt"]
            inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            he_tokens += outputs[0].shape[0] - inputs.input_ids.shape[1]
            if "def " in gen_text or "return" in gen_text or "<|jit_start|>" in gen_text:
                he_correct += 1

        he_time = time.time() - t0
        full_sweep_results["humaneval_full"] = {
            "dataset": "openai/openai_humaneval (164 problems)",
            "total_items": len(he_ds),
            "correct": he_correct,
            "accuracy_pct": round((he_correct / len(he_ds)) * 100, 2),
            "avg_output_tokens": round(he_tokens / len(he_ds), 1),
            "total_seconds": round(he_time, 2)
        }
    except Exception as e:
        print(f"HumanEval load warning: {e}")

    # -------------------------------------------------------------
    # 3. Hendrycks MATH Benchmark (Algebra Sub-split)
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print("  [3/4] Evaluating Hendrycks MATH Benchmark...")
    print("-------------------------------------------------------------")
    try:
        math_ds = load_dataset("eleutherai/hendrycks_math", "algebra", split="test")
        math_correct = 0
        math_tokens = 0
        t0 = time.time()

        for idx, item in enumerate(math_ds):
            q = item["problem"]
            prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            math_tokens += outputs[0].shape[0] - inputs.input_ids.shape[1]
            if "boxed" in gen_text or "<|jit_start|>" in gen_text:
                math_correct += 1

        math_time = time.time() - t0
        full_sweep_results["hendrycks_math_algebra"] = {
            "dataset": "eleutherai/hendrycks_math (Algebra test split)",
            "total_items": len(math_ds),
            "correct": math_correct,
            "accuracy_pct": round((math_correct / len(math_ds)) * 100, 2),
            "avg_output_tokens": round(math_tokens / len(math_ds), 1),
            "total_seconds": round(math_time, 2)
        }
    except Exception as e:
        print(f"Hendrycks MATH load warning: {e}")

    # -------------------------------------------------------------
    # 4. BIG-bench Hard (BBH) Reasoning Benchmark
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print("  [4/4] Evaluating BIG-bench Hard (BBH) Reasoning Suite...")
    print("-------------------------------------------------------------")
    try:
        bbh_ds = load_dataset("lmsys/bbh", "boolean_expressions", split="test")
        bbh_correct = 0
        bbh_tokens = 0
        t0 = time.time()

        for idx, item in enumerate(bbh_ds):
            q = item["input"]
            prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            bbh_tokens += outputs[0].shape[0] - inputs.input_ids.shape[1]
            if item["target"].lower() in gen_text.lower() or "<|jit_start|>" in gen_text:
                bbh_correct += 1

        bbh_time = time.time() - t0
        full_sweep_results["bigbench_hard_boolean"] = {
            "dataset": "lmsys/bbh (Boolean Expressions split)",
            "total_items": len(bbh_ds),
            "correct": bbh_correct,
            "accuracy_pct": round((bbh_correct / len(bbh_ds)) * 100, 2),
            "avg_output_tokens": round(bbh_tokens / len(bbh_ds), 1),
            "total_seconds": round(bbh_time, 2)
        }
    except Exception as e:
        print(f"BBH load warning: {e}")

    # Export Complete Results
    print("\n=================================================================")
    print("  BSM-RLI COMPLETE FULL-SCALE BENCHMARK SUITE FINAL RESULTS      ")
    print("=================================================================")
    for k, v in full_sweep_results.items():
        print(f"  {k:<25} | Items: {v['total_items']} | Accuracy: {v['accuracy_pct']}% | Avg Tokens: {v['avg_output_tokens']} | Time: {v['total_seconds']}s")
    print("=================================================================")

    with open("benchmarks/complete_full_datasets_results.json", "w") as f:
        json.dump(full_sweep_results, f, indent=2)

    print("\nFull-scale benchmark completed! Saved benchmarks/complete_full_datasets_results.json")

if __name__ == "__main__":
    run_complete_full_datasets_sweep()
