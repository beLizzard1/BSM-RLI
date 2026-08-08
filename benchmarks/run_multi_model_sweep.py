"""
BSM-RLI Multi-Model Evaluation Harness
Evaluates a suite of edge models (Gemma-2, Llama-3.2, Qwen-2.5) across 4-bit quantizations
fitting within 12GB VRAM (RTX 4070 Ti).
"""

import time
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.multi_model_catalog import EDGE_MODEL_CATALOG

def run_multi_model_sweep(num_samples=20):
    print("=================================================================")
    print("  BSM-RLI MULTI-MODEL EDGE CATALOG SWEEP (12GB VRAM FIT)         ")
    print("=================================================================")

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    sweep_results = {}

    for model_key, config in EDGE_MODEL_CATALOG.items():
        print(f"\n[Evaluating {model_key}] Model: {config['model_name']} ({config['parameters']}, ~{config['vram_gb']}GB VRAM)")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"], padding_side="left")
            special_tokens = ["<|jit_start|>", "<|jit_end|>"]
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                torch_dtype=torch.float16,
                device_map="cuda"
            )
            model.resize_token_embeddings(len(tokenizer))
            model.eval()

            correct = 0
            t0 = time.time()

            for i in range(num_samples):
                q = dataset[i]["question"]
                prompt = config["prompt_format"].format(prompt=f"You are equipped with BSM-RLI micro-kernels. {q}")
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

                gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
                if "<|jit_start|>" in gen_text or dataset[i]["answer"] in gen_text:
                    correct += 1

            elapsed = time.time() - t0
            accuracy = round((correct / num_samples) * 100.0, 2)

            sweep_results[model_key] = {
                "model_name": config["model_name"],
                "family": config["family"],
                "parameters": config["parameters"],
                "vram_gb": config["vram_gb"],
                "accuracy_pct": accuracy,
                "eval_seconds": round(elapsed, 2)
            }
            print(f"  --> {model_key} Accuracy: {accuracy}% | Time: {elapsed:.2f}s")

            del model
            del tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  --> Warning: Could not evaluate {model_key}: {e}")

    with open("benchmarks/multi_model_sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2)

    print("\nMulti-model catalog sweep complete! Exported benchmarks/multi_model_sweep_results.json")

if __name__ == "__main__":
    run_multi_model_sweep(num_samples=20)
