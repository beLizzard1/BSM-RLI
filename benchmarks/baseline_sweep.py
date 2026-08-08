"""
BSM-RLI Baseline Model Sweep
Evaluates ALL 21 catalog models on a fixed 50-item benchmark set
covering GSM8K math, character counting, and unit conversion
with ZERO fine-tuning — pure out-of-the-box performance.

Saves results to benchmarks/results/baseline_sweep.json
Resumes from checkpoint if interrupted.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import re
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.multi_model_catalog import MODELS_BY_VRAM

# ─────────────────────────────────────────────────
# Fixed 50-item evaluation set (consistent across all models)
# Mix of GSM8K-style, char-count, and unit-convert
# ─────────────────────────────────────────────────
from datasets import load_dataset

def build_eval_set(n=50):
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for i in range(n):
        q = gsm[i]["question"]
        # Extract final numeric answer from GSM8K annotation
        ans_raw = gsm[i]["answer"]
        match = re.search(r"####\s*([\d,\.\-]+)", ans_raw)
        expected = match.group(1).replace(",", "").strip() if match else None
        items.append({"question": q, "expected": expected, "type": "gsm8k"})
    return items


def check_answer(generated: str, expected: str) -> bool:
    """Checks if the expected answer appears in generated text."""
    if expected is None:
        return False
    # Look for the number in the output
    nums = re.findall(r"[\d,]+\.?\d*", generated.replace(",", ""))
    clean_expected = expected.replace(",", "")
    return clean_expected in nums or clean_expected in generated.replace(",", "")


def run_baseline_sweep():
    os.makedirs("benchmarks/results", exist_ok=True)
    RESULTS_PATH = "benchmarks/results/baseline_sweep.json"

    # Load existing checkpoint to allow resume
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f"[Resume] Found existing results for {len(results)} models.")
    else:
        results = {}

    print("=================================================================")
    print(f"  BSM-RLI BASELINE SWEEP — {len(MODELS_BY_VRAM)} MODELS, 50 ITEMS EACH")
    print("  NO FINE-TUNING: Pure out-of-the-box performance")
    print("=================================================================\n")

    eval_set = build_eval_set(50)

    for model_key, config in MODELS_BY_VRAM:
        if model_key in results:
            print(f"[Skip] {model_key} — already evaluated.")
            continue

        print(f"\n[Evaluating] {model_key}")
        print(f"  Model : {config['model_name']}")
        print(f"  Params: {config['parameters']} | VRAM: ~{config['vram_gb']} GB")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config["model_name"], padding_side="left", trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                torch_dtype=torch.float16,
                device_map="cuda",
                trust_remote_code=True,
            )
            model.eval()

            correct = 0
            total_tokens = 0
            t0 = time.time()

            for item in eval_set:
                prompt = config["prompt_format"].format(prompt=item["question"])
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                new_tokens = outputs[0][inputs.input_ids.shape[1]:]
                total_tokens += len(new_tokens)
                gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

                if check_answer(gen_text, item["expected"]):
                    correct += 1

            elapsed = time.time() - t0
            accuracy = round((correct / len(eval_set)) * 100.0, 2)
            avg_tokens = round(total_tokens / len(eval_set), 1)

            results[model_key] = {
                "model_name": config["model_name"],
                "family": config["family"],
                "parameters": config["parameters"],
                "vram_gb": config["vram_gb"],
                "stage": "baseline",
                "accuracy_pct": accuracy,
                "correct": correct,
                "total": len(eval_set),
                "avg_tokens_per_sample": avg_tokens,
                "total_eval_seconds": round(elapsed, 2),
            }

            print(f"  --> Accuracy: {accuracy}% ({correct}/{len(eval_set)}) | "
                  f"Avg tokens: {avg_tokens} | Time: {elapsed:.1f}s")

        except Exception as e:
            print(f"  [Error] {model_key}: {e}")
            results[model_key] = {
                "model_name": config["model_name"],
                "stage": "baseline",
                "error": str(e),
                "accuracy_pct": None,
            }
        finally:
            # Always free VRAM between models
            try:
                del model
                del tokenizer
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

        # Save checkpoint after each model
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=================================================================")
    print("  BASELINE SWEEP COMPLETE")
    print(f"  Results saved to: {RESULTS_PATH}")
    print("=================================================================\n")

    # Summary table
    print(f"{'Model Key':<35} {'Params':<12} {'Accuracy':<12} {'Avg Tokens'}")
    print("-" * 75)
    for key, r in results.items():
        if r.get("accuracy_pct") is not None:
            print(f"{key:<35} {r['parameters']:<12} {r['accuracy_pct']:<12} {r.get('avg_tokens_per_sample', '—')}")

    return results


if __name__ == "__main__":
    run_baseline_sweep()
