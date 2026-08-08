"""
BSM-RLI Baseline Model Sweep — Adaptive Token Budget + Batched Inference
Evaluates ALL 21 catalog models on a fixed 50-item benchmark set.

Parallelisation strategy (single RTX 4070 Ti 12GB):
  - Batched GPU inference: batch size scales inversely with model VRAM
    ≤1GB VRAM  → batch_size=32
    ≤2GB VRAM  → batch_size=16
    ≤4GB VRAM  → batch_size=8
    ≤6GB VRAM  → batch_size=4
    >6GB VRAM  → batch_size=2
  - Thinking models (DeepSeek-R1, Qwen3) capped at half batch due to longer outputs
  - Adaptive token budgets: 256 → 512 → 1024 → 2048 with auto-retry on saturation
  - Left-padding for correct batch alignment

Checkpoint-safe: resumes from benchmarks/results/baseline_sweep.json
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import time
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from models.multi_model_catalog import MODELS_BY_VRAM

THINKING_FAMILIES = {"deepseek-r1", "qwen3"}
TOKEN_BUDGETS = [256, 512, 1024, 2048]


def get_batch_size(vram_gb: float, is_thinking: bool) -> int:
    """Scale batch size inversely with model VRAM footprint."""
    if vram_gb <= 1.0:
        batch = 32
    elif vram_gb <= 2.0:
        batch = 16
    elif vram_gb <= 4.0:
        batch = 8
    elif vram_gb <= 6.0:
        batch = 4
    else:
        batch = 2
    # Thinking models emit much longer outputs — halve batch to avoid OOM
    if is_thinking:
        batch = max(1, batch // 2)
    return batch


def build_eval_set(n=50):
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for i in range(n):
        q = gsm[i]["question"]  # type: ignore
        ans_raw = gsm[i]["answer"]  # type: ignore
        match = re.search(r"####\s*([\d,\.\-]+)", ans_raw)  # type: ignore
        expected = match.group(1).replace(",", "").strip() if match else None
        items.append({"question": q, "expected": expected})
    return items


def check_answer(generated: str, expected: str) -> bool:
    if expected is None:
        return False
    nums = set(re.findall(r"\b\d[\d,]*\.?\d*\b", generated.replace(",", "")))
    return expected.replace(",", "") in nums or expected in generated.replace(",", "")


def is_budget_saturated(avg_tokens: float, budget: int, texts: list) -> bool:
    if avg_tokens < budget * 0.92:
        return False
    # Check if majority of outputs are mid-sentence truncations
    truncated = sum(1 for t in texts if t and not re.search(r"[\d\.\!\?]$", t.rstrip()))
    return truncated > len(texts) * 0.5


def evaluate_batched(model, tokenizer, config, eval_set, max_new_tokens, batch_size):
    """Batched inference over eval_set. Returns (correct, avg_tokens, texts, throughput)."""
    correct = 0
    total_tokens = 0
    all_texts = []
    prompts = [
        config["prompt_format"].format(prompt=item["question"]) for item in eval_set
    ]

    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_expected = [item["expected"] for item in eval_set[i : i + batch_size]]

        encoding = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **encoding,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens (strip off the input)
        input_len = encoding.input_ids.shape[1]
        for j, output in enumerate(outputs):
            new_tokens = output[input_len:]
            total_tokens += len(new_tokens)
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_texts.append(text)
            if check_answer(text, batch_expected[j]):
                correct += 1

    elapsed = time.time() - t0
    avg_tokens = round(total_tokens / len(eval_set), 1)
    throughput = round(len(eval_set) / elapsed, 2)
    return correct, avg_tokens, all_texts, elapsed, throughput


def run_baseline_sweep():
    os.makedirs("benchmarks/results", exist_ok=True)
    RESULTS_PATH = "benchmarks/results/baseline_sweep.json"

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f"[Resume] Found existing results for {len(results)} models.")
    else:
        results = {}

    print("=================================================================")
    print(f"  BSM-RLI BASELINE SWEEP — {len(MODELS_BY_VRAM)} MODELS (BATCHED)")
    print(f"  Token budgets: {TOKEN_BUDGETS} | Batch size: adaptive by VRAM")
    print("=================================================================\n")

    eval_set = build_eval_set(50)

    for model_key, config in MODELS_BY_VRAM:
        if model_key in results:
            print(f"[Skip] {model_key} — already evaluated.")
            continue

        is_thinking = config["family"] in THINKING_FAMILIES
        batch_size = get_batch_size(config["vram_gb"], is_thinking)
        starting_budget_idx = 1 if is_thinking else 0
        budgets_to_try = TOKEN_BUDGETS[starting_budget_idx:]

        print(f"\n[Evaluating] {model_key}")
        print(f"  Model    : {config['model_name']}")
        print(f"  Params   : {config['parameters']} | VRAM: ~{config['vram_gb']} GB")
        print(f"  Batch    : {batch_size} | Thinking: {'Yes' if is_thinking else 'No'}")

        model = None
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config["model_name"], padding_side="left", trust_remote_code=True
            )
            tokenizer.add_special_tokens({"additional_special_tokens": ["<|jit_start|>", "<|jit_end|>"]})
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                torch_dtype=torch.float16,
                device_map="cuda",
                trust_remote_code=True,
            )
            model.resize_token_embeddings(len(tokenizer))
            model.eval()

            final_correct, final_avg_tokens, final_budget = 0, 0, budgets_to_try[0]
            total_elapsed, final_throughput = 0, 0

            for budget in budgets_to_try:
                print(
                    f"  [Budget={budget}, Batch={batch_size}] Running...",
                    end=" ",
                    flush=True,
                )
                correct, avg_tokens, texts, elapsed, throughput = evaluate_batched(
                    model, tokenizer, config, eval_set, budget, batch_size
                )
                print(
                    f"{correct}/50 correct | avg {avg_tokens} tokens | "
                    f"{throughput} samples/s"
                )

                final_correct, final_avg_tokens, final_budget = (
                    correct,
                    avg_tokens,
                    budget,
                )
                total_elapsed += elapsed
                final_throughput = throughput

                if not is_budget_saturated(avg_tokens, budget, texts):
                    print(f"  ✓ Budget {budget} sufficient")
                    break
                elif budget == budgets_to_try[-1]:
                    print(f"  ⚠ Max budget {budget} reached — using best result")
                else:
                    print(f"  ↑ Saturated ({avg_tokens}/{budget}) — stepping up budget")

            accuracy = round((final_correct / len(eval_set)) * 100.0, 2)
            results[model_key] = {
                "model_name": config["model_name"],
                "family": config["family"],
                "parameters": config["parameters"],
                "vram_gb": config["vram_gb"],
                "stage": "baseline",
                "accuracy_pct": accuracy,
                "correct": final_correct,
                "total": len(eval_set),
                "avg_tokens_per_sample": final_avg_tokens,
                "token_budget_used": final_budget,
                "batch_size": batch_size,
                "throughput_samples_per_sec": final_throughput,
                "total_eval_seconds": round(total_elapsed, 2),
            }
            print(
                f"  ✅ {accuracy}% | budget={final_budget} | "
                f"throughput={final_throughput} samples/s | {total_elapsed:.1f}s"
            )

        except Exception as e:
            print(f"  [Error] {model_key}: {e}")
            results[model_key] = {
                "model_name": config["model_name"],
                "stage": "baseline",
                "error": str(e),
                "accuracy_pct": None,
            }
        finally:
            try:
                if model is not None:
                    del model
                if tokenizer is not None:
                    del tokenizer
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

        try:
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"  [Warn] could not write checkpoint: {e}")

    print("\n=================================================================")
    print("  BASELINE SWEEP COMPLETE")
    print("=================================================================\n")

    print(
        f"{'Model Key':<35} {'Params':<10} {'Accuracy':>10} {'Budget':>8} "
        f"{'Batch':>6} {'samples/s':>10} {'Avg Tok':>8}"
    )
    print("-" * 95)
    for key, r in results.items():
        if r.get("accuracy_pct") is not None:
            avg_tok = r.get("avg_tokens_per_sample", 0) or 0
            budget = r.get("token_budget_used", 0) or 0
            saturated = (
                "⚠"
                if (avg_tok and budget and avg_tok >= budget * 0.92)
                else " "
            )
            print(
                f"{key:<35} {r.get('parameters', '—'):<10} "
                f"{r['accuracy_pct']:>9}% {r.get('token_budget_used', '—'):>8} "
                f"{r.get('batch_size', '—'):>6} {r.get('throughput_samples_per_sec', '—'):>10} "
                f"{r.get('avg_tokens_per_sample', '—'):>8} {saturated}"
            )


if __name__ == "__main__":
    run_baseline_sweep()
