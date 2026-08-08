"""
BSM-RLI Fine-Tuning + Evaluation Sweep (Batched Inference & CoT Loss Masking)
For each model in the catalog:
  1. Load the model via Unsloth
  2. Fine-tune with BSM-RLI curriculum using completion loss-masking
     - Standard models target direct triggers
     - Reasoning models (Qwen3, DeepSeek-R1) target <think> traces + triggers
  3. Evaluate on fixed 50-item benchmark with batched inference & adaptive token budgets
  4. Save LoRA weights to models/finetuned/<model_key>/

Checkpoint-safe: saves results to benchmarks/results/finetuned_sweep.json
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import re
import torch
import gc
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from models.multi_model_catalog import MODELS_BY_VRAM
from benchmarks.baseline_sweep import get_batch_size

CURRICULUM_PATH = "dataset/bsm_rli_curriculum_75k.json"
RESULTS_PATH = "benchmarks/results/finetuned_sweep.json"
LORA_OUTPUT_ROOT = "models/finetuned"

SKIP_FINETUNE_VRAM_THRESHOLD = 9.0  # GB

SKIP_MODEL_KEYS = {
    "phi-4-mini-4bit",
    "gemma-4-e4b-4bit",
    "ministral-3-8b-4bit",
}


class CoTCompletionDataCollator:
    """Masks input prompt tokens with -100 so loss applies strictly to assistant responses."""

    def __init__(self, tokenizer, response_template):
        self.tokenizer = tokenizer
        self.response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def __call__(self, examples):
        batch_input_ids = [
            torch.tensor(e["input_ids"])
            if isinstance(e["input_ids"], list)
            else e["input_ids"]
            for e in examples
        ]
        padded = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask = (padded != self.pad_token_id).long()
        labels = padded.clone()

        for i, ids in enumerate(batch_input_ids):
            ids_list = ids.tolist()
            match_idx = -1
            for k in range(len(ids_list) - len(self.response_template_ids) + 1):
                if (
                    ids_list[k : k + len(self.response_template_ids)]
                    == self.response_template_ids
                ):
                    match_idx = k + len(self.response_template_ids)
                    break
            if match_idx != -1:
                labels[i, :match_idx] = -100
            else:
                labels[i, : len(ids_list) // 2] = -100
            labels[i, padded[i] == self.pad_token_id] = -100

        return {
            "input_ids": padded,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def get_response_template(chat_template):
    if chat_template == "gemma":
        return "<start_of_turn>model\n"
    elif chat_template == "llama3":
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif chat_template == "mistral":
        return "[/INST]"
    elif chat_template == "phi4":
        return "<|assistant|>\n"
    else:
        return "<|im_start|>assistant\n"


def build_eval_set(n=50):
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for i in range(n):
        q = gsm[i]["question"]
        ans_raw = gsm[i]["answer"]
        match = re.search(r"####\s*([\d,\.\-]+)", ans_raw)
        expected = match.group(1).replace(",", "").strip() if match else None
        items.append({"question": q, "expected": expected})
    return items


def check_answer(generated: str, expected: str) -> bool:
    if expected is None:
        return False
    nums = re.findall(r"[\d,]+\.?\d*", generated.replace(",", ""))
    clean_expected = expected.replace(",", "")
    return clean_expected in nums or clean_expected in generated.replace(",", "")


def finetune_and_eval(model_key, config, eval_set, curriculum_data):
    lora_out = os.path.join(LORA_OUTPUT_ROOT, model_key.replace("/", "_"))
    os.makedirs(lora_out, exist_ok=True)

    print(f"\n  [Fine-tune] Loading {config['model_name']} via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=512,
        load_in_4bit=True,
        dtype=torch.float16,
        trust_remote_code=True,
    )

    special_tokens = ["<|jit_start|>", "<|jit_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    is_thinking = config["family"] in {"deepseek-r1", "qwen3"}
    resp_template = get_response_template(config.get("chat_template", "chatml"))

    formatted = []
    for item in curriculum_data[:4500]:
        resp = (
            item.get("cot_response")
            if is_thinking and "cot_response" in item
            else item["response"]
        )
        text = config["prompt_format"].format(prompt=item["instruction"]) + resp
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)
    collator = CoTCompletionDataCollator(tokenizer, resp_template)

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
            logging_steps=30,
            output_dir=lora_out,
            save_strategy="no",
            report_to="none",
        ),
    )

    print("  [Fine-tune] Training 150 steps with CoT completion loss masking...")
    trainer.train()

    model.save_pretrained(lora_out)
    tokenizer.save_pretrained(lora_out)
    print(f"  [Fine-tune] LoRA weights saved to {lora_out}")

    # ── Evaluate the fine-tuned model (batched) ──────────────────
    batch_size = get_batch_size(config["vram_gb"], is_thinking)
    max_new_tokens = 1024 if is_thinking else 256
    print(
        f"  [Eval] Batched evaluation (batch={batch_size}, budget={max_new_tokens})..."
    )

    correct = 0
    total_tokens = 0
    t0 = time.time()
    prompts = [
        config["prompt_format"].format(prompt=item["question"]) for item in eval_set
    ]

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

        input_len = encoding.input_ids.shape[1]
        for j, output in enumerate(outputs):
            new_tokens = output[input_len:]
            total_tokens += len(new_tokens)
            gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            if check_answer(gen_text, batch_expected[j]):
                correct += 1

    elapsed = time.time() - t0
    accuracy = round((correct / len(eval_set)) * 100.0, 2)
    avg_tokens = round(total_tokens / len(eval_set), 1)
    throughput = round(len(eval_set) / elapsed, 2)

    return {
        "model_name": config["model_name"],
        "family": config["family"],
        "parameters": config["parameters"],
        "vram_gb": config["vram_gb"],
        "stage": "finetuned_cot_preserving_150steps",
        "accuracy_pct": accuracy,
        "correct": correct,
        "total": len(eval_set),
        "avg_tokens_per_sample": avg_tokens,
        "token_budget_used": max_new_tokens,
        "batch_size": batch_size,
        "throughput_samples_per_sec": throughput,
        "total_eval_seconds": round(elapsed, 2),
        "lora_weights": lora_out,
    }


def run_finetuning_sweep():
    os.makedirs("benchmarks/results", exist_ok=True)
    os.makedirs(LORA_OUTPUT_ROOT, exist_ok=True)

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f"[Resume] Found existing results for {len(results)} models.")
    else:
        results = {}

    print("=================================================================")
    print("  BSM-RLI CoT-PRESERVING FINE-TUNING SWEEP — 21 MODELS           ")
    print("  150-Step Unsloth QLoRA | Completion Loss Masking Active         ")
    print("=================================================================\n")

    with open(CURRICULUM_PATH) as f:
        curriculum_data = json.load(f)
    print(f"[Dataset] Loaded {len(curriculum_data)} curriculum pairs.\n")

    eval_set = build_eval_set(50)

    for model_key, config in MODELS_BY_VRAM:
        if model_key in SKIP_MODEL_KEYS:
            print(f"[Skip] {model_key} — known unsupported/crash risk.")
            continue

        if model_key in results and results[model_key].get("stage") == "finetuned_cot_preserving_150steps":
            print(f"[Skip] {model_key} — already evaluated with CoT-preserving SFT.")
            continue

        if config["vram_gb"] > SKIP_FINETUNE_VRAM_THRESHOLD:
            print(
                f"[Skip] {model_key} — VRAM {config['vram_gb']}GB exceeds training threshold ({SKIP_FINETUNE_VRAM_THRESHOLD}GB)."
            )
            results[model_key] = {
                "model_name": config["model_name"],
                "parameters": config["parameters"],
                "vram_gb": config["vram_gb"],
                "stage": "skipped_vram_limit",
                "accuracy_pct": None,
            }
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)
            continue

        print(
            f"\n[Model {model_key}] {config['parameters']} | ~{config['vram_gb']} GB VRAM"
        )

        try:
            result = finetune_and_eval(model_key, config, eval_set, curriculum_data)
            results[model_key] = result
            print(
                f"  --> Fine-tuned Accuracy: {result['accuracy_pct']}% | "
                f"Avg tokens: {result['avg_tokens_per_sample']} | "
                f"Time: {result['total_eval_seconds']}s"
            )

        except Exception as e:
            print(f"  [Error] {model_key}: {e}")
            results[model_key] = {
                "model_name": config["model_name"],
                "stage": "finetuned_cot_preserving_150steps",
                "error": str(e),
                "accuracy_pct": None,
            }
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=================================================================")
    print("  FINE-TUNING SWEEP COMPLETE")
    print(f"  Results saved to: {RESULTS_PATH}")
    print("=================================================================\n")

    print(f"{'Model Key':<35} {'Params':<12} {'FT Accuracy':<14} {'Avg Tokens'}")
    print("-" * 75)
    for key, r in results.items():
        if r.get("accuracy_pct") is not None:
            print(
                f"{key:<35} {r['parameters']:<12} {r['accuracy_pct']:<14} {r.get('avg_tokens_per_sample', '—')}"
            )

    return results


if __name__ == "__main__":
    run_finetuning_sweep()
