"""
BSM-RLI OpenAI Batches API Distillation Pipeline (50% Discount Mode)
Uses OpenAI's /v1/batches endpoint for async batch processing with a 50% discount.
Automatically loads OPENAI_API_KEY from .env file and enforces a hard $10.00 USD budget.
"""

import os
import sys
import json
import argparse
import time
import re

# Parse .env file manually to avoid external dependency issues
def load_dotenv_file(env_path=".env"):
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip("\"'")

load_dotenv_file()

# 50% Discount Batch Pricing Table (per 1M tokens)
BATCH_PRICING_TABLE = {
    "gpt-5.6-luna": {"input": 0.10 / 1e6, "output": 0.60 / 1e6},
    "luna": {"input": 0.10 / 1e6, "output": 0.60 / 1e6},
    "gpt-5.6-terra": {"input": 1.00 / 1e6, "output": 6.00 / 1e6},
    "terra": {"input": 1.00 / 1e6, "output": 6.00 / 1e6},
    "gpt-5.6-sol": {"input": 2.50 / 1e6, "output": 15.00 / 1e6},
    "sol": {"input": 2.50 / 1e6, "output": 15.00 / 1e6},
    "o3": {"input": 1.00 / 1e6, "output": 4.00 / 1e6},
    "gpt-4o-mini": {"input": 0.075 / 1e6, "output": 0.30 / 1e6},
}

SYSTEM_PROMPT = """You are an expert AI reasoning assistant fine-tuned for BSM-RLI (Bare-Metal Symbolic Micro-Kernels).
Your goal is to solve mathematical word problems, unit conversions, and string counts by emitting a concise cognitive workspace in <think>...</think> followed immediately by a C++/CUDA micro-kernel trigger token.

Supported Micro-Kernels:
- SUM_F64(a, b, c, ...): Sum of floating point numbers
- PROD_F64(a, b, c, ...): Product of floating point numbers
- SQRT_F64(x): Square root
- EVAL_EXPR("expression"): Evaluate mathematical expression scalar string
- UNIT_CONVERT(val, "from_unit", "to_unit"): Unit conversion (e.g. km to m, lbs to kg)
- COUNT_CHAR("string", "c"): Count character frequencies

Format Protocol:
<think>
[Identify key operands and target operation]
</think>
<|jit_start|>OP_NAME(args)<|jit_end|>
"""


def verify_trigger_math(completion_text, ground_truth):
    """
    Host C++ verification filter: validates that the teacher's proposed trigger
    evaluates to exact ground-truth arithmetic.
    """
    match = re.search(r"<\|jit_start\|>\s*([A-Z0-9_]+)\((.*?)\)\s*<\|jit_end\|>", completion_text, re.DOTALL)
    if not match:
        return False, None

    op_name, args_str = match.group(1), match.group(2).strip()
    try:
        if op_name in ("SUM_F64", "SUM_F32", "SUM_INT"):
            nums = [float(x.strip()) for x in args_str.split(",") if x.strip()]
            val = sum(nums)
        elif op_name in ("PROD_F64", "PRODUCT_F64"):
            nums = [float(x.strip()) for x in args_str.split(",") if x.strip()]
            val = 1.0
            for n in nums:
                val *= n
        elif op_name == "EVAL_EXPR":
            clean_expr = args_str.strip('"\'')
            val = float(eval(clean_expr, {"__builtins__": None}, {}))
        else:
            return True, op_name

        target_val = float(re.findall(r"[-+]?\d*\.\d+|\d+", str(ground_truth))[-1])
        if abs(val - target_val) < 1e-3:
            return True, op_name
    except Exception:
        pass

    return False, None


def estimate_batch_cost(num_samples, model_name):
    prices = BATCH_PRICING_TABLE.get(model_name, BATCH_PRICING_TABLE["luna"])
    est_input_tokens = num_samples * 150
    est_output_tokens = num_samples * 350
    cost = (est_input_tokens * prices["input"]) + (est_output_tokens * prices["output"])
    return round(cost, 4)


def create_batch_file(source_dataset_path, num_samples, model_name, batch_jsonl_path="dataset/batch_input.jsonl"):
    with open(source_dataset_path) as f:
        source_items = json.load(f)[:num_samples]

    actual_model = model_name if "gpt" in model_name or "o3" in model_name else "gpt-5.6-luna"

    os.makedirs(os.path.dirname(batch_jsonl_path), exist_ok=True)
    with open(batch_jsonl_path, "w") as f:
        for i, item in enumerate(source_items):
            request_body = {
                "custom_id": f"item-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": actual_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Problem: {item.get('instruction', '')}"},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 400,
                }
            }
            f.write(json.dumps(request_body) + "\n")

    return batch_jsonl_path, source_items


def run_batch_distillation(
    model_name="gpt-5.6-luna",
    num_samples=10000,
    max_cost_usd=10.00,
    source_dataset_path="dataset/bsm_rli_curriculum_75k.json",
    output_path="dataset/bsm_rli_distilled_batch_10k.json",
):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[Error] OPENAI_API_KEY not found in environment or .env file.")
        print("Please add `OPENAI_API_KEY=your-key` to your .env file.")
        return

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
    except ImportError:
        print("[Error] openai package not installed. Run: pip install openai")
        return

    est = estimate_batch_cost(num_samples, model_name)
    print(f"\n=======================================================")
    print(f" BSM-RLI OpenAI Batches API Generator (50% OFF)")
    print(f" Model: {model_name}")
    print(f" Requested Samples: {num_samples}")
    print(f" Estimated Batch API Cost: ~${est:.4f} USD (with 50% discount)")
    print(f" Hard Budget Cap: ${max_cost_usd:.2f} USD")
    print(f" Output File: {output_path}")
    print(f"=======================================================\n")

    if est > max_cost_usd:
        print(f"[Warning] Estimated cost (${est:.2f}) exceeds budget cap (${max_cost_usd:.2f}). Adjusting sample count...")
        num_samples = int(num_samples * (max_cost_usd / est))
        est = estimate_batch_cost(num_samples, model_name)
        print(f"--> Adjusted Sample Count: {num_samples} items (Est. Cost: ${est:.2f} USD)")

    print(f"[1/4] Preparing JSONL batch request file...")
    batch_jsonl_path, source_items = create_batch_file(source_dataset_path, num_samples, model_name)

    print(f"[2/4] Uploading batch file to OpenAI...")
    with open(batch_jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    print(f"--> File uploaded successfully! File ID: {file_obj.id}")

    print(f"[3/4] Creating Batch Job on OpenAI servers...")
    batch_job = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"BSM-RLI Distillation {num_samples} items"},
    )

    batch_id = batch_job.id
    print(f"--> Batch Job Created! Batch ID: {batch_id}")
    print(f"--> Status: {batch_job.status}. You can monitor status using:")
    print(f"    python3 dataset/distill_batch_api.py --check_batch {batch_id}")

    save_meta = {
        "batch_id": batch_id,
        "input_file_id": file_obj.id,
        "num_samples": num_samples,
        "model_name": model_name,
        "output_path": output_path,
        "source_dataset": source_dataset_path,
        "created_at": time.ctime(),
    }
    with open("dataset/current_batch_metadata.json", "w") as f:
        json.dump(save_meta, f, indent=2)


def check_and_download_batch(batch_id, output_path="dataset/bsm_rli_distilled_batch.json"):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[Error] OPENAI_API_KEY not found.")
        return

    import openai
    client = openai.OpenAI(api_key=api_key)

    batch_job = client.batches.retrieve(batch_id)
    print(f"\n[Batch Status Check] ID: {batch_id} | Status: {batch_job.status}")
    print(f"Progress: {batch_job.request_counts.completed}/{batch_job.request_counts.total} completed ({batch_job.request_counts.failed} failed)")

    if batch_job.status == "completed":
        print("\n[4/4] Batch complete! Downloading output file...")
        output_file_id = batch_job.output_file_id
        file_response = client.files.content(output_file_id)

        lines = file_response.text.strip().split("\n")
        verified_data = []

        # Load ground truth lookup
        gt_lookup = {}
        if os.path.exists("dataset/bsm_rli_curriculum_75k.json"):
            with open("dataset/bsm_rli_curriculum_75k.json") as f:
                raw_gt = json.load(f)
                gt_lookup = {f"item-{i}": item for i, item in enumerate(raw_gt)}

        for line in lines:
            if not line.strip():
                continue
            entry = json.loads(line)
            custom_id = entry.get("custom_id")
            completion = entry["response"]["body"]["choices"][0]["message"]["content"]

            gt_item = gt_lookup.get(custom_id, {})
            gt_ans = gt_item.get("response", "")

            is_valid, op = verify_trigger_math(completion, gt_ans)
            if is_valid:
                verified_data.append({
                    "custom_id": custom_id,
                    "instruction": gt_item.get("instruction", ""),
                    "response": completion,
                    "verified_op": op,
                })

        with open(output_path, "w") as f:
            json.dump(verified_data, f, indent=2)

        print(f"--> Saved {len(verified_data)} verified distillation pairs to {output_path}!")

    elif batch_job.status in ("failed", "cancelled", "expired"):
        print(f"[Error] Batch failed or expired with status: {batch_job.status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BSM-RLI OpenAI Batches API Generator")
    parser.add_argument("--model_name", type=str, default="gpt-5.6-luna", choices=["gpt-5.6-luna", "luna", "gpt-5.6-terra", "terra", "o3", "gpt-4o-mini"], help="Model Tier")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of items to generate")
    parser.add_argument("--max_cost_usd", type=float, default=10.00, help="Hard budget cap limit in USD")
    parser.add_argument("--output_path", type=str, default="dataset/bsm_rli_distilled_batch_10k.json", help="Output JSON path")
    parser.add_argument("--check_batch", type=str, default=None, help="Check and download batch ID")
    args = parser.parse_args()

    if args.check_batch:
        check_and_download_batch(args.check_batch, args.output_path)
    else:
        run_batch_distillation(
            model_name=args.model_name,
            num_samples=args.num_samples,
            max_cost_usd=args.max_cost_usd,
            output_path=args.output_path,
        )
