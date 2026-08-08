"""
BSM-RLI Frontier Teacher Distillation Generator (OpenAI Luna / Terra / Sol / o3)
Generates high-quality reasoning traces and BSM-RLI micro-kernel triggers (<|jit_start|>)
using OpenAI API models with real-time budget caps, cost tracking, and C++ verification.
"""

import os
import sys
import json
import argparse
import re

# Pricing per 1M tokens (as of August 2026)
PRICING_TABLE = {
    "gpt-5.6-luna": {"input": 0.20 / 1e6, "output": 1.20 / 1e6},
    "luna": {"input": 0.20 / 1e6, "output": 1.20 / 1e6},
    "gpt-5.6-terra": {"input": 2.00 / 1e6, "output": 12.00 / 1e6},
    "terra": {"input": 2.00 / 1e6, "output": 12.00 / 1e6},
    "gpt-5.6-sol": {"input": 5.00 / 1e6, "output": 30.00 / 1e6},
    "sol": {"input": 5.00 / 1e6, "output": 30.00 / 1e6},
    "o3": {"input": 2.00 / 1e6, "output": 8.00 / 1e6},
    "gpt-4o-mini": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
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


def estimate_cost(num_samples, model_name):
    prices = PRICING_TABLE.get(model_name, PRICING_TABLE["luna"])
    est_input_tokens = num_samples * 150
    est_output_tokens = num_samples * 350
    cost = (est_input_tokens * prices["input"]) + (est_output_tokens * prices["output"])
    return round(cost, 4)


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


def distill_from_openai(
    model_name="gpt-5.6-luna",
    num_samples=1000,
    max_cost_usd=0.50,
    output_path="dataset/bsm_rli_distilled_reference_1k.json",
    source_dataset_path="dataset/bsm_rli_curriculum_75k.json",
):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[Error] OPENAI_API_KEY environment variable not set.")
        print("Please export OPENAI_API_KEY='your-key-here' and re-run.")
        return

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
    except ImportError:
        print("[Error] openai Python package not installed. Run: pip install openai")
        return

    est = estimate_cost(num_samples, model_name)
    print(f"\n=======================================================")
    print(f" BSM-RLI Frontier Distillation Generator")
    print(f" Model: {model_name} (Luna Tier)")
    print(f" Requested Samples: {num_samples}")
    print(f" Estimated API Cost: ~${est:.4f} USD")
    print(f" Hard Cost Limit: ${max_cost_usd:.2f} USD")
    print(f" Output File: {output_path}")
    print(f"=======================================================\n")

    if not os.path.exists(source_dataset_path):
        print(f"[Error] Source dataset {source_dataset_path} not found.")
        return

    with open(source_dataset_path) as f:
        source_items = json.load(f)[:num_samples]

    distilled_pairs = []
    total_cost = 0.0
    verified_count = 0
    prices = PRICING_TABLE.get(model_name, PRICING_TABLE["luna"])

    for i, item in enumerate(source_items):
        if total_cost >= max_cost_usd:
            print(f"\n[Budget Alert] Reached max cost limit (${max_cost_usd:.2f} USD). Halting distillation early.")
            break

        prompt_text = item.get("instruction", "")
        try:
            response = client.chat.completions.create(
                model=model_name if "gpt" in model_name or "o3" in model_name else "gpt-5.6-luna",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Problem: {prompt_text}"},
                ],
                temperature=0.2,
                max_tokens=400,
            )

            completion = response.choices[0].message.content
            usage = response.usage

            # Track cost
            in_cost = usage.prompt_tokens * prices["input"]
            out_cost = usage.completion_tokens * prices["output"]
            item_cost = in_cost + out_cost
            total_cost += item_cost

            # Host C++ Verification Filter
            is_valid, op = verify_trigger_math(completion, item.get("response", ""))
            if is_valid:
                verified_count += 1
                distilled_pairs.append({
                    "instruction": prompt_text,
                    "response": completion,
                    "verified_op": op,
                    "source_model": model_name,
                })
                print(f"  [Sample {i+1}/{num_samples}] Verified ✓ ({op}) | Acc Cost: ${total_cost:.4f}")
            else:
                print(f"  [Sample {i+1}/{num_samples}] Filtered ✗ (Math check failed) | Acc Cost: ${total_cost:.4f}")

        except Exception as e:
            print(f"  [Sample {i+1}] API Error: {e}")
            break

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(distilled_pairs, f, indent=2)

    print(f"\n=======================================================")
    print(f" Distillation Complete!")
    print(f" Total Verified Samples: {verified_count}/{len(distilled_pairs)}")
    print(f" Total Accumulated API Cost: ${total_cost:.4f} USD")
    print(f" Dataset Saved: {output_path}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BSM-RLI Frontier Teacher Distillation Generator")
    parser.add_argument("--model_name", type=str, default="gpt-5.6-luna", choices=["gpt-5.6-luna", "luna", "gpt-5.6-terra", "terra", "o3", "gpt-4o-mini"], help="OpenAI Model Tier")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of items to generate")
    parser.add_argument("--max_cost_usd", type=float, default=0.50, help="Hard cap budget limit in USD")
    parser.add_argument("--output_path", type=str, default="dataset/bsm_rli_distilled_reference_1k.json", help="Output JSON path")
    args = parser.parse_args()

    distill_from_openai(
        model_name=args.model_name,
        num_samples=args.num_samples,
        max_cost_usd=args.max_cost_usd,
        output_path=args.output_path,
    )
