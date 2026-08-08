"""
Synthetic Dataset Generator for BSM-RLI (50,000+ Scaled Hybrid Dataset)
Generates 50,000 micro-kernel instruction pairs + 10,000 general conversational rejection pairs.
"""

import json
import random
import sys

def generate_math_samples(count=15000):
    samples = []
    for _ in range(count):
        vals = [round(random.uniform(1.0, 500.0), 2) for _ in range(random.randint(3, 8))]
        expected_sum = sum(vals)
        vals_str = ", ".join(map(str, vals))
        
        sample = {
            "instruction": f"Add the following numbers: {vals_str}",
            "response": f"The total sum is <|jit_start|>SUM_F64({vals_str})<|jit_end|> {expected_sum:.2f}.",
            "ground_truth": f"{expected_sum:.2f}"
        }
        samples.append(sample)
    return samples

def generate_char_count_samples(count=15000):
    words = ["strawberry", "microkernel", "interception", "unsloth", "symbolic", "architecture", "deterministic", "co-processor"]
    chars = ['r', 'e', 'c', 's', 'i', 't', 'a']
    samples = []
    for _ in range(count):
        word = random.choice(words)
        char = random.choice(chars)
        cnt = word.count(char)
        
        sample = {
            "instruction": f"How many times does the letter '{char}' appear in the word '{word}'?",
            "response": f"The letter '{char}' appears <|jit_start|>COUNT_CHAR(\"{word}\", '{char}')<|jit_end|> {cnt} times in '{word}'.",
            "ground_truth": str(cnt)
        }
        samples.append(sample)
    return samples

def generate_bitwise_samples(count=10000):
    ops = ["AND", "OR", "XOR"]
    samples = []
    for _ in range(count):
        a = random.randint(1, 255)
        b = random.randint(1, 255)
        op = random.choice(ops)
        if op == "AND": res = a & b
        elif op == "OR": res = a | b
        else: res = a ^ b

        sample = {
            "instruction": f"Compute bitwise {op} between {a} and {b}.",
            "response": f"The bitwise result is <|jit_start|>BITWISE_OP({a}, \"{op}\", {b})<|jit_end|> {res}.",
            "ground_truth": str(res)
        }
        samples.append(sample)
    return samples

def generate_sat_samples(count=10000):
    samples = []
    for _ in range(count):
        num_vars = random.randint(2, 4)
        sample = {
            "instruction": f"Solve Boolean SAT formula for {num_vars} variables with clauses [[1, 2], [-1, 2]].",
            "response": f"The satisfying assignment is <|jit_start|>SOLVE_SAT({num_vars}, [[1, 2], [-1, 2]])<|jit_end|> SAT: [0,1].",
            "ground_truth": "SAT: [0,1]"
        }
        samples.append(sample)
    return samples

def generate_rejection_samples(count=10000):
    rejection_prompts = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Write a haiku about nature.", "Green trees gently sway,\nSoft breeze whispers through the leaves,\nNature wakes in peace."),
        ("Explain the theory of relativity in simple terms.", "Einstein's theory of relativity explains how time and space are connected for objects moving at consistent speeds."),
        ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
        ("What is photosyntesis?", "Photosynthesis is the process used by plants to convert sunlight into energy.")
    ]
    samples = []
    for _ in range(count):
        prompt, resp = random.choice(rejection_prompts)
        sample = {
            "instruction": prompt,
            "response": resp,
            "ground_truth": resp
        }
        samples.append(sample)
    return samples

if __name__ == "__main__":
    scale_size = 50000
    if len(sys.argv) > 1:
        try:
            scale_size = int(sys.argv[1])
        except ValueError:
            pass

    print(f"Generating {scale_size}+ scaled hybrid synthetic dataset...")
    
    math_cnt = int(scale_size * 0.3)
    char_cnt = int(scale_size * 0.3)
    bit_cnt = int(scale_size * 0.2)
    sat_cnt = int(scale_size * 0.2)
    rejection_cnt = 10000

    dataset = (generate_math_samples(math_cnt) + 
               generate_char_count_samples(char_cnt) + 
               generate_bitwise_samples(bit_cnt) + 
               generate_sat_samples(sat_cnt) + 
               generate_rejection_samples(rejection_cnt))
    
    random.shuffle(dataset)

    output_path = "dataset/bsm_rli_sft_50k.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Successfully generated {len(dataset)} training samples (50k micro-kernel + 10k rejection) in {output_path}")
