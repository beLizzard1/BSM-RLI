"""
BSM-RLI Unsloth GRPO (Group Relative Policy Optimization) Alignment Script
Aligns model preference toward 2-token micro-kernel execution over verbose CoT scratchpads.
"""

import re
import torch
from unsloth import FastLanguageModel

# Reward Functions for GRPO

def reward_jit_trigger_format(completions, **kwargs):
    """
    Rewards completions that emit valid <|jit_start|>KERNEL(...)<|jit_end|> tags.
    """
    rewards = []
    pattern = r"<\|jit_start\|>([A-Z_]+)\((.*?)\)<\|jit_end\|>"
    for completion in completions:
        if re.search(pattern, completion):
            rewards.append(1.5)
        elif "<|jit_start|>" in completion:
            rewards.append(0.2)  # Partial match
        else:
            rewards.append(0.0)
    return rewards

def reward_token_economy(completions, **kwargs):
    """
    Penalizes verbose CoT responses when a concise kernel trigger is used.
    """
    rewards = []
    for completion in completions:
        token_count = len(completion.split())
        if "<|jit_start|>" in completion and token_count < 30:
            rewards.append(1.0)  # High efficiency reward
        elif token_count > 150:
            rewards.append(-0.5) # Penalty for verbose CoT bloat
        else:
            rewards.append(0.0)
    return rewards

def reward_ground_truth_match(completions, ground_truths, **kwargs):
    """
    Rewards exact mathematical and logical correctness of execution result.
    """
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        if str(gt) in completion:
            rewards.append(2.0)
        else:
            rewards.append(-1.0)
    return rewards

print("Unsloth GRPO reward functions initialized for BSM-RLI preference alignment!")
