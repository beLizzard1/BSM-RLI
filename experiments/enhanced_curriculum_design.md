# Research & Curriculum Specification: Enhanced 4-Phase Fine-Tuning & Alignment Curriculum

> **Architectural Curriculum Design Targeting All 4 Empirical Failure Modes of Small Language Models (SLMs) in BSM-RLI.**

---

## 1. Problem Statement: SLM Failure Modes Addressed

Our stress-testing identified 4 specific breaking points where a 1B–3B parameter SLM fails when interfacing with micro-kernels:

1. **Formula Framing Collapse (32% failure on MATH Level 5)**: Misinterprets multi-clause word problems.
2. **Sequential State Drift (32% failure on Multi-Kernel Chaining)**: Drops intermediate register variables across 3+ steps.
3. **Parameter Injection Noise (18% failure on Distractor Contexts)**: Injects decoy numbers from noisy prompts into micro-kernel calls.
4. **CoT Fallback Collapse (86% failure on Unmapped Domains)**: Reverts to slow, buggy CoT text when a pre-compiled kernel is missing.

---

## 2. The Enhanced 4-Phase Curriculum Matrix (75,000 Total Pairs)

| Curriculum Phase | Dataset Size | Targeted SLM Failure Mode | Training Input & Response Format | Target Accuracy After Curriculum |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: MATH Level 5 Complex Framing** | **18,750 pairs** | Formula Framing Collapse | Multi-clause competition word problems mapped directly into `<|jit_start|>EVAL_EXPR("...")<|jit_end|>`. | **`97.5%`** (up from 68.0%) |
| **Phase 2: Multi-Kernel Register Chaining** | **18,750 pairs** | Sequential State Drift | Multi-step workflows emitting `<|jit_start|>REG_STORE(...)<|jit_end|>` and `<|jit_start|>REG_READ(...)<|jit_end|>`. | **`96.0%`** (up from 68.0%) |
| **Phase 3: Hardened Distractor Noise Filtering** | **18,750 pairs** | Parameter Injection Noise | Prompts containing 500-word financial noise and decoy numbers, training attention heads to isolate exact parameters. | **`99.0%`** (up from 82.0%) |
| **Phase 4: Dynamic Runtime C++ JIT Fallback** | **18,750 pairs** | CoT Fallback Collapse | Unmapped domain queries mapped into raw C++ code snippets inside `<|jit_start|>DYN_CPP("...")<|jit_end|>`. | **`88.5%`** (up from 14.0%) |
| **TOTAL ENHANCED CURRICULUM** | **75,000 pairs** | **All 4 Failure Modes** | Dataset artifact: [`dataset/bsm_rli_curriculum_75k.json`](file:///home/liz/Projects/BSM-RLI/dataset/bsm_rli_curriculum_75k.json) | **`95.25% Average`** |

---

## 3. Curriculum Pipeline Implementation

1. **Synthetic Generator**: [`dataset/generate_enhanced_curriculum.py`](file:///home/liz/Projects/BSM-RLI/dataset/generate_enhanced_curriculum.py) producing 75,000 multi-phase JSON pairs in [`dataset/bsm_rli_curriculum_75k.json`](file:///home/liz/Projects/BSM-RLI/dataset/bsm_rli_curriculum_75k.json).
2. **Fine-Tuning Script**: [`training/train_unsloth_sft.py`](file:///home/liz/Projects/BSM-RLI/training/train_unsloth_sft.py) updating `unsloth/Llama-3.2-1B-Instruct` or `Gemma-2B`.
3. **GRPO Preference Alignment**: [`training/train_unsloth_grpo.py`](file:///home/liz/Projects/BSM-RLI/training/train_unsloth_grpo.py) penalizing distractor parameter injection and CoT token bloat.
