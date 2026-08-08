"""
BSM-RLI Edge Model Catalog & Quantization Matrix (< 12GB VRAM Fit, RTX 4070 Ti)
Covers:
  - Google Gemma 4 (E2B, E4B, 12B) — April 2026
  - Qwen3 (0.6B, 1.7B, 4B, 8B) — Q2 2025
  - Meta Llama 3.3 (70B 4-bit fits in ~40GB but 8B fits here)
  - Meta Llama 3.2 (1B, 3B)
  - DeepSeek-R1 Distill (1.5B, 7B, 8B)
  - HuggingFace SmolLM2 (135M, 360M, 1.7B)
  - Microsoft Phi-4 Mini (3.8B) & Phi-4 (14B 4-bit)
  - Mistral / Ministral (8B)
  - Alibaba Qwen 2.5 (0.5B, 1.5B, 7B)
"""

EDGE_MODEL_CATALOG = {

    # ─────────────────────────────────────────────────────────────
    # GOOGLE GEMMA 4 (April 2026) — SentencePiece, multimodal
    # ─────────────────────────────────────────────────────────────
    "gemma-4-e2b-4bit": {
        "model_name": "unsloth/gemma-4-E2B-it-bnb-4bit",
        "family": "gemma4",
        "parameters": "2.0B (MoE)",
        "vram_gb": 2.0,
        "chat_template": "gemma",
        "multimodal": True,
        "prompt_format": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
        "notes": "Google Gemma 4 E2B MoE — 2B active params, vision-capable",
    },
    "gemma-4-e4b-4bit": {
        "model_name": "unsloth/gemma-4-E4B-it-bnb-4bit",
        "family": "gemma4",
        "parameters": "4.0B (MoE)",
        "vram_gb": 3.8,
        "chat_template": "gemma",
        "multimodal": True,
        "prompt_format": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
        "notes": "Google Gemma 4 E4B MoE — 4B active params, vision-capable",
    },
    "gemma-4-12b-4bit": {
        "model_name": "unsloth/gemma-4-12B-it-bnb-4bit",
        "family": "gemma4",
        "parameters": "12.0B",
        "vram_gb": 8.0,
        "chat_template": "gemma",
        "multimodal": True,
        "prompt_format": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
        "notes": "Google Gemma 4 12B dense — fits 12GB VRAM at 4-bit",
    },
    # Gemma 2 (stable prior gen)
    "gemma-2-2b-4bit": {
        "model_name": "unsloth/gemma-2-2b-it-bnb-4bit",
        "family": "gemma2",
        "parameters": "2.61B",
        "vram_gb": 1.8,
        "chat_template": "gemma",
        "multimodal": False,
        "prompt_format": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
    },
    "gemma-2-9b-4bit": {
        "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
        "family": "gemma2",
        "parameters": "9.24B",
        "vram_gb": 5.8,
        "chat_template": "gemma",
        "multimodal": False,
        "prompt_format": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
    },

    # ─────────────────────────────────────────────────────────────
    # ALIBABA QWEN 3 (Q2 2025) — ChatML template
    # ─────────────────────────────────────────────────────────────
    "qwen3-0.6b-4bit": {
        "model_name": "unsloth/Qwen3-0.6B-bnb-4bit",
        "family": "qwen3",
        "parameters": "0.60B",
        "vram_gb": 0.8,
        "chat_template": "chatml",
        "multimodal": False,
        "prompt_format": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "notes": "Tiny on-device reasoning model",
    },
    "qwen3-1.7b-4bit": {
        "model_name": "unsloth/Qwen3-1.7B-bnb-4bit",
        "family": "qwen3",
        "parameters": "1.70B",
        "vram_gb": 1.4,
        "chat_template": "chatml",
        "multimodal": False,
        "prompt_format": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    },
    "qwen3-4b-4bit": {
        "model_name": "unsloth/Qwen3-4B-bnb-4bit",
        "family": "qwen3",
        "parameters": "4.0B",
        "vram_gb": 2.9,
        "chat_template": "chatml",
        "multimodal": False,
        "prompt_format": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    },
    "qwen3-8b-4bit": {
        "model_name": "unsloth/Qwen3-8B-bnb-4bit",
        "family": "qwen3",
        "parameters": "8.0B",
        "vram_gb": 5.5,
        "chat_template": "chatml",
        "multimodal": False,
        "prompt_format": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    },
    # Qwen 2.5 (prior gen, stable)
    "qwen-2.5-0.5b-4bit": {
        "model_name": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "family": "qwen2.5",
        "parameters": "0.49B",
        "vram_gb": 0.7,
        "chat_template": "chatml",
        "multimodal": False,
        "prompt_format": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    },
    "qwen-2.5-7b-4bit": {
        "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "family": "qwen2.5",
        "parameters": "7.61B",
        "vram_gb": 4.8,
        "chat_template": "chatml",
        "multimodal": False,
        "prompt_format": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    },

    # ─────────────────────────────────────────────────────────────
    # META LLAMA 3.2 & 3.3 — Llama3 header template
    # ─────────────────────────────────────────────────────────────
    "llama-3.2-1b-4bit": {
        "model_name": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "family": "llama3",
        "parameters": "1.23B",
        "vram_gb": 1.2,
        "chat_template": "llama3",
        "multimodal": False,
        "prompt_format": "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "llama-3.2-3b-4bit": {
        "model_name": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "family": "llama3",
        "parameters": "3.21B",
        "vram_gb": 2.4,
        "chat_template": "llama3",
        "multimodal": False,
        "prompt_format": "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    },

    # ─────────────────────────────────────────────────────────────
    # DEEPSEEK-R1 DISTILL FAMILY
    # ─────────────────────────────────────────────────────────────
    "deepseek-r1-qwen-1.5b-4bit": {
        "model_name": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit",
        "family": "deepseek-r1",
        "parameters": "1.54B",
        "vram_gb": 1.5,
        "chat_template": "deepseek",
        "multimodal": False,
        "prompt_format": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n",
        "notes": "DeepSeek R1 reasoning distill into Qwen-1.5B",
    },
    "deepseek-r1-qwen-7b-4bit": {
        "model_name": "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit",
        "family": "deepseek-r1",
        "parameters": "7.61B",
        "vram_gb": 4.8,
        "chat_template": "deepseek",
        "multimodal": False,
        "prompt_format": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n",
    },
    "deepseek-r1-llama-8b-4bit": {
        "model_name": "unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit",
        "family": "deepseek-r1",
        "parameters": "8.03B",
        "vram_gb": 5.2,
        "chat_template": "deepseek",
        "multimodal": False,
        "prompt_format": "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<think>\n</think>\n",
    },

    # ─────────────────────────────────────────────────────────────
    # HUGGING FACE SMOLLM2 (Ultra-Lightweight MCU-Grade)
    # ─────────────────────────────────────────────────────────────
    "smollm2-135m-4bit": {
        "model_name": "unsloth/SmolLM2-135M-Instruct-bnb-4bit",
        "family": "smollm2",
        "parameters": "0.14B",
        "vram_gb": 0.3,
        "chat_template": "chatml",
        "multimodal": False,
        "prompt_format": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "notes": "MCU-grade — 300MB VRAM footprint",
    },
    "smollm2-360m-4bit": {
        "model_name": "unsloth/SmolLM2-360M-Instruct-bnb-4bit",
        "family": "smollm2",
        "parameters": "0.36B",
        "vram_gb": 0.5,
        "chat_template": "chatml",
        "multimodal": False,
        "prompt_format": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    },
    "smollm2-1.7b-4bit": {
        "model_name": "unsloth/SmolLM2-1.7B-Instruct-bnb-4bit",
        "family": "smollm2",
        "parameters": "1.71B",
        "vram_gb": 1.4,
        "chat_template": "chatml",
        "multimodal": False,
        "prompt_format": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    },

    # ─────────────────────────────────────────────────────────────
    # MICROSOFT PHI-4 MINI & PHI-4
    # ─────────────────────────────────────────────────────────────
    "phi-4-mini-4bit": {
        "model_name": "unsloth/Phi-4-mini-instruct-bnb-4bit",
        "family": "phi4",
        "parameters": "3.84B",
        "vram_gb": 2.9,
        "chat_template": "phi4",
        "multimodal": False,
        "prompt_format": "<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
        "notes": "Phi-4 Mini — strong math & reasoning for size",
    },

    # ─────────────────────────────────────────────────────────────
    # MISTRAL / MINISTRAL
    # ─────────────────────────────────────────────────────────────
    "ministral-8b-4bit": {
        "model_name": "unsloth/Ministral-8B-Instruct-2410-bnb-4bit",
        "family": "mistral",
        "parameters": "8.02B",
        "vram_gb": 5.1,
        "chat_template": "mistral",
        "multimodal": False,
        "prompt_format": "[INST] {prompt} [/INST]",
    },
}

# VRAM budget sorted listing
MODELS_BY_VRAM = sorted(
    EDGE_MODEL_CATALOG.items(),
    key=lambda x: x[1]["vram_gb"]
)

def get_model_config(key):
    return EDGE_MODEL_CATALOG.get(key, EDGE_MODEL_CATALOG["llama-3.2-1b-4bit"])

def list_catalog():
    print(f"\n{'Model Key':<35} {'Family':<12} {'Params':<12} {'VRAM (GB)':<12} {'Multimodal'}")
    print("-" * 90)
    for key, cfg in MODELS_BY_VRAM:
        mm = "✅" if cfg.get("multimodal") else "—"
        print(f"{key:<35} {cfg['family']:<12} {cfg['parameters']:<12} {cfg['vram_gb']:<12} {mm}")

if __name__ == "__main__":
    list_catalog()
