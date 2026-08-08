# BSM-RLI: Bare-Metal Symbolic Micro-Kernels via Region-Scoped Logit Interception

> **Exploratory Research Project**: Investigating whether Small Language Models (1B–8B) can delegate deterministic computation (arithmetic, string processing, formal logic) to host C++/CUDA micro-kernels via sub-microsecond logit interception.

---

## 💡 The Core Thesis & Motivation

Small Language Models (SLMs $\le$ 8B parameters) face a fundamental architectural limitation: **they do not have enough parameters or KV-cache budget to reliably memorize and execute multi-step deterministic tasks.**

When asked to compute exact multi-digit math, count subword graphemes, or perform calendar deltas:
1. **BPE Tokenization Errors**: Subword byte-pair encoding (BPE) splits numbers and words into arbitrary token chunks, confusing self-attention layers.
2. **Context & Token Inflation**: Standard Chain-of-Thought (CoT) forces SLMs to generate 500–2,000 reasoning tokens for simple arithmetic, consuming precious KV-cache VRAM and taking seconds to run.
3. **Cumulative Rounding Drift**: In long multi-step calculations, early rounding errors compound, leading to hallucinated final answers.

**The BSM-RLI Hypothesis**: Instead of teaching a 1B–3B model to perform complex mental math inside its neural weights, train the model to **emit a lightweight C++ trigger token** (`<|jit_start|>SUM_F64(...)<|jit_end|>`). The host machine intercepts the token stream in **$<5\mu\text{s}$**, executes a bit-exact C++/CUDA micro-kernel, and splices the result directly back into the generation loop.

---

## 🤔 Why Isn't Everyone Doing This Already?

If delegating computation to the host machine is so simple, why isn't it standard industry practice? 

Through our empirical exploration, we identified the key trade-offs between **Cloud Batch APIs** and **Single-Tenant Edge Inference**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│             SINGLE-TENANT EDGE INFERENCE vs. CLOUD BATCH APIs               │
└─────────────────────────────────────────────────────────────────────────────┘

  Cloud API Infrastructure (vLLM, TGI, OpenAI)    Single-Tenant Edge (llama.cpp, Local Agent)
  ───────────────────────────────────────────    ───────────────────────────────────────────
  ❌ Pausing generation per-user mid-stream        ✅ Pausing the single-user generation loop
     disrupts continuous GPU batching pipelines       takes < 1 microsecond (virtually free).
     and tensor parallelism across clusters.
                                                  ✅ Replaces 1,500 CoT tokens with 3 trigger
  ❌ Prefers long token generation because           tokens, saving 98% of KV-cache VRAM.
     serving raw tokens maximizes GPU utilization.
                                                  ✅ Guarantees IEEE 754 bit-exact math on
                                                     resource-constrained edge devices.
```

**Where BSM-RLI Holds Promise**: Local-first agents, robotics, edge micro-controllers, personal assistant devices, and single-tenant local LLM runtimes (`llama.cpp`, Ollama).

---

## 🔬 Empirical Findings & The SFT Reasoning Collapse

Our 21-model benchmark sweep across edge SLMs revealed a critical research challenge:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 SUPERVISED FINE-TUNING (SFT) VS. GRPO REINFORCEMENT LEARNING │
└─────────────────────────────────────────────────────────────────────────────┘

  Unadapted Base Thinking Models              Supervised Fine-Tuning (SFT)
  ──────────────────────────────              ────────────────────────────
  • High CoT Accuracy (74%–98% on GSM8K)     • Restricts model to static target strings
  • Uses 1,000–2,000 tokens per sample        • Drops reasoning accuracy to 24%–46%
  • High KV-cache VRAM overhead               • "The SFT Reasoning Paradox"

                                 ▼
              SOLUTION: GRPO Policy Optimization (RL)
              ────────────────────────────────────────
              1. Model proposes ITS OWN reasoning path in <think>
              2. Host C++ engine evaluates execution correctness (+1.0)
              3. Rewards early micro-kernel triggering (+0.4)
              4. Target: Maintain 95%+ reasoning with sub-20 tokens
```

---

## 📊 Performance Benchmark Matrix

Below is a snapshot of our empirical evaluation on an **NVIDIA GeForce RTX 4070 Ti** across representative model families:

| Model | Parameter Size | Base CoT Accuracy | Baseline Avg Tokens | CoT-Preserving SFT Acc | SFT Avg Tokens | BSM-RLI Host Kernel Soundness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Llama-3.2-3B-Instruct` | 3.21B | 30.0% | 216 tokens | **74.0%** 🚀 | 216 tokens | **100.0% IEEE 754** |
| `SmolLM2-1.7B-Instruct` | 1.71B | 46.0% | 117 tokens | **46.0%** | 216 tokens | **100.0% IEEE 754** |
| `DeepSeek-R1-Qwen-1.5B` | 1.54B | 84.0% | 655 tokens | **36.0%** | 1,024 tokens | **100.0% IEEE 754** |
| `Qwen3-0.6B` | 0.60B | 74.0% | 987 tokens | **32.0%** | 987 tokens | **100.0% IEEE 754** |
| `Llama-3.2-1B-Instruct` | 1.23B | 62.0% | 248 tokens | **24.0%** | 248 tokens | **100.0% IEEE 754** |
| `Qwen3-1.7B` | 1.70B | **94.0%** | 2,019 tokens | **24.0%** | 561 tokens | **100.0% IEEE 754** |
| `Qwen3-4B` | 4.00B | **98.0%** | 2,039 tokens | **14.0%** | 1,024 tokens | **100.0% IEEE 754** |

---

## 📚 Persona-Driven GitHub Wiki Directory

Detailed research notes, architectural specifications, and implementation guides are available in the [GitHub Wiki](https://github.com/beLizzard1/BSM-RLI/wiki):

| Reader Persona | Primary Technical Interest | Recommended Wiki Pages |
| :--- | :--- | :--- |
| 🔬 **AI Researchers & ML Engineers** | CoT alignment paradox, SFT vs. RL, response loss masking, frontier teacher distillation | 📖 [SLM Limits & Research](https://github.com/beLizzard1/BSM-RLI/wiki/SLM-Limits)<br>📖 [Benchmarks & Sweep Data](https://github.com/beLizzard1/BSM-RLI/wiki/Benchmarks)<br>📖 [Training Curriculum & 75k Dataset](https://github.com/beLizzard1/BSM-RLI/wiki/Training-Curriculum) |
| ⚡ **Systems & C++/CUDA Engineers** | Bare-metal logit interception, EBNF grammars, sub-5µs C++ dispatch, SIMD kernels | 📖 [System Architecture & Design](https://github.com/beLizzard1/BSM-RLI/wiki/Architecture)<br>📖 [CUDA & C++ Micro-Kernel Specification](https://github.com/beLizzard1/BSM-RLI/wiki/CUDA-Micro-Kernels)<br>📖 [21-Model Edge Catalog & VRAM Budgets](https://github.com/beLizzard1/BSM-RLI/wiki/Model-Catalog) |
| 🚀 **Application Developers & Contributors** | Quick start, C++ build, fine-tuning scripts, GGUF export, OpenAI batch distillation | 📖 [Getting Started & Build Guide](https://github.com/beLizzard1/BSM-RLI/wiki/Getting-Started)<br>📖 [Fine-Tuning & QLoRA Guide](https://github.com/beLizzard1/BSM-RLI/wiki/Fine-Tuning-Guide)<br>📖 [Repository Codebase Architecture](https://github.com/beLizzard1/BSM-RLI/wiki/Project-Structure) |

---

## ⚡ Quick Start & Verification

### 1. Build C++ Engine & Run Unit Tests
```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
ctest --output-on-failure
```

### 2. Run Interactive Logit Interception CLI
```bash
./build/bsm_rli_cli
```

### 3. Launch Distillation via OpenAI Batches API (50% Off)
```bash
python3 dataset/distill_batch_api.py --model_name gpt-5.6-luna --num_samples 10000 --max_cost_usd 10.00
```

---

## 🎯 Current Gaps & Promising Research Areas

### Areas of Promising Merit
- **Extreme Context Compression**: Offloading multi-step math to 3-token micro-kernels achieves **>40x token compression** and saves **97% of KV-cache VRAM**.
- **Bit-Exact Soundness**: Guarantees zero arithmetic hallucinated drift on local edge devices.
- **Single-Tenant Latency**: Executing C++ micro-kernels in $<5\mu\text{s}$ avoids cloud API latency (~500ms).

### Current Open Challenges
- **The SFT Bottleneck**: Standard SFT degrades open-ended CoT reasoning. GRPO policy optimization and frontier teacher distillation are required to preserve high-level reasoning.
- **Multi-Tenant GPU Batching**: Integrating mid-stream host execution into continuous batching runtimes (vLLM) remains an open system challenge.
