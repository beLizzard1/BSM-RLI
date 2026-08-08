# Capability Specification: Google Gemma Edge Model Integration

> **Gemma-2B/9B & PaliGemma SentencePiece Formatting, Header Alignment, and Logit Masking**

---

## 1. Specification Overview

This specification details the tokenizer formatting and host logit masking requirements for running BSM-RLI on Google Gemma open-weights edge models (`Gemma-2B`, `Gemma-2-2B`, `Gemma-2-9B`, `PaliGemma-3B`).

---

## 2. Requirements & Guarantees

### Requirement 1: SentencePiece Header Format
- **MUST** support Gemma's native SentencePiece chat template tags: `<start_of_turn>user`, `<end_of_turn>`, `<start_of_turn>model`.

```text
<start_of_turn>user
Evaluate 1000 * (1 + 0.05/12)^(12*10) - 250<end_of_turn>
<start_of_turn>model
<|jit_start|>EVAL_EXPR("1000 * (1 + 0.05/12)^(12*10) - 250")<|jit_end|>
```

### Requirement 2: Special Token Registration
- **MUST** register `<|jit_start|>` (ID: `128256` or extended vocab) and `<|jit_end|>` as control tokens without breaking Gemma's default embedding matrix.

### Requirement 3: C++ GBNF Grammar Enforcement
- **MUST** enforce `grammars/bsm_rli.gbnf` in Gemma inference engines (`llama.cpp`, `vLLM`, `ollama`) to restrict trigger output space during math, search, and coding tasks.
