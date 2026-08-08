## Context

To prevent neural generation errors during kernel trigger emission, BSM-RLI incorporates an FSM-based logit mask adapter (`LLGuidanceAdapter`) compatible with `llguidance` and `XGrammar`. Inside `<|jit_start|>` regions, candidate logits are clamped to `-infinity` for any token ID that violates the active EBNF grammar.

## Goals / Non-Goals

**Goals:**
- **llguidance / XGrammar Adapter Interface**: Provide [`include/bsm_rli/llguidance_adapter.hpp`](file:///home/liz/Projects/BSM-RLI/include/bsm_rli/llguidance_adapter.hpp) defining FSM token matching and logit mask clamping.
- **Vocabulary Bitmask Masking**: Implement `apply_logit_mask(float* logits, size_t vocab_size, const std::vector<std::string>& vocab)` to set `-std::numeric_limits<float>::infinity()` on invalid token indices.
- **Region Scoping**: Ensure logit masking is active **only** between `<|jit_start|>` and `<|jit_end|>`.

**Non-Goals:**
- Applying logit masks to general text outside JIT regions.

## Decisions

### 1. FSM Token Transition State Machine
- `LLGuidanceAdapter` parses the EBNF grammar produced by `KernelRegistry::generate_ebnf_grammar()`.
- During token sampling, `LLGuidanceAdapter::compute_valid_token_mask(...)` evaluates next-token string prefixes against valid EBNF rules.
- Invalid token IDs receive `-inf` in the logit array, guaranteeing 100% syntactically valid token generation.

## Risks / Trade-offs

- **[Risk]** Large vocabulary size ($32k - 128k$ tokens) causing logit mask calculation latency.  
  *Mitigation*: Pre-index token prefix trees (trie-based filtering) to ensure bitmask computation completes in $< 10\mu\text{s}$.
