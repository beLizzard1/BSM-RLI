## MODIFIED Requirements

### Requirement: Region-Scoped Logit Mask Enforcement
The system SHALL keep next-token sampling unconstrained during normal text generation and transition into constrained EBNF logit sampling immediately upon encountering `<|jit_start|>`.

#### Scenario: Intercepting JIT trigger token
- **WHEN** the inference engine samples the token `<|jit_start|>`
- **THEN** the system SHALL clamp all subsequent candidate token logits strictly to the registered EBNF grammar using `LLGuidanceAdapter` FSM bitmasks

#### Scenario: Exiting JIT region and injecting KV cache result
- **WHEN** the `<|jit_end|>` token is reached
- **THEN** the system SHALL execute the intercepted micro-kernel, write the ground-truth result tokens into the KV cache, and restore unconstrained decoding
