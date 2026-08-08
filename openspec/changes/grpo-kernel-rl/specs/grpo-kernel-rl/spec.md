## ADDED Requirements

### Requirement: GRPO Composite Reward Evaluation Engine
The system SHALL evaluate candidate generation outputs from Small Language Models using a composite reward engine incorporating host C++/CUDA micro-kernel execution accuracy ($R_{\text{correctness}}$), trigger syntax format ($R_{\text{validity}}$), kernel selection directness ($R_{\text{kernel\_select}}$), and early token offloading economy ($R_{\text{economy}}$).

#### Scenario: Successful C++ micro-kernel execution reward
- **WHEN** a candidate generation output emits a valid `<|jit_start|>SUM_F64(...)<|jit_end|>` trigger and host C++ execution yields exact ground truth
- **THEN** the reward engine returns a high positive reward (+1.0 correctness + 0.3 validity + 0.5 kernel_select + economy bonus)

#### Scenario: Malformed trigger syntax penalty
- **WHEN** a candidate completion emits unclosed or syntactically invalid trigger tags
- **THEN** the reward engine assigns a format penalty (-0.5 validity reward)

#### Scenario: Redundant dynamic C++ code penalty
- **WHEN** a candidate completion writes a multi-line dynamic C++ loop for a simple arithmetic operation where a direct kernel (`SUM_F64`) exists
- **THEN** the reward engine assigns 0.0 kernel_select reward, favoring direct micro-kernel invocation

### Requirement: Multi-Candidate Policy Advantage Calculation
The system SHALL sample $G=8$ candidate reasoning traces per prompt in parallel during GRPO training and compute relative advantages normalized across the group without requiring a separate Critic model network.

#### Scenario: Relative advantage normalization
- **WHEN** $G=8$ completions are generated for a micro-batch prompt
- **THEN** the GRPO trainer normalizes rewards $R_i$ into standardized group advantages $A_i = \frac{R_i - \mu_R}{\sigma_R}$ to update policy weights
