## ADDED Requirements

### Requirement: Context Window & Attention Economy Optimization
The system SHALL intercept multi-step calculations, string scans, calendar operations, and constraint solvers using region-scoped JIT triggers, compressing context consumption by an average of > 50x and reducing KV-cache VRAM allocation by > 97%.

#### Scenario: Vector math context compression
- **WHEN** evaluating a 10+ element vector summation
- **THEN** the system SHALL consume <= 3 context tokens (`<|jit_start|>SUM_F64(...)<|jit_end|>`), achieving >= 40x token compression compared to 125+ token Chain-of-Thought scratchpads

#### Scenario: Constraint solver context compression
- **WHEN** evaluating a Boolean SAT or Integer Linear Programming model
- **THEN** the system SHALL consume <= 4 context tokens (`<|jit_start|>SOLVE_SAT(...)<|jit_end|>`), achieving >= 100x token compression compared to 400+ token Chain-of-Thought search paths
