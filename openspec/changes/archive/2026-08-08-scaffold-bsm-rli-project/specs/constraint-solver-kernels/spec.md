## ADDED Requirements

### Requirement: Embedded Logic and Constraint Solvers
The system SHALL provide pre-compiled C++ wrappers around SAT (`SOLVE_SAT`), Integer Linear Programming (`SOLVE_ILP`), and Satisfiability Modulo Theories (`SOLVE_SMT`) solvers.

#### Scenario: Solving Boolean satisfiability formula
- **WHEN** `SOLVE_SAT([[1, 2], [-1, 3], [-2, -3]])` is executed
- **THEN** the system SHALL invoke the embedded SAT solver and return the satisfying variable assignment in < 50 microseconds

#### Scenario: Evaluating Integer Linear Programming model
- **WHEN** `SOLVE_ILP(...)` is called with objective function and constraints
- **THEN** the system SHALL compute the optimal integer allocation vector without context drift or CoT hallucinations
