## Context

To support high-dimensional constraint solving without context drift, BSM-RLI embeds C++ solver primitives (`SOLVE_SAT`, `SOLVE_ILP`, `SOLVE_SMT`) into the host kernel registry (`src/solver_kernels.cpp`).

## Goals / Non-Goals

**Goals:**
- **SOLVE_ILP Implementation**: Provide Integer Linear Programming solver execution returning optimal variable vectors for knapsack and resource optimization.
- **SOLVE_SMT Implementation**: Provide Satisfiability Modulo Theories solver execution for multi-type logic, linear integer arithmetic, and equality logic.
- **Enhanced SOLVE_SAT**: Provide fast DPLL / CDCL unit propagation solver execution.

**Non-Goals:**
- Slow external Python solver process wrappers (all solvers must run in-process in C++).

## Decisions

### 1. In-Process C++ Solver Dispatch
- `solve_ilp(...)` parses objective coefficients, constraint coefficient matrices, and upper/lower bounds, solving them using simplex / branch-and-bound logic.
- `solve_smt(...)` parses linear equality and inequality assertions and evaluates variable satisfiability in microsecond execution time.

## Risks / Trade-offs

- **[Risk]** Computational timeout on NP-hard constraint instances.  
  *Mitigation*: Enforce a strict solver timeout ($50\text{ms}$) returning `TIMEOUT` if bounds cannot be resolved within budget.
