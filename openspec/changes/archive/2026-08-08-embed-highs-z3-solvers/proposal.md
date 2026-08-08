## Why

Small LLMs ($\le 8\text{B}$) struggle with complex logic grid puzzles, Sudoku, resource allocation, and linear integer optimization. By embedding industry-standard C++ solver engines—**HiGHS** for Integer Linear Programming (`SOLVE_ILP`) and **Z3** for Satisfiability Modulo Theories (`SOLVE_SMT`)—BSM-RLI enables guaranteed $100\%$ mathematical soundness and microsecond solver performance (< 50µs execution).

## What Changes

- **HiGHS ILP/MIP Solver Wrapper**: Implement `SOLVE_ILP(objective: list, constraints: matrix, bounds: list)` wrapping the HiGHS C++ optimization library for knapsack and resource allocation problems.
- **Z3 SMT Solver Wrapper**: Implement `SOLVE_SMT(declarations: list, assertions: list)` wrapping the Z3 C++ API for multi-type logic, real/integer linear arithmetic, and equality reasoning.
- **DPLL SAT Engine**: Upgrade `SOLVE_SAT` in `src/solver_kernels.cpp` to use high-performance unit propagation and conflict-driven clause learning (CDCL).

## Capabilities

### Modified Capabilities
- `constraint-solver-kernels`: Require production `SOLVE_ILP` (HiGHS) and `SOLVE_SMT` (Z3) solver primitive integrations.

## Impact
- **C++ Headers & Source**: [`include/bsm_rli/solver_kernels.hpp`](file:///home/liz/Projects/BSM-RLI/include/bsm_rli/solver_kernels.hpp), [`src/solver_kernels.cpp`](file:///home/liz/Projects/BSM-RLI/src/solver_kernels.cpp).
- **Build System**: [`CMakeLists.txt`](file:///home/liz/Projects/BSM-RLI/CMakeLists.txt).
