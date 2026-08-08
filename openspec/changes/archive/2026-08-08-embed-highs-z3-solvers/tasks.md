## 1. Update Solver Header

- [x] 1.1 Update `include/bsm_rli/solver_kernels.hpp` adding `solve_ilp` and `solve_smt` function signatures and parameter structures

## 2. Implement ILP & SMT Kernels

- [x] 2.1 Update `src/solver_kernels.cpp` implementing `SOLVE_ILP` integer linear programming optimizer and registering kernel
- [x] 2.2 Update `src/solver_kernels.cpp` implementing `SOLVE_SMT` satisfiability modulo theories solver and registering kernel

## 3. Unit Tests & Verification

- [x] 3.1 Update `tests/test_solvers.cpp` adding GoogleTest cases for `SOLVE_ILP` and `SOLVE_SMT`
- [x] 3.2 Build C++ library and run CTest suite (`ctest`)
