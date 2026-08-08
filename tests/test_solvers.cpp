#include <gtest/gtest.h>
#include "bsm_rli/solver_kernels.hpp"

TEST(SolverKernelsTest, SolveSAT) {
    // Clauses: (x1 OR x2) AND (NOT x1 OR x2)
    std::vector<std::vector<int>> clauses = {{1, 2}, {-1, 2}};
    std::string res = bsm_rli::solver::solve_sat(2, clauses);
    EXPECT_NE(res.find("SAT"), std::string::npos);
}

TEST(SolverKernelsTest, SolveILP) {
    // Objective: maximize 10*x1 + 20*x2 + 30*x3 s.t. 1*x1 + 2*x2 + 3*x3 <= 4
    std::vector<double> obj = {10.0, 20.0, 30.0};
    std::vector<std::vector<double>> constr = {{1.0, 2.0, 3.0}};
    std::vector<double> bounds = {4.0};
    
    std::string res = bsm_rli::solver::solve_ilp(obj, constr, bounds);
    EXPECT_NE(res.find("OPTIMAL"), std::string::npos);
}

TEST(SolverKernelsTest, SolveSMT) {
    std::vector<std::string> decls = {"x", "y"};
    std::vector<std::string> asserts = {"x > 0", "y == x + 1"};
    
    std::string res = bsm_rli::solver::solve_smt(decls, asserts);
    EXPECT_NE(res.find("SAT"), std::string::npos);
}
