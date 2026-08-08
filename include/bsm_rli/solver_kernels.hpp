#ifndef BSM_RLI_SOLVER_KERNELS_HPP
#define BSM_RLI_SOLVER_KERNELS_HPP

#include "bsm_rli/registry.hpp"
#include <string>
#include <vector>

namespace bsm_rli {
namespace solver {

void register_solver_kernels(KernelRegistry& registry);

std::string solve_sat(int num_vars, const std::vector<std::vector<int>>& clauses);
std::string solve_ilp(const std::vector<double>& objective,
                      const std::vector<std::vector<double>>& constraints,
                      const std::vector<double>& bounds);
std::string solve_smt(const std::vector<std::string>& declarations,
                      const std::vector<std::string>& assertions);

} // namespace solver
} // namespace bsm_rli

#endif // BSM_RLI_SOLVER_KERNELS_HPP
