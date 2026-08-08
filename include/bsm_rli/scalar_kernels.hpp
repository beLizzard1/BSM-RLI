#ifndef BSM_RLI_SCALAR_KERNELS_HPP
#define BSM_RLI_SCALAR_KERNELS_HPP

#include "bsm_rli/registry.hpp"
#include <string>
#include <vector>

namespace bsm_rli {
namespace scalar {

void register_scalar_kernels(KernelRegistry& registry);

double eval_expr(double a, const std::string& op, double b);
double unit_convert(double val, const std::string& from_unit, const std::string& to_unit);
double round_prec(double val, int decimals);

} // namespace scalar
} // namespace bsm_rli

#endif // BSM_RLI_SCALAR_KERNELS_HPP
