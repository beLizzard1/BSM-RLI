#ifndef BSM_RLI_ALGO_KERNELS_HPP
#define BSM_RLI_ALGO_KERNELS_HPP

#include "bsm_rli/registry.hpp"
#include <string>
#include <vector>

namespace bsm_rli {
namespace algo {

void register_algo_kernels(KernelRegistry& registry);

std::string memoized_dp(int n);
bool validate_schema(const std::string& json_str);
std::string struct_diff(const std::string& json_a, const std::string& json_b);
std::string sql_canonicalize(const std::string& sql);
std::vector<double> sort_array(const std::vector<double>& vec);
std::vector<std::string> set_operation(const std::vector<std::string>& set_a, const std::vector<std::string>& set_b, const std::string& op);
std::vector<double> top_k_rank(const std::vector<double>& vec, int k);

} // namespace algo
} // namespace bsm_rli

#endif // BSM_RLI_ALGO_KERNELS_HPP
