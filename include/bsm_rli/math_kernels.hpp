#ifndef BSM_RLI_MATH_KERNELS_HPP
#define BSM_RLI_MATH_KERNELS_HPP

#include "bsm_rli/registry.hpp"
#include <vector>
#include <string>

namespace bsm_rli {
namespace math {

void register_math_kernels(KernelRegistry& registry);

double sum_f64(const std::vector<double>& vec);
float sum_f32(const std::vector<float>& vec);
int64_t sum_int(const std::vector<int64_t>& vec);
double avg_f32(const std::vector<float>& vec);
double std_dev_f32(const std::vector<float>& vec);
std::pair<float, float> min_max_f32(const std::vector<float>& vec);
double product_f64(const std::vector<double>& vec);
float product_f32(const std::vector<float>& vec);
double dot_product(const std::vector<double>& a, const std::vector<double>& b);
double percent_delta(double old_val, double new_val);
std::string stats_summary(const std::vector<double>& vec);

} // namespace math
} // namespace bsm_rli

#endif // BSM_RLI_MATH_KERNELS_HPP
