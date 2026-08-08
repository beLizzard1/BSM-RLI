#ifndef BSM_RLI_TEMPORAL_KERNELS_HPP
#define BSM_RLI_TEMPORAL_KERNELS_HPP

#include "bsm_rli/registry.hpp"
#include <string>
#include <vector>

namespace bsm_rli {
namespace temporal {

void register_temporal_kernels(KernelRegistry& registry);

std::string date_add(const std::string& base_date, int val, const std::string& unit);
int date_diff(const std::string& date_a, const std::string& date_b, const std::string& unit);
std::string day_of_week(const std::string& date);
std::string tz_convert(const std::string& timestamp, const std::string& src_tz, const std::string& dst_tz);

} // namespace temporal
} // namespace bsm_rli

#endif // BSM_RLI_TEMPORAL_KERNELS_HPP
