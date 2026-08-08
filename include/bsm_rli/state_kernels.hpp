#ifndef BSM_RLI_STATE_KERNELS_HPP
#define BSM_RLI_STATE_KERNELS_HPP

#include "bsm_rli/registry.hpp"
#include <string>
#include <vector>

namespace bsm_rli {
namespace state {

void register_state_kernels(KernelRegistry& registry);

int count_char(const std::string& text, char target);
size_t len_char(const std::string& text);
std::string reverse_str(const std::string& text);
int substring_index(const std::string& text, const std::string& query);
std::string concat_str(const std::vector<std::string>& parts, const std::string& delim);
std::string case_transform(const std::string& text, const std::string& mode);

} // namespace state
} // namespace bsm_rli

#endif // BSM_RLI_STATE_KERNELS_HPP
