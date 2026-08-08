#ifndef BSM_RLI_REGEX_KERNELS_HPP
#define BSM_RLI_REGEX_KERNELS_HPP

#include "bsm_rli/registry.hpp"
#include <string>
#include <vector>

namespace bsm_rli {
namespace regex {

void register_regex_kernels(KernelRegistry& registry);

bool regex_match(const std::string& text, const std::string& pattern);
std::vector<std::string> regex_extract(const std::string& text, const std::string& pattern);
std::string regex_replace(const std::string& text, const std::string& pattern, const std::string& repl);
std::string sanitize_url(const std::string& url_str);

} // namespace regex
} // namespace bsm_rli

#endif // BSM_RLI_REGEX_KERNELS_HPP
