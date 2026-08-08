#include "bsm_rli/registry.hpp"
#include "bsm_rli/llguidance_adapter.hpp"
#include <fstream>
#include <iostream>

namespace bsm_rli {

bool export_ebnf_file(const KernelRegistry& registry, const std::string& filepath) {
    std::ofstream out(filepath);
    if (!out.is_open()) return false;
    out << registry.generate_ebnf_grammar();
    return true;
}

std::unique_ptr<LLGuidanceAdapter> create_llguidance_adapter(const KernelRegistry& registry) {
    std::string grammar = registry.generate_ebnf_grammar();
    return std::make_unique<LLGuidanceAdapter>(grammar);
}

} // namespace bsm_rli
