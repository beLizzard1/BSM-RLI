#include "bsm_rli/registry.hpp"
#include <sstream>
#include <stdexcept>

namespace bsm_rli {

KernelRegistry& KernelRegistry::instance() {
    static KernelRegistry instance_;
    return instance_;
}

bool KernelRegistry::register_kernel(const std::string& id,
                                      const std::string& signature,
                                      const std::string& description,
                                      KernelFunction fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (kernels_.find(id) != kernels_.end()) {
        return false;
    }
    kernels_[id] = KernelInfo{id, signature, description, std::move(fn)};
    return true;
}

bool KernelRegistry::has_kernel(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return kernels_.find(id) != kernels_.end();
}

std::string KernelRegistry::execute(const std::string& id, const std::vector<std::string>& args) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = kernels_.find(id);
    if (it == kernels_.end()) {
        throw std::runtime_error("Kernel identifier not found: " + id);
    }
    return it->second.fn(args);
}

std::vector<KernelInfo> KernelRegistry::get_all_kernels() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<KernelInfo> result;
    result.reserve(kernels_.size());
    for (const auto& [_, info] : kernels_) {
        result.push_back(info);
    }
    return result;
}

std::string KernelRegistry::generate_ebnf_grammar() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::stringstream ss;
    ss << "root ::= \"<|jit_start|>\" kernel_call \"<|jit_end|>\"\n";
    ss << "kernel_call ::=\n";
    
    bool first = true;
    for (const auto& [id, info] : kernels_) {
        if (!first) ss << "            | ";
        else ss << "              ";
        ss << "\"" << id << "(\" args \")\"\n";
        first = false;
    }
    ss << "args ::= [^)]*\n";
    return ss.str();
}

} // namespace bsm_rli
