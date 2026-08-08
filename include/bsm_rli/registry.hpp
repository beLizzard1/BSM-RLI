#ifndef BSM_RLI_REGISTRY_HPP
#define BSM_RLI_REGISTRY_HPP

#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <variant>
#include <string_view>

namespace bsm_rli {

using KernelArg = std::variant<int64_t, double, std::string, std::vector<double>, std::vector<std::string>>;
using KernelFunction = std::function<std::string(const std::vector<std::string>& args)>;

struct KernelInfo {
    std::string identifier;
    std::string signature;
    std::string description;
    KernelFunction fn;
};

class KernelRegistry {
public:
    static KernelRegistry& instance();

    bool register_kernel(const std::string& id,
                        const std::string& signature,
                        const std::string& description,
                        KernelFunction fn);

    bool has_kernel(const std::string& id) const;
    std::string execute(const std::string& id, const std::vector<std::string>& args) const;

    std::vector<KernelInfo> get_all_kernels() const;
    std::string generate_ebnf_grammar() const;

private:
    KernelRegistry() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, KernelInfo> kernels_;
};

} // namespace bsm_rli

#endif // BSM_RLI_REGISTRY_HPP
