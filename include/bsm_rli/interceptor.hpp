#ifndef BSM_RLI_INTERCEPTOR_HPP
#define BSM_RLI_INTERCEPTOR_HPP

#include "bsm_rli/registry.hpp"
#include "bsm_rli/llguidance_adapter.hpp"
#include <string>
#include <string_view>
#include <memory>

namespace bsm_rli {

enum class InterceptorState {
    Unconstrained,
    ConstrainedJIT
};

class RegionScopedInterceptor {
public:
    explicit RegionScopedInterceptor(KernelRegistry& registry);

    InterceptorState state() const { return state_; }

    bool process_token(const std::string& token);
    std::string current_buffer() const { return buffer_; }
    void reset();

    void apply_logit_mask(float* logits, size_t vocab_size, const std::vector<std::string>& vocab) const;

    std::string execute_intercepted();

private:
    KernelRegistry& registry_;
    InterceptorState state_{InterceptorState::Unconstrained};
    std::string buffer_;
    std::unique_ptr<LLGuidanceAdapter> adapter_;
    
    static constexpr std::string_view JIT_START = "<|jit_start|>";
    static constexpr std::string_view JIT_END = "<|jit_end|>";
};

} // namespace bsm_rli

#endif // BSM_RLI_INTERCEPTOR_HPP
