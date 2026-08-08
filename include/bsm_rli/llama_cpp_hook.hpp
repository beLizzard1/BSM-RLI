#ifndef BSM_RLI_LLAMA_CPP_HOOK_HPP
#define BSM_RLI_LLAMA_CPP_HOOK_HPP

#include "bsm_rli/interceptor.hpp"
#include "bsm_rli/registry.hpp"
#include <string>
#include <vector>

namespace bsm_rli {

class LlamaCppHook {
public:
    explicit LlamaCppHook(KernelRegistry& registry)
        : interceptor_(registry) {}

    // Called for each sampled token during llama.cpp inference loop
    bool on_token_sampled(int32_t token_id, const std::string& token_text) {
        bool state_changed = interceptor_.process_token(token_text);
        return state_changed;
    }

    bool is_in_jit_region() const {
        return interceptor_.state() == InterceptorState::ConstrainedJIT;
    }

    std::string execute_and_reset() {
        std::string result = interceptor_.execute_intercepted();
        interceptor_.reset();
        return result;
    }

private:
    RegionScopedInterceptor interceptor_;
};

} // namespace bsm_rli

#endif // BSM_RLI_LLAMA_CPP_HOOK_HPP
