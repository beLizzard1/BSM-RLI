#include "bsm_rli/registry.hpp"
#include "bsm_rli/math_kernels.hpp"
#include "bsm_rli/crypto_kernels.hpp"
#include "bsm_rli/state_kernels.hpp"
#include "bsm_rli/solver_kernels.hpp"
#include "bsm_rli/interceptor.hpp"
#include <iostream>

int main() {
    std::cout << "========================================================\n";
    std::cout << "  BSM-RLI: Bare-Metal Symbolic Micro-Kernel Engine v0.1  \n";
    std::cout << "========================================================\n\n";

    auto& registry = bsm_rli::KernelRegistry::instance();
    bsm_rli::math::register_math_kernels(registry);
    bsm_rli::crypto::register_crypto_kernels(registry);
    bsm_rli::state::register_state_kernels(registry);
    bsm_rli::solver::register_solver_kernels(registry);

    std::cout << "Registered Micro-Kernels (" << registry.get_all_kernels().size() << "):\n";
    for (const auto& k : registry.get_all_kernels()) {
        std::cout << " - " << k.identifier << " : " << k.description << "\n";
    }

    std::cout << "\n--- Simulating Region-Scoped Logit Interception ---\n";
    bsm_rli::RegionScopedInterceptor interceptor(registry);

    std::string mock_stream = "The result is <|jit_start|>SUM_F64(12.5, 45.0, 100.25)<|jit_end|>";
    std::cout << "Input stream: " << mock_stream << "\n";

    for (char c : mock_stream) {
        std::string token(1, c);
        interceptor.process_token(token);
    }

    if (interceptor.state() == bsm_rli::InterceptorState::Unconstrained) {
        std::string result = interceptor.execute_intercepted();
        std::cout << "Intercepted Result: " << result << "\n";
    }

    std::cout << "\n--- Generated EBNF Logit Mask Grammar ---\n";
    std::cout << registry.generate_ebnf_grammar();

    return 0;
}
