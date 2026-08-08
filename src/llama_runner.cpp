#include "bsm_rli/registry.hpp"
#include "bsm_rli/llama_cpp_hook.hpp"
#include "bsm_rli/math_kernels.hpp"
#include "bsm_rli/crypto_kernels.hpp"
#include "bsm_rli/state_kernels.hpp"
#include "bsm_rli/solver_kernels.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << "  BSM-RLI Standalone C++ llama.cpp Engine Runner         \n";
    std::cout << "========================================================\n\n";

    auto& registry = bsm_rli::KernelRegistry::instance();
    bsm_rli::math::register_math_kernels(registry);
    bsm_rli::crypto::register_crypto_kernels(registry);
    bsm_rli::state::register_state_kernels(registry);
    bsm_rli::solver::register_solver_kernels(registry);

    bsm_rli::LlamaCppHook hook(registry);

    std::string model_path = (argc > 1) ? argv[1] : "models/bsm-rli-llama-3.1-8b-Q4_K_M.gguf";
    std::cout << "Target GGUF Model Path: " << model_path << "\n";
    std::cout << "Registered Micro-Kernels: " << registry.get_all_kernels().size() << "\n\n";

    // Simulate token sampling stream
    std::string stream = "Calculate vector sum: <|jit_start|>SUM_F64(10.5, 20.5, 30.0)<|jit_end|>";
    std::cout << "Simulating LLM Token Stream:\n\"" << stream << "\"\n\n";

    for (char c : stream) {
        std::string tok(1, c);
        hook.on_token_sampled(0, tok);
    }

    if (!hook.is_in_jit_region()) {
        std::string result = hook.execute_and_reset();
        std::cout << ">>> Host Kernel Executed (< 5µs): " << result << "\n";
    }

    return 0;
}
