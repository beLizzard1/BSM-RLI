#include <gtest/gtest.h>
#include "bsm_rli/llguidance_adapter.hpp"
#include "bsm_rli/registry.hpp"
#include "bsm_rli/interceptor.hpp"
#include <vector>
#include <cmath>

TEST(LLGuidanceTest, PrefixValidation) {
    bsm_rli::LLGuidanceAdapter adapter("root ::= \"SUM_F64(\"");
    EXPECT_TRUE(adapter.is_valid_prefix("<|jit_start|>SUM_F64(10, 20)"));
    EXPECT_FALSE(adapter.is_valid_prefix("<|jit_start|>SUM_F64(10, 20))"));
}

TEST(LLGuidanceTest, ApplyLogitMask) {
    auto& registry = bsm_rli::KernelRegistry::instance();
    bsm_rli::RegionScopedInterceptor interceptor(registry);

    // Transition into Constrained JIT mode
    interceptor.process_token("<|jit_start|>");
    EXPECT_EQ(interceptor.state(), bsm_rli::InterceptorState::ConstrainedJIT);

    std::vector<std::string> vocab = {"SUM_F64(10, 20)", "SUM_F64(10, 20))", "INVALID)"};
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};

    interceptor.apply_logit_mask(logits.data(), vocab.size(), vocab);

    EXPECT_EQ(logits[0], 1.0f); // Valid token
    EXPECT_TRUE(std::isinf(logits[1]) || std::isinf(logits[2])); // Invalid tokens clamped
}
