#include <gtest/gtest.h>
#include "bsm_rli/math_kernels.hpp"
#include "bsm_rli/state_kernels.hpp"
#include "bsm_rli/regex_kernels.hpp"
#include "bsm_rli/temporal_kernels.hpp"
#include "bsm_rli/scalar_kernels.hpp"
#include "bsm_rli/algo_kernels.hpp"

TEST(FullSpecSuiteTest, MathKernels) {
    EXPECT_EQ(bsm_rli::math::sum_f32({1.0f, 2.0f, 3.0f}), 6.0f);
    EXPECT_EQ(bsm_rli::math::sum_int({10, 20, 30}), 60);
    EXPECT_EQ(bsm_rli::math::avg_f32({10.0f, 20.0f}), 15.0);
    EXPECT_EQ(bsm_rli::math::percent_delta(100.0, 150.0), 50.0);
}

TEST(FullSpecSuiteTest, StringKernels) {
    EXPECT_EQ(bsm_rli::state::len_char("strawberry"), 10);
    EXPECT_EQ(bsm_rli::state::reverse_str("abc"), "cba");
    EXPECT_EQ(bsm_rli::state::substring_index("hello world", "world"), 6);
}

TEST(FullSpecSuiteTest, RegexKernels) {
    EXPECT_TRUE(bsm_rli::regex::regex_match("user@example.com", ".*@.*"));
    EXPECT_EQ(bsm_rli::regex::sanitize_url("https://example.com/api?user=123"), "https://example.com/api");
}

TEST(FullSpecSuiteTest, TemporalKernels) {
    EXPECT_EQ(bsm_rli::temporal::date_add("2026-08-08", 2, "DAYS"), "2026-08-10");
    EXPECT_EQ(bsm_rli::temporal::day_of_week("2026-08-08"), "Saturday");
}

TEST(FullSpecSuiteTest, ScalarKernels) {
    EXPECT_EQ(bsm_rli::scalar::eval_expr(10.0, "ADD", 5.0), 15.0);
    EXPECT_NEAR(bsm_rli::scalar::unit_convert(100.0, "lbs", "kg"), 45.3592, 0.01);
    EXPECT_EQ(bsm_rli::scalar::round_prec(12.3456, 2), 12.35);
}

TEST(FullSpecSuiteTest, AlgoKernels) {
    EXPECT_EQ(bsm_rli::algo::memoized_dp(5), "5");
    EXPECT_TRUE(bsm_rli::algo::validate_schema("{\"key\":\"value\"}"));
    auto sorted = bsm_rli::algo::sort_array({5.0, 1.0, 3.0});
    EXPECT_EQ(sorted[0], 1.0);
    EXPECT_EQ(sorted[2], 5.0);
}
