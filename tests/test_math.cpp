#include <gtest/gtest.h>
#include "bsm_rli/math_kernels.hpp"

TEST(MathKernelsTest, SumF64) {
    std::vector<double> vals = {1.5, 2.5, 3.0};
    EXPECT_EQ(bsm_rli::math::sum_f64(vals), 7.0);
}

TEST(MathKernelsTest, ProductF64) {
    std::vector<double> vals = {2.0, 3.0, 4.0};
    EXPECT_EQ(bsm_rli::math::product_f64(vals), 24.0);
}

TEST(MathKernelsTest, DotProduct) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 5.0, 6.0};
    EXPECT_EQ(bsm_rli::math::dot_product(a, b), 32.0);
}

TEST(MathKernelsTest, StatsSummary) {
    std::vector<double> vals = {10.0, 20.0, 30.0};
    std::string res = bsm_rli::math::stats_summary(vals);
    EXPECT_NE(res.find("\"mean\":20"), std::string::npos);
}
