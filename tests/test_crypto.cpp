#include <gtest/gtest.h>
#include "bsm_rli/crypto_kernels.hpp"

TEST(CryptoKernelsTest, BitwiseOp) {
    EXPECT_EQ(bsm_rli::crypto::bitwise_op(0b1100, "AND", 0b1010), 0b1000u);
    EXPECT_EQ(bsm_rli::crypto::bitwise_op(0b1100, "OR", 0b1010), 0b1110u);
    EXPECT_EQ(bsm_rli::crypto::bitwise_op(0b1100, "XOR", 0b1010), 0b0110u);
}

TEST(CryptoKernelsTest, Base64Encode) {
    EXPECT_EQ(bsm_rli::crypto::base64_encode("Hello"), "SGVsbG8=");
}
