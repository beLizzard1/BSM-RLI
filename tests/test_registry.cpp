#include <gtest/gtest.h>
#include "bsm_rli/registry.hpp"
#include "bsm_rli/interceptor.hpp"

TEST(RegistryTest, RegisterAndExecute) {
    auto& registry = bsm_rli::KernelRegistry::instance();
    bool registered = registry.register_kernel(
        "TEST_ADD",
        "TEST_ADD(a: int, b: int) -> int",
        "Test addition kernel",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 2) return "0";
            return std::to_string(std::stoi(args[0]) + std::stoi(args[1]));
        }
    );
    EXPECT_TRUE(registered || registry.has_kernel("TEST_ADD"));
    EXPECT_EQ(registry.execute("TEST_ADD", {"10", "20"}), "30");
}

TEST(InterceptorTest, ProcessAndIntercept) {
    auto& registry = bsm_rli::KernelRegistry::instance();
    if (!registry.has_kernel("TEST_ADD")) {
        registry.register_kernel(
            "TEST_ADD",
            "TEST_ADD(a: int, b: int) -> int",
            "Test addition kernel",
            [](const std::vector<std::string>& args) -> std::string {
                if (args.size() < 2) return "0";
                return std::to_string(std::stoi(args[0]) + std::stoi(args[1]));
            }
        );
    }
    bsm_rli::RegionScopedInterceptor interceptor(registry);

    std::string stream = "<|jit_start|>TEST_ADD(15, 25)<|jit_end|>";
    for (char c : stream) {
        interceptor.process_token(std::string(1, c));
    }
    EXPECT_EQ(interceptor.execute_intercepted(), "40");
}
