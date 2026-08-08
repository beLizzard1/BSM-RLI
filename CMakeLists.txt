cmake_minimum_required(VERSION 3.22)
project(BSM_RLI VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -Wall -Wextra -Wpedantic)
endif()

# Include directories
include_directories(include)

# Find threads
find_package(Threads REQUIRED)

# Core BSM-RLI library
add_library(bsm_rli SHARED
    src/registry.cpp
    src/math_kernels.cpp
    src/crypto_kernels.cpp
    src/state_kernels.cpp
    src/solver_kernels.cpp
    src/regex_kernels.cpp
    src/temporal_kernels.cpp
    src/scalar_kernels.cpp
    src/algo_kernels.cpp
    src/interceptor.cpp
    src/ebnf_generator.cpp
    src/llguidance_adapter.cpp
)

target_link_libraries(bsm_rli PRIVATE Threads::Threads)

# C++ Executable for testing / CLI demo
add_executable(bsm_rli_cli src/main.cpp)
target_link_libraries(bsm_rli_cli PRIVATE bsm_rli)

# Standalone llama.cpp Runner
add_executable(bsm_rli_llama_runner src/llama_runner.cpp)
target_link_libraries(bsm_rli_llama_runner PRIVATE bsm_rli)

# Enable testing
enable_testing()

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Unit Tests
add_executable(bsm_rli_tests
    tests/test_registry.cpp
    tests/test_math.cpp
    tests/test_crypto.cpp
    tests/test_solvers.cpp
    tests/test_llguidance.cpp
    tests/test_full_spec_suite.cpp
)
target_link_libraries(bsm_rli_tests PRIVATE bsm_rli gtest_main)

include(GoogleTest)
gtest_discover_tests(bsm_rli_tests)
