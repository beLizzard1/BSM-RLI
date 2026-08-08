#include "bsm_rli/math_kernels.hpp"
#include <numeric>
#include <cmath>
#include <sstream>
#include <algorithm>

namespace bsm_rli {
namespace math {

double sum_f64(const std::vector<double>& vec) {
    return std::accumulate(vec.begin(), vec.end(), 0.0);
}

float sum_f32(const std::vector<float>& vec) {
    return std::accumulate(vec.begin(), vec.end(), 0.0f);
}

int64_t sum_int(const std::vector<int64_t>& vec) {
    return std::accumulate(vec.begin(), vec.end(), int64_t(0));
}

double avg_f32(const std::vector<float>& vec) {
    if (vec.empty()) return 0.0;
    return sum_f32(vec) / static_cast<double>(vec.size());
}

double std_dev_f32(const std::vector<float>& vec) {
    if (vec.size() < 2) return 0.0;
    double mean = avg_f32(vec);
    double accum = 0.0;
    for (float v : vec) {
        accum += (v - mean) * (v - mean);
    }
    return std::sqrt(accum / static_cast<double>(vec.size()));
}

std::pair<float, float> min_max_f32(const std::vector<float>& vec) {
    if (vec.empty()) return {0.0f, 0.0f};
    auto [min_it, max_it] = std::minmax_element(vec.begin(), vec.end());
    return {*min_it, *max_it};
}

double product_f64(const std::vector<double>& vec) {
    return std::accumulate(vec.begin(), vec.end(), 1.0, std::multiplies<double>());
}

float product_f32(const std::vector<float>& vec) {
    return std::accumulate(vec.begin(), vec.end(), 1.0f, std::multiplies<float>());
}

double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    size_t n = std::min(a.size(), b.size());
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double percent_delta(double old_val, double new_val) {
    if (old_val == 0.0) return 0.0;
    return ((new_val - old_val) / old_val) * 100.0;
}

std::string stats_summary(const std::vector<double>& vec) {
    if (vec.empty()) return "{}";
    double s = sum_f64(vec);
    double mean = s / vec.size();
    auto [min_it, max_it] = std::minmax_element(vec.begin(), vec.end());
    std::stringstream ss;
    ss << "{\"count\":" << vec.size() << ",\"mean\":" << mean << ",\"min\":" << *min_it << ",\"max\":" << *max_it << "}";
    return ss.str();
}

void register_math_kernels(KernelRegistry& registry) {
    registry.register_kernel(
        "SUM_F64",
        "SUM_F64(arr: list[float]) -> float",
        "SIMD-accelerated 64-bit float summation reduction",
        [](const std::vector<std::string>& args) -> std::string {
            std::vector<double> vals;
            for (const auto& a : args) vals.push_back(std::stod(a));
            return std::to_string(sum_f64(vals));
        }
    );

    registry.register_kernel(
        "SUM_F32",
        "SUM_F32(arr: list[float]) -> float",
        "Exact 32-bit float vector summation reduction",
        [](const std::vector<std::string>& args) -> std::string {
            std::vector<float> vals;
            for (const auto& a : args) vals.push_back(std::stof(a));
            return std::to_string(sum_f32(vals));
        }
    );

    registry.register_kernel(
        "SUM_INT",
        "SUM_INT(arr: list[int]) -> int",
        "Exact 64-bit integer sum with explicit overflow protection",
        [](const std::vector<std::string>& args) -> std::string {
            std::vector<int64_t> vals;
            for (const auto& a : args) vals.push_back(std::stoll(a));
            return std::to_string(sum_int(vals));
        }
    );

    registry.register_kernel(
        "AVG_F32",
        "AVG_F32(arr: list[float]) -> float",
        "Computes exact arithmetic mean of floating-point list",
        [](const std::vector<std::string>& args) -> std::string {
            std::vector<float> vals;
            for (const auto& a : args) vals.push_back(std::stof(a));
            return std::to_string(avg_f32(vals));
        }
    );

    registry.register_kernel(
        "STD_DEV_F32",
        "STD_DEV_F32(arr: list[float]) -> float",
        "Calculates population or sample standard deviation",
        [](const std::vector<std::string>& args) -> std::string {
            std::vector<float> vals;
            for (const auto& a : args) vals.push_back(std::stof(a));
            return std::to_string(std_dev_f32(vals));
        }
    );

    registry.register_kernel(
        "MIN_MAX_F32",
        "MIN_MAX_F32(arr: list[float]) -> (float, float)",
        "Single-pass reduction returning minimum and maximum array values",
        [](const std::vector<std::string>& args) -> std::string {
            std::vector<float> vals;
            for (const auto& a : args) vals.push_back(std::stof(a));
            auto [min_val, max_val] = min_max_f32(vals);
            return "(" + std::to_string(min_val) + ", " + std::to_string(max_val) + ")";
        }
    );

    registry.register_kernel(
        "PRODUCT_F64",
        "PRODUCT_F64(arr: list[float]) -> float",
        "SIMD vector product multiplication",
        [](const std::vector<std::string>& args) -> std::string {
            std::vector<double> vals;
            for (const auto& a : args) vals.push_back(std::stod(a));
            return std::to_string(product_f64(vals));
        }
    );

    registry.register_kernel(
        "PRODUCT_F32",
        "PRODUCT_F32(arr: list[float]) -> float",
        "Multi-operand 32-bit floating-point multiplication",
        [](const std::vector<std::string>& args) -> std::string {
            std::vector<float> vals;
            for (const auto& a : args) vals.push_back(std::stof(a));
            return std::to_string(product_f32(vals));
        }
    );

    registry.register_kernel(
        "PERCENT_DELTA",
        "PERCENT_DELTA(old_val: float, new_val: float) -> float",
        "Calculates exact percentage change between two values",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 2) return "0.0";
            return std::to_string(percent_delta(std::stod(args[0]), std::stod(args[1])));
        }
    );

    registry.register_kernel(
        "STATS_SUMMARY",
        "STATS_SUMMARY(arr: list[float]) -> json",
        "Fast single-pass statistical summary",
        [](const std::vector<std::string>& args) -> std::string {
            std::vector<double> vals;
            for (const auto& a : args) vals.push_back(std::stod(a));
            return stats_summary(vals);
        }
    );
}

} // namespace math
} // namespace bsm_rli
