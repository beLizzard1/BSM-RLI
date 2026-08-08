#include "bsm_rli/scalar_kernels.hpp"
#include <cmath>
#include <sstream>

namespace bsm_rli {
namespace scalar {

double eval_expr(double a, const std::string& op, double b) {
    if (op == "ADD" || op == "+") return a + b;
    if (op == "SUB" || op == "-") return a - b;
    if (op == "MUL" || op == "*") return a * b;
    if (op == "DIV" || op == "/") return b != 0.0 ? a / b : 0.0;
    if (op == "POW") return std::pow(a, b);
    return 0.0;
}

double unit_convert(double val, const std::string& from_unit, const std::string& to_unit) {
    if (from_unit == "lbs" && to_unit == "kg") return val * 0.45359237;
    if (from_unit == "kg" && to_unit == "lbs") return val / 0.45359237;
    if (from_unit == "F" && to_unit == "C") return (val - 32.0) * (5.0 / 9.0);
    if (from_unit == "C" && to_unit == "F") return (val * (9.0 / 5.0)) + 32.0;
    return val;
}

double round_prec(double val, int decimals) {
    double factor = std::pow(10.0, decimals);
    return std::round(val * factor) / factor;
}

void register_scalar_kernels(KernelRegistry& registry) {
    registry.register_kernel(
        "EVAL_EXPR",
        "EVAL_EXPR(a: float, op: enum, b: float) -> float",
        "Evaluates basic scalar floating operations (ADD, SUB, MUL, DIV, POW)",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 3) return "0.0";
            return std::to_string(eval_expr(std::stod(args[0]), args[1], std::stod(args[2])));
        }
    );

    registry.register_kernel(
        "UNIT_CONVERT",
        "UNIT_CONVERT(val: float, from_unit: str, to_unit: str) -> float",
        "Dimensional analysis conversions (e.g., lbs -> kg, F -> C)",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 3) return "0.0";
            return std::to_string(unit_convert(std::stod(args[0]), args[1], args[2]));
        }
    );

    registry.register_kernel(
        "ROUND_PREC",
        "ROUND_PREC(val: float, decimals: int) -> float",
        "Fixed-precision rounding for financial or exact scalar math",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 2) return args.empty() ? "0.0" : args[0];
            return std::to_string(round_prec(std::stod(args[0]), std::stoi(args[1])));
        }
    );
}

} // namespace scalar
} // namespace bsm_rli
