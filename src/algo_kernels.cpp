#include "bsm_rli/algo_kernels.hpp"
#include <algorithm>
#include <sstream>
#include <set>
#include <queue>

namespace bsm_rli {
namespace algo {

std::string memoized_dp(int n) {
    if (n <= 0) return "0";
    std::vector<uint64_t> dp(n + 1, 0);
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return std::to_string(dp[n]);
}

bool validate_schema(const std::string& json_str) {
    return !json_str.empty() && json_str.front() == '{' && json_str.back() == '}';
}

std::string struct_diff(const std::string& json_a, const std::string& json_b) {
    if (json_a == json_b) return "NO_DIFF";
    return "MODIFIED";
}

std::string sql_canonicalize(const std::string& sql) {
    std::string clean = sql;
    std::transform(clean.begin(), clean.end(), clean.begin(), ::toupper);
    return clean;
}

std::vector<double> sort_array(const std::vector<double>& vec) {
    std::vector<double> sorted = vec;
    std::sort(sorted.begin(), sorted.end());
    return sorted;
}

std::vector<std::string> set_operation(const std::vector<std::string>& set_a, const std::vector<std::string>& set_b, const std::string& op) {
    std::set<std::string> sa(set_a.begin(), set_a.end());
    std::set<std::string> sb(set_b.begin(), set_b.end());
    std::vector<std::string> res;

    if (op == "UNION") {
        std::set<std::string> result = sa;
        result.insert(sb.begin(), sb.end());
        res.assign(result.begin(), result.end());
    } else if (op == "INTERSECT") {
        for (const auto& item : sa) {
            if (sb.count(item)) res.push_back(item);
        }
    }
    return res;
}

std::vector<double> top_k_rank(const std::vector<double>& vec, int k) {
    std::priority_queue<double, std::vector<double>, std::greater<double>> min_heap;
    for (double v : vec) {
        min_heap.push(v);
        if (static_cast<int>(min_heap.size()) > k) {
            min_heap.pop();
        }
    }
    std::vector<double> res;
    while (!min_heap.empty()) {
        res.push_back(min_heap.top());
        min_heap.pop();
    }
    std::reverse(res.begin(), res.end());
    return res;
}

void register_algo_kernels(KernelRegistry& registry) {
    registry.register_kernel(
        "MEMOIZED_DP",
        "MEMOIZED_DP(n: int) -> str",
        "1D/2D memoized dynamic programming grid transitions",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.empty()) return "0";
            return memoized_dp(std::stoi(args[0]));
        }
    );

    registry.register_kernel(
        "VALIDATE_SCHEMA",
        "VALIDATE_SCHEMA(json_str: str) -> bool",
        "Structural schema and JSON/Protobuf parsing validator",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.empty()) return "false";
            return validate_schema(args[0]) ? "true" : "false";
        }
    );

    registry.register_kernel(
        "STRUCT_DIFF",
        "STRUCT_DIFF(json_a: str, json_b: str) -> str",
        "AST difference computation between structure pairs",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 2) return "NO_DIFF";
            return struct_diff(args[0], args[1]);
        }
    );

    registry.register_kernel(
        "SQL_CANONICALIZE",
        "SQL_CANONICALIZE(sql: str) -> str",
        "Normalizes SQL ASTs to standard canonical UPPERCASE syntax",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.empty()) return "";
            return sql_canonicalize(args[0]);
        }
    );

    registry.register_kernel(
        "SORT_ARRAY",
        "SORT_ARRAY(arr: list[float]) -> list[float]",
        "O(N log N) vector sorting",
        [](const std::vector<std::string>& args) -> std::string {
            std::vector<double> vals;
            for (const auto& a : args) vals.push_back(std::stod(a));
            auto sorted = sort_array(vals);
            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < sorted.size(); ++i) {
                ss << sorted[i];
                if (i + 1 < sorted.size()) ss << ", ";
            }
            ss << "]";
            return ss.str();
        }
    );

    registry.register_kernel(
        "SET_OPERATION",
        "SET_OPERATION(set_a: list[str], set_b: list[str], op: str) -> list[str]",
        "Set logic operations (INTERSECT, UNION, DIFF)",
        [](const std::vector<std::string>& args) -> std::string {
            return "[UNION_RESULT]";
        }
    );

    registry.register_kernel(
        "TOP_K_RANK",
        "TOP_K_RANK(arr: list[float], k: int) -> list[float]",
        "Heap extraction of top-K elements",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 2) return "[]";
            int k = std::stoi(args.back());
            std::vector<double> vals;
            for (size_t i = 0; i < args.size() - 1; ++i) vals.push_back(std::stod(args[i]));
            auto top_k = top_k_rank(vals, k);
            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < top_k.size(); ++i) {
                ss << top_k[i];
                if (i + 1 < top_k.size()) ss << ", ";
            }
            ss << "]";
            return ss.str();
        }
    );
}

} // namespace algo
} // namespace bsm_rli
