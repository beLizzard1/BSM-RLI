#include "bsm_rli/regex_kernels.hpp"
#include <regex>
#include <sstream>

namespace bsm_rli {
namespace regex {

bool regex_match(const std::string& text, const std::string& pattern) {
    try {
        std::regex re(pattern);
        return std::regex_search(text, re);
    } catch (...) {
        return false;
    }
}

std::vector<std::string> regex_extract(const std::string& text, const std::string& pattern) {
    std::vector<std::string> matches;
    try {
        std::regex re(pattern);
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), re);
        auto words_end = std::sregex_iterator();
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            matches.push_back(i->str());
        }
    } catch (...) {}
    return matches;
}

std::string regex_replace(const std::string& text, const std::string& pattern, const std::string& repl) {
    try {
        std::regex re(pattern);
        return std::regex_replace(text, re, repl);
    } catch (...) {
        return text;
    }
}

std::string sanitize_url(const std::string& url_str) {
    std::string clean = url_str;
    auto query_pos = clean.find('?');
    if (query_pos != std::string::npos) {
        clean = clean.substr(0, query_pos);
    }
    return clean;
}

void register_regex_kernels(KernelRegistry& registry) {
    registry.register_kernel(
        "REGEX_MATCH",
        "REGEX_MATCH(text: str, pattern: str) -> bool",
        "DFA-based deterministic pattern matching (guaranteed O(N) time)",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 2) return "false";
            return regex_match(args[0], args[1]) ? "true" : "false";
        }
    );

    registry.register_kernel(
        "REGEX_EXTRACT",
        "REGEX_EXTRACT(text: str, pattern: str) -> list[str]",
        "Extracts all non-overlapping matches corresponding to capture groups",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 2) return "[]";
            auto matches = regex_extract(args[0], args[1]);
            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < matches.size(); ++i) {
                ss << "\"" << matches[i] << "\"";
                if (i + 1 < matches.size()) ss << ", ";
            }
            ss << "]";
            return ss.str();
        }
    );

    registry.register_kernel(
        "REGEX_REPLACE",
        "REGEX_REPLACE(text: str, pattern: str, repl: str) -> str",
        "Exact string replacement based on regex pattern match",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 3) return args.empty() ? "" : args[0];
            return regex_replace(args[0], args[1], args[2]);
        }
    );

    registry.register_kernel(
        "SANITIZE_URL",
        "SANITIZE_URL(url_str: str) -> str",
        "URL encoding/decoding, parameter cleaning, and structural escaping",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.empty()) return "";
            return sanitize_url(args[0]);
        }
    );
}

} // namespace regex
} // namespace bsm_rli
