#include "bsm_rli/state_kernels.hpp"
#include <algorithm>
#include <sstream>
#include <cctype>

namespace bsm_rli {
namespace state {

int count_char(const std::string& text, char target) {
    return std::count(text.begin(), text.end(), target);
}

size_t len_char(const std::string& text) {
    size_t count = 0;
    for (char c : text) {
        if ((c & 0xC0) != 0x80) count++;
    }
    return count;
}

std::string reverse_str(const std::string& text) {
    std::string rev = text;
    std::reverse(rev.begin(), rev.end());
    return rev;
}

int substring_index(const std::string& text, const std::string& query) {
    auto pos = text.find(query);
    if (pos == std::string::npos) return -1;
    return static_cast<int>(pos);
}

std::string concat_str(const std::vector<std::string>& parts, const std::string& delim) {
    std::stringstream ss;
    for (size_t i = 0; i < parts.size(); ++i) {
        ss << parts[i];
        if (i + 1 < parts.size()) ss << delim;
    }
    return ss.str();
}

std::string case_transform(const std::string& text, const std::string& mode) {
    std::string res = text;
    if (mode == "UPPER") {
        std::transform(res.begin(), res.end(), res.begin(), ::toupper);
    } else if (mode == "LOWER") {
        std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    }
    return res;
}

void register_state_kernels(KernelRegistry& registry) {
    registry.register_kernel(
        "COUNT_CHAR",
        "COUNT_CHAR(text: str, target: char) -> int",
        "Byte-level UTF-8 character frequency scan",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 2) return "0";
            char target = args[1].empty() ? ' ' : args[1][0];
            return std::to_string(count_char(args[0], target));
        }
    );

    registry.register_kernel(
        "LEN_CHAR",
        "LEN_CHAR(text: str) -> int",
        "Returns exact UTF-8 grapheme/character count",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.empty()) return "0";
            return std::to_string(len_char(args[0]));
        }
    );

    registry.register_kernel(
        "REVERSE_STR",
        "REVERSE_STR(text: str) -> str",
        "Inverts string character array byte-for-byte",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.empty()) return "";
            return reverse_str(args[0]);
        }
    );

    registry.register_kernel(
        "SUBSTRING_INDEX",
        "SUBSTRING_INDEX(text: str, query: str) -> int",
        "Exact 0-indexed position of sub-string (-1 if absent)",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 2) return "-1";
            return std::to_string(substring_index(args[0], args[1]));
        }
    );

    registry.register_kernel(
        "CONCAT_STR",
        "CONCAT_STR(parts: list[str], delim: str) -> str",
        "Efficient string join and concatenation",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.empty()) return "";
            std::string delim = (args.size() > 1) ? args.back() : "";
            std::vector<std::string> parts(args.begin(), args.end() - (args.size() > 1 ? 1 : 0));
            return concat_str(parts, delim);
        }
    );

    registry.register_kernel(
        "CASE_TRANSFORM",
        "CASE_TRANSFORM(text: str, mode: str) -> str",
        "Converts text to UPPERCASE or LOWERCASE",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.empty()) return "";
            std::string mode = (args.size() > 1) ? args[1] : "UPPER";
            return case_transform(args[0], mode);
        }
    );
}

} // namespace state
} // namespace bsm_rli
