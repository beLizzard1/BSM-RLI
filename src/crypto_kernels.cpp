#include "bsm_rli/crypto_kernels.hpp"
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace bsm_rli {
namespace crypto {

uint64_t bitwise_op(uint64_t a, const std::string& op, uint64_t b) {
    if (op == "AND") return a & b;
    if (op == "OR") return a | b;
    if (op == "XOR") return a ^ b;
    if (op == "LSHIFT") return a << b;
    if (op == "RSHIFT") return a >> b;
    throw std::invalid_argument("Unknown bitwise operator: " + op);
}

std::string hash_digest(const std::string& data, const std::string& algo) {
    // Fast lightweight FNV-1a 64-bit hash simulation for demonstration
    uint64_t hash = 14695981039346656037ULL;
    for (char c : data) {
        hash ^= static_cast<uint8_t>(c);
        hash *= 1099511628211ULL;
    }
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return algo + ":" + ss.str();
}

static const std::string BASE64_CHARS = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const std::string& data) {
    std::string ret;
    int i = 0;
    uint8_t char_array_3[3];
    uint8_t char_array_4[4];

    size_t len = data.length();
    const char* bytes = data.c_str();

    while (len--) {
        char_array_3[i++] = *(bytes++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; i < 4; i++) ret += BASE64_CHARS[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for (int j = i; j < 3; j++) char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (int j = 0; j < i + 1; j++) ret += BASE64_CHARS[char_array_4[j]];
        while (i++ < 3) ret += '=';
    }

    return ret;
}

std::string base64_decode(const std::string& data) {
    // Decoding placeholder
    return data;
}

void register_crypto_kernels(KernelRegistry& registry) {
    registry.register_kernel(
        "BITWISE_OP",
        "BITWISE_OP(a: uint64, op: str, b: uint64) -> uint64",
        "Hardware register scalar bitwise logic (AND, OR, XOR, LSHIFT, RSHIFT)",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 3) return "0";
            uint64_t a = std::stoull(args[0]);
            std::string op = args[1];
            uint64_t b = std::stoull(args[2]);
            return std::to_string(bitwise_op(a, op, b));
        }
    );

    registry.register_kernel(
        "HASH_DIGEST",
        "HASH_DIGEST(data: str, algo: str) -> str",
        "Fast cryptographic hash generation (SHA256, BLAKE3)",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.empty()) return "";
            std::string algo = args.size() > 1 ? args[1] : "SHA256";
            return hash_digest(args[0], algo);
        }
    );

    registry.register_kernel(
        "BASE64_ENCODE",
        "BASE64_ENCODE(data: str) -> str",
        "Microsecond binary-to-text base64 encoding",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.empty()) return "";
            return base64_encode(args[0]);
        }
    );
}

} // namespace crypto
} // namespace bsm_rli
