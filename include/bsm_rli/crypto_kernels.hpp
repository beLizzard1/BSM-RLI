#ifndef BSM_RLI_CRYPTO_KERNELS_HPP
#define BSM_RLI_CRYPTO_KERNELS_HPP

#include "bsm_rli/registry.hpp"
#include <string>
#include <cstdint>

namespace bsm_rli {
namespace crypto {

void register_crypto_kernels(KernelRegistry& registry);

uint64_t bitwise_op(uint64_t a, const std::string& op, uint64_t b);
std::string hash_digest(const std::string& data, const std::string& algo);
std::string base64_encode(const std::string& data);
std::string base64_decode(const std::string& data);

} // namespace crypto
} // namespace bsm_rli

#endif // BSM_RLI_CRYPTO_KERNELS_HPP
