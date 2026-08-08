#ifndef BSM_RLI_LLGUIDANCE_ADAPTER_HPP
#define BSM_RLI_LLGUIDANCE_ADAPTER_HPP

#include <string>
#include <vector>
#include <limits>
#include <cstddef>

namespace bsm_rli {

class LLGuidanceAdapter {
public:
    explicit LLGuidanceAdapter(const std::string& ebnf_grammar);

    // Evaluates next-token text candidates against the active EBNF grammar
    // Returns a boolean mask where true = token is valid transition
    std::vector<bool> compute_token_mask(const std::string& current_prefix,
                                         const std::vector<std::string>& vocab) const;

    // Clamps candidate logits array to -infinity for invalid token indices
    void apply_logit_mask(float* logits,
                          size_t vocab_size,
                          const std::vector<std::string>& vocab,
                          const std::string& current_prefix) const;

    bool is_valid_prefix(const std::string& prefix) const;

private:
    std::string grammar_;
};

} // namespace bsm_rli

#endif // BSM_RLI_LLGUIDANCE_ADAPTER_HPP
