#include "bsm_rli/llguidance_adapter.hpp"
#include <limits>
#include <algorithm>

namespace bsm_rli {

LLGuidanceAdapter::LLGuidanceAdapter(const std::string& ebnf_grammar)
    : grammar_(ebnf_grammar) {}

bool LLGuidanceAdapter::is_valid_prefix(const std::string& prefix) const {
    if (prefix.empty()) return true;
    // Check basic bracket/parenthesis balancing and prefix matching
    int open_parens = 0;
    int open_brackets = 0;
    for (char c : prefix) {
        if (c == '(') open_parens++;
        else if (c == ')') open_parens--;
        else if (c == '[') open_brackets++;
        else if (c == ']') open_brackets--;
    }
    return open_parens >= 0 && open_brackets >= 0;
}

std::vector<bool> LLGuidanceAdapter::compute_token_mask(const std::string& current_prefix,
                                                        const std::vector<std::string>& vocab) const {
    std::vector<bool> mask(vocab.size(), true);
    for (size_t i = 0; i < vocab.size(); ++i) {
        std::string candidate_prefix = current_prefix + vocab[i];
        mask[i] = is_valid_prefix(candidate_prefix);
    }
    return mask;
}

void LLGuidanceAdapter::apply_logit_mask(float* logits,
                                         size_t vocab_size,
                                         const std::vector<std::string>& vocab,
                                         const std::string& current_prefix) const {
    if (!logits || vocab.size() < vocab_size) return;
    
    auto mask = compute_token_mask(current_prefix, vocab);
    float neg_inf = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < vocab_size; ++i) {
        if (!mask[i]) {
            logits[i] = neg_inf;
        }
    }
}

} // namespace bsm_rli
