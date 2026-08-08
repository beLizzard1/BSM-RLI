#include "bsm_rli/interceptor.hpp"
#include <iostream>
#include <sstream>

namespace bsm_rli {

RegionScopedInterceptor::RegionScopedInterceptor(KernelRegistry& registry)
    : registry_(registry),
      adapter_(std::make_unique<LLGuidanceAdapter>(registry.generate_ebnf_grammar())) {}

bool RegionScopedInterceptor::process_token(const std::string& token) {
    buffer_ += token;

    if (state_ == InterceptorState::Unconstrained) {
        if (buffer_.find(JIT_START) != std::string::npos) {
            state_ = InterceptorState::ConstrainedJIT;
            return true;
        }
    } else if (state_ == InterceptorState::ConstrainedJIT) {
        if (buffer_.find(JIT_END) != std::string::npos) {
            state_ = InterceptorState::Unconstrained;
            return true;
        }
    }
    return false;
}

void RegionScopedInterceptor::reset() {
    state_ = InterceptorState::Unconstrained;
    buffer_.clear();
}

void RegionScopedInterceptor::apply_logit_mask(float* logits, size_t vocab_size, const std::vector<std::string>& vocab) const {
    if (state_ == InterceptorState::ConstrainedJIT && adapter_) {
        adapter_->apply_logit_mask(logits, vocab_size, vocab, buffer_);
    }
}

std::string RegionScopedInterceptor::execute_intercepted() {
    auto start_pos = buffer_.find(JIT_START);
    auto end_pos = buffer_.find(JIT_END);

    if (start_pos == std::string::npos || end_pos == std::string::npos) {
        return "";
    }

    size_t call_start = start_pos + JIT_START.length();
    std::string call_str = buffer_.substr(call_start, end_pos - call_start);

    // Basic parsing: ID(arg1, arg2, ...)
    auto paren_pos = call_str.find('(');
    if (paren_pos == std::string::npos) return "PARSE_ERROR";

    std::string kernel_id = call_str.substr(0, paren_pos);
    std::string args_str = call_str.substr(paren_pos + 1);
    if (!args_str.empty() && args_str.back() == ')') {
        args_str.pop_back();
    }

    std::vector<std::string> args;
    std::stringstream ss(args_str);
    std::string arg;
    while (std::getline(ss, arg, ',')) {
        size_t first = arg.find_first_not_of(" \"'");
        size_t last = arg.find_last_not_of(" \"'");
        if (first != std::string::npos && last != std::string::npos) {
            args.push_back(arg.substr(first, (last - first + 1)));
        }
    }

    return registry_.execute(kernel_id, args);
}

} // namespace bsm_rli
