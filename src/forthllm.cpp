#include "forthllm/forthllm.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace forthllm {

    // Simple attention-based importance scoring
    // In a real implementation, this would use actual attention weights
    static float compute_attention_score(const std::vector<int> & tokens, size_t token_idx) {
        // Placeholder: give recent tokens higher scores
        float recency = 1.0f / (1.0f + (tokens.size() - token_idx - 1) * 0.1f);
        return recency;
    }

    // Simple mean pooling for chunk summary
    static std::vector<float> mean_pool(const std::vector<std::vector<float>> & embeddings, size_t dim) {
        std::vector<float> result(dim, 0.0f);
        for (const auto & emb : embeddings) {
            for (size_t i = 0; i < dim && i < emb.size(); ++i) {
                result[i] += emb[i];
            }
        }
        for (size_t i = 0; i < dim; ++i) {
            result[i] /= embeddings.size();
        }
        return result;
    }

    struct context_manager::impl {
        config cfg;
        std::vector<chunk_summary> stack;
        size_t next_chunk_id;

        impl(const config & cfg) : cfg(cfg), next_chunk_id(0) {}
    };

    context_manager::context_manager(const config & cfg) : p(new impl(cfg)) {}
    context_manager::~context_manager() = default;

    void context_manager::process_chunk(const std::vector<int> & tokens) {
        if (tokens.empty()) {
            return;
        }

        // Create a chunk summary
        chunk_summary summary;
        summary.chunk_id = p->next_chunk_id++;
        summary.start_token = 0;  // Will be set by caller
        summary.end_token = tokens.size();
        summary.embedding = std::vector<float>(p->cfg.summary_dim, 0.0f);
        summary.attention_score = 0.0f;
        summary.evicted = false;

        // Compute attention score (simplified)
        for (size_t i = 0; i < tokens.size(); ++i) {
            summary.attention_score += compute_attention_score(tokens, i);
        }
        summary.attention_score /= tokens.size();

        // In a real implementation, we would:
        // 1. Embed the tokens
        // 2. Run through transformer layers
        // 3. Extract attention weights
        // 4. Create a compressed summary

        // For now, use a simple mean of random values as a placeholder
        for (size_t i = 0; i < p->cfg.summary_dim; ++i) {
            summary.embedding[i] = static_cast<float>(std::sin(i + summary.chunk_id)) * 0.1f;
        }

        push(summary);
    }

    void context_manager::push(const chunk_summary & summary) {
        p->stack.push_back(summary);

        // Evict if over max depth
        if (p->stack.size() > p->cfg.max_stack_depth) {
            evict();
        }
    }

    chunk_summary context_manager::pop() {
        if (p->stack.empty()) {
            throw std::runtime_error("pop() on empty stack");
        }
        chunk_summary result = p->stack.back();
        p->stack.pop_back();
        return result;
    }

    const chunk_summary & context_manager::peek() const {
        if (p->stack.empty()) {
            throw std::runtime_error("peek() on empty stack");
        }
        return p->stack.back();
    }

    size_t context_manager::depth() const {
        return p->stack.size();
    }

    void context_manager::evict() {
        if (p->stack.size() <= p->cfg.max_stack_depth) {
            return;
        }

        // LRU eviction: remove oldest entries
        size_t to_remove = p->stack.size() - p->cfg.max_stack_depth;
        for (size_t i = 0; i < to_remove; ++i) {
            p->stack[i].evicted = true;
        }
        p->stack.erase(p->stack.begin(), p->stack.begin() + to_remove);
    }

    std::vector<float> context_manager::combine_context() const {
        if (p->stack.empty()) {
            return std::vector<float>(p->cfg.summary_dim, 0.0f);
        }

        // Collect all embeddings
        std::vector<std::vector<float>> embeddings;
        for (const auto & summary : p->stack) {
            if (!summary.evicted) {
                embeddings.push_back(summary.embedding);
            }
        }

        if (embeddings.empty()) {
            return std::vector<float>(p->cfg.summary_dim, 0.0f);
        }

        // Mean pool across all summaries
        return mean_pool(embeddings, p->cfg.summary_dim);
    }

    void context_manager::reset() {
        p->stack.clear();
        p->next_chunk_id = 0;
    }

    const config & context_manager::get_config() const {
        return p->cfg;
    }

} // namespace forthllm