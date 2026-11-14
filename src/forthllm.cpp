// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Alfred Leung
//
// ForthLLM engine implementation: chunked KV-row summarization (attention/mean/last),
// stack management with LRU/LFU/fixed-window eviction, and config validation.

#include "forthllm/forthllm.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace forthllm {

    void config::validate() const {
        if (chunk_size == 0) {
            throw std::runtime_error("forthllm: chunk_size must be > 0");
        }
        if (summary_dim == 0) {
            throw std::runtime_error("forthllm: summary_dim must be > 0");
        }
        if (summary_dim > chunk_size) {
            throw std::runtime_error("forthllm: summary_dim (" + std::to_string(summary_dim) +
                                     ") must be <= chunk_size (" + std::to_string(chunk_size) + ")");
        }
        if (max_stack_depth == 0) {
            throw std::runtime_error("forthllm: max_stack_depth must be > 0");
        }
        squeezer_from_str(squeezer); // throws on unknown
        if (squeezer == "jl-bwt" && jl_dim == 0) {
            throw std::runtime_error("forthllm: jl_dim must be > 0 when the squeezer is jl-bwt");
        }
        // summary and eviction are enums, so they are always valid
    }

    summary_method summary_method_from_str(const std::string & s) {
        std::string l = s;
        std::transform(l.begin(), l.end(), l.begin(), ::tolower);
        if (l == "attention") {
            return summary_method::attention;
        }
        if (l == "mean") {
            return summary_method::mean;
        }
        if (l == "last") {
            return summary_method::last;
        }
        throw std::runtime_error("forthllm: unknown summary method '" + s + "' (use attention|mean|last)");
    }

    const char * summary_method_name(summary_method m) {
        switch (m) {
            case summary_method::attention: return "attention";
            case summary_method::mean:      return "mean";
            case summary_method::last:      return "last";
        }
        return "unknown";
    }

    std::string eviction_from_str(const std::string & s) {
        std::string l = s;
        std::transform(l.begin(), l.end(), l.begin(), ::tolower);
        if (l == "lru" || l == "lfu" || l == "fixed-window") {
            return l;
        }
        throw std::runtime_error("forthllm: unknown eviction policy '" + s + "' (use lru|lfu|fixed-window)");
    }

    std::string squeezer_from_str(const std::string & s) {
        std::string l = s;
        std::transform(l.begin(), l.end(), l.begin(), ::tolower);
        if (l == "none" || l == "jl-bwt") {
            return l;
        }
        throw std::runtime_error("forthllm: unknown squeezer '" + s + "' (use none|jl-bwt)");
    }

    void summarize(
            const float * k, size_t n_heads_k, size_t head_dim_k,
            const float * v, size_t n_heads_v, size_t head_dim_v,
                   size_t C, size_t S,
            summary_method method,
            std::vector<float> & k_out, std::vector<float> & v_out) {
        if (C == 0 || S == 0) {
            k_out.clear();
            v_out.clear();
            return;
        }
        S = std::min(S, C);
        const size_t kv_dim_k = n_heads_k * head_dim_k;
        const size_t kv_dim_v = n_heads_v * head_dim_v;
        k_out.assign(S * kv_dim_k, 0.0f);
        v_out.assign(S * kv_dim_v, 0.0f);

        if (method == summary_method::last) {
            for (size_t s = 0; s < S; ++s) {
                const size_t i = C - S + s;
                std::memcpy(&k_out[s * kv_dim_k], &k[i * kv_dim_k], kv_dim_k * sizeof(float));
                std::memcpy(&v_out[s * kv_dim_v], &v[i * kv_dim_v], kv_dim_v * sizeof(float));
            }
            return;
        }

        // split the chunk into S contiguous sub-chunks (even split; the boundaries are
        // floor(i*C/S) so every sub-chunk is non-empty); pool each sub-chunk with
        // per-head weights (head h's K and V rows are pooled with head h's weights)
        for (size_t s = 0; s < S; ++s) {
            const size_t i0 = (s * C) / S;
            const size_t i1 = ((s + 1) * C) / S;
            const size_t n  = i1 - i0;

            // per-head weights from the K rows (last-key similarity within the
            // sub-chunk: the sub-chunk's last key stands in for the query, so rows
            // similar to it get the most weight)
            std::vector<std::vector<float>> W(n_heads_k, std::vector<float>(n, 1.0f / (float) n));
            if (method == summary_method::attention) {
                for (size_t h = 0; h < n_heads_k; ++h) {
                    const float * q = &k[(i1 - 1) * kv_dim_k + h * head_dim_k];
                    float max_score = -INFINITY;
                    for (size_t i = 0; i < n; ++i) {
                        const float * kr = &k[(i0 + i) * kv_dim_k + h * head_dim_k];
                        float dot = 0.0f;
                        for (size_t j = 0; j < head_dim_k; ++j) {
                            dot += kr[j] * q[j];
                        }
                        W[h][i] = dot / std::sqrt((float) head_dim_k);
                        max_score = std::max(max_score, W[h][i]);
                    }
                    float sum = 0.0f;
                    for (size_t i = 0; i < n; ++i) {
                        W[h][i] = std::exp(W[h][i] - max_score);
                        sum += W[h][i];
                    }
                    for (size_t i = 0; i < n; ++i) {
                        W[h][i] /= sum;
                    }
                }
            }

            for (size_t h = 0; h < n_heads_k; ++h) {
                for (size_t j = 0; j < head_dim_k; ++j) {
                    float ak = 0.0f;
                    for (size_t i = 0; i < n; ++i) {
                        ak += W[h][i] * k[(i0 + i) * kv_dim_k + h * head_dim_k + j];
                    }
                    k_out[s * kv_dim_k + h * head_dim_k + j] = ak;
                }
            }
            for (size_t h = 0; h < n_heads_v; ++h) {
                for (size_t j = 0; j < head_dim_v; ++j) {
                    float av = 0.0f;
                    for (size_t i = 0; i < n; ++i) {
                        av += W[h][i] * v[(i0 + i) * kv_dim_v + h * head_dim_v + j];
                    }
                    v_out[s * kv_dim_v + h * head_dim_v + j] = av;
                }
            }
        }
    }

    stack::stack(const config & cfg) : cfg_(cfg) {}

    void stack::push(const chunk_info & info) {
        stack_.push_back(info);
    }

    std::vector<size_t> stack::evict() {
        std::vector<size_t> to_drop;
        if (stack_.size() <= cfg_.max_stack_depth) {
            return to_drop;
        }
        if (cfg_.eviction == eviction_policy::fixed_window) {
            // keep only the most recent max_stack_depth chunks
            const size_t n_drop = stack_.size() - cfg_.max_stack_depth;
            for (size_t i = 0; i < n_drop; ++i) {
                to_drop.push_back(stack_[i].chunk_id);
            }
            stack_.erase(stack_.begin(), stack_.begin() + n_drop);
            return to_drop;
        }
        // lru (and lfu, which falls back to lru until per-chunk attention counts are
        // available): drop the least important, ties broken by oldest
        while (stack_.size() > cfg_.max_stack_depth) {
            size_t worst = 0;
            for (size_t i = 1; i < stack_.size(); ++i) {
                if (stack_[i].importance < stack_[worst].importance) {
                    worst = i;
                }
            }
            to_drop.push_back(stack_[worst].chunk_id);
            stack_.erase(stack_.begin() + worst);
        }
        return to_drop;
    }

    size_t stack::size() const {
        return stack_.size();
    }

    const chunk_info & stack::at(size_t i) const {
        return stack_.at(i);
    }

    void stack::clear() {
        stack_.clear();
    }

} // namespace forthllm
