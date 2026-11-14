// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Alfred Leung

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ForthLLM: chunked context compression (model-agnostic math).
//
// Old prompt chunks (C tokens) are replaced by a small number (S << C) of KV summary
// rows. Each summary row is a weighted combination of the chunk's K/V rows (per layer,
// per KV head), so the row shape is preserved and the attention graph is untouched: an
// old chunk occupies S KV cells instead of C.
//
// This header holds the self-contained math (summarization, chunk stack, parse
// helpers). The KV-cache surgery (reading rows, writing summaries back, freeing cells,
// shifting positions) and the C API live in llama.cpp (src/forthllm.cpp +
// llama_kv_cache::forthllm_compact). The squeezer/BWT/JL/FM-index used for the
// stored/offloaded representation are in "forthllm/squeezer.h".

namespace forthllm {

    // Summary method for compressing a chunk's KV rows into S summary rows
    enum class summary_method {
        attention,   // per-head last-key similarity weights within each sub-chunk
        mean,        // uniform pooling of each sub-chunk
        last         // keep the last S rows verbatim
    };

    // Eviction policy for the chunk stack
    enum class eviction_policy {
        lru,           // least important (ties: oldest)
        lfu,           // falls back to lru until per-chunk attention counts exist
        fixed_window   // keep only the most recent max_stack_depth chunks
    };

    // Configuration for ForthLLM
    struct config {
        size_t chunk_size      = 128;  // tokens per chunk (C)
        size_t summary_dim     = 1;    // summary rows per chunk (S)
        summary_method summary = summary_method::attention;
        eviction_policy eviction = eviction_policy::lru;
        size_t max_stack_depth  = 64;  // maximum number of summarized chunks to keep
        std::string squeezer    = "none"; // "none" | "jl-bwt"
        size_t jl_dim           = 128;    // JL projection output dim (squeezer)

        // throws std::runtime_error with a descriptive message on invalid values
        void validate() const;
    };

    // one summarized chunk on the stack
    struct chunk_info {
        size_t chunk_id   = 0;    // prompt-order index of the chunk
        size_t n_tokens   = 0;    // raw tokens in the chunk
        size_t n_summary  = 0;    // summary rows in the KV cache
        float  importance  = 1.0f; // eviction importance (higher = keep)
    };

    // Summarize one chunk of KV rows (all KV heads of one layer) into S summary rows.
    // k: [C][n_heads_k*head_dim_k], v: [C][n_heads_v*head_dim_v], row-major
    // (head_dim_k and head_dim_v may differ; n_heads_k == n_heads_v == n_head_kv).
    //   attention: the chunk is split into S contiguous sub-chunks; for each, per-head
    //              weights w_i = softmax(k_i[h] . k_last[h] / sqrt(head_dim_k)) (last-key
    //              similarity within the sub-chunk) are applied to both K and V rows
    //   mean:      the chunk is split into S contiguous sub-chunks, each uniformly pooled
    //   last:      the last S rows are kept verbatim
    // k_out/v_out: [S][...]. S is clamped to C.
    void summarize(
            const float * k, size_t n_heads_k, size_t head_dim_k,
            const float * v, size_t n_heads_v, size_t head_dim_v,
                   size_t C, size_t S,
            summary_method method,
            std::vector<float> & k_out, std::vector<float> & v_out);

    // Stack of summarized chunks with an eviction policy. When the stack exceeds
    // max_stack_depth, evict() returns the chunk_ids whose summaries should be dropped
    // from the KV cache entirely (the chunk's content leaves the context).
    class stack {
    public:
        explicit stack(const config & cfg);

        void push(const chunk_info & info);
        std::vector<size_t> evict(); // chunk_ids to drop; may be called repeatedly
        size_t size() const;
        const chunk_info & at(size_t i) const;
        void clear();

    private:
        config cfg_;
        std::vector<chunk_info> stack_;
    };

    // parse helpers (case-insensitive); throw std::runtime_error on unknown
    summary_method summary_method_from_str(const std::string & s);
    const char * summary_method_name(summary_method m);
    std::string eviction_from_str(const std::string & s);
    std::string squeezer_from_str(const std::string & s);

} // namespace forthllm
