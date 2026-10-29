#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ForthLLM: Stack-like context processing for LLMs
// Processes input in chunks, replaces intermediate values with compressed
// representations, and maintains a fixed-size hidden state.

namespace forthllm {

    // Chunk processing strategy
    enum class chunk_strategy {
        fixed_size,       // Fixed number of tokens per chunk
        sentence,         // Process by sentence boundaries
        paragraph,        // Process by paragraph boundaries
        semantic          // Process by semantic units (future)
    };

    // Summary method for chunk compression
    enum class summary_method {
        attention,        // Use attention weights to identify important tokens
        mean,             // Average embeddings of chunk
        last,             // Use last token embedding
        learned           // Train a small summarization model (future)
    };

    // Eviction policy for stack entries
    enum class eviction_policy {
        lru,              // Least recently used
        lfu,              // Least frequently used
        fixed_window,     // Fixed-size sliding window
        attention_score   // Evict based on attention relevance (future)
    };

    // Configuration for ForthLLM context manager
    struct config {
        // Chunk size in tokens (for fixed_size strategy)
        size_t chunk_size = 128;

        // Strategy for determining chunk boundaries
        chunk_strategy strategy = chunk_strategy::fixed_size;

        // Method for compressing chunks into summaries
        summary_method summary = summary_method::attention;

        // Eviction policy for stack entries
        eviction_policy eviction = eviction_policy::lru;

        // Maximum stack depth (number of summaries to keep)
        size_t max_stack_depth = 64;

        // Summary embedding dimension (typically 768 or 1024)
        size_t summary_dim = 768;

        // Enable verbose logging
        bool verbose = false;
    };

    // Represents a processed chunk and its summary
    struct chunk_summary {
        size_t chunk_id;
        size_t start_token;
        size_t end_token;
        std::vector<float> embedding;  // Compressed representation
        float attention_score;         // Importance score
        bool evicted;
    };

    // ForthLLM context manager
    class context_manager {
    public:
        explicit context_manager(const config & cfg);
        ~context_manager();

        context_manager(const context_manager &) = delete;
        context_manager & operator=(const context_manager &) = delete;

        // Process a new chunk of tokens
        void process_chunk(const std::vector<int> & tokens);

        // Push a processed chunk onto the stack
        void push(const chunk_summary & summary);

        // Pop the most recent chunk from the stack
        chunk_summary pop();

        // Peek at the most recent chunk without removing it
        const chunk_summary & peek() const;

        // Get the current stack depth
        size_t depth() const;

        // Evict least important entries based on policy
        void evict();

        // Combine all stack entries into a single context representation
        std::vector<float> combine_context() const;

        // Reset the context manager
        void reset();

        // Get configuration
        const config & get_config() const;

    private:
        struct impl;
        std::unique_ptr<impl> p;
    };

} // namespace forthllm