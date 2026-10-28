#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace forthllm {

    // Johnson-Lindenstrauss projector: reduces dimensionality while
    // preserving distances (with high probability).
    class jl_projector {
    public:
        // Create a projector that maps input_dim -> output_dim
        jl_projector(size_t input_dim, size_t output_dim);

        // Project a high-dimensional vector to lower dimensions
        std::vector<float> project(const std::vector<float> & input) const;

        size_t input_dim() const { return input_dim_; }
        size_t output_dim() const { return output_dim_; }

    private:
        size_t input_dim_;
        size_t output_dim_;
        std::vector<std::vector<float>> matrix_;
    };

    // Burrows-Wheeler Transform: lossless data transformation that
    // rearranges data into runs of similar characters, making it
    // more compressible.
    class bwt_transform {
    public:
        // Apply the BWT to a string
        static std::string transform(const std::string & input);

        // Invert the BWT to recover the original string
        static std::string invert(const std::string & transformed);
    };

    // FM-index: full-text index based on the Burrows-Wheeler Transform.
    // Allows efficient substring searching without storing the full text.
    class fm_index {
    public:
        // Build the FM-index from a string
        fm_index(const std::string & text);

        // Search for a pattern in the indexed text
        // Returns the number of occurrences
        size_t count(const std::string & pattern) const;

        // Find all positions of a pattern in the indexed text
        std::vector<size_t> locate(const std::string & pattern) const;

    private:
        std::string bwt_;
        std::vector<int> c_;
        std::vector<std::vector<int>> occ_;
        std::vector<int> suffix_array_;
    };

    // Integrated squeezer: JL projection + BWT compression + FM-index
    class squeezer {
    public:
        // Create a squeezer for embeddings of the given dimension
        squeezer(size_t input_dim, size_t output_dim);

        // Compress an embedding
        std::string squeeze(const std::vector<float> & embedding);

        // Decompress a compressed embedding
        std::vector<float> expand(const std::string & compressed);

        // Add a compressed embedding to the index
        void index_add(size_t id, const std::string & compressed);

        // Search for similar embeddings
        std::vector<size_t> search(const std::vector<float> & query, size_t top_k);

    private:
        jl_projector projector_;
        fm_index index_;
        std::vector<std::pair<size_t, std::string>> indexed_data_;
    };

} // namespace forthllm