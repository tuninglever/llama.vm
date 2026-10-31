#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace forthllm {

    // Johnson-Lindenstrauss projector: reduces dimensionality while preserving
    // distances (with high probability). The projection matrix is a deterministic
    // Gaussian N(0, 1/output_dim) drawn from a fixed seed, so the same (input_dim,
    // output_dim, seed) always gives the same projection. Not invertible: the output
    // is within the JL/quantization error of the true projection.
    class jl_projector {
    public:
        // Create a projector that maps input_dim -> output_dim
        jl_projector(size_t input_dim, size_t output_dim, uint32_t seed = 0);

        // Project a high-dimensional vector to lower dimensions
        std::vector<float> project(const std::vector<float> & input) const;

        size_t input_dim()  const { return input_dim_; }
        size_t output_dim() const { return output_dim_; }

    private:
        size_t input_dim_;
        size_t output_dim_;
        uint32_t seed_;
        std::vector<std::vector<float>> matrix_; // [output_dim][input_dim]
    };

    // Burrows-Wheeler Transform: lossless data transformation that rearranges data
    // into runs of similar bytes, making it more compressible. transform() appends the
    // original row index (a uint32_t) so invert() is exact for any byte content,
    // including bytes >= 128 (the rotation order is compared as unsigned bytes).
    class bwt_transform {
    public:
        // Apply the BWT to a string (returns input.size() + 4 bytes)
        static std::string transform(const std::string & input);

        // Invert the BWT to recover the original string (exact)
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

    // Integrated squeezer: JL projection -> per-vector 8-bit quantization -> BWT.
    // The in-cache summary rows stay head-dim; the squeezer is used for the
    // stored/offloaded representation. squeeze() stores a min/max scale with the blob
    // so expand() recovers the dequantized JL projection (within quantization error).
    class squeezer {
    public:
        // Create a squeezer for embeddings of the given dimension
        squeezer(size_t input_dim, size_t output_dim, uint32_t seed = 0);

        // Compress an embedding (JL -> 8-bit quant -> BWT -> min/max scale)
        std::string squeeze(const std::vector<float> & embedding) const;

        // Decompress a compressed embedding (returns the dequantized JL projection,
        // output_dim floats)
        std::vector<float> expand(const std::string & compressed) const;

        // Add a compressed embedding to the index
        void index_add(size_t id, const std::string & compressed);

        // Search for similar embeddings (simple byte-distance heuristic over the
        // compressed blobs; not part of the core ForthLLM path)
        std::vector<size_t> search(const std::vector<float> & query, size_t top_k) const;

    private:
        jl_projector projector_;
        fm_index index_;
        std::vector<std::pair<size_t, std::string>> indexed_data_;
    };

} // namespace forthllm
