#include "forthllm/squeezer.h"
#include <cmath>
#include <sstream>

namespace forthllm {

    squeezer::squeezer(size_t input_dim, size_t output_dim)
        : projector_(input_dim, output_dim), index_("") {}

    std::string squeezer::squeeze(const std::vector<float> & embedding) {
        // Step 1: JL projection
        std::vector<float> projected = projector_.project(embedding);

        // Step 2: Quantize to bytes (0-255)
        std::string quantized;
        for (float val : projected) {
            // Normalize to [0, 1] then scale to [0, 255]
            float normalized = (val + 1.0f) / 2.0f;
            unsigned char byte = static_cast<unsigned char>(normalized * 255.0f);
            quantized += static_cast<char>(byte);
        }

        // Step 3: BWT transform
        std::string transformed = bwt_transform::transform(quantized);

        return transformed;
    }

    std::vector<float> squeezer::expand(const std::string & compressed) {
        // Step 1: Invert BWT
        std::string quantized = bwt_transform::invert(compressed);

        // Step 2: Dequantize to floats
        std::vector<float> projected(quantized.size());
        for (size_t i = 0; i < quantized.size(); ++i) {
            unsigned char byte = static_cast<unsigned char>(quantized[i]);
            float normalized = byte / 255.0f;
            projected[i] = normalized * 2.0f - 1.0f;
        }

        // Note: We can't perfectly invert the JL projection,
        // so we return the projected vector as-is.
        return projected;
    }

    void squeezer::index_add(size_t id, const std::string & compressed) {
        indexed_data_.push_back({id, compressed});

        // Rebuild the FM-index with all indexed data
        std::string combined;
        for (const auto & [id, data] : indexed_data_) {
            combined += data;
            combined += "\n";  // Separator
        }
        index_ = fm_index(combined);
    }

    std::vector<size_t> squeezer::search(const std::vector<float> & query, size_t top_k) {
        // Project the query
        std::vector<float> projected = projector_.project(query);

        // Quantize to bytes
        std::string quantized;
        for (float val : projected) {
            float normalized = (val + 1.0f) / 2.0f;
            unsigned char byte = static_cast<unsigned char>(normalized * 255.0f);
            quantized += static_cast<char>(byte);
        }

        // Use FM-index to find similar entries
        // This is a simplified search; in practice, you'd use
        // approximate nearest neighbor search on the compressed data
        std::vector<size_t> results;
        for (const auto & [id, data] : indexed_data_) {
            // Compare quantized representations
            if (data.size() == quantized.size()) {
                size_t differences = 0;
                for (size_t i = 0; i < data.size(); ++i) {
                    if (data[i] != quantized[i]) {
                        differences++;
                    }
                }
                // Include if sufficiently similar
                if (differences < quantized.size() / 4) {
                    results.push_back(id);
                }
            }
        }

        // Return top_k results
        if (results.size() > top_k) {
            results.resize(top_k);
        }

        return results;
    }

} // namespace forthllm