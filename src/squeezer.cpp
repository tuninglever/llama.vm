#include "forthllm/squeezer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace forthllm {

    squeezer::squeezer(size_t input_dim, size_t output_dim, uint32_t seed)
        : projector_(input_dim, output_dim, seed), index_("") {}

    std::string squeezer::squeeze(const std::vector<float> & embedding) const {
        // Step 1: JL projection
        const std::vector<float> projected = projector_.project(embedding);

        // Step 2: 8-bit quantization with a per-vector min/max scale (real KV data is
        // not confined to [-1, 1])
        float lo = projected.front();
        float hi = projected.front();
        for (float x : projected) {
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
        const float scale = (hi > lo) ? 255.0f / (hi - lo) : 0.0f;
        std::string bytes(projected.size(), '\0');
        for (size_t i = 0; i < projected.size(); ++i) {
            bytes[i] = (char) (uint8_t) std::lround((projected[i] - lo) * scale);
        }

        // Step 3: BWT transform, then append the min/max scale for expand()
        std::string blob = bwt_transform::transform(bytes);
        blob.append(reinterpret_cast<const char *>(&lo), sizeof(lo));
        blob.append(reinterpret_cast<const char *>(&hi), sizeof(hi));
        return blob;
    }

    std::vector<float> squeezer::expand(const std::string & compressed) const {
        if (compressed.size() < 8) {
            throw std::runtime_error("forthllm: invalid squeezer blob");
        }
        const size_t n_bwt = compressed.size() - 8;
        float lo, hi;
        std::memcpy(&lo, compressed.data() + n_bwt, sizeof(lo));
        std::memcpy(&hi, compressed.data() + n_bwt + sizeof(hi), sizeof(hi));
        const std::string bytes = bwt_transform::invert(compressed.substr(0, n_bwt));
        const float scale = (hi > lo) ? (hi - lo) / 255.0f : 0.0f;
        std::vector<float> out(bytes.size());
        for (size_t i = 0; i < bytes.size(); ++i) {
            out[i] = lo + (float) (uint8_t) bytes[i] * scale;
        }
        return out;
    }

    void squeezer::index_add(size_t id, const std::string & compressed) {
        indexed_data_.push_back({id, compressed});

        // Rebuild the FM-index over the indexed blobs (auxiliary; the core search is
        // the byte-distance heuristic in search())
        std::string combined;
        for (const auto & [id, data] : indexed_data_) {
            combined += data;
            combined += "\n"; // Separator
        }
        index_ = fm_index(combined);
    }

    std::vector<size_t> squeezer::search(const std::vector<float> & query, size_t top_k) const {
        // Compress the query the same way as the indexed entries, then rank by the
        // number of differing bytes (a simple heuristic, not a true ANN search)
        const std::string q = squeeze(query);
        std::vector<size_t> results;
        for (const auto & [id, data] : indexed_data_) {
            if (data.size() != q.size()) {
                continue;
            }
            size_t differences = 0;
            for (size_t i = 0; i < data.size(); ++i) {
                if (data[i] != q[i]) {
                    differences++;
                }
            }
            if (differences < q.size() / 4) {
                results.push_back(id);
            }
        }
        if (results.size() > top_k) {
            results.resize(top_k);
        }
        return results;
    }

} // namespace forthllm
