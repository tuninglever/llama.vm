#include "forthllm/squeezer.h"

#include <cmath>
#include <random>
#include <stdexcept>

namespace forthllm {

    jl_projector::jl_projector(size_t input_dim, size_t output_dim, uint32_t seed)
        : input_dim_(input_dim), output_dim_(output_dim), seed_(seed) {
        if (input_dim == 0 || output_dim == 0) {
            throw std::runtime_error("forthllm: jl_projector dims must be > 0");
        }
        // deterministic Gaussian projection N(0, 1/output_dim): the same
        // (input_dim, output_dim, seed) always gives the same matrix
        std::mt19937 gen(seed_);
        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt((float) output_dim_));
        matrix_.resize(output_dim_, std::vector<float>(input_dim_));
        for (auto & row : matrix_) {
            for (auto & m : row) {
                m = dist(gen);
            }
        }
    }

    std::vector<float> jl_projector::project(const std::vector<float> & input) const {
        if (input.size() != input_dim_) {
            throw std::runtime_error("forthllm: jl_projector input dim mismatch (" +
                                     std::to_string(input.size()) + " != " +
                                     std::to_string(input_dim_) + ")");
        }
        std::vector<float> result(output_dim_, 0.0f);
        for (size_t r = 0; r < output_dim_; ++r) {
            const std::vector<float> & row = matrix_[r];
            float acc = 0.0f;
            for (size_t j = 0; j < input_dim_; ++j) {
                acc += row[j] * input[j];
            }
            result[r] = acc;
        }
        return result;
    }

} // namespace forthllm
