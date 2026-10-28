#include "forthllm/squeezer.h"
#include <cmath>
#include <random>

namespace forthllm {

    jl_projector::jl_projector(size_t input_dim, size_t output_dim)
        : input_dim_(input_dim), output_dim_(output_dim) {
        // Initialize projection matrix with random values from N(0, 1/output_dim)
        matrix_.resize(output_dim, std::vector<float>(input_dim, 0.0f));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(output_dim)));

        for (auto & row : matrix_) {
            for (auto & val : row) {
                val = dist(gen);
            }
        }
    }

    std::vector<float> jl_projector::project(const std::vector<float> & input) const {
        if (input.size() != input_dim_) {
            throw std::runtime_error("input dimension mismatch");
        }

        std::vector<float> result(output_dim_, 0.0f);
        for (size_t i = 0; i < output_dim_; ++i) {
            for (size_t j = 0; j < input_dim_; ++j) {
                result[i] += matrix_[i][j] * input[j];
            }
        }
        return result;
    }

} // namespace forthllm