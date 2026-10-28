#include "forthllm/squeezer.h"
#include <algorithm>
#include <stdexcept>

namespace forthllm {

    std::string bwt_transform::transform(const std::string & input) {
        if (input.empty()) {
            return "";
        }

        // Generate all rotations
        std::vector<std::string> rotations;
        for (size_t i = 0; i < input.size(); ++i) {
            rotations.push_back(input.substr(i) + input.substr(0, i));
        }

        // Sort the rotations
        std::sort(rotations.begin(), rotations.end());

        // Extract the last column
        std::string result;
        for (const auto & rot : rotations) {
            result += rot.back();
        }

        return result;
    }

    std::string bwt_transform::invert(const std::string & transformed) {
        if (transformed.empty()) {
            return "";
        }

        size_t n = transformed.size();

        // Build the table of sorted rotations
        std::vector<std::string> table(n, "");
        for (auto & row : table) {
            row = transformed;
        }

        for (size_t i = 0; i < n; ++i) {
            for (auto & row : table) {
                row = row[0] + row.substr(1);
            }
            std::sort(table.begin(), table.end());
        }

        // Find the original string (ends with the newline or is the first row)
        for (const auto & row : table) {
            if (row[0] == transformed[0]) {
                return row;
            }
        }

        throw std::runtime_error("BWT inversion failed");
    }

} // namespace forthllm