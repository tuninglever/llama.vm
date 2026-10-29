#include "forthllm/squeezer.h"
#include <algorithm>
#include <map>

namespace forthllm {

    fm_index::fm_index(const std::string & text) {
        // Compute the BWT
        bwt_ = bwt_transform::transform(text);

        size_t n = bwt_.size();

        // Build the C array (cumulative counts)
        // C[c] = number of characters in BWT that are lexicographically smaller than c
        c_.resize(256, 0);
        std::map<char, int> char_counts;
        for (char c : bwt_) {
            char_counts[c]++;
        }

        int total = 0;
        for (int i = 0; i < 256; ++i) {
            char ch = static_cast<char>(i);
            c_[i] = total;
            if (char_counts.find(ch) != char_counts.end()) {
                total += char_counts[ch];
            }
        }

        // Build the OCC array (occurrence counts)
        // OCC[i][c] = number of occurrences of c in BWT[0..i-1]
        for (size_t i = 0; i <= n; ++i) {
            occ_.push_back(std::vector<int>(256, 0));
        }

        for (size_t i = 0; i < n; ++i) {
            occ_[i + 1] = occ_[i];
            unsigned char ch = static_cast<unsigned char>(bwt_[i]);
            occ_[i + 1][ch]++;
        }

        // Build the suffix array
        suffix_array_.resize(n);
        int row = 0;
        for (size_t i = 0; i < n; ++i) {
            unsigned char ch = static_cast<unsigned char>(bwt_[row]);
            row = c_[ch] + occ_[row][ch];
            suffix_array_[i] = row;
        }
    }

    size_t fm_index::count(const std::string & pattern) const {
        size_t n = bwt_.size();
        int top = 0;
        int bottom = n - 1;

        for (int i = pattern.size() - 1; i >= 0; --i) {
            unsigned char c = static_cast<unsigned char>(pattern[i]);
            top = c_[c] + occ_[top][c];
            bottom = c_[c] + occ_[bottom + 1][c] - 1;

            if (top > bottom) {
                return 0;
            }
        }

        return bottom - top + 1;
    }

    std::vector<size_t> fm_index::locate(const std::string & pattern) const {
        std::vector<size_t> positions;
        size_t n = bwt_.size();
        int top = 0;
        int bottom = n - 1;

        for (int i = pattern.size() - 1; i >= 0; --i) {
            unsigned char c = static_cast<unsigned char>(pattern[i]);
            top = c_[c] + occ_[top][c];
            bottom = c_[c] + occ_[bottom + 1][c] - 1;

            if (top > bottom) {
                return positions;
            }
        }

        for (int i = top; i <= bottom; ++i) {
            positions.push_back(n - 1 - suffix_array_[i]);
        }

        return positions;
    }

} // namespace forthllm