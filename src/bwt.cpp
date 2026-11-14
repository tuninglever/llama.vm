// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Alfred Leung
//
// Burrows-Wheeler Transform implementation: reversible permutation for improved
// compressibility, with exact invert via LF mapping.

#include "forthllm/squeezer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace forthllm {

    std::string bwt_transform::transform(const std::string & input) {
        const size_t n = input.size();
        if (n == 0) {
            return "";
        }
        // sort the rotations; the start index breaks ties so the transform is
        // invertible for any byte content (no unique sentinel needed)
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            // compare as uint8_t: char is signed on most platforms, and signed
            // comparison corrupts the rotation order for bytes >= 128 (invert()
            // assumes the unsigned byte order)
            for (size_t i = 0; i < n; ++i) {
                const uint8_t ca = (uint8_t) input[(a + i) % n];
                const uint8_t cb = (uint8_t) input[(b + i) % n];
                if (ca != cb) {
                    return ca < cb;
                }
            }
            return a < b;
        });
        std::string out(n, '\0');
        for (size_t r = 0; r < n; ++r) {
            out[r] = input[(order[r] + n - 1) % n];
        }
        // store the row of the original string (rotation 0) for invert()
        uint32_t r0 = 0;
        for (size_t r = 0; r < n; ++r) {
            if (order[r] == 0) {
                r0 = (uint32_t) r;
                break;
            }
        }
        out.append(reinterpret_cast<const char *>(&r0), sizeof(r0));
        return out;
    }

    std::string bwt_transform::invert(const std::string & blob) {
        if (blob.size() < sizeof(uint32_t)) {
            throw std::runtime_error("forthllm: invalid BWT blob");
        }
        const size_t n = blob.size() - sizeof(uint32_t);
        const std::string & L = blob.substr(0, n);
        uint32_t r0;
        std::memcpy(&r0, blob.data() + n, sizeof(r0));
        if (r0 >= n) {
            throw std::runtime_error("forthllm: invalid BWT row index");
        }
        // LF mapping: LF(r) = C[L[r]] + Occ(L[r], r), where C[c] is the number of
        // characters smaller than c and Occ counts c in L[0..r)
        std::array<uint32_t, 256> count{};
        for (char c : L) {
            count[(uint8_t) c]++;
        }
        std::array<uint32_t, 256> C{};
        uint32_t acc = 0;
        for (int c = 0; c < 256; ++c) {
            C[c] = acc;
            acc += count[c];
        }
        std::vector<uint32_t> occ_before(n, 0);
        std::array<uint32_t, 256> cnt{};
        for (size_t r = 0; r < n; ++r) {
            const uint8_t c = (uint8_t) L[r];
            occ_before[r] = cnt[c];
            cnt[c]++;
        }
        std::string out(n, '\0');
        uint32_t r = r0;
        for (size_t i = n; i-- > 0;) {
            const uint8_t c = (uint8_t) L[r];
            out[i] = (char) c;
            r = C[c] + occ_before[r];
        }
        return out;
    }

} // namespace forthllm
