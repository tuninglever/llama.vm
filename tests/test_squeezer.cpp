// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Alfred Leung

#include "forthllm/squeezer.h"

#include <algorithm>
#include <cassert>
#include <iostream>

int main() {
    std::cout << "ForthLLM Squeezer Test" << std::endl;

    // Test 1: JL Projection
    {
        std::cout << "Test 1: JL Projection" << std::endl;
        forthllm::jl_projector projector(1024, 128);

        std::vector<float> input(1024, 0.1f);
        auto projected = projector.project(input);

        assert(projected.size() == 128);
        // deterministic: the same projector gives the same projection
        auto projected2 = projector.project(input);
        assert(projected == projected2);
        std::cout << "  Input dim: 1024, Output dim: 128" << std::endl;
        std::cout << "  Passed" << std::endl;
    }

    // Test 2: BWT Transform (ASCII round-trip)
    {
        std::cout << "Test 2: BWT Transform (ASCII)" << std::endl;
        std::string original = "banana";
        std::string transformed = forthllm::bwt_transform::transform(original);
        std::string inverted = forthllm::bwt_transform::invert(transformed);

        assert(inverted == original);
        assert(transformed.size() == original.size() + 4); // + uint32 row index
        std::cout << "  Original: " << original << std::endl;
        std::cout << "  Transformed size: " << transformed.size() << " (input + 4)" << std::endl;
        std::cout << "  Inverted: " << inverted << std::endl;
        std::cout << "  Passed" << std::endl;
    }

    // Test 3: BWT Transform (bytes >= 128 — the uint8_t comparator fix)
    {
        std::cout << "Test 3: BWT Transform (high bytes)" << std::endl;
        std::string original;
        original += (char) 200;
        original += (char) 255;
        original += (char) 0;
        original += (char) 128;
        original += (char) 1;
        original += (char) 250;
        std::string transformed = forthllm::bwt_transform::transform(original);
        std::string inverted = forthllm::bwt_transform::invert(transformed);

        assert(inverted == original);
        std::cout << "  Round-tripped " << original.size() << " high-byte values exactly" << std::endl;
        std::cout << "  Passed" << std::endl;
    }

    // Test 4: FM-index
    {
        std::cout << "Test 4: FM-index" << std::endl;
        forthllm::fm_index index("banana");

        size_t count = index.count("ana");
        assert(count == 2);
        std::cout << "  Count of 'ana': " << count << " (expected 2)" << std::endl;

        auto positions = index.locate("ana");
        assert(positions.size() == 2);
        std::cout << "  Positions of 'ana': " << positions[0] << ", " << positions[1] << std::endl;
        std::cout << "  Passed" << std::endl;
    }

    // Test 5: Integrated Squeezer (JL -> 8-bit quant -> BWT)
    {
        std::cout << "Test 5: Integrated Squeezer" << std::endl;
        forthllm::squeezer sq(1024, 128);

        std::vector<float> embedding(1024, 0.1f);
        std::string compressed = sq.squeeze(embedding);

        assert(!compressed.empty());
        // 128 projected bytes + 4 (BWT row index) + 8 (min/max scale)
        assert(compressed.size() == 128 + 4 + 8);
        std::cout << "  Compressed size: " << compressed.size() << " bytes" << std::endl;

        auto expanded = sq.expand(compressed);
        assert(expanded.size() == 128);
        std::cout << "  Expanded size: " << expanded.size() << " floats" << std::endl;
        std::cout << "  Passed" << std::endl;
    }

    // Test 6: Squeezer preserves values outside [-1, 1] (per-vector min/max scale)
    {
        std::cout << "Test 6: Squeezer min/max scaling" << std::endl;
        forthllm::squeezer sq(64, 32);

        // values well outside [-1, 1]: the old [-1,1] assumption would clip these
        std::vector<float> embedding(64);
        for (size_t i = 0; i < embedding.size(); ++i) {
            embedding[i] = (float) i * 10.0f; // 0, 10, 20, ... 630
        }
        std::string compressed = sq.squeeze(embedding);
        auto expanded = sq.expand(compressed);
        assert(expanded.size() == 32);
        // the min and max of the JL projection must be recoverable (within 1/255 scale)
        float lo = expanded[0], hi = expanded[0];
        for (float x : expanded) {
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
        assert(hi - lo > 1.0f); // not collapsed to a point (would happen if clipped to [-1,1])
        std::cout << "  Recovered range: [" << lo << ", " << hi << "]" << std::endl;
        std::cout << "  Passed" << std::endl;
    }

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
