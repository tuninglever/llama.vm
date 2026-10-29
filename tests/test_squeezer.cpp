#include "forthllm/squeezer.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "ForthLLM Squeezer Test" << std::endl;

    // Test 1: JL Projection
    {
        std::cout << "Test 1: JL Projection" << std::endl;
        forthllm::jl_projector projector(1024, 128);

        std::vector<float> input(1024, 0.1f);
        auto projected = projector.project(input);

        assert(projected.size() == 128);
        std::cout << "  Input dim: 1024, Output dim: 128" << std::endl;
        std::cout << "  Passed" << std::endl;
    }

    // Test 2: BWT Transform
    {
        std::cout << "Test 2: BWT Transform" << std::endl;
        std::string original = "banana";
        std::string transformed = forthllm::bwt_transform::transform(original);
        std::string inverted = forthllm::bwt_transform::invert(transformed);

        assert(inverted == original);
        std::cout << "  Original: " << original << std::endl;
        std::cout << "  Transformed: " << transformed << std::endl;
        std::cout << "  Inverted: " << inverted << std::endl;
        std::cout << "  Passed" << std::endl;
    }

    // Test 3: FM-index
    {
        std::cout << "Test 3: FM-index" << std::endl;
        forthllm::fm_index index("banana");

        size_t count = index.count("ana");
        assert(count == 2);
        std::cout << "  Count of 'ana': " << count << " (expected 2)" << std::endl;

        auto positions = index.locate("ana");
        assert(positions.size() == 2);
        std::cout << "  Positions of 'ana': " << positions[0] << ", " << positions[1] << std::endl;
        std::cout << "  Passed" << std::endl;
    }

    // Test 4: Integrated Squeezer
    {
        std::cout << "Test 4: Integrated Squeezer" << std::endl;
        forthllm::squeezer sq(1024, 128);

        std::vector<float> embedding(1024, 0.1f);
        std::string compressed = sq.squeeze(embedding);

        assert(!compressed.empty());
        std::cout << "  Compressed size: " << compressed.size() << " bytes" << std::endl;

        auto expanded = sq.expand(compressed);
        assert(expanded.size() == 128);
        std::cout << "  Expanded size: " << expanded.size() << " floats" << std::endl;
        std::cout << "  Passed" << std::endl;
    }

    std::cout << "All tests passed!" << std::endl;
    return 0;
}