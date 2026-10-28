#include "forthllm/forthllm.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "ForthLLM: Stack-like context processing test" << std::endl;

    // Test 1: Basic push/pop
    {
        std::cout << "Test 1: Basic push/pop" << std::endl;
        forthllm::config cfg;
        cfg.max_stack_depth = 10;
        forthllm::context_manager cm(cfg);

        // Push some chunks
        for (int i = 0; i < 5; ++i) {
            std::vector<int> tokens = {i, i+1, i+2};
            cm.process_chunk(tokens);
        }

        assert(cm.depth() == 5);
        std::cout << "  Stack depth: " << cm.depth() << " (expected 5)" << std::endl;

        // Pop one
        auto summary = cm.pop();
        assert(cm.depth() == 4);
        std::cout << "  After pop, depth: " << cm.depth() << " (expected 4)" << std::endl;
        std::cout << "  Passed" << std::endl;
    }

    // Test 2: Eviction
    {
        std::cout << "Test 2: Eviction" << std::endl;
        forthllm::config cfg;
        cfg.max_stack_depth = 3;
        forthllm::context_manager cm(cfg);

        // Push more than max depth
        for (int i = 0; i < 5; ++i) {
            std::vector<int> tokens = {i, i+1, i+2};
            cm.process_chunk(tokens);
        }

        assert(cm.depth() == 3);
        std::cout << "  Stack depth after eviction: " << cm.depth() << " (expected 3)" << std::endl;
        std::cout << "  Passed" << std::endl;
    }

    // Test 3: Combine context
    {
        std::cout << "Test 3: Combine context" << std::endl;
        forthllm::config cfg;
        cfg.summary_dim = 768;
        forthllm::context_manager cm(cfg);

        for (int i = 0; i < 3; ++i) {
            std::vector<int> tokens = {i, i+1, i+2};
            cm.process_chunk(tokens);
        }

        auto context = cm.combine_context();
        assert(context.size() == 768);
        std::cout << "  Combined context size: " << context.size() << " (expected 768)" << std::endl;
        std::cout << "  Passed" << std::endl;
    }

    // Test 4: Reset
    {
        std::cout << "Test 4: Reset" << std::endl;
        forthllm::config cfg;
        forthllm::context_manager cm(cfg);

        for (int i = 0; i < 3; ++i) {
            std::vector<int> tokens = {i, i+1, i+2};
            cm.process_chunk(tokens);
        }

        assert(cm.depth() == 3);
        cm.reset();
        assert(cm.depth() == 0);
        std::cout << "  Stack depth after reset: " << cm.depth() << " (expected 0)" << std::endl;
        std::cout << "  Passed" << std::endl;
    }

    std::cout << "All tests passed!" << std::endl;
    return 0;
}