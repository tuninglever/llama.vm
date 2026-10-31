#include "forthllm/forthllm.h"

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>

namespace {

    bool close(float a, float b, float tol = 1e-3f) {
        return std::fabs(a - b) <= tol;
    }

    bool throws(std::function<void()> f) {
        try {
            f();
        } catch (const std::exception &) {
            return true;
        }
        return false;
    }

} // namespace

int main() {
    std::cout << "ForthLLM: chunked KV summarization + stack test" << std::endl;

    // Shared fixture: 1 head, head_dim 4, C=4 rows, distinct one-hot K and V rows
    const size_t n_heads = 1, head_dim = 4, C = 4, S = 2;
    std::vector<float> k = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    std::vector<float> v = k;

    // Test 1: "last" keeps the last S rows verbatim
    {
        std::cout << "Test 1: summarize(last)" << std::endl;
        std::vector<float> k_out, v_out;
        forthllm::summarize(k.data(), n_heads, head_dim, v.data(), n_heads, head_dim,
                            C, S, forthllm::summary_method::last, k_out, v_out);
        assert(k_out.size() == S * n_heads * head_dim);
        assert(close(k_out[0], 0) && close(k_out[1], 0) && close(k_out[2], 1) && close(k_out[3], 0));
        assert(close(k_out[4], 0) && close(k_out[5], 0) && close(k_out[6], 0) && close(k_out[7], 1));
        assert(v_out == k_out);
        std::cout << "  Passed" << std::endl;
    }

    // Test 2: "mean" uniformly pools each sub-chunk
    {
        std::cout << "Test 2: summarize(mean)" << std::endl;
        std::vector<float> k_out, v_out;
        forthllm::summarize(k.data(), n_heads, head_dim, v.data(), n_heads, head_dim,
                            C, S, forthllm::summary_method::mean, k_out, v_out);
        // sub-chunk 0 = rows {0,1} -> (0.5,0.5,0,0); sub-chunk 1 = rows {2,3} -> (0,0,0.5,0.5)
        assert(close(k_out[0], 0.5f) && close(k_out[1], 0.5f) && close(k_out[2], 0) && close(k_out[3], 0));
        assert(close(k_out[4], 0) && close(k_out[5], 0) && close(k_out[6], 0.5f) && close(k_out[7], 0.5f));
        std::cout << "  Passed" << std::endl;
    }

    // Test 3: "attention" pools with per-head last-key similarity weights
    {
        std::cout << "Test 3: summarize(attention)" << std::endl;
        std::vector<float> k_out, v_out;
        forthllm::summarize(k.data(), n_heads, head_dim, v.data(), n_heads, head_dim,
                            C, S, forthllm::summary_method::attention, k_out, v_out);
        // sub-chunk 0: query = row 1; weights softmax([0, 0.5]) = (0.3775, 0.6225)
        assert(close(k_out[0], 0.37751f) && close(k_out[1], 0.62249f));
        // sub-chunk 1: query = row 3; same weights on rows {2,3}
        assert(close(k_out[6], 0.37751f) && close(k_out[7], 0.62249f));
        std::cout << "  Passed" << std::endl;
    }

    // Test 4: S is clamped to C
    {
        std::cout << "Test 4: summarize clamps S to C" << std::endl;
        std::vector<float> k_out, v_out;
        forthllm::summarize(k.data(), n_heads, head_dim, v.data(), n_heads, head_dim,
                            C, 10, forthllm::summary_method::last, k_out, v_out);
        assert(k_out.size() == C * n_heads * head_dim); // clamped to 4 rows
        std::cout << "  Passed" << std::endl;
    }

    // Test 5: stack LRU eviction drops the least important
    {
        std::cout << "Test 5: stack lru eviction" << std::endl;
        forthllm::config cfg;
        cfg.max_stack_depth = 3;
        cfg.eviction = forthllm::eviction_policy::lru;
        forthllm::stack st(cfg);
        for (size_t i = 0; i < 5; ++i) {
            forthllm::chunk_info info;
            info.chunk_id = i;
            info.n_tokens = 128;
            info.n_summary = 1;
            info.importance = (float) (i + 1); // chunk 0 is least important
            st.push(info);
        }
        assert(st.size() == 5);
        auto drop = st.evict();
        assert(st.size() == 3);
        assert(drop.size() == 2 && drop[0] == 0 && drop[1] == 1); // least important first
        std::cout << "  Dropped chunk_ids: " << drop[0] << ", " << drop[1] << " (expected 0, 1)" << std::endl;
        std::cout << "  Passed" << std::endl;
    }

    // Test 6: stack fixed-window eviction drops the oldest
    {
        std::cout << "Test 6: stack fixed-window eviction" << std::endl;
        forthllm::config cfg;
        cfg.max_stack_depth = 3;
        cfg.eviction = forthllm::eviction_policy::fixed_window;
        forthllm::stack st(cfg);
        for (size_t i = 0; i < 5; ++i) {
            forthllm::chunk_info info;
            info.chunk_id = i;
            info.importance = 1.0f;
            st.push(info);
        }
        auto drop = st.evict();
        assert(st.size() == 3);
        assert(drop.size() == 2 && drop[0] == 0 && drop[1] == 1); // oldest first
        assert(st.at(0).chunk_id == 2 && st.at(2).chunk_id == 4);  // newest kept
        std::cout << "  Passed" << std::endl;
    }

    // Test 7: parse helpers
    {
        std::cout << "Test 7: parse helpers" << std::endl;
        assert(forthllm::summary_method_from_str("Attention") == forthllm::summary_method::attention);
        assert(forthllm::summary_method_from_str("mean") == forthllm::summary_method::mean);
        assert(forthllm::summary_method_from_str("last") == forthllm::summary_method::last);
        assert(throws([] { forthllm::summary_method_from_str("bogus"); }));
        assert(forthllm::eviction_from_str("LRU") == "lru");
        assert(forthllm::eviction_from_str("fixed-window") == "fixed-window");
        assert(throws([] { forthllm::eviction_from_str("bogus"); }));
        assert(forthllm::squeezer_from_str("none") == "none");
        assert(forthllm::squeezer_from_str("jl-bwt") == "jl-bwt");
        assert(throws([] { forthllm::squeezer_from_str("bogus"); }));
        std::cout << "  Passed" << std::endl;
    }

    // Test 8: config validation
    {
        std::cout << "Test 8: config validation" << std::endl;
        forthllm::config ok;
        ok.validate(); // default config is valid
        forthllm::config bad1;
        bad1.summary_dim = 200;
        bad1.chunk_size = 128;
        assert(throws([&] { bad1.validate(); })); // summary_dim > chunk_size
        forthllm::config bad2;
        bad2.squeezer = "jl-bwt";
        bad2.jl_dim = 0;
        assert(throws([&] { bad2.validate(); })); // jl_dim must be > 0 for jl-bwt
        std::cout << "  Passed" << std::endl;
    }

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
