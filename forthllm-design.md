# ForthLLM: Stack-like Context Processing for LLMs

## Origin of the Idea

The idea emerged from considering how Forth's stack-based computation model could apply to LLM prompt processing. In Forth:

- Operations pop operands from the stack
- Operations push results back onto the stack
- Intermediate values are consumed and discarded as needed
- The entire computation is driven by what's currently on the stack

**Key insight**: Do LLMs need all values in matrices throughout the entire computation? Or can we process in chunks, replace intermediate values with compressed representations, and only keep what's needed?

## Why This Matters

- Models are getting larger, but so are the contexts they need to process
- KV cache grows linearly with sequence length, creating a memory bottleneck
- MoE addresses compute scaling but not context/memory scaling
- Long conversations, document processing, and codebase understanding need efficient context management

## Architecture: Stack-like Context Processing

### Core Concept

Instead of keeping all tokens in the KV cache, process in chunks and replace each chunk with a compressed representation:

```forth
"Hello, how are you?"
  chunk-process -> [summary-of-greeting]
"Today I went to the store."
  chunk-process -> [summary-of-day]
[summary-of-greeting] [summary-of-day]
  combine -> [full-context-summary]
"What did I do today?"
  answer-from -> "You went to the store."
```

### Implementation Approach

Build on top of llama.cpp as a library (not a server) with a custom context management layer:

```cpp
class ForthContextManager {
    llama_context* ctx;           // llama.cpp as a library
    std::vector<Token> stack;     // Stack-like state
    std::vector<ChunkSummary> summaries;  // Compressed context
    // ... eviction, summarization, etc.
};
```

### Key Techniques

1. **Chunk processing**: Process input in fixed-size or semantic chunks
2. **Summary replacement**: Replace processed chunks with compressed representations
3. **Stack discipline**: Maintain a fixed-size hidden state that gets updated
4. **Eviction policies**: LRU on attention relevance, or fixed-size window

### Relationship to llama.vm

| | llama.vm (weights) | ForthLLM (context) |
|---|---|---|
| **What's managed** | Model weights (static, large) | KV cache (grows with sequence) |
| **Access pattern** | Read during forward pass | Read/write during generation |
| **Problem solved** | Model doesn't fit in memory | Long context doesn't fit in memory |
| **Eviction policy** | LRU on weight tensors | LRU on attention relevance |

These are **complementary** optimizations. Together, they enable long conversations with large models on memory-constrained hardware.

## Research Context

This approach aligns with several active research directions:

- **State space models** (Mamba, RWKV): Fixed-size state, no attention matrix
- **Recurrent transformers**: Transformer layers + recurrent state update
- **KV cache compression**: Quantize, evict, or summarize cached values
- **Chunked attention**: Process in windows with inter-chunk communication

## Implementation Plan

### Phase 1: Proof of Concept

- Implement basic chunk processing and summary replacement
- Test with simple prompts and short conversations
- Measure memory savings vs. accuracy trade-off

### Phase 2: Integration with llama.vm

- Combine weight paging (llama.vm) with context paging (ForthLLM)
- Test with long conversations and large models
- Optimize eviction policies

### Phase 3: Production Readiness

- Benchmark performance
- Optimize for edge devices
- Add configurable strategies (window size, summary method, etc.)

## Trade-offs

| | Attention (current) | Stack-like (ForthLLM) |
|---|---|---|
| **Memory** | O(n) with sequence length | O(1) fixed-size state |
| **Latency** | O(n²) attention computation | O(1) state update |
| **Expressiveness** | Full global attention | Limited context window |
| **Quality** | Best for long-range dependencies | Good for local dependencies |

The trade-off is between memory/latency efficiency and expressiveness. For many practical tasks (conversation, code completion, summarization), stack-like processing may be sufficient.

## Future Work

- Explore hybrid approaches (attention for recent tokens, stack for older)
- Investigate learned summarization (train a small model to summarize chunks)
- Compare with state space models and recurrent transformers
- Optimize for specific hardware targets (ARM, RISC-V, etc.)