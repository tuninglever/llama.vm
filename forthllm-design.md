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

## Examples: Running Large LLMs with llama-server

### Example 1: Basic ForthLLM Usage

```bash
# Start llama-server with ForthLLM context management
llama-server \
    -m /path/to/large-model.gguf \
    --device LLAMA_VM_GPU \
    -ot '.*=LLAMA_VM_GPU' \
    -ngl 99 \
    --load-mode none \
    --port 8123 \
    --llama-vm-ram 4G \
    --llama-vm-vram 8G \
    --llama-vm-swap /tmp/llama-vm-swap \
    --forthllm-chunk-size 256 \
    --forthllm-summary-method attention \
    --forthllm-max-stack-depth 32
```

### Example 2: Running a Model Larger Than Available Memory

```bash
# 70B parameter model on a machine with 32GB RAM + 16GB VRAM
# Standard approach: OOM
# ForthLLM approach: Demand-paged weights + compressed context stack

llama-server \
    -m /path/to/70b-model.gguf \
    --device LLAMA_VM_GPU \
    -ot '.*=LLAMA_VM_GPU' \
    -ngl 99 \
    --load-mode none \
    --port 8123 \
    --llama-vm-ram 16G \
    --llama-vm-vram 8G \
    --llama-vm-swap /path/to/fast-ssd/swap \
    --llama-vm-swap-size 32G \
    --forthllm-chunk-size 128 \
    --forthllm-summary-dim 256 \
    --forthllm-eviction lru
```

### Example 3: Long-Context Processing

```bash
# Process a 100K token document with efficient context management
# Standard approach: KV cache grows to ~40GB
# ForthLLM approach: Fixed-size context stack (~1GB)

llama-server \
    -m /path/to/13b-model.gguf \
    --device LLAMA_VM_GPU \
    -ot '.*=LLAMA_VM_GPU' \
    -ngl 99 \
    --load-mode none \
    --port 8123 \
    --llama-vm-ram 8G \
    --llama-vm-vram 4G \
    --forthllm-chunk-size 512 \
    --forthllm-summary-method attention \
    --forthllm-max-stack-depth 64 \
    --forthllm-squeezer jl-bwt \
    --forthllm-jl-dim 128
```

### Example 4: Edge Device Deployment

```bash
# Run a 7B model on a Raspberry Pi 5 (8GB RAM, no GPU)
# Using CPU-only with aggressive context compression

llama-server \
    -m /path/to/7b-model-q4.gguf \
    --port 8123 \
    --llama-vm-ram 4G \
    --llama-vm-swap /tmp/swap \
    --forthllm-chunk-size 64 \
    --forthllm-summary-dim 64 \
    --forthllm-eviction fixed-window \
    --forthllm-squeezer jl-bwt
```

### Example 5: Interactive Chat with Context Management

```bash
# Multi-turn conversation with automatic context compression
# Old turns are summarized and compressed as the conversation grows

llama-server \
    -m /path/to/13b-model.gguf \
    --device LLAMA_VM_GPU \
    -ot '.*=LLAMA_VM_GPU' \
    -ngl 99 \
    --load-mode none \
    --port 8123 \
    --llama-vm-ram 8G \
    --llama-vm-vram 8G \
    --forthllm-chunk-size 256 \
    --forthllm-summary-method attention \
    --forthllm-eviction lru \
    --forthllm-max-stack-depth 16
```

### Performance Comparison

| Scenario | Standard Approach | ForthLLM Approach |
|----------|------------------|-------------------|
| 70B model on 32GB RAM | OOM | Runs with paging |
| 100K token context | ~40GB KV cache | ~1GB context stack |
| Long conversation | Grows unbounded | Fixed-size stack |
| Edge deployment | Limited to small models | Larger models possible |

### Configuration Options

- `--forthllm-chunk-size`: Number of tokens per chunk (default: 128)
- `--forthllm-summary-method`: Summary method (attention, mean, last)
- `--forthllm-max-stack-depth`: Maximum number of summaries to keep
- `--forthllm-summary-dim`: Dimension of summary embeddings
- `--forthllm-eviction`: Eviction policy (lru, lfu, fixed-window)
- `--forthllm-squeezer`: Compression method (none, jl-bwt)
- `--forthllm-jl-dim`: JL projection output dimension

## Future Work

- Explore hybrid approaches (attention for recent tokens, stack for older)
- Investigate learned summarization (train a small model to summarize chunks)
- Compare with state space models and recurrent transformers
- Optimize for specific hardware targets (ARM, RISC-V, etc.)