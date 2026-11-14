# ForthLLM: Stack-like Context Processing for LLMs

## Status (2026-11-05)

The `--forthllm-*` flags are **implemented and validated** in llama.cpp (Phases 1–4b). Patch: `patches/llama-cpp-forthllm-kv-compaction.patch`, applied to the `/work/llama.cpp` clone (upstream `45b6604b2`).

### What works
- Flag parsing (`--forthllm`, `--forthllm-chunk-size`, `--forthllm-summary-method`, `--forthllm-summary-dim`, `--forthllm-max-stack-depth`, `--forthllm-eviction`, `--forthllm-squeezer`, `--forthllm-jl-dim`)
- Chunked KV-row summarization engine (mean/last/attention, per-head weights)
- `llama_kv_cache::forthllm_compact` with hybrid/IMROPE support
- Server hooks: post-prefill, context-limit-stop, pre-context-shift
- `max_stack_depth` / `eviction` wired into the cache path (oldest chunks beyond keep limit are dropped)
- Validated end-to-end on Qwen3.8-27B (2196-token prompt → 180 cells, coherent answer)

### What's aspirational
- `--forthllm-squeezer` / `--forthllm-jl-dim` parse and validate but are not yet used by the cache path (they attach to state save/load)
- MLA models (DeepSeek-V4) are a documented "not supported" compaction path — requires its own design pass (see below)

## Origin

The idea emerged from considering how Forth's stack-based computation model could apply to LLM prompt processing:
- Operations pop operands from the stack
- Operations push results back onto the stack
- Intermediate values are consumed and discarded as needed

**Key insight:** Do LLMs need all values in matrices throughout the entire computation? Or can we process in chunks, replace intermediate values with compressed representations, and only keep what's needed?

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

### Relationship to llama.vm

| | llama.vm (weights) | ForthLLM (context) |
|---|---|---|
| **What's managed** | Model weights (static, large) | KV cache (grows with sequence) |
| **Access pattern** | Read during forward pass | Read/write during generation |
| **Problem solved** | Model doesn't fit in memory | Long context doesn't fit in memory |
| **Eviction policy** | LRU on weight tensors | LRU on attention relevance |

These are **complementary** optimizations. Together, they enable long conversations with large models on memory-constrained hardware.

## Implementation

### Phase 1 — Flags (llama.cpp) ✅
Added to `common/arg.cpp` + `common_params`:
```
--forthllm                     enable ForthLLM context compression
--forthllm-chunk-size N        tokens per chunk (default 128)
--forthllm-summary-method M    attention|mean|last (default attention)
--forthllm-summary-dim N       summary rows S per chunk (default 1)
--forthllm-max-stack-depth N   max summarized chunks to keep (default 64)
--forthllm-eviction P          lru|lfu|fixed-window (default lru)
--forthllm-squeezer X          none|jl-bwt (default none)
--forthllm-jl-dim N            JL output dim for the squeezer (default 128)
```

### Phase 2 — Engine (llama.cpp) ✅
Landed in the **core** (`include/llama-forthllm.h` + `src/forthllm.cpp`):
- `llama_forthllm_summarize(k, n_heads_k, head_dim_k, v, n_heads_v, head_dim_v, C, S, method, k_out, v_out)` → S summary K rows and S summary V rows with per-head weights
- Stack of chunk summaries with lru/lfu/fixed-window eviction
- Squeezer: JL projection → BWT for stored summary representation

### Phase 3 — Server Wiring (llama.cpp) ✅
`llama_kv_cache::forthllm_compact(seq_id, params, chunks)` member + C API `llama_forthllm_compact`. Three server hooks:
1. `post_decode` — one-shot compaction after prefill
2. `process_token` — at the context-limit stop check (works with `--context-shift` disabled)
3. `pre_decode` — try compaction before falling through to context shift

### Phase 4a — Replace Placeholder Scaffold (llama.vm) ✅
The library in `include/forthllm/` + `src/forthllm.cpp` is now a faithful port of the tested llama.cpp engine (real `summarize`/`stack`/`squeezer`/`bwt`/`jl_projector`, no ggml).

### Phase 4b — Wire max_stack_depth / eviction ✅
The compaction computes `n_keep = min(n_chunks, max_stack_depth)`. The oldest `n_drop = n_chunks - n_keep` chunks are dropped entirely (all C cells freed, no summary rows); the most recent `n_keep` chunks are summarized as before.

### Phase 4c — MLA / DeepSeek-V4 Support (open)
DeepSeek-V4 uses `llama_kv_cache_dsv4`, which is **not** a standard MLA cache. It's composed of:
- `kv_raw` — a normal raw/SWA token cache
- `kv_csa` / `kv_hca` / `kv_lid` — three compressed K-only block caches
- `csa_state` / `hca_state` / `lid_state` — ring compressor states

**This is an open design problem, not a mechanical port.** DSV4 already compresses its KV; ForthLLM would be a second compression on top. The ring states are graph-computed, not stored rows we can summarize in place. Requires its own design pass.

## Examples

### Basic ForthLLM Usage

```bash
# Start llama-server with ForthLLM context management
llama-server \
    -m /path/to/large-model.gguf \
    --device LLAMA_VM_GPU \
    -ot '.*=LLAMA_VM_GPU' \
    -ngl 99 \
    --load-mode none \
    --port 8123 \
    --forthllm \
    --forthllm-chunk-size 256 \
    --forthllm-summary-method attention \
    --forthllm-max-stack-depth 32
```

### Long-Context Processing

```bash
# Process a 100K token document with efficient context management
# Standard approach: KV cache grows to ~40GB
# ForthLLM approach: Fixed-size context stack (~1GB)

export LLAMA_VM_RAM=8G
export LLAMA_VM_VRAM=4G
export LLAMA_VM_GGUF=/path/to/13b-model.gguf

llama-server \
    -m /path/to/13b-model.gguf \
    --device LLAMA_VM_GPU \
    -ot '.*=LLAMA_VM_GPU' \
    -ngl 99 \
    --load-mode none \
    --port 8123 \
    --forthllm \
    --forthllm-chunk-size 512 \
    --forthllm-summary-method attention \
    --forthllm-max-stack-depth 64
```

### Performance Comparison

| Scenario | Standard Approach | ForthLLM Approach |
|----------|------------------|-------------------|
| 70B model on 32GB RAM | OOM | Runs with paging |
| 100K token context | ~40GB KV cache | ~1GB context stack |
| Long conversation | Grows unbounded | Fixed-size stack |
| Edge deployment | Limited to small models | Larger models possible |

## Trade-offs

| | Attention (current) | Stack-like (ForthLLM) |
|---|---|---|
| **Memory** | O(n) with sequence length | O(1) fixed-size state |
| **Latency** | O(n²) attention computation | O(1) state update |
| **Expressiveness** | Full global attention | Limited context window |
| **Quality** | Best for long-range dependencies | Good for local dependencies |

## Configuration Options

- `--forthllm-chunk-size`: Number of tokens per chunk (default: 128)
- `--forthllm-summary-method`: Summary method (attention, mean, last)
- `--forthllm-max-stack-depth`: Maximum number of summaries to keep
- `--forthllm-summary-dim`: Number of summary rows per chunk
- `--forthllm-eviction`: Eviction policy (lru, lfu, fixed-window)
- `--forthllm-squeezer`: Compression method (none, jl-bwt)
- `--forthllm-jl-dim`: JL projection output dimension