# Architecture

## Objective

Run models **much larger than available RAM+VRAM** through llama.cpp, with the memory
footprint **dictated by the user** (e.g. a 144 GiB model on a 91 GiB machine with an
operator-chosen RAM + VRAM budget), and the **pager — not the OS — managing what is
resident**: LRU eviction across the RAM and VRAM tiers, file-backed re-faults, swap for
modified data, and prefetch.

## Core Concepts

### Page Table
Every attached region (each GGUF tensor is one region) is split into fixed-size pages (default 4 MiB). Pages are the unit of faulting, caching, and eviction. `page_starts` maps virtual addresses to pages.

### Tiers
RAM and VRAM are independent caches, each backed by one aligned buffer and an `arena` allocator that hands out per-page slices. A page may be resident in both tiers; the most recently modified tier is authoritative, clean pages fall back to the source file. VRAM is a ggml backend buffer (`hip` / `cuda` / `vulkan` / `cpu` / `auto`).

### LRU + Smart Eviction
Each tier keeps an LRU list (MRU at front). When space is needed, the pager evicts the least recently used victim, preferring victims that need no swap write-back and deferring dirty last-copy pages to a second pass.

### Eviction Unit
A `tensor_data()` call gathers a whole tensor into **one** arena allocation shared by its pages (this is what makes the returned pointer contiguous). Consequently eviction and the gather free-loop operate on the whole block: every resident page in the block is detached, and each dirty last-copy page in the block is written back to the swap file first. Pinned pages (or pinned co-tenants in the same block) are never evicted.

### Swap
Optional swap file, one slot per page. Only dirty pages whose last copy is being evicted are written back; re-faults prefer the swap slot over the source file. Swap is crash-recoverable via a validity bitmap and `fdatasync`.

### Safe Access
- `read()`/`write()` copy through the pager and are never invalidated by eviction
- `resolve_page()`/`tensor_data()` return direct pointers that stay valid only until the page is evicted by a later operation

### Prefetch
Optional background worker (`cfg.prefetch`) that faults in the next sequential page on a fault, plus explicit work: `prefetch_range()` enqueues a page range and `prefetch_tensor()` enqueues a whole tensor, staged as one contiguous arena block so a later `tensor_data()` finds the pages already contiguous and reuses the block without copying.

### MoE Expert Slices
`tensor_data_range` faults only the pages covering a byte sub-range of a tensor, so a huge MoE expert tensor is paged per-expert (per view) rather than pulled in whole.

### Host Regions
`attach_host_region()` maps a region with **no** backing file (data is written in via `write()`). Host pages are never evicted and map linearly: a page's data lives at `base + virtual_offset`. This is the basis for the ggml buffer backend.

### Captured Regions
`attach_captured_region()` maps a region with no backing file whose data is written in via `write()` (e.g. weights captured by the GPU buffer at load time). Unlike host regions, its pages are **paged like file-backed pages**: LRU-evicted, and a dirty last-copy page is swapped out — so a captured region larger than the tiers needs a swap file. Pages carry a `written` flag; faulting, reading, or `tensor_data`-ing an unwritten page throws.

### Sharded GGUF
`from_gguf` handles split models (the `gguf.split.count` / `gguf.split_no` keys). The first split holds the full metadata plus its own tensors; every other split lists only the tensors it stores, with offsets relative to its own data section. `from_gguf` requires the first split, derives the sibling paths, and attaches each shard's tensors backed by that shard's file.

## Buffer Backends

### Host Buffer (`LLAMA_VM`)
A ggml `buffer_type` ("LLAMA_VM", `is_host`) backed by a pager. A buffer's memory is the pager's RAM tier; `set_tensor`/`get_tensor` route through `pager::write`/`read` at `base + offset`. The buffer owns its pager (freed by `free_buffer`). Registered as a memory-only backend (one device, no compute).

**Limitation:** Must hold the whole model in RAM. A budget below the model size is rejected instead of silently malloc'ing it.

### GPU Weight Buffer (`LLAMA_VM_GPU`)
A non-host `buffer_type` whose base is the pager's VRAM tier — a user-capped **window** (`LLAMA_VM_VRAM`, required), not the whole model.

**Two data modes:**
- **Capture (default):** The loader's `set_tensor` writes are captured into per-tensor captured regions in a user-capped RAM tier (`LLAMA_VM_RAM`, default: the whole model) and swapped out (`LLAMA_VM_SWAP`) as it fills.
- **File (`LLAMA_VM_GGUF=<model>`):** The pager is built from the GGUF file, the loader's writes are ignored, and weights fault straight from the file into the window at compute time.

**Wrapper backend:** Intercepts `graph_compute` and splits the graph into **batches**: the longest run of consecutive ops whose unique weight tensors can be resident in the window at once. For each batch, it faults those weights in (`tensor_data`, pinning each), re-points `tensor->data` at the contiguous window blocks, computes the batch as one real graph on the inner backend, and synchronizes once.

**Inner compute backend:** Chosen by `LLAMA_VM_GPU=auto|hip|cuda|vulkan|cpu`, optionally with a device index within the backend (`cuda1` = the second CUDA device, registration order).

## llama.cpp Integration

Weights are routed to the buffer via `llama_model_params.tensor_buft_overrides` (a `{regex, buft}` list; our buft is used verbatim, bypassing the `supports_op` gate).

**Important:** Load with `LLAMA_LOAD_MODE_NONE`. With mmap, the loader swaps a host `buffer_type` for the stock CPU buft, which would silently disable demand paging.

## Capacity Governor

`plan_capacity(model_size, avail_ram, avail_vram)` sizes the tiers from the model: RAM is filled first, then VRAM with the model's remainder; the total is capped at the capacity the model needs (the arena rounds each page's slice up to 4 KiB). `fits` reports whether the whole model can be cached.

## Invariants and API Constraints

- `tensor_data(name, t)` requires the **entire tensor** to fit in tier `t` at once
- `pager` owns a **copy** of its `config`; callers may pass temporaries
- `arena` allocations are 4 KiB aligned and first-fit; page slices packed inside a single allocation are only ever freed as a whole block
- GGUF tensors are aligned in the file, so the virtual address space does **not** map 1:1 onto file offsets — always translate via `page_info::file_offset` / `tensor_info`
- Host regions have no backing file: never evicted, must fit in RAM tier capacity
- Captured regions have no backing file: a page's data exists only after `write()`; a region larger than the tiers needs a swap file

## Repository Layout

```
include/llama_vm/llama_vm.h   public API (config, pager, tensor_info, page_info, stats,
                                attach_host_region, tier_base, plan_capacity)
src/arena.h                   per-tier fixed-size allocator (first-fit, 4 KiB aligned)
src/pager-impl.h              pager internals (page table, LRU lists, prefetch, swap bitmap)
src/pager.cpp                 paging engine: faults, eviction, swap (crash-recoverable),
                               read/write, host regions
src/vram.{h,cpp}              VRAM tier backend (ggml buffer; hip/cuda/vulkan/cpu/auto)
src/gguf.cpp                  GGUF backend: tensors -> regions (single-file + sharded/split)
src/ggml_buffer.{h,cpp}       ggml buffer backend: a host buffer_type backed by a pager
src/ggml_buffer_gpu.cpp       GPU weight buffer (LLAMA_VM_GPU) + wrapper compute backend
src/ggml_backend_shim.cpp     dlopen shim -> libggml-llamavm-cpu.so (ggml_backend_init)
src/ggml_backend_gpu_shim.cpp dlopen shim -> libggml-llamavm-gpu.so (wrapper backend)
src/vm_env.h                  K/M/G size parsing + env-var size helpers (budgets)
tests/                        test_arena, test_pager, test_concurrent, test_ggml_buffer,
                                 test_captured, test_gpu_buffer, test_llama_integration,
                                 test_gguf, test_vram (ctest)
tools/llama-vm.cpp            CLI: info / verify / bench / pages / plan / tiered / demo
include/forthllm/, src/forthllm.cpp, src/squeezer.cpp, src/bwt.cpp, src/fm_index.cpp,
                                src/jl_projector.cpp   ForthLLM model-agnostic math
                                (summarize/stack/squeezer/BWT/JL — a faithful port of
                                the tested llama.cpp engine, no ggml; the KV-cache
                                surgery + C API live in llama.cpp, see below)
patches/                      llama.cpp patches
docs/                         documentation
```