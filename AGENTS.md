# Llama Virtual Memory Manager

## Objective (the single objective)

Run models **much larger than available RAM+VRAM** through llama.cpp, with the memory
footprint **dictated by the user** (e.g. a 144 GiB model on a 91 GiB machine with an
operator-chosen RAM + VRAM budget), and the **pager — not the OS — managing what is
resident**: LRU eviction across the RAM and VRAM tiers, file-backed re-faults, swap for
modified data, and prefetch.

Done means: `llama-server -m <model> -ot '.*=LLAMA_VM' <pager budget flags>` runs a real
forward pass with model size >> budget, without OOM, for both model > RAM (CPU) and
model > VRAM (GPU) configurations. Every feature in this repo is judged against this;
anything that only works when the whole model fits in RAM does not advance the objective.

llama.cpp's existing oversize-model support is the baseline, not the goal:
`--load-mode mmap` (OS page cache — no user control, no VRAM tier, no swap),
`--n-gpu-layers` (static per-layer split), `--cpu-moe`/`--n-cpu-moe` (static MoE expert
placement), `--fit` (one-shot load-time fitting). llama.cpp has **no runtime weight
paging**: no user budget, no operator-controlled LRU, no swap. That gap is the project.

## Overview
The Llama Virtual Memory Manager (llama.vm) manages large language models that exceed
available RAM/VRAM capacity. It implements a demand-paging system over GGUF model files:
a page table maps a flat virtual address space (one entry per tensor) to pages, which are
lazily faulted into a RAM and/or VRAM tier, cached with an LRU policy, and — for modified
data — swapped to a swap file. A CLI tool (`llama-vm`) exercises and benchmarks the pager.

Status against the objective: the pager engine is complete and validated standalone
(faults/LRU/swap/prefetch, both tiers — the VRAM tier on a real ROCm/HIP APU — sharded
GGUF up to ~434 GB / 2573 tensors / 11 shards; `test_pager`, `test_gguf`,
`llama-vm bench|tiered`). The capacity governor (`plan_capacity`) sizes tiers from a
model + available memory. The demand-paging GPU weight buffer (`src/ggml_buffer_gpu.cpp`,
buft `LLAMA_VM_GPU`) runs a model **larger than the VRAM window** through llama.cpp,
reliably and fast: with batch compute (the wrapper faults a batch of ops' weights into
the window and computes the batch as one real graph), `llama-server` (HIP) ran a 28 GB
Qwen3.6-27B Q8_0 through a 3 GiB window — a completion took 2.4 s, pager stats
faults=30760 / evicts=30153, and repeated runs complete reproducibly. The fault phase is
now at the machine's memory-bandwidth floor: the pager reads GGUF bytes straight into the
window in one pass on iGPUs (`vram.cpu_accessible`), tier copies and swap → RAM reads are
one-pass, and the background prefetch worker stages the next batch's weight tensors as
contiguous blocks. When the operator's window covers the model, the whole model faults in
once and re-faults drop to zero (28 G window: faults=7683, 12.8 t/s vs 2.3 t/s at the
3 GiB window). The host buffer (`LLAMA_VM`) still must hold the whole model in RAM, but it
now rejects an operator budget (`LLAMA_VM_RAM`) below the model size instead of silently
malloc'ing it. The budgets are exposed as llama-server/llama-cli flags
(`--llama-vm-ram`, `--llama-vm-vram`, `--llama-vm-swap`, `--llama-vm-gguf`), which
`common_params_parse` maps onto the `LLAMA_VM_*` env vars (a flag overrides the env var).
**The objective is not yet met**: model > RAM (CPU) still relies on the OS page cache.
See *Next steps*.

## Repository layout
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
design.md, planning.md        original design and roadmap
```

## Architecture
- **Page table.** Every attached region (each GGUF tensor is one region) is split into
  fixed-size pages (default 4 MiB; the last page of a tensor may be partial). Pages are the
  unit of faulting, caching, and eviction. `page_starts` maps virtual addresses to pages.
- **Tiers.** RAM and VRAM are independent caches, each backed by one aligned buffer and an
  `arena` allocator that hands out per-page slices. A page may be resident in both tiers;
  the most recently modified tier is authoritative, clean pages fall back to the source
  file. VRAM is a ggml backend buffer; `vram_backend = "cpu"` gives a plain host buffer,
  which lets the paging logic be tested without a GPU.
- **LRU + smart eviction.** Each tier keeps an LRU list (MRU at front). When space is
  needed, the pager evicts the least recently used victim, preferring victims that need no
  swap write-back and deferring dirty last-copy pages to a second pass.
- **Eviction unit is the arena block.** A `tensor_data()` call gathers a whole tensor into
  *one* arena allocation shared by its pages (this is what makes the returned pointer
  contiguous). Consequently eviction and the gather free-loop operate on the whole block:
  every resident page in the block is detached, and each dirty last-copy page in the block
  is written back to the swap file first. Pinned pages (or pinned co-tenants in the same
  block) are never evicted. Do not "fix" this to per-page frees — freeing one page's
  offset frees the shared block and silently dangles the co-resident pages (this bug was
  found by corrupted-data test failures).
- **Swap.** Optional swap file, one slot per page. Only dirty pages whose last copy is
  being evicted are written back; re-faults prefer the swap slot over the source file.
- **Safe access.** `read()`/`write()` copy through the pager and are never invalidated by
  eviction; `resolve_page()`/`tensor_data()` return direct pointers that stay valid only
  until the page is evicted by a later operation.
- **Prefetch.** Optional background worker (`cfg.prefetch`) that faults in the next
  sequential page on a fault, plus explicit work: `prefetch_range()` enqueues a page
  range and `prefetch_tensor()` enqueues a whole tensor, staged as one contiguous arena
  block so a later `tensor_data()` finds the pages already contiguous and reuses the
  block without copying (the gather's fast path). `cancel_prefetch()` drops pending work.
  Best effort: resident pages are skipped; when the RAM tier is full the worker makes
  room by evicting pages that are also resident in the other tier (staging duplicates —
  the other copy remains, so no swap is needed); the VRAM window is never evicted for a
  prefetch.
- **Host regions.** `attach_host_region()` maps a region with *no* backing file (data is
  written in via `write()` / the buffer's `set_tensor`). Host pages are never evicted and
  map linearly: a page's data lives at `base + virtual_offset`, so the RAM tier base is a
  plain linear buffer. This is the basis for the ggml buffer backend.
- **Captured regions.** `attach_captured_region()` maps a region with *no* backing file
  whose data is written in via `write()` (eg. weights captured by the GPU buffer at load
  time). Unlike host regions its pages are **paged like file-backed pages**: LRU-evicted,
  and a dirty last-copy page is swapped out — so a captured region larger than the tiers
  needs a swap file. Pages carry a `written` flag; faulting, reading, or `tensor_data`-ing
  an unwritten page throws (the data does not exist anywhere).
- **Sharded GGUF.** `from_gguf` handles split models (the `gguf.split.count` /
  `gguf.split_no` keys). The first split holds the full metadata (arch/name) plus its own
  tensors; every other split lists only the tensors it stores, with offsets relative to its
  own data section. `from_gguf` requires the first split, derives the sibling paths with the
  llama.cpp convention (`<prefix>-NNNNN-of-MMMMM.gguf`), and attaches each shard's tensors
  backed by that shard's file (the pager already supports one file per region, so no pager
  change was needed). Each `tensor_info::file` records the shard, which is how the CLI and
  `test_gguf` verify a tensor against its own shard.
- **ggml buffer backend.** `src/ggml_buffer.cpp` exposes the pager as a ggml
  `buffer_type` ("LLAMA_VM", `is_host`). A buffer's memory is the pager's RAM tier;
  `set_tensor`/`get_tensor` route through `pager::write`/`read` at `base + offset`, and
  `cpy_tensor`/`clear` use direct host access. The buffer owns its pager (freed by
  `free_buffer`). It is registered as a memory-only backend (one device, no compute) and
  is loadable by llama.cpp either by linking `libllama_vm` (`llama_vm_ggml_buft()`) or
  dlopen-ing `libggml-llamavm-cpu.so` (`ggml_backend_init`). The operator can cap the RAM
  tier with `LLAMA_VM_RAM`; a host buffer cannot page, so a budget below the model size is
  rejected (the buffer alloc fails) instead of silently malloc'ing the model.
- **GPU weight buffer (`LLAMA_VM_GPU`).** `src/ggml_buffer_gpu.cpp` exposes a
  *non-host* `buffer_type` whose base is the pager's VRAM tier — a user-capped **window**
  (`LLAMA_VM_VRAM`, required), not the whole model. Two data modes: *capture* (default) —
  the loader's `set_tensor` writes are captured into per-tensor captured regions in a
  user-capped RAM tier (`LLAMA_VM_RAM`, default: the whole model) and swapped out
  (`LLAMA_VM_SWAP`) as it fills — a RAM budget below the model size *requires* a swap
  file; *file* (`LLAMA_VM_GGUF=<model>`) — the pager is built from the GGUF file, the
   loader's writes are ignored, and weights fault straight from the file into the window at
   compute time (no RAM tier or swap needed for the weights). On a host-writable window
   (an iGPU under the HMM, detected via the device type; `LLAMA_VM_VRAM_DIRECT_IO=0`
   overrides) the pager reads the file **straight into the window** — one pass instead of
   file → scratch → device — and the RAM tier defaults to 0; on a discrete GPU the RAM
   tier (default: one window) stages the next batch's weights so a fault is a one-pass
   RAM → VRAM copy. The wrapper's prefetch: after launching a batch's compute it enqueues
   the next batch's weight tensors (the tensors whose pages follow the batch in the
   virtual address space, up to one window's worth) for the pager's background worker,
   which stages them as contiguous blocks into the window as space frees up. A wrapper ggml backend (one
  registry, two devices: a compute device presenting the inner backend and a memory-only
  device for buft name resolution) intercepts `graph_compute` and splits the graph into
  **batches**: the longest run of consecutive ops whose unique weight tensors can be
  resident in the window at once. For each batch it faults those weights in
  (`tensor_data`, pinning each as it goes so a later fault cannot evict an earlier
  member), re-points `tensor->data` at the contiguous window blocks, computes the batch
  as one real graph on the inner backend, and synchronizes once. View/reshape/permute/
  transpose/empty/none nodes are skipped only as single-node batches (the inner backend
  hangs on them as single-node graphs); inside a multi-node batch they are fine. The
  inner compute backend is chosen by `LLAMA_VM_GPU=auto|hip|cuda|vulkan|cpu`. A weight is
  only resident while its batch runs, so the window holds one batch's weights (plus
  nothing else: activations live in the inner buft). `test_gpu_buffer` exercises the
  buffer + wrapper (dedupe, pin boundary, cross-batch reuse) with `vram_backend = "cpu"`;
  the dlopen-able form is `libggml-llamavm-gpu.so`.
- **llama.cpp integration.** Weights are routed to the buffer via
  `llama_model_params.tensor_buft_overrides` (a `{regex, buft}` list; our buft is used
  verbatim, bypassing the `supports_op` gate). Because the host buffer is `is_host`, the
  loader writes GGUF bytes straight into `tensor->data` (= `base + offset`) and CPU
  kernels read `tensor->data` directly — so it must be a contiguous in-RAM region that
  holds the whole model. **Load with `LLAMA_LOAD_MODE_NONE`**: with mmap the loader swaps
  a host `buffer_type` for the stock CPU buft. `test_llama_integration` exercises this
  end-to-end. For a model larger than RAM, use the `LLAMA_VM_GPU` buffer instead: it is
  non-host, so the loader's writes are captured (or ignored in file mode) and the wrapper
  backend faults weights into the VRAM window at compute time.
- **Capacity governor.** `plan_capacity(model_size, avail_ram, avail_vram)` sizes the
   tiers from the model: RAM is filled first, then VRAM with the model's remainder; the
   total is capped at the capacity the model needs (the arena rounds each page's slice up
   to 4 KiB, so that is `model_size` rounded up to 4 KiB). `fits` reports whether the whole
   model can be cached. `llama-vm plan` and the unit tests exercise it.

## Invariants and API constraints (read before changing)
- `tensor_data(name, t)` requires the *entire tensor* to fit in tier `t` at once — the
  capacity check is enforced and throws a descriptive error otherwise.
- `pager` owns a **copy** of its `config`; callers may pass temporaries. Do not store
  references to caller-owned config in the impl (a dangling reference here broke the CLI).
- `arena` allocations are 4 KiB aligned and first-fit; page slices packed inside a single
  allocation (gather blocks) are only ever freed as a whole block (see eviction unit above).
- GGUF tensors are aligned in the file, so the virtual address space does **not** map 1:1
  onto file offsets — always translate via `page_info::file_offset` / `tensor_info`.
- Host regions (`attach_host_region`) have no backing file: they are **never evicted** and
  must fit entirely in the RAM tier capacity (enforced at attach). Their pages map linearly
  (`base + virtual_offset`), so do not route them through the arena like file-backed pages.
- Captured regions (`attach_captured_region`) have no backing file: a page's data exists
  only after `write()` (the `written` flag), and a region larger than the tiers **needs a
  swap file** (a dirty last-copy page with nowhere to go throws).

## Build and test
Requires a llama.cpp tree with prebuilt ggml libraries, passed with
`-DLLlamaVM_LLAMA_CPP_DIR=/path/to/llama.cpp`.

Note: llama.cpp may need a small **load-only stub for ggml type 66**
(`GGML_TYPE_KIMI_Q1_0`, see `patches/ggml-type66-kimi-q1-0-stub.patch`). Kimi-K3's
`UD-Q1_0` quant uses type 66, which is not in upstream ggml; the stub gives it the measured
19/128 bytes/elem so GGUF files using it parse and page correctly (no dequant kernels).
Rebuild the ggml libs after touching it: `cmake --build <llama.cpp>/build --target ggml-base ggml -j`.

```sh
cmake -S . -B build -DLLlamaVM_LLAMA_CPP_DIR=/path/to/llama.cpp
cmake --build build -j

# unit tests (test_gguf needs a model; set LLAMA_VM_TEST_MODEL or it auto-skips)
ctest --test-dir build --output-on-failure

# individual runs
build/test_arena
build/test_pager
build/test_concurrent
build/test_ggml_buffer
build/test_captured
build/test_gpu_buffer             # GPU buffer + wrapper backend, VRAM tier on a host buffer
build/test_llama_integration      # needs libllama + a model (LLAMA_VM_TEST_MODEL)
LLAMA_VM_TEST_VRAM=1 LLAMA_VM_VRAM_BACKEND=cpu build/test_vram
LLAMA_VM_TEST_MODEL=/path/to/model.gguf build/test_gguf

# CLI
build/llama-vm info   <model.gguf> [--ram 64M] [--vram 32M] [--page 4M] [--backend cpu]
build/llama-vm verify <model.gguf> [--ram 16M] [--vram 64M] [--tensor NAME]
build/llama-vm bench  <model.gguf> [--ram 32M] [--pattern seq|rand] [--pass N] [--swap FILE]
build/llama-vm plan   <model.gguf> --ram AVAIL_RAM --vram AVAIL_VRAM   # capacity plan (here --ram/--vram are available memory)
build/llama-vm tiered <model.gguf> --ram 1G --vram 1G --backend hip    # exercise RAM + VRAM + file + tier copies, then verify every page
build/llama-vm demo                     # self-contained paging/swap demo, no model needed
```

Size arguments accept K/M/G suffixes. `--backend cpu` runs the VRAM tier on a host buffer
(useful for testing without a GPU). `tiered` needs both `--ram` and `--vram`; set the
capacities below the model size to force LRU evictions and re-faults from the file.

### Using the VRAM tier on a real GPU (ROCm/HIP)

The pager loads the VRAM backend **at runtime** via ggml's backend registry
(`vram.cpp`), so it does not need to be rebuilt for a GPU — it just needs a
`libggml-hip.so` on disk:

```sh
# 1. build the HIP backend (separate tree so the CPU build stays intact)
cmake -S <llama.cpp> -B <llama.cpp>/build-hip -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build <llama.cpp>/build-hip --target ggml-hip -j

# 2. point the pager at that bin dir
export LLAMA_VM_BACKEND_DIR=<llama.cpp>/build-hip/bin
build/llama-vm tiered <model.gguf> --ram 2G --vram 2G --backend hip
```

Two gotchas (both now handled, see `src/vram.cpp`):
- `ggml_backend_init_by_name` matches a **device** name (`ROCm0`), not the bare backend
  name, and a HIP build registers its device as **ROCm** (GGML_CUDA_NAME). `vram.cpp`
  enumerates devices and matches by prefix, so `--backend hip` finds it.
- The registry loader only accepts a backend .so that exports the generic
  `ggml_backend_init` entry, which requires building with **GGML_BACKEND_DL=ON**. A
  backend built without it (the default) exports the CUDA-named `ggml_backend_cuda_reg`;
  in that case a tiny shim providing `ggml_backend_init`/`ggml_backend_score` that
  delegates to `ggml_backend_cuda_reg` is enough (the loader also requires the file to
  match `libggml-hip-*.so`).

The ggml buffer backend builds into `libllama_vm.so` (used by `test_ggml_buffer`) and as
the standalone dlopen-able `build/libggml-llamavm-cpu.so`. llama.cpp can load the latter
via `GGML_BACKEND_PATH=.../libggml-llamavm-cpu.so` or `ggml_backend_load()`.

### Using the GPU weight buffer with llama.cpp (demand paging, model > VRAM)

The `LLAMA_VM_GPU` buffer + wrapper backend run a model **larger than the VRAM window**
through llama.cpp (see the *GPU weight buffer* architecture entry):

```sh
export LD_LIBRARY_PATH=$PWD/build:/usr/local/lib
export GGML_BACKEND_PATH=$PWD/build/libggml-llamavm-gpu.so
export GGML_CUDA_DISABLE_GRAPHS=1
export LLAMA_VM_GPU=hip          # inner compute backend (auto|hip|cuda|vulkan|cpu)
export LLAMA_VM_VRAM=3G          # the VRAM window: one batch's weights
export LLAMA_VM_GGUF=/path/to/model.gguf    # file mode: fault weights from the model file
# capture mode instead: unset LLAMA_VM_GGUF; set LLAMA_VM_RAM=<budget> and, if the
# budget is below the model size, LLAMA_VM_SWAP=/path/to/swap-file
llama-cli -m /path/to/model.gguf --device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' -ngl 99 \
          --n-predict 1 --prompt "Hi" --temp 0 --single-turn
# or the server (the objective's acceptance path):
llama-server -m /path/to/model.gguf --device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' -ngl 99 \
             --port 8123
curl -s http://127.0.0.1:8123/completion \
     -d '{"prompt":"Hi","n_predict":1,"temperature":0,"stream":false}'
```

**Use `--single-turn` with llama-cli**: without it, after `--n-predict` is satisfied
llama-cli enters the interactive loop; with stdin at EOF (background/nohup runs) it
busy-loops printing the `> ` prompt forever — see *Debugging notes*.

`LLAMA_VM_DEBUG=1` writes a per-batch heartbeat (with pager faults/evictions and
per-batch fault/compute times) to `/tmp/llamavm_debug.log`. Validated on a 28 GB
Qwen3.6-27B Q8_0 on a Radeon 890M APU:
- **3 GiB window (model >> budget):** `llama-cli` completes reproducibly in ~15 s total
  (fault phase ~7.8 s / compute ~1.8 s over 4 graphs; faults=30760 / evicts=30153), and
  `llama-server` completes a request in 2.4 s. The fault phase is at the machine's memory
  bandwidth ceiling (~16 GiB/s): every graph re-reads the whole 28 GB working set from
  the page cache, and on the iGPU the pager reads it straight into the window in one
  pass.
- **28 G window (model fits the budget):** the whole model faults in **once**
  (faults=7683, then zero re-faults — later graphs hit the gather's fast path), and the
  prompt runs at 12.8 t/s vs 2.3 t/s with the 3 GiB window. When the operator's budget
  covers the model, paging disappears; when it does not, the per-graph re-read is the
  floor.

Also validated on a real **MoE** model: Qwen3.8-27B (16.4 GB, `qwen35` arch, fused
`attn_qkv` + per-head views) runs end-to-end through the GPU buffer with a **1 GiB
window** (16x smaller than the model) — `llama-cli --device LLAMA_VM_GPU -ot
'.*=LLAMA_VM_GPU' -ngl 99 --n-predict 8` completes and exits normally. The wrapper's
fault path is exercised for both whole tensors (`attn_qkv`, `token_embd`) and views
(`attn_q/k/v` are slices of `attn_qkv`, faulted via `tensor_data_range`). Two
operational notes from this run:

- **The window must be >= the largest non-view tensor.** `token_embd.weight` (715 MB
  here) is a plain tensor faulted whole via `tensor_data`, so a window below it throws
  `tier capacity is too small for this allocation`. A 512 MiB window fails on this
  model; 1 GiB works. Views (MoE expert slices, per-head QKV) are faulted as slices and
  only need their slice, but any single non-view tensor is a hard floor on the window.
- **Generation is slow at a tiny window on an APU.** At a 1 GiB window the model
  re-reads its whole 16.4 GB working set from the page cache every token (0.1 t/s; the
  window competes with the page cache for the shared HMM memory). This is the expected
  bandwidth-bound re-read, not a correctness issue — an earlier "hang" was just this
  slowness.

## Performance findings (Radeon 890M APU, iGPU)

Measured on a 28 GB Qwen3.6-27B Q8_0 (model on the page cache):
- The APU's "VRAM" (47 GiB) is a carve-out of the 91 GiB system RAM (HMM): the pager's
  window (`hipMalloc`) and the page cache compete for the **same** physical memory.
- The host copy ceiling is ~15.6 GiB/s (warm `memcpy`); `pread` from the page cache is
  ~11–13 GiB/s. The fault phase runs at this ceiling — it is bandwidth-bound, not
  overhead-bound.
- **Window far below the model (3 GiB):** every forward pass re-reads the whole working
  set from the page cache. The fault phase is at the memory-bandwidth floor, and
  background prefetch cannot beat it: the window is full during compute, so staging
  races the wrapper's own faults (each page is read exactly once, by whichever thread
  gets there first).
- **Window covers the model (28 G):** the model faults in once; later passes hit the
  gather's fast path with zero re-faults.
- **The operator's budget is the lever.** When the window covers the model, paging
  disappears; when it does not, the per-pass re-read at the memory-bandwidth floor is the
  cost. On an iGPU, size the window so the model still fits in the page cache
  (`window + model <= available RAM`) — if the model is on a slow disk, spilling it out
  of the page cache makes faults ~100x slower, so a small window that keeps the model
  cached is essential, not a compromise.

## Conventions
- C++17, 4-space indent, snake_case members/functions, no exceptions in hot paths
  (errors throw `std::runtime_error` from API boundaries).
- No comments unless they explain *why*; keep the public API in `llama_vm.h` small and
  documented (see the pointer-validity notes on `resolve_page`/`tensor_data`).
- Keep tests deterministic and self-contained; `test_gguf` auto-skips when no model is
  available, so the suite stays green on machines without big models.

## Next steps
1. **Performance: cut the bytes moved.** The fault traffic is now at the machine's
   memory-bandwidth floor: every graph re-reads the whole working set from the page
   cache (28 GB per 3655-op graph on the 28 GB / 3 GiB fixture; the fault phase runs at
   ~16 GiB/s ≈ the memcpy ceiling), and the next-batch prefetch + one-pass file → window
   read (iGPU HMM) are already in place. Remaining levers: a window that covers the
   model (re-faults drop to zero — validated above), fewer graphs per generation
   (llama.cpp / MTP side), and the file-mode load phase (the loader still reads the whole
   model out of the GGUF and the buffer discards it — a llama.cpp hook to skip
   `set_tensor` for bufts that ignore writes would save one full model read per load).
2. **User-dictated RAM+VRAM budget in the llama.cpp path.** The budgets are exposed as
   llama-server/llama-cli flags (`--llama-vm-ram`, `--llama-vm-vram`, `--llama-vm-swap`,
   `--llama-vm-gguf`), which `common_params_parse` maps onto the `LLAMA_VM_*` env vars (a
   flag overrides the env var; the host buffer rejects a sub-model budget, the GPU buffer
   pages at the budget). Size the tiers via `plan_capacity`.
3. **CPU constraint (why the GPU path is primary).** A *host* buffer must hold the whole
   model in RAM: the CPU reads weights via a direct `tensor->data` host pointer
   (`ggml/src/ggml-cpu/ggml-cpu.c`, e.g. MUL_MAT reads `src1->data`), and the CPU backend
   accepts a buffer type only if it is host or a CPU-extra type
   (`ggml_backend_cpu_device_supports_buft`); every CPU-extra type (repack, AMX, KleidiAI)
   is itself a *host* buffer read via `tensor->data`. There is no fault-in path for CPU
   weight reads, so a model larger than RAM on CPU relies on the OS page cache
   (`--load-mode mmap`) — the baseline llama.cpp already provides. Do not try to "fix"
   this by making the host buffer smaller; it dangles. The pager's CPU value is
   standalone (`test_pager`, `test_gguf`, `llama-vm bench`) and as the RAM tier of the
   GPU path (item 1).
4. **Performance** — `read()` copies page-by-page through `scratch`; large sequential reads
   could bypass the per-page path. Profile `bench --pattern seq` with big pages.
5. **Eviction policy** — LRU only; consider clock/LFU or tensor-affinity (evict whole
   tensors preferentially, which the block-granular eviction already supports cheaply).
6. **Swap hardening** — crash recovery (validity bitmap + `fdatasync`) is implemented;
   still handle disk-full on write-back gracefully (currently throws).
7. **MLX backend** — low priority (see `planning.md`).

## Debugging notes
- **A "stalled" llama-cli is usually the stdin-EOF prompt loop, not the GPU.** After
  `--n-predict` is satisfied, llama-cli (without `--single-turn`) prints the `> ` prompt
  and reads stdin; with stdin at EOF (nohup/background runs) it busy-loops printing the
  prompt at ~40 MB/s (the process shows `R`, the output file grows by GBs, and the
  compute log is frozen at the last op — all of which looks exactly like a GPU hang).
  Check `ls -l /proc/<pid>/fd/0` and the tail of the output file before suspecting the
  pager. This false alarm cost a full investigation once.
- Corrupted-data failures: add temporary overlap checks over the page table (two pages with
  overlapping `[off, off+size)` in one tier) — the corruption that bit us was a freed
  shared arena block being reallocated, not a bad load.
- `stats` (`get_stats()`) exposes faults/evictions/swap counters; `test_gguf` prints them
  on success, which makes regressions (e.g. unexpected swap traffic) visible.
