# llama.vm — a virtual memory manager for LLM weights

llama.vm runs large language models that are **much larger than the available
RAM + VRAM** through llama.cpp, with the memory footprint **dictated by the
operator** (a RAM budget and a VRAM window), and a pager — not the OS — deciding
what is resident.

It is a demand-paging system over GGUF model files: a page table maps a flat
virtual address space (one entry per tensor) to pages, which are lazily faulted
into a RAM and/or VRAM tier, cached with an LRU policy, and — for modified data —
swapped to a swap file.

> **Status.** The pager engine is complete and validated standalone (faults /
> LRU / swap / prefetch, both tiers). The `LLAMA_VM_GPU` weight buffer runs a
> model **larger than the VRAM window** through llama.cpp: on a Radeon 890M APU,
> a 28 GB Qwen3.6-27B Q8_0 ran through a 3 GiB window via `llama-server`
> (a completion took 2.4 s; faults=30760 / evicts=30153). The gap llama.cpp
> still leaves open is **runtime weight paging with an operator budget** —
> llama.cpp's `--load-mode mmap` (OS page cache), `--n-gpu-layers`,
> `--cpu-moe`, and `--fit` are all static; none offer a user budget, an
> operator-controlled LRU, or swap. That gap is this project.

## How it works

- **Page table.** Every attached region (each GGUF tensor is one region) is
  split into fixed-size pages (default 4 MiB). Pages are the unit of faulting,
  caching, and eviction.
- **Tiers.** RAM and VRAM are independent caches, each backed by one aligned
  buffer and an arena allocator. A page may be resident in both tiers; the most
  recently modified tier is authoritative, clean pages fall back to the source
  file. VRAM is a ggml backend buffer (`hip` / `cuda` / `vulkan` / `cpu` / `auto`).
- **LRU + smart eviction.** Each tier keeps an LRU list. When space is needed the
  pager evicts the least recently used victim, preferring victims that need no
  swap write-back and deferring dirty last-copy pages to a second pass.
- **Swap.** Optional swap file, one slot per page. Only dirty pages whose last
  copy is being evicted are written back; re-faults prefer the swap slot over
  the source file.
- **Prefetch.** Optional background worker plus explicit
  `prefetch_range` / `prefetch_tensor` / `cancel_prefetch`.
- **MoE expert slices.** `tensor_data_range` faults only the pages covering a
  byte sub-range of a tensor, so a huge MoE expert tensor is paged per-expert
  (per view) rather than pulled in whole.

## Repository layout

```
include/llama_vm/llama_vm.h   public API (config, pager, tensor_info, stats, plan_capacity)
src/arena.h                   per-tier fixed-size allocator (first-fit, 4 KiB aligned)
src/pager-impl.h              pager internals (page table, LRU lists, prefetch, swap bitmap)
src/pager.cpp                 paging engine: faults, eviction, swap, read/write, host regions
src/vram.{h,cpp}              VRAM tier backend (ggml buffer; hip/cuda/vulkan/cpu/auto)
src/gguf.cpp                  GGUF backend: tensors -> regions (single-file + sharded/split)
src/ggml_buffer.{h,cpp}       ggml buffer backend: a host buffer_type backed by a pager
src/ggml_buffer_gpu.cpp       GPU weight buffer (LLAMA_VM_GPU) + wrapper compute backend
src/ggml_backend_shim.cpp     dlopen shim -> libggml-llamavm-cpu.so
src/ggml_backend_gpu_shim.cpp dlopen shim -> libggml-llamavm-gpu.so
src/vm_env.h                  K/M/G size parsing + env-var size helpers (budgets)
tests/                        ctest suite (test_pager, test_gguf, test_gpu_buffer, ...)
tools/llama-vm.cpp            CLI: info / verify / bench / pages / plan / tiered / demo
patches/                      optional llama.cpp patches (see below)
design.md, planning.md        original design and roadmap
```

## Building

Requires a llama.cpp tree with prebuilt ggml libraries.

```sh
# 1. (optional) apply the llama.cpp patches for the full feature set
patches/apply-llama.cpp-patches.sh /path/to/llama.cpp
cmake --build /path/to/llama.cpp/build --target ggml-base ggml -j

# 2. build llama.vm
cmake -S . -B build -DLlamaVM_LLAMA_CPP_DIR=/path/to/llama.cpp
cmake --build build -j

# 3. run the tests (test_gguf/test_llama_integration need a model and auto-skip otherwise)
ctest --test-dir build --output-on-failure
```

### The llama.cpp patches

llama.vm integrates with llama.cpp through the ggml backend registry (dlopen),
so it does not need to live inside the llama.cpp tree. Two small patches are
required for the full feature set — see [`patches/README.md`](patches/README.md):

- **type-66 stub** — lets Kimi-K3 `UD-Q1_0` GGUF files (a quant not yet in
  upstream ggml) parse and page.
- **MoE ids host buffer** — lets the scheduler read a MoE `ids` tensor directly
  from host memory on an integrated GPU.

Without them, core paging over standard GGUF still works.

## CLI

```sh
build/llama-vm info   <model.gguf> [--ram 64M] [--vram 32M] [--page 4M] [--backend cpu]
build/llama-vm verify <model.gguf> [--ram 16M] [--vram 64M] [--tensor NAME]
build/llama-vm bench  <model.gguf> [--ram 32M] [--pattern seq|rand] [--pass N] [--swap FILE]
build/llama-vm plan   <model.gguf> --ram AVAIL_RAM --vram AVAIL_VRAM
build/llama-vm tiered <model.gguf> --ram 1G --vram 1G --backend hip
build/llama-vm demo
```

`--backend cpu` runs the VRAM tier on a host buffer (useful without a GPU).
Size arguments accept `K`/`M`/`G` suffixes.

## Running a model larger than VRAM through llama.cpp

The `LLAMA_VM_GPU` buffer + wrapper backend demand-page weights into a
user-capped VRAM window:

```sh
export LD_LIBRARY_PATH=$PWD/build:/usr/local/lib
export GGML_BACKEND_PATH=$PWD/build/libggml-llamavm-gpu.so
export GGML_CUDA_DISABLE_GRAPHS=1
export LLAMA_VM_GPU=hip          # inner compute backend (auto|hip|cuda|vulkan|cpu)
export LLAMA_VM_VRAM=3G          # the VRAM window: one batch's weights
export LLAMA_VM_GGUF=/path/to/model.gguf   # file mode: fault weights from the model file
# capture mode instead: unset LLAMA_VM_GGUF; set LLAMA_VM_RAM=<budget> and, if the
# budget is below the model size, LLAMA_VM_SWAP=/path/to/swap-file
llama-server -m /path/to/model.gguf --device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' -ngl 99 --port 8123
```

The wrapper splits the graph into batches — the longest run of consecutive ops
whose unique weight tensors can be resident in the window at once — faults those
weights in, re-points `tensor->data` at the window, and computes the batch as one
real graph on the inner backend.

## License

MIT — see [LICENSE](LICENSE).
