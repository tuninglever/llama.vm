# llama.vm — a virtual memory manager for LLM weights

llama.vm runs large language models that are **much larger than the available RAM + VRAM** through llama.cpp, with the memory footprint **dictated by the operator** (a RAM budget and a VRAM window), and a pager — not the OS — deciding what is resident.

It is a demand-paging system over GGUF model files: a page table maps a flat virtual address space (one entry per tensor) to pages, which are lazily faulted into a RAM and/or VRAM tier, cached with an LRU policy, and — for modified data — swapped to a swap file.

> **Status.** The pager engine is complete and validated standalone (faults / LRU / swap / prefetch, both tiers). The `LLAMA_VM_GPU` weight buffer runs a model **larger than the VRAM window** through llama.cpp: on a Radeon 890M APU, a 28 GB Qwen3.6-27B Q8_0 ran through a 3 GiB window via `llama-server` (a completion took 2.4 s; faults=30760 / evicts=30153). The gap llama.cpp still leaves open is **runtime weight paging with an operator budget** — llama.cpp's `--load-mode mmap` (OS page cache), `--n-gpu-layers`, `--cpu-moe`, and `--fit` are all static; none offer a user budget, an operator-controlled LRU, or swap. That gap is this project.

## Quick Start

This guide takes you from cloning the repo to running a model through llama-server.

### 1. Clone the repos

```bash
git clone https://github.com/ggml-org/llama.cpp.git
git clone https://github.com/ggml-org/llama.vm.git
cd llama.vm
```

### 2. Apply the llama.cpp patches

```bash
# Apply the type-66 stub + ForthLLM feature patch
patches/apply-llama.cpp-patches.sh ../llama.cpp
```

### 3. Build llama.cpp with your platform's backend

**ROCm (AMD GPUs):**
```bash
cd ../llama.cpp
cmake -S . -B build-hip -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip --target ggml-hip -j
```

**CUDA (NVIDIA GPUs):**
```bash
cd ../llama.cpp
cmake -S . -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --target ggml-cuda -j
```

**MLX (Apple Silicon):**
```bash
brew install mlx
cd ../llama.cpp
cmake -S . -B build-mlx -DGGML_MLX=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/homebrew:/opt/homebrew/opt/libomp \
      -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp" \
      -DOpenMP_CXX_LIBRARIES="/opt/homebrew/opt/libomp/lib/libomp.dylib"
cmake --build build-mlx --target ggml -j
```

### 4. Build llama.vm

```bash
cd ../llama.vm
cmake -S . -B build -DLlamaVM_LLAMA_CPP_DIR=../llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 5. Run a model through llama-server

**ROCm:**
```bash
export LD_LIBRARY_PATH=$PWD/build:/path/to/llama.cpp/build-hip/bin:/usr/local/lib
export GGML_BACKEND_PATH=$PWD/build/libggml-llamavm-gpu.so
export LLAMA_VM_GPU=hip
export LLAMA_VM_VRAM=3G
export LLAMA_VM_GGUF=/path/to/model.gguf

llama-server -m /path/to/model.gguf --device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' \
             -ngl 99 --load-mode none --port 8123
```

**CUDA:**
```bash
export LD_LIBRARY_PATH=$PWD/build:/path/to/llama.cpp/build-cuda/bin:/usr/local/lib
export GGML_BACKEND_PATH=$PWD/build/libggml-llamavm-gpu.so
export GGML_CUDA_DISABLE_GRAPHS=1
export LLAMA_VM_GPU=cuda
export LLAMA_VM_VRAM=3G
export LLAMA_VM_GGUF=/path/to/model.gguf

llama-server -m /path/to/model.gguf --device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' \
             -ngl 99 --load-mode none --port 8123
```

**MLX:**
```bash
export LD_LIBRARY_PATH=$PWD/build:/path/to/llama.cpp/build-mlx/bin:/opt/homebrew/lib
export GGML_BACKEND_PATH=$PWD/build/libggml-llamavm-gpu.dylib
export LLAMA_VM_GPU=mlx
export LLAMA_VM_VRAM=3G
export LLAMA_VM_GGUF=/path/to/model.gguf

llama-server -m /path/to/model.gguf --device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' \
             -ngl 99 --load-mode none --port 8123
```

### 6. Test the server

```bash
curl -s http://127.0.0.1:8123/completion \
     -d '{"prompt":"Hi","n_predict":1,"temperature":0,"stream":false}'
```

## How it works

- **Page table.** Every attached region (each GGUF tensor is one region) is split into fixed-size pages (default 4 MiB). Pages are the unit of faulting, caching, and eviction.
- **Tiers.** RAM and VRAM are independent caches, each backed by one aligned buffer and an arena allocator. A page may be resident in both tiers; the most recently modified tier is authoritative, clean pages fall back to the source file.
- **LRU + smart eviction.** Each tier keeps an LRU list. When space is needed the pager evicts the least recently used victim, preferring victims that need no swap write-back and deferring dirty last-copy pages to a second pass.
- **Swap.** Optional swap file, one slot per page. Only dirty pages whose last copy is being evicted are written back; re-faults prefer the swap slot over the source file.
- **Prefetch.** Optional background worker plus explicit `prefetch_range` / `prefetch_tensor` / `cancel_prefetch`.
- **MoE expert slices.** `tensor_data_range` faults only the pages covering a byte sub-range of a tensor, so a huge MoE expert tensor is paged per-expert (per view) rather than pulled in whole.

## Environment Variables

| Variable | Meaning |
|---|---|
| `LLAMA_VM_VRAM` | The VRAM window (required; buffer creation fails without it) |
| `LLAMA_VM_GGUF` | Set → **file mode**: pager built from the GGUF, loader writes ignored, weights fault from the file at compute time |
| `LLAMA_VM_RAM` | File mode: staging tier for the background prefetch (discrete GPU; default = one window; 0 on host-writable iGPU windows). Capture mode: the capture tier (default = whole model; a budget below the model size **requires** `LLAMA_VM_SWAP`) |
| `LLAMA_VM_SWAP` | Swap file (capture mode only) |
| `LLAMA_VM_GPU` | Inner compute backend: `auto\|hip\|cuda\|vulkan\|cpu` + optional device index (`cuda1` = second CUDA device) |
| `LLAMA_VM_BACKEND_DIR` | Directory containing the inner backend's .so (if not in the llama.cpp build) |

## Key Considerations

- **Window sizing:** `LLAMA_VM_VRAM` must be at least the largest *plain* tensor of the model (e.g. `token_embd.weight` / `output.weight`); a window below it throws at the first fault. MoE expert tensors are paged as per-expert slices, so the window does not need to hold a whole expert tensor.
- **Split models:** Pass the **first shard** to `-m` (and `LLAMA_VM_GGUF`); the pager derives the sibling shards itself.
- **Load mode:** Use `--load-mode none`: with mmap (the `auto` default) the loader swaps any buft that is a device's host buffer type for the stock CPU buft, which would silently disable demand paging.
- **Device selection:** `--device` must be the wrapper device name `LLAMA_VM_GPU` (it appears in `llama-server --list-devices`). The *physical* GPU is chosen by the `LLAMA_VM_GPU` env var.

## Documentation

- [Architecture](docs/architecture.md) — detailed architecture, invariants, and API constraints
- [Building](BUILD.md) — platform-specific build instructions
- [Testing](docs/testing.md) — unit tests, CLI tools, and model testing
- [ForthLLM](docs/forthllm.md) — stack-like context processing for LLMs
- [Debugging](docs/debugging.md) — debugging notes and known issues
- [Patches](patches/README.md) — llama.cpp patches

## License

MIT — see [LICENSE](LICENSE).