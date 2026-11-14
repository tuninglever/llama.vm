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

**Documentation:** [README](README.md) (quick start), [BUILD](BUILD.md) (platform builds),
[Architecture](docs/architecture.md), [Testing](docs/testing.md), [ForthLLM](docs/forthllm.md),
[Debugging](docs/debugging.md).

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
malloc'ing it. The budgets are read by the buffers from the `LLAMA_VM_*` env vars
(`LLAMA_VM_RAM`, `LLAMA_VM_VRAM`, `LLAMA_VM_SWAP`, `LLAMA_VM_GGUF`). Note: the
`--llama-vm-*` CLI flags are **not** present in any llama.cpp build (upstream or
patched) — passing them fails with `error: invalid argument`; set the env vars
instead. The `/work/llama.cpp` clone carries the llama.vm patches (we do not own
llama.cpp; the patches live in `patches/` and are applied with
`patches/apply-llama.cpp-patches.sh`, which applies the ggml type-66 stub and the
ForthLLM feature patch (`--forthllm-*` flags + KV-cache compaction) by default,
`--pager-only` for the stub only), applied and rebuilt 2026-11-05 against upstream
`45b6604b2` (re-validated E2E on Qwen3.8-27B). If the clone is updated,
re-run the apply script. See
`llama-server-forthllm-investigation.md`.
**The objective is not yet met**: model > RAM (CPU) still relies on the OS page cache.
See *Next steps*.

## Repository layout

See [Architecture](docs/architecture.md) for the complete repository layout and detailed
component descriptions.

Key directories:
- `src/` — pager engine, buffer backends, VRAM tier, GGUF backend
- `include/` — public API (`llama_vm.h`) and ForthLLM library
- `tests/` — ctest suite
- `tools/llama-vm.cpp` — CLI tool
- `patches/` — llama.cpp patches
- `docs/` — documentation (architecture, testing, ForthLLM, debugging)

## Architecture

See [docs/architecture.md](docs/architecture.md) for the complete architecture, including
page table, tiers, LRU eviction, swap, prefetch, host/captured regions, sharded GGUF,
buffer backends, llama.cpp integration, capacity governor, and API constraints.

Key concepts:
- **Page table** maps virtual addresses (one entry per tensor) to pages
- **Tiers** are RAM and VRAM caches with LRU eviction
- **Swap** handles dirty page write-back
- **Host buffer** (`LLAMA_VM`) must hold the whole model in RAM
- **GPU weight buffer** (`LLAMA_VM_GPU`) demand-pages weights into a user-capped VRAM window

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

See [BUILD.md](BUILD.md) for platform-specific build instructions (ROCm, CUDA, MLX) and
[docs/testing.md](docs/testing.md) for the full test suite and CLI tools.

Quick build:
```sh
# Apply patches
patches/apply-llama.cpp-patches.sh /path/to/llama.cpp

# Build llama.cpp (platform-specific)
cd /path/to/llama.cpp && cmake -S . -B build-<platform> -DGGML_<BACKEND>=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-<platform> --target ggml-<backend> -j

# Build llama.vm
cd /path/to/llama.vm
cmake -S . -B build -DLlamaVM_LLAMA_CPP_DIR=/path/to/llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run tests
ctest --test-dir build --output-on-failure
```

## Performance findings

On a 28 GB Qwen3.6-27B Q8_0 on a Radeon 890M APU (iGPU):
- **Window far below the model (3 GiB):** every forward pass re-reads the whole working set from the page cache. The fault phase is at the memory-bandwidth floor (~16 GiB/s).
- **Window covers the model (28 G):** the model faults in once; later passes hit the gather's fast path with zero re-faults.
- **The operator's budget is the lever.** When the window covers the model, paging disappears; when it does not, the per-pass re-read at the memory-bandwidth floor is the cost.

## Conventions
- C++17, 4-space indent, snake_case members/functions
- No exceptions in hot paths; errors throw `std::runtime_error` from API boundaries
- No comments unless they explain *why*
- Keep tests deterministic and self-contained

## Next steps

1. **Performance:** Cut the bytes moved — the fault traffic is at the memory-bandwidth floor. Remaining levers: window that covers the model, fewer graphs per generation, file-mode load phase optimization.
2. **User-dictated budget in llama.cpp path:** The `--llama-vm-*` CLI flags are not in the current llama.cpp build — either add them to `common_params_parse` or keep using env vars.
3. **CPU constraint:** A host buffer must hold the whole model in RAM. The pager's CPU value is standalone and as the RAM tier of the GPU path.
4. **Performance:** `read()` copies page-by-page through `scratch`; large sequential reads could bypass the per-page path.
5. **Eviction policy:** LRU only; consider clock/LFU or tensor-affinity.
6. **Swap hardening:** Handle disk-full on write-back gracefully.
7. **ForthLLM Phase 4c:** Add MLA (DeepSeek-V4) support — requires its own design pass.

See [docs/forthllm.md](docs/forthllm.md) for the full ForthLLM status and design.

## Debugging notes

See [docs/debugging.md](docs/debugging.md) for detailed debugging notes, including:
- "Stalled" llama-cli is usually the stdin-EOF prompt loop, not the GPU
- Corrupted-data failures: add temporary overlap checks
- Swap counter monitoring
- After `git pull`, the ForthLLM patch can appear to "not fit"
- CUDA device indices are not nvidia-smi indices
- Swap bugs found and fixed
- Eviction policy: LRU only
