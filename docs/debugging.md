# Debugging Notes

## A "stalled" llama-cli is usually the stdin-EOF prompt loop, not the GPU

After `--n-predict` is satisfied, llama-cli (without `--single-turn`) prints the `> ` prompt and reads stdin; with stdin at EOF (nohup/background runs) it busy-loops printing the prompt at ~40 MB/s (the process shows `R`, the output file grows by GBs, and the compute log is frozen at the last op — all of which looks exactly like a GPU hang).

Check `ls -l /proc/<pid>/fd/0` and the tail of the output file before suspecting the pager.

## Corrupted-data failures

Add temporary overlap checks over the page table (two pages with overlapping `[off, off+size)` in one tier) — the corruption that bit us was a freed shared arena block being reallocated, not a bad load.

## Swap counter monitoring

`stats` (`get_stats()`) exposes faults/evictions/swap counters; `test_gguf` prints them on success, which makes regressions (e.g. unexpected swap traffic) visible.

## After `git pull` in the llama.cpp clone, the ForthLLM patch can appear to "not fit"

The patch creates three *new* files (`include/llama-forthllm.h`, `src/forthllm.cpp`, `tests/test-forthllm.cpp`); a pull discards the tracked edits but leaves those untracked files behind, and `git apply` refuses to create a file that already exists — so the whole patch is rejected with a misleading context error.

Fix: delete the three untracked files, re-run `patches/apply-llama.cpp-patches.sh`, then rebuild the ggml libs + server in each platform's build dir.

## CUDA device indices are not nvidia-smi indices

The wrapper's `LLAMA_VM_GPU=cudaN` (and the VRAM window's device) select by the CUDA *runtime* device order, which on a multi-GPU box can differ from `nvidia-smi`'s bus ordering. On the P6000/3060 box, runtime `cuda0` = the RTX 3060 (12 GiB), `cuda1` = a P6000 (24 GiB) — the reverse of what the nvidia-smi index suggests.

A `cudaMalloc failed: out of memory` on a card that `nvidia-smi` shows as mostly free usually means the window landed on the wrong (smaller) device. Verify with a `cudaGetDeviceProperties` probe, not nvidia-smi.

## Swap bugs found and fixed

Two swap bugs were found during a swap audit (commit `eba05a6`):

### Bug 1 — stale co-resident copy (silent corruption)
`dirty` is per-page, not per-tier, so a `write()` to a page resident in both tiers left the other tier's copy stale. The gather fast path serves tier-resident bytes without re-checking, so `tensor_data(..., vram)` would return the pre-write bytes.

Fix: `pager::write()` now propagates the written bytes to the co-resident copy.

### Bug 2 — crash recovery for captured regions (spurious "data missing")
The `written` flag is in-memory only, so after a restart `read()`/`tensor_data()`/`tensor_data_range()` rejected pages whose data was actually sitting in the swap file.

Fix: `impl::data_exists()` — written **or** resident in a tier **or** held in the swap file per the recovered validity bitmap.

## Eviction policy: LRU only

A per-decode-call "hot set" eviction bias was tried and reverted: in the wrapper's batch-compute path every tensor faulted during a decode call is "hot", so `evict_one` pass 0 found no victim and the no-swap guard aborted file-mode runs with the misleading "only dirty pages remain and no swap file is configured" (the pages are clean file-backed pages). LRU already keeps recently used pages at the MRU end, so the bias bought nothing.