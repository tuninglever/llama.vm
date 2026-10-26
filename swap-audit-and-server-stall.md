# Swap audit and server-stall diagnosis

Date: 2026-10-24
Scope: (1) audit of the pager's swap data-preservation, (2) diagnosis of why a
llama-server completion on this machine "takes ~1 hour and returns nothing".
Outcome: two swap bugs found and fixed (commit `eba05a6`, regression tests in
`tests/test_captured.cpp`, ctest 9/9); the server stall is unrelated to the pager.

## 1. Swap mechanism: audited — two real data-preservation bugs found and fixed

The swap path in `src/pager.cpp` was read end to end (eviction, swap write-back,
re-fault, crash recovery) and stress-tested. The core invariant holds: a dirty
last-copy page is swap-written + header + `fdatasync` **before** its in-memory
copy drops, and re-faults prefer the swap slot over the source file. Two bugs
could still corrupt or lose data (both reachable through the public API on
captured regions, i.e. the GPU-buffer capture mode):

### Bug 1 — stale co-resident copy (silent corruption)

`dirty` is per-page, not per-tier, so a `write()` to a page resident in both
tiers left the other tier's copy stale. The gather fast path serves
tier-resident bytes without re-checking, so `tensor_data(..., vram)` would
return the pre-write bytes — and a stale last copy could be swapped out as if
it were the newest (permanent loss).

Fix: `pager::write()` now propagates the written bytes to the co-resident copy
(`src/pager.cpp`).

### Bug 2 — crash recovery for captured regions (spurious "data missing")

The `written` flag is in-memory only, so after a restart
`read()`/`tensor_data()`/`tensor_data_range()` rejected pages whose data was
actually sitting in the swap file.

Fix: `impl::data_exists()` — written **or** resident in a tier **or** held in
the swap file per the recovered validity bitmap (trusted only once the bitmap
is known to describe the current page table). Used at the three check sites
(`src/pager.cpp`, declaration in `src/pager-impl.h`).

### Verification

- `tests/test_captured.cpp` gains both regression cases:
  - patched bytes must be visible through the gather fast path (same contiguous
    VRAM block, new bytes present, neighbours intact);
  - a restarted pager (same swap file) must serve swapped-out pages, report
    RAM-only pages as missing, and serve the whole tensor again after the
    missing tail is rewritten.
- Each new test was confirmed to **fail** with the corresponding fix disabled
  (stale-copy: the patch check fails; recovery: throws the original
  "read of a captured page before its data was written in").
- Full suite: 9/9 ctest pass. Committed as `eba05a6`
  ("pager: keep co-resident tier copies consistent, recover captured data from swap").
- Nuance for the llama.cpp flow: swap only matters for weights in **capture
  mode** (no `LLAMA_VM_GGUF`). In file mode (the README default), weights fault
  straight from the GGUF and swap is not involved.

Eviction-order note (verified while building the restart test): `evict_one`'s
pass 0 prefers clean victims (no swap write) over dirty last-copy pages,
scanning from the LRU tail — so a read-back of a captured region swaps out only
the first dirty page it must displace, not the whole displaced run. The restart
test's expected swap bitmap (pages 0..24 recovered, 25..31 missing) matches
this behaviour exactly.

## 2. Why the server returns nothing: the pager is not in that server at all

### The running server is stock llama-server

PID 1273352 (port 8080, ~28 h uptime):

- Command line: `llama-server -hf unsloth/Qwen3.8-27B-GGUF --context-shift
  --models-dir /AI/models --ctx-size 256000 --cache-type-k q8_0
  --cache-type-v q8_0 --load-mode mmap --op-offload --temp 0.7 --top-p 0.95
  --top-k 40 --min-p 0.01 -ngl all --host 0.0.0.0 --port 8080 --tools all --agent`
  — **no** `--device LLAMA_VM_GPU`, **no** `-ot '.*=LLAMA_VM_GPU'`.
- `libggml-llamavm-gpu.so` is **not loaded** in the process (zero matches in
  `/proc/1273352/maps`). `GGML_BACKEND_PATH` points at a stale Oct 19 build in
  `/work/llama.vm`; the `LLAMA_VM_*` env vars are leftovers pointing at
  **DeepSeek** while the server loads **Qwen** — they do nothing here.
- Model: Qwen3.8-27B UD-Q4_K_M (16.4 GB), mmap, all layers on the Radeon 890M
  APU, 256K context, q8_0 KV.

### What it is actually doing

From `/slots`, GPU sampling, and the server source in this tree
(`tools/server/server-context.cpp`):

- It is in a continuous agent loop: task IDs keep climbing (83K → 178K over the
  day) and the agent conversation keeps growing (95K → 145K → 125K tokens;
  `--context-shift` truncates it when it nears the 256K cap).
- Each turn = a short GPU burst (~2–5 s at ~85%), then idle ~20 s while the
  client thinks/retries.
- The `/slots` field `n_prompt_tokens_processed: 0` is **misleading**:
  `slot.reset()` clears stats when a task releases, so it shows the *cleared*
  stats of the last task, not evidence of zero work. The tasks are completing —
  the loop only continues if responses are coming back.

Conclusion: "an hour, nothing" = the user's question is buried in a
125K–145K-token agent conversation that gets re-rendered and re-prefilled every
turn, at ~1–2 turns/minute on this APU. Nothing is hung; it is an enormous
prompt being churned.

### Recommendations

1. **If the pager is wanted (README recipe), this command line is not using
   it.** It needs `--device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' --load-mode
   none`, and `LLAMA_VM_GGUF` must point at the *same* model being loaded (it
   currently points at DeepSeek while Qwen loads — in file mode that mismatch
   breaks weight faulting).
2. **Cap the context** — `--ctx-size 256000` lets the agent conversation
   balloon to 145K tokens. Something like 32K–64K (plus history truncation on
   the agent side) would cut per-turn cost by ~3–4x.
3. **Watch the real logs** in the konsole terminal running the server (per-task
   `prompt eval time` lines).
4. Reference performance measured on this machine with the pager (file mode,
   HIP): Qwen3.8-27B at a 1 GiB window does ~0.65 t/s with coherent output; the
   155 GB DeepSeek-V4-Flash at a 3 GiB window does ~0.006 t/s (this NVMe is
   ~0.9 GiB/s, ~6.5x slower than the machine in AGENTS.md) — so the big model
   is impractical here regardless of paging.

Nothing was killed or modified on the server during the investigation.
