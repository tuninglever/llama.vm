# Investigation: llama-server `--llama-vm-*` / `--forthllm-*` failure

Date: 2026-10-30
Model under test: DeepSeek-V4-Flash `UD-Q4_K_XL` (5 shards, `deepseek4` arch, MLA),
via the `LLAMA_VM_GPU` demand-paging buffer, launched from `test-forth-deepseek()`
in `~/.bashrc.ai`.

## Symptom

**Command A (fails):**

```sh
$LLAMA_SERVER_ROOT/build-cuda/bin/llama-server \
    -m "$LLAMA_VM_GGUF" \
    --device LLAMA_VM_GPU \
    -ot '.*=LLAMA_VM_GPU' \
    --ctx-size 100000 \
    --host 0.0.0.0 --port 8888 \
    --tools all --agent \
    --load-mode none --n-gpu-layers all \
    --llama-vm-ram 8G --llama-vm-vram 8G \
    --forthllm-chunk-size 512 --forthllm-summary-method attention \
    --forthllm-max-stack-depth 64 --forthllm-squeezer jl-bwt --forthllm-jl-dim 128
```

```
llama.vm GPU backend: unknown LLAMA_VM_GPU target 'LLAMA_VM_GPU' (use auto|hip|cuda|vulkan|cpu,
optionally with a device index, eg. cuda1); falling back to auto
error: invalid argument: --llama-vm-ram
```

**Command B (works):** the same, minus the `--llama-vm-*` and `--forthllm-*` flags
(`do-deepseek()` in `~/.bashrc.ai`, with `--context-shift`).

## TL;DR

Three independent problems. None of them is in the pager:

1. **`--llama-vm-ram` / `--llama-vm-vram` do not exist in the llama.cpp build in use.**
   The llama.vm buffers read their budgets from the `LLAMA_VM_*` **env vars**
   (`LLAMA_VM_RAM`, `LLAMA_VM_VRAM`, `LLAMA_VM_SWAP`, `LLAMA_VM_GGUF`). The CLI flags
   are documented in `AGENTS.md` but were never committed to the llama.cpp tree this
   machine builds from.
2. **`~/.bashrc.ai` exports `LLAMA_VM_GPU=LLAMA_VM_GPU`** — the variable's own name as
   its value. Non-fatal (the wrapper falls back to `auto`), but wrong; on this machine
   it should be `cuda0` (Quadro P6000, 24 GB).
3. **The `--forthllm-*` flags exist only in `forthllm-design.md` examples.** ForthLLM
   is a placeholder library in this repo and has never been wired into llama-server.
   The arg parser aborts at the first unknown flag, so Command A dies at
   `--llama-vm-ram` before it ever reaches the `--forthllm-*` flags — but those would
   fail identically (reproduced below).

   **Status (2026-10-30, later in the session): the `--forthllm-*` flags are now real.**
   Phases 1–3 of the implementation plan below are done and validated end-to-end:
   the flags parse, the summarization engine + KV-cache compaction live in the llama
   core (`include/llama-forthllm.h`, `src/forthllm.cpp`, `llama_kv_cache::forthllm_compact`),
   and `llama-server` compacts a prefilled prompt into summary rows (post-prefill, and
   under context pressure — at the context-limit stop check and before a context
   shift). Validated on Qwen3.8-27B (hybrid attention+recurrent, IMROPE): a
   2196-token prompt compacted to 180 cells (16 chunks, 2016 cells freed) and the
   model still answered correctly. The patch is in
   `patches/llama-cpp-forthllm-kv-compaction.patch`. DeepSeek-V4-Flash (MLA) is still
   a documented "not supported" path.

## Mechanism: how the pieces fit together

### 1. The `llama-server` binary is a stub; the real code is in shared libs

`/work/llama.cpp/build-cuda/bin/llama-server` is an **18 KB stub** (it only contains
`main` → `llama_server(argc, argv)` plus the build-info symbols). `ldd` shows where the
real implementation comes from:

```
libllama-server-impl.so => /usr/local/lib/libllama-server-impl.so
libllama-common.so.0    => /usr/local/lib/libllama-common.so.0
libllama.so.0           => /usr/local/lib/libllama.so.0
libggml*.so.0           => /usr/local/lib/...
```

So which `llama-server` behavior you get depends entirely on which `libllama-common.so.0`
the dynamic linker resolves — `LD_LIBRARY_PATH` first, then the default paths.

**Key finding:** the libs in `/work/llama.cpp/build-cuda/bin` and the ones in
`/usr/local/lib` are **the same build** (byte-identical sizes and timestamps:
`libllama-common.so.0.8.0` = 6,098,864 bytes, Oct 29 17:07, in both places; same for
`libllama-server-impl.so`). `/usr/local/lib` was installed from `build-cuda`, and
`/usr/local/bin/llama-server` is the same stub (identical BuildID). Changing
`LD_LIBRARY_PATH` between the two dirs therefore changes **nothing** — both sets reject
the `--llama-vm-*` flags.

### 2. The llama.cpp tree is clean upstream — the flags were never committed

`/work/llama.cpp` is at `b11989-14-g36a0e2ffc` with a clean working tree (only an
untracked `asdf` file). Verified by search:

- `grep -rEn 'llama-vm|LLAMA_VM|forthllm'` over the whole tree (`.c/.h/.cpp/.md`):
  **zero matches**.
- `git log --all --grep=...` over every branch: no related commits.
- `strings -n 8 libllama-common.so.0 | grep -Ei 'llama-vm|LLAMA_VM|forthllm'`:
  **zero matches** — the compiled parser doesn't know these options either.

`AGENTS.md` (lines 46–48 and 387–391) describes `--llama-vm-ram` etc. as if they exist
in `common_params_parse`. That is not true for this tree — either they were implemented
in a different checkout that is no longer here, or the doc describes intended work.
The doc has been corrected to match the verified state.

**How the arg parser fails:** llama.cpp's option parser treats an unrecognized long
option as a hard error and stops parsing there. In Command A, `--llama-vm-ram` appears
before the `--forthllm-*` flags, so the run dies there. Reproduced directly:

```
$ llama-server --llama-vm-ram 8G            # → error: invalid argument: --llama-vm-ram
$ llama-server --forthllm-chunk-size 512    # → error: invalid argument: --forthllm-chunk-size
```

So even a llama.cpp build that had the `--llama-vm-*` flags would still fail on
`--forthllm-chunk-size`.

### 3. The budgets are read from env vars by the buffer, not from CLI flags

`src/ggml_buffer_gpu.cpp` (`buf_alloc`, lines 249–312) is where the `LLAMA_VM_GPU`
buffer is created, and it reads everything from the environment via `env_size()`
(`src/vm_env.h`, K/M/G suffix parsing):

| Env var | Meaning |
|---|---|
| `LLAMA_VM_VRAM` | the VRAM window (required; buffer creation fails without it) |
| `LLAMA_VM_GGUF` | set → **file mode**: pager built from the GGUF, loader writes ignored, weights fault from the file at compute time |
| `LLAMA_VM_RAM` | file mode: staging tier for the background prefetch (discrete GPU; default = one window; 0 on host-writable iGPU windows). Capture mode: the capture tier (default = whole model; a budget below the model size **requires** `LLAMA_VM_SWAP`) |
| `LLAMA_VM_SWAP` | swap file (capture mode only) |
| `LLAMA_VM_GPU` | inner compute backend: `auto\|hip\|cuda\|vulkan\|cpu` + optional device index (`cuda1` = second CUDA device) |

This is why Command B works: `do-deepseek()` sets `LLAMA_VM_VRAM=3G` in the
environment, and `LLAMA_VM_GGUF` selects file mode. The env-var path needs no llama.cpp
support at all — it is entirely inside `libggml-llamavm-gpu.so`.

### 4. Where `LLAMA_VM_GPU=LLAMA_VM_GPU` comes from

`~/.bashrc` sources `~/.bashrc.ai` (lines 129–130). In **both** `do-deepseek()`
(line 248) and `test-forth-deepseek()` (line 274):

```sh
export LLAMA_VM_GPU=LLAMA_VM_GPU    # inner compute backend: auto|hip|cuda|vulkan|cpu,
                                    # optionally with a device index (cuda1 = 2nd CUDA device)
```

The value is the variable's own name — the comment on the next line even lists the
valid values. `gpu_target()` (`src/ggml_buffer_gpu.cpp:63`) does
`getenv("LLAMA_VM_GPU")`, and `find_inner_device()` (lines 962–1006) only recognizes
`hip|cuda|vulkan|cpu|auto` (plus a numeric suffix), so the literal string
`LLAMA_VM_GPU` hits the fallback at line 989:

```
llama.vm GPU backend: unknown LLAMA_VM_GPU target 'LLAMA_VM_GPU' ...; falling back to auto
```

`auto` then picks the first GPU device found in the registry, which is why the run
proceeded to the arg-parsing error instead of dying on backend init. Note the warning
fires in the "working" run too — it was just not noticed.

### 5. The `LLAMA_VM_GPU` buft resolution (why Command B works end-to-end)

`-ot '.*=LLAMA_VM_GPU'` routes every tensor to the `LLAMA_VM_GPU` buffer type, and
`--device LLAMA_VM_GPU` registers the device name. llama.cpp resolves the buft name
across the registered ggml backends; the wrapper backend is loaded from
`GGML_BACKEND_PATH=$PWD/build/libggml-llamavm-gpu.so` (the llama.vm build), which
registers both the compute device (presenting the inner CUDA/HIP device) and the
memory-only buft. `--load-mode none` is required so the loader's `set_tensor` writes go
to the buffer (file mode ignores them; capture mode captures them). This path is
independent of the llama.cpp version's CLI flags — it only needs the buft-override
mechanism, which the build has.

## ForthLLM: what actually exists

**The `--forthllm-*` flags are not implemented anywhere.** They appear only in the
*Examples* section of `forthllm-design.md` (added by commit `af48854` "Add llama-server
usage examples to ForthLLM design doc") — aspirational examples, not a working
interface.

The ForthLLM code in this repo (commits `4bc6101`, `29986ee`, merged `d098204`) is a
**placeholder scaffold**, not an implementation:

- `src/forthllm.cpp:42-74` — `context_manager::process_chunk()` never touches a model.
  The "summary embedding" is filled with `sin(i + chunk_id) * 0.1f`; the code comment
  says it verbatim: *"For now, use a simple mean of random values as a placeholder"*,
  and lists what a real implementation would do (embed tokens, run the layers, extract
  attention weights, build a compressed summary).
- `src/forthllm.cpp:11-15` — `compute_attention_score()` is a recency heuristic
  (`1/(1 + 0.1·age)`), not attention weights.
- The squeezer (`src/squeezer.cpp`, JL projection → 8-bit quantization assuming
  values in [-1,1] → BWT → FM-index) is real math, but it is built for the placeholder
  pipeline: `expand()` explicitly notes *"We can't perfectly invert the JL projection,
  so we return the projected vector as-is"* — squeeze→expand is lossy by design and
  does not round-trip to the original dimension.
- `tests/test_forthllm.cpp` asserts only stack bookkeeping (depth, pop, reset,
  embedding size) — nothing about content quality.

**Conclusion:** linking the existing library into llama.cpp would give flags that do
nothing useful (or worse, replace real context with fake sinusoidal summaries). The
real work is a new engine plus server wiring — see the plan below.

### What a real implementation can hook into (llama.cpp API survey)

Public API (`include/llama.h:594-822`): sequence-level only —
`llama_memory_seq_rm/cp/keep/add/div`, `llama_memory_clear`, `llama_memory_seq_pos_min/max`,
`llama_memory_can_shift`. **No public access to raw K/V rows.**

Internals (`src/llama-kv-cache.h`, `class llama_kv_cache : llama_memory_i`):

- Per-layer storage is ggml tensors: `struct kv_layer { uint32_t il; ggml_tensor * k;
  ggml_tensor * v; std::vector<ggml_tensor *> k_stream, v_stream; }` (lines 257–267),
  with `v_trans` (transposed V layout) and per-head sizes `n_embd_head_k_all/v_all`.
  Cache types are quantized (`type_k()/type_v()`: f16, q8_0, ...), so row reads need
  dequantization.
- Cell bookkeeping: `llama_kv_cells` (`src/llama-kv-cells.h`) — per-position
  `pos[]`, `ext[]` (tok, 2D positions), `seq[]` bitsets, `shift[]` (the position-shift
  mechanism used by `--context-shift`), plus a `used` set.
- Graph-side access: `get_k/get_v(ctx, il, n_kv, sinfo)` and `cpy_k/cpy_v(...)`
  (lines 196–201) build views/copies for the attention graph; `slot_info` maps ubatch
  tokens to cell indices.
- `llama_kv_cache_context` (line 356) is the per-decode context; `update(lctx,
  do_shift, sc_info)` applies pending stream copies/shifts.

The compaction primitive a ForthLLM integration needs — *replace C cells in
[p0, p1) with S summary cells and shift the tail by (C−S)* — is not a single existing
call; it would be composed from cell-level ops plus the `shift[]` machinery (the same
machinery `--context-shift` uses). That is the main engineering risk of the wiring.

## Machine state at time of investigation

- `/work/llama.cpp`: clean upstream `b11989-14`; `build-cuda` built Oct 29 17:0x
  (GGML_CUDA); `/usr/local/lib` + `/usr/local/bin/llama-server` installed from it.
- GPUs: `0: Quadro P6000 24 GB` (11.6 GB used), `1: RTX 3060 12 GB` (5 GB used),
  `2: Quadro P6000 24 GB` (12.7 GB used). GGML registers them in this order, so
  `cuda0` = P6000 #1, `cuda1` = RTX 3060, `cuda2` = P6000 #2. (AGENTS.md's older
  "cuda1 = the P6000" note predates the 3-GPU layout.)
- Unrelated running process: `/usr/local/bin/llama-server` serving
  `unsloth/Qwen3.8-27B-GGUF` on port 8080 (plain CUDA, no llama.vm).

## Fixes applied (this session)

1. `~/.bashrc.ai`:
   - `export LLAMA_VM_GPU=LLAMA_VM_GPU` → `export LLAMA_VM_GPU=cuda0` in both
     functions (with a comment mapping cuda0/1/2 to the physical GPUs).
   - `test-forth-deepseek()`: `--llama-vm-ram 8G` / `--llama-vm-vram 8G` replaced by
     `export LLAMA_VM_RAM=8G` / `export LLAMA_VM_VRAM=8G` (the env vars the buffer
     actually reads). The `--forthllm-*` flags are now **real** (see below) and can be
     uncommented — though DeepSeek-V4-Flash (MLA) is a documented "not supported"
     compaction path, so they are a no-op (with a clear log) on that model.
2. `AGENTS.md`: corrected the two `--llama-vm-*` flag claims to the verified state
   (env vars are the interface today; the flags are not in the current build) and added
   a ForthLLM status note + next-steps item.
3. `forthllm-design.md`: added a status banner and a warning on the *Examples* section
   so the aspirational commands are not mistaken for runnable ones.
4. **ForthLLM Phases 1–3 implemented in llama.cpp** (this session): the `--forthllm-*`
   flags parse (`common/arg.cpp`), the engine + KV-cache compaction live in the core
   (`include/llama-forthllm.h`, `src/forthllm.cpp`, `llama_kv_cache::forthllm_compact`,
   C API `llama_forthllm_compact` incl. hybrid memory), and `llama-server` has the
   post-prefill + context-pressure hooks. Validated end-to-end on Qwen3.8-27B. Patch:
   `patches/llama-cpp-forthllm-kv-compaction.patch`.

The corrected `test-forth-deepseek()` is now equivalent to the working
`do-deepseek()` with an 8 GB window + 8 GB staging tier, plus the ForthLLM flags
(active on full-attention/hybrid models, a logged no-op on the MLA DeepSeek-V4-Flash).

## Implementation plan: making the `--forthllm-*` flags real

Design goal (from `forthllm-design.md`): replace processed prompt chunks with compact
representations so the KV cache stops growing linearly with context. The representation
that actually works inside a transformer is **summary KV rows**: a chunk of C positions
is replaced by S ≪ C rows that are linear combinations of the chunk's rows, per layer
and per head. Shape is preserved (head-dim), so the attention graph is untouched. This
is the same family as KV-eviction/summarization work (H2O/SnapKV-style).

### Phase 1 — flags (llama.cpp)

Add to `common/arg.cpp` + `common_params` (this tree parses options in
`common/arg.cpp`):

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

Notes on semantics (documented in help text):
- `--forthllm-summary-dim` is reinterpreted from the design doc's "summary embedding
  dimension" to *number of summary rows per chunk* — rows must keep head-dim shape to
  be consumable by attention.
- Any `--forthllm-*` flag (or bare `--forthllm`) enables the feature.
- Print the active config in the server banner; reject nonsense combos
  (e.g. `jl-bwt` without a sane `jl-dim`, `chunk-size < summary-dim`).

### Phase 2 — engine (llama.cpp, model-agnostic core) — **done**

Landed in the **core** (not `common/`), because `src/` cannot include `common/`:
`include/llama-forthllm.h` (C API + C++ engine) + `src/forthllm.cpp`. The `common/`
glue is thin (`common_forthllm_compact` in `common.cpp`); the original
`common/forthllm.{h,cpp}` was deleted.

- `llama_forthllm_summarize(k, n_heads_k, head_dim_k, v, n_heads_v, head_dim_v, C, S,
  method, k_out, v_out)` → S summary K rows and S summary V rows, with **per-head**
  weights (head h's K weights are applied to both K head h and V head h):
  - `mean`: uniform weights;
  - `last`: the last S rows;
  - `attention`: per-head weights `w_i = softmax(k_i[h]·k_last[h]/√d)` over the
    chunk's keys (last-key similarity — a real, cache-data-only proxy for attention
    relevance; true attention weights would require graph surgery, out of scope for
    the MVP).
- Stack of chunk summaries with `lru`/`lfu`/`fixed-window` eviction at
  `max_stack_depth` (LRU on the prompt = oldest chunks; LFU falls back to LRU with a
  log note in the MVP).
- Squeezer: JL projection → BWT applied to the **stored** summary representation for
  offload/persistence; in-cache rows stay head-dim. `--forthllm-squeezer none` skips
  it. (The BWT LF-mapping comparator must compare `uint8_t` — signed `char` corrupts
  the order for bytes ≥ 128; that was a 17-failure bug, now fixed.)
- Deterministic unit test (`tests/test-forthllm`, ctest, no model needed):
  mean/last/attention summaries, BWT round-trip, squeezer bounds, stack eviction,
  config validation — **all pass**.

### Phase 3 — server wiring (the risky part) — **done**

The surgery is a `llama_kv_cache::forthllm_compact(seq_id, params, chunks)` member
(it needs the private `v_cells`/`v_heads`/`layers`/`seq_to_stream`); the C API
`llama_forthllm_compact` (in `src/llama-context.cpp`) resolves the memory via
`dynamic_cast` to `llama_kv_cache` **or** `llama_memory_hybrid` → `get_mem_attn()` /
`get_mem_recr()`.

1. Walk the sequence's cells in position order; group into chunks of `chunk_size`;
   keep the last `chunk_size` positions raw (recent window); a leading remainder
   smaller than `chunk_size` stays raw.
2. For each older chunk: read its K/V rows per layer (dequantized to f32 via the
   ggml type traits), compute S summary rows, write them into the first S cells of
   the chunk (the summary rows keep the chunk's head positions), free the C−S cells,
   and shift subsequent cells by (C−S) with `cells.pos_add(i, d)` — the same
   `shift[]` machinery `--context-shift` uses. The pending re-rotation is applied by
   the K-shift graph at the next `memory_update` (start of the next compute); the
   graph treats MROPE/IMROPE as 1D NEOX over `n_rot` (upstream workaround, PR #13870).
3. Log the compaction (`compacted N chunk(s), freed M cells, prompt X -> Y tokens`).

**Gating (MVP)** — otherwise a clear "not supported" log + negative return:
full-attention `llama_kv_cache` (bare or hybrid-wrapped), flash attention enabled
(`!v_trans`, contiguous V rows), host-resident KV cache, no SWA, K-shift-capable
(1D RoPE or **IMROPE** — see below), exclusive single-seq cell ownership, contiguous
chunk positions.

**Hybrid support (QWEN35 and friends).** QWEN35 is a hybrid arch →
`llama_memory_hybrid`. Recurrent state is a per-seq running state (position-
independent), so attn-only compaction is functionally correct; after compaction the
wrapper calls `recr->seq_add(seq_id, 0, -1, -n_freed)` to shift the recurrent tail
cell's pos (prevents a spurious "non-consecutive token position" warning and a stale
`hybrid::seq_pos_max`). The hybrid update context delegates to `attn->init_update`,
so the K-shift propagates. `llama_memory_hybrid_iswa`/`_idx` are not supported (MVP).

**IMROPE K-shift gate (the blocker that took the longest).** `get_can_shift()`
returned false for QWEN35 because `n_pos_per_embd()` is 4 for IMROPE, and the K-shift
apply path `GGML_ABORT`s on a pending shift when `!get_can_shift()`. Verified that
IMROPE text is 1D-equivalent: text positions expand to `[p, p, p, 0]` (all non-zero
sections carry the same 1D token position; QWEN35's `rope.dimension_sections =
[11, 11, 10, 0]` leaves no un-rotated section), and the shift graph already re-rotates
as 1D NEOX over `n_rot`. So `get_can_shift()` now allows `LLAMA_ROPE_TYPE_IMROPE`
(STEP35 and true MROPE stay rejected).

**Server hooks (three).** All in `tools/server/server-context.cpp`, all gated on
`params_base.forthllm.enabled && !has_mtmd && !can_speculate()`:
1. `post_decode`, at the DONE_PROMPT→GENERATING transition: one-shot compaction right
   after prefill (also requires `slot.n_predict_max != 0`, else the pending K-shift
   would never be applied before the prompt is cached).
2. `process_token`, at the "running out of context" stop check (the
   `!ctx_shift && prompt.n_tokens() + 1 >= n_ctx` branch): try compaction before
   stopping the slot. This is the hook that makes compaction work with
   `--context-shift` **disabled** — the stop check runs before the next `pre_decode`,
   so without it the slot would be stopped first (verified: a long generation stopped
   exactly at `n_ctx` with no compaction until this hook was added).
3. `pre_decode`, in the context-pressure branch (`prompt.n_tokens() + 1 >= n_ctx`):
   try compaction before falling through to the context shift (the
   `--context-shift`-enabled path).

Token sync (helper `forthllm_compact_slot`): copy `slot.prompt.tokens`, erase the
compacted ranges `[pos0+S, pos0+n_tokens)` back-to-front, `slot.prompt.clear()` +
`insert()` (clears position-tied checkpoints — the context-shift precedent), set
`slot.truncated = true`.

**Validated end-to-end (2026-10-30).** Qwen3.8-27B `UD-Q4_K_M` (qwen35, hybrid,
IMROPE), CPU (`-ngl 0 --no-kv-offload`), `--ctx-size 4096`,
`--forthllm --forthllm-chunk-size 128 --forthllm-summary-dim 2 --forthllm-summary-method
attention`:

*Post-prefill (post_decode hook), 2196-token prompt:*
- `forthllm: compacted 16 chunk(s), freed 2016 cells, prompt 2196 -> 180 tokens`
  (16×128 compacted, 16×2 summary rows kept, 2016 freed — the arithmetic checks out);
- no "non-consecutive token position" warning (the hybrid recr sync works);
- the model answered the trailing factual question correctly (**"Paris"**) and even
  noted the long prefix was irrelevant — coherent output through the compacted cache.

*Context pressure (process_token hook), 2238-token prompt + `n_predict=4000` (no
`--context-shift`):*
- post-prefill: `compacted 16 chunk(s), freed 2016 cells, prompt 2238 -> 222 tokens`;
- at the context limit (4095+1 ≥ 4096): `compacted 30 chunk(s), freed 3780 cells,
  prompt 4095 -> 315 tokens` (30×126 = 3780; 4095−3780 = 315);
- generation then ran to the full `n_predict=4000` (`stop_type: limit` = the
  n_predict cap, not the context) — without the hook it stopped at 3874 with the
  output cut off mid-sentence.

**MVP limitations (documented "not supported" paths):** MLA caches (DeepSeek-V4:
`llama_kv_cache_dsv4` — the user's actual model), `hybrid_iswa`/`hybrid_idx`,
speculative slots (draft cache not compacted), shared cells (n_cmpl>1 parent/child),
multimodal prompts, and the stored-representation options (`max_stack_depth` /
`eviction` / `squeezer`) which validate at parse time but are not yet used by the
cache path.

### Phase 4 — replace the llama.vm placeholder

Once the engine proves out in llama.cpp, replace `src/forthllm.cpp`'s placeholder math
with the real engine (or make the llama.vm copy a thin re-export), and update
`forthllm-design.md` examples to the actually-implemented flag set. Keep the patch in
`patches/` (repo convention) so it applies to a fresh llama.cpp checkout.

### Open questions

- Quality: how much does S=1 per 512-token chunk hurt on real benchmarks? (The design
  doc's "Performance Comparison" table is unmeasured.)
- `lfu` eviction needs per-chunk attention counts — that requires a callback into the
  attention path; defer beyond MVP.
- Interaction with `--context-shift` and KV cache quantization (`--cache-type-k q8_0`):
  summaries written back must be quantized to the cache type.
- MLA (DeepSeek-V4) latent rows: summarizing the latent is shape-compatible, but the
  rope-key component needs separate handling — verify on the actual model.

## Reproduction / verification commands used

```sh
# stub vs real libs
ldd /work/llama.cpp/build-cuda/bin/llama-server
file /usr/local/bin/llama-server            # same BuildID as build-cuda stub

# flags absent from the build
strings -n 8 /usr/local/lib/libllama-common.so.0 | grep -Ei 'llama-vm|LLAMA_VM|forthllm'   # (empty)
strings -n 6 /work/llama.cpp/build-cuda/bin/libllama-common.so.0.8.0 | grep -Ei 'forthllm|llama-vm'  # (empty)

# tree is clean upstream
cd /work/llama.cpp && git log --oneline -3 && git status --short
grep -rEn 'llama-vm|LLAMA_VM|forthllm' /work/llama.cpp --include='*.cpp' --include='*.h' --include='*.md'  # (empty)

# parser behavior (before the patch)
/work/llama.cpp/build-cuda/bin/llama-server --llama-vm-ram 8G           # error: invalid argument
/work/llama.cpp/build-cuda/bin/llama-server --forthllm-chunk-size 512   # error: invalid argument
```

### ForthLLM build + end-to-end verification (after the patch)

```sh
# build (reconfigure first — CMakeLists changed)
cd /work/llama.cpp/build-cuda && cmake . && \
    cmake --build . --target test-forthllm llama-server -j$(nproc)

# unit tests (no model needed)
LD_LIBRARY_PATH=$PWD/bin ./bin/test-forthllm        # "All ForthLLM tests passed"

# end-to-end: Qwen3.8-27B (qwen35, hybrid, IMROPE), CPU, 4096 ctx, ForthLLM on
LD_LIBRARY_PATH=$PWD/bin ./bin/llama-server \
    -m /AI/cache/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/<hash>/Qwen3.8-27B-UD-Q4_K_M.gguf \
    -ngl 0 --no-kv-offload --ctx-size 4096 \
    --host 127.0.0.1 --port 8899 --parallel 1 \
    --forthllm --forthllm-chunk-size 128 --forthllm-summary-dim 2 \
    --forthllm-summary-method attention &

# a ~2196-token prompt ending in a factual question
curl -s http://127.0.0.1:8899/completion -d @req.json
# server log:  forthllm: compacted 16 chunk(s), freed 2016 cells, prompt 2196 -> 180 tokens
# answer:      "Paris" (coherent; no "non-consecutive token position" warning)
```
