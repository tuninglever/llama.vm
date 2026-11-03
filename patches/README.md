# llama.cpp patches

llama.vm integrates with llama.cpp via the ggml backend registry (dlopen), so
it does **not** need to live inside the llama.cpp tree. However, a few small
llama.cpp patches are required for the full feature set. They are tracked here
so a public checkout can reproduce the exact tree llama.vm was validated against.

| Patch | What it enables | Applied by the script |
|-------|-----------------|-----------------------|
| `ggml-type66-kimi-q1-0-stub.patch` | Adds `GGML_TYPE_KIMI_Q1_0 = 66` (a load-only stub) so Kimi-K3 `UD-Q1_0` GGUF files parse and page correctly. Kimi-K3's quant uses type 66, which is not in upstream ggml. Load-only: no dequant/quant kernels. | always |
| `llama-cpp-forthllm-kv-compaction.patch` | ForthLLM KV-cache compaction: `--forthllm*` CLI options, the chunk-summarization engine (`llama-forthllm.h` / `forthllm.cpp`), `llama_kv_cache::forthllm_compact` (compacts old chunks of a full-attention KV cache into S summary rows, re-rotating the shifted keys via the K-shift graph; works for bare and hybrid attention+recurrent memories, incl. IMROPE text), and the `llama-server` hooks (one-shot post-prefill compaction + context-pressure compaction before a context shift). Validated end-to-end on Qwen3.8-27B (qwen35, hybrid, IMROPE). Not required by the pager. | by default (`--pager-only` skips it) |
| `llama-cpp-mlx-patch.patch` | Adds the `GGML_MLX` CMake option to a llama.cpp tree that predates the MLX backend. | never (Apple-only; apply manually if your tree lacks `GGML_MLX`) |

Apply them with:

```sh
patches/apply-llama.cpp-patches.sh /path/to/llama.cpp          # all patches (default)
patches/apply-llama.cpp-patches.sh --pager-only /path/to/llama.cpp   # type-66 stub only
```

then rebuild in **your platform's build directory** — the `build` you
configured for your target: `build` (CPU), `build-hip` (ROCm), `build-cuda`
(CUDA), or `build-mlx` (MLX). If you have several build directories, repeat
for each one you use:

```sh
LLAMA_BUILD=/path/to/llama.cpp/build-hip   # your platform's build dir
cmake --build "$LLAMA_BUILD" --target ggml-base ggml -j
```

Since the ForthLLM patch is applied by default, also reconfigure and rebuild
the server + test (the patch changes `src/CMakeLists.txt`):

```sh
cmake -S /path/to/llama.cpp -B "$LLAMA_BUILD"
cmake --build "$LLAMA_BUILD" --target llama-server test-forthllm -j
```

The patches are generated against the llama.cpp revision llama.vm was developed
against (re-rolled 2026-11-02 for upstream `785322d17`, which the
`/work/llama.cpp` clone is at); if a newer tree does not apply cleanly, the
change may already be upstream (check
`git log --oneline -S GGML_TYPE_KIMI_Q1_0 -- ggml/include/ggml.h` for the
type-66 stub), or the surrounding code may have moved and the patch needs a
small re-roll.
