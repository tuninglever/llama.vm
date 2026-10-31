# llama.cpp patches

llama.vm integrates with llama.cpp via the ggml backend registry (dlopen), so
it does **not** need to live inside the llama.cpp tree. However, two small
llama.cpp patches are required for the full feature set. They are tracked here
so a public checkout can reproduce the exact tree llama.vm was validated against.

| Patch | What it enables |
|-------|-----------------|
| `ggml-type66-kimi-q1-0-stub.patch` | Adds `GGML_TYPE_KIMI_Q1_0 = 66` (a load-only stub) so Kimi-K3 `UD-Q1_0` GGUF files parse and page correctly. Kimi-K3's quant uses type 66, which is not in upstream ggml. Load-only: no dequant/quant kernels. |
| `ggml-backend-moe-ids-host-buffer.patch` | Lets the scheduler read a MoE `ids` tensor directly from a host buffer (e.g. pinned host memory on an integrated GPU) instead of requiring a device copy, which the backend would reject. |
| `llama-cpp-forthllm-kv-compaction.patch` | ForthLLM KV-cache compaction (feature patch, not required by the pager): `--forthllm*` CLI options, the chunk-summarization engine (`llama-forthllm.h` / `forthllm.cpp`), `llama_kv_cache::forthllm_compact` (compacts old chunks of a full-attention KV cache into S summary rows, re-rotating the shifted keys via the K-shift graph; works for bare and hybrid attention+recurrent memories, incl. IMROPE text), and the `llama-server` hooks (one-shot post-prefill compaction + context-pressure compaction before a context shift). Validated end-to-end on Qwen3.8-27B (qwen35, hybrid, IMROPE). |

Apply them with:

```sh
patches/apply-llama.cpp-patches.sh /path/to/llama.cpp
```

then rebuild the ggml libraries:

```sh
cmake --build /path/to/llama.cpp/build --target ggml-base ggml -j
```

The patches are generated against the llama.cpp revision llama.vm was developed
against; if they do not apply cleanly to a newer tree, they may already be
upstream (check `git log --oneline -S GGML_TYPE_KIMI_Q1_0 -- ggml/include/ggml.h`
for the type-66 stub), or the surrounding code may have moved and they need a
small re-roll.
