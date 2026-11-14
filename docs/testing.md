# Testing

## Unit Tests

```sh
# Build
cmake -S . -B build -DLlamaVM_LLAMA_CPP_DIR=/path/to/llama.cpp
cmake --build build -j

# Run all tests
ctest --test-dir build --output-on-failure

# Individual tests
build/test_arena
build/test_pager
build/test_concurrent
build/test_ggml_buffer
build/test_captured
build/test_gpu_buffer             # GPU buffer + wrapper backend, VRAM tier on a host buffer
build/test_llama_integration      # needs libllama + a model (LLAMA_VM_TEST_MODEL)
LLAMA_VM_TEST_VRAM=1 LLAMA_VM_VRAM_BACKEND=cpu build/test_vram
LLAMA_VM_TEST_MODEL=/path/to/model.gguf build/test_gguf
```

## CLI Tools

```sh
# Model info
build/llama-vm info   <model.gguf> [--ram 64M] [--vram 32M] [--page 4M] [--backend cpu]

# Verify pages
build/llama-vm verify <model.gguf> [--ram 16M] [--vram 64M] [--tensor NAME]

# Benchmark
build/llama-vm bench  <model.gguf> [--ram 32M] [--vram 32M] [--backend cuda] [--pattern seq|rand] [--pass N] [--swap FILE]

# Page table
build/llama-vm pages  <model.gguf> [--ram 64M] [--vram 32M] [--page 4M] [--backend cpu]

# Capacity plan
build/llama-vm plan   <model.gguf> --ram AVAIL_RAM --vram AVAIL_VRAM

# Tiered test (exercise RAM + VRAM + file + tier copies, then verify every page)
build/llama-vm tiered <model.gguf> --ram 1G --vram 1G --backend hip

# Self-contained demo (no model needed)
build/llama-vm demo
```

`--backend cpu` runs the VRAM tier on a host buffer (useful without a GPU). Size arguments accept `K`/`M`/`G` suffixes.

## Model Testing

Use three tiers of models to validate the MLX backend:

### 1. Small Model Test (Basic Functionality)
**Model:** GLM-5 UD-IQ2_XXS (~100-150 GB)
**Purpose:** Verify basic MLX backend functionality with a model that fits in memory

```bash
export LLAMA_VM_GPU=mlx
export LLAMA_VM_VRAM=160G
export LLAMA_VM_GGUF=unsloth/GLM-5-GGUF/GLM-5-UD-IQ2_XXS-00001-of-00006.gguf
llama-cli -m unsloth/GLM-5-GGUF/GLM-5-UD-IQ2_XXS-00001-of-00006.gguf \
          --device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' -ngl 99 \
          --n-predict 8 --prompt "Hi" --temp 0 --single-turn
```

### 2. Medium Model Test (Demand Paging)
**Model:** DeepSeek-V4-Flash Q4 (~200-250 GB)
**Purpose:** Force demand paging with a smaller VRAM window

```bash
export LLAMA_VM_GPU=mlx
export LLAMA_VM_VRAM=50G
export LLAMA_VM_GGUF=ds4/gguf/DeepSeek-V4-Flash-Q4_K_M-00001-of-NN.gguf
llama-cli -m ds4/gguf/DeepSeek-V4-Flash-Q4_K_M-00001-of-NN.gguf \
          --device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' -ngl 99 \
          --n-predict 8 --prompt "Hi" --temp 0 --single-turn
```

### 3. Huge Model Test (Scale)
**Model:** Kimi-K3 UD-IQ1_M (~648.9 GB)
**Purpose:** Demonstrate that even massive models can run with demand paging
**Note:** Requires the ggml type 66 stub for `UD-Q1_0` quantization

```bash
export LLAMA_VM_GPU=mlx
export LLAMA_VM_VRAM=32G
export LLAMA_VM_GGUF=unsloth/Kimi-K3-GGUF/Kimi-K3-UD-IQ1_M-00001-of-00011.gguf
llama-cli -m unsloth/Kimi-K3-GGUF/Kimi-K3-UD-IQ1_M-00001-of-00011.gguf \
          --device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' -ngl 99 \
          --n-predict 4 --prompt "Hi" --temp 0 --single-turn
```

## Testing Notes

- **Use `--single-turn`** with llama-cli to avoid the stdin-EOF prompt loop
- **VRAM window size** is the key parameter - smaller windows force more demand paging
- **Performance expectation:** Smaller windows will be slower due to more frequent page faults, but should still complete successfully
- **Monitor pager stats** with `LLAMA_VM_DEBUG=1` to verify demand paging is working correctly

## Expected Results

| Test | Expected Outcome |
|------|------------------|
| Small model | Completes quickly, no or minimal page faults |
| Medium model | Completes with visible page faults, demonstrates demand paging |
| Huge model | Completes with heavy page faults, demonstrates scale of demand paging |

## Troubleshooting

- If the model doesn't fit in the specified VRAM window, increase `LLAMA_VM_VRAM`
- If you see "fatal error" messages, check that the ggml type 66 stub is built (for Kimi-K3)
- If the model hangs, verify that `--single-turn` is used with llama-cli
- Use `LLAMA_VM_DEBUG=1` to enable detailed logging for debugging