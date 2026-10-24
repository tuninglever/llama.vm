# Model Testing Guide for MLX Backend

This guide provides recommendations for testing the MLX backend at different scales, from basic functionality to massive model demand paging.

## Testing Strategy

Use three tiers of models to validate the MLX backend:

1. **Small model** - Fits entirely in RAM/VRAM (basic functionality test)
2. **Medium model** - Forces demand paging with smaller VRAM window (paging mechanism test)
3. **Huge model** - Demonstrates demand paging at scale (scale test)

## 1. Small Model Test (Basic Functionality)

**Model:** GLM-5 UD-IQ2_XXS (smallest quantization)
- Size: ~100-150 GB
- Purpose: Verify basic MLX backend functionality with a model that fits in memory

**Download:**
```bash
pip install -U huggingface_hub
hf download unsloth/GLM-5-GGUF \
    --local-dir unsloth/GLM-5-GGUF \
    --include "*UD-IQ2_XXS*"
```

**Test command:**
```bash
export LLAMA_VM_GPU=mlx
export LLAMA_VM_VRAM=160G
export LLAMA_VM_GGUF=unsloth/GLM-5-GGUF/GLM-5-UD-IQ2_XXS-00001-of-00006.gguf
llama-cli -m unsloth/GLM-5-GGUF/GLM-5-UD-IQ2_XXS-00001-of-00006.gguf \
          --device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' -ngl 99 \
          --n-predict 8 --prompt "Hi" --temp 0 --single-turn
```

## 2. Medium Model Test (Demand Paging)

**Model:** DeepSeek-V4-Flash Q4
- Size: ~200-250 GB
- Purpose: Force demand paging with a smaller VRAM window to test the paging mechanism

**Download:**
```bash
git clone https://github.com/antirez/ds4
cd ds4
./download_model.sh q4
```

**Test command (with smaller VRAM window to force paging):**
```bash
export LLAMA_VM_GPU=mlx
export LLAMA_VM_VRAM=50G
export LLAMA_VM_GGUF=ds4/gguf/DeepSeek-V4-Flash-Q4_K_M-00001-of-NN.gguf
llama-cli -m ds4/gguf/DeepSeek-V4-Flash-Q4_K_M-00001-of-NN.gguf \
          --device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' -ngl 99 \
          --n-predict 8 --prompt "Hi" --temp 0 --single-turn
```

## 3. Huge Model Test (Scale)

**Model:** Kimi-K3 UD-IQ1_M (1-bit dynamic GGUF)
- Size: ~648.9 GB
- Purpose: Demonstrate that even massive models can run with demand paging
- Note: Kimi-K3 requires the ggml type 66 stub mentioned in AGENTS.md for `UD-Q1_0` quantization

**Download:**
```bash
hf download unsloth/Kimi-K3-GGUF \
    --local-dir unsloth/Kimi-K3-GGUF \
    --include "*UD-IQ1_M*"
```

**Test command (with very small VRAM window to force heavy paging):**
```bash
export LLAMA_VM_GPU=mlx
export LLAMA_VM_VRAM=32G
export LLAMA_VM_GGUF=unsloth/Kimi-K3-GGUF/Kimi-K3-UD-IQ1_M-00001-of-00011.gguf
llama-cli -m unsloth/Kimi-K3-GGUF/Kimi-K3-UD-IQ1_M-00001-of-00011.gguf \
          --device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' -ngl 99 \
          --n-predict 4 --prompt "Hi" --temp 0 --single-turn
```

## Server-Based Testing

For all model tests, you can use llama-server instead of llama-cli:

```bash
export LLAMA_VM_GPU=mlx
export LLAMA_VM_VRAM=<size>
export LLAMA_VM_GGUF=/path/to/model.gguf
llama-server -m /path/to/model.gguf --device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' -ngl 99 --port 8123
```

Then test with curl:
```bash
curl -s http://127.0.0.1:8123/completion \
     -d '{"prompt":"Hi","n_predict":1,"temperature":0,"stream":false}'
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
