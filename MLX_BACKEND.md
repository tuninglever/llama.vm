# MLX Backend for llama.vm

The MLX backend enables demand-paged model execution on Apple Silicon using MLX as the compute backend.

## Architecture

The MLX backend (`llama.cpp/ggml/src/ggml-mlx/`) implements the ggml backend API using the MLX C++ API. It follows the same pattern as other ggml backends (CUDA, HIP, Metal) but uses MLX's graph compilation and execution model.

## Building

### Prerequisites

1. Install MLX:
   ```bash
   brew install mlx
   ```

2. Build llama.cpp with MLX support:
   ```bash
   cmake -S llama.cpp -B llama.cpp/build -DGGML_MLX=ON -DCMAKE_BUILD_TYPE=Release
   cmake --build llama.cpp/build --target ggml-mlx -j
   ```

### Building llama.vm

```bash
cmake -S . -B build -DGGML_MLX=ON
cmake --build build -j
```

## Usage

Set the `LLAMA_VM_GPU` environment variable to `mlx` to use the MLX backend:

```bash
export LLAMA_VM_GPU=mlx
export LLAMA_VM_VRAM=4G
export LLAMA_VM_GGUF=/path/to/model.gguf

llama-cli -m /path/to/model.gguf --device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' -ngl 99 \
          --n-predict 1 --prompt "Hi" --temp 0 --single-turn
```

## Notes

- MLX uses graph compilation, which is different from CUDA/HIP's kernel launch model
- The backend is registered with ggml's backend registry as "MLX"
- The pager's VRAM tier supports MLX as a backend option
- Demand paging works the same way as with other backends: weights are faulted into the VRAM window as needed

## Status

The MLX backend has been implemented with the following features:

1. **Backend structure**: Complete ggml backend API implementation
2. **Buffer management**: MLX allocator integration for device memory
3. **Device interface**: MLX GPU device registration
4. **Graph compute**: Operation dispatch with actual MLX calls for:
   - MUL_MAT (matrix multiplication via `mlx::core::matmul`)
   - SOFT_MAX (softmax via `mlx::core::softmax`)
   - ADD (addition via MLX operator overloading)
   - MUL (multiplication via MLX operator overloading)
   - DIV (division via MLX operator overloading)
   - RMS_NORM (RMS normalization via MLX operations)
   - SIN (via `mlx::core::sin`)
   - COS (via `mlx::core::cos`)
   - LOG (via `mlx::core::log`)
   - Unary operations (via `GGML_OP_UNARY` dispatch):
     - SILU (SiLU activation)
     - EXP (exponential)
     - TANH (hyperbolic tangent)
     - RELU (rectified linear unit)
     - GELU (Gaussian error linear unit)
     - SIGMOID (sigmoid)
   - ROPE (Rotary Position Embedding, with YaRN support)
   - CLAMP (clamp values)
   - MEAN (mean reduction)
   - SUM (sum reduction)
   - REPEAT (repeat tensor)
   - CONCAT (concatenate tensors)
   - TRANSPOSE (transpose tensor)
   - PERMUTE (permute tensor dimensions)
   - SCALE (scale tensor)
   - RESHAPE (reshape tensor)
5. **Synchronization**: MLX stream synchronization
6. **Tensor conversion**: Helper functions to convert between ggml tensors and MLX arrays (supports f32, f16, bf16, i32, i8 dtypes)

The MLX backend library (`libggml-mlx.dylib`) compiles successfully with:
- MLX library: `/opt/homebrew/lib/libmlx.dylib`
- MLX include dir: `/opt/homebrew/include`

### Remaining work

- Test with llama.vm's demand paging system (Phase 3)
- Validate with real models (Phase 3)
- Performance optimization (Phase 4)

See [NEXT_STEPS.md](NEXT_STEPS.md) for detailed priority order and implementation guidance.
