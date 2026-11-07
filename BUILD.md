# Building llama.vm

llama.vm supports three compute backends: ROCm (AMD GPUs), CUDA (NVIDIA GPUs), and MLX (Apple Silicon). This document describes how to build for each platform.

## Prerequisites

All platforms require:
- CMake 3.18+
- A C++17 compiler
- llama.cpp source tree with prebuilt ggml libraries

### Patch llama.cpp (all platforms)

Before building, apply the llama.vm patches to your llama.cpp tree — the ggml
type-66 stub plus the ForthLLM feature patch (`--forthllm-*` server flags +
KV-cache compaction) by default; `--pager-only` applies just the stub. See
[patches/README.md](patches/README.md):

```bash
cd /path/to/llama.vm
patches/apply-llama.cpp-patches.sh /path/to/llama.cpp
```

Then rebuild in **your platform's build directory** — the one your platform
section below configures: `build-hip` for ROCm, `build-cuda` for CUDA,
`build-mlx` for MLX (plain `build` for a CPU-only tree). If you have several
build directories, repeat this for each one you use:

```bash
LLAMA_BUILD=/path/to/llama.cpp/build-hip   # your platform's build dir
cmake --build "$LLAMA_BUILD" --target ggml-base ggml -j
cmake -S /path/to/llama.cpp -B "$LLAMA_BUILD"    # reconfigure (the ForthLLM patch changes CMakeLists)
cmake --build "$LLAMA_BUILD" --target llama-server test-forthllm -j
```

## ROCm (AMD GPUs)

### Build llama.cpp with ROCm support

```bash
cd /path/to/llama.cpp
cmake -S . -B build-hip \
  -DGGML_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip --target ggml-hip -j
```

### Build llama.vm

```bash
cd /path/to/llama.vm
cmake -S . -B build \
  -DLlamaVM_LLAMA_CPP_DIR=/path/to/llama.cpp \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Test

```bash
ctest --test-dir build --output-on-failure
```

### Run a model

```bash
export LD_LIBRARY_PATH=$PWD/build:/path/to/llama.cpp/build-hip/bin:/usr/local/lib
export GGML_BACKEND_PATH=$PWD/build/libggml-llamavm-gpu.so
export LLAMA_VM_GPU=hip
export LLAMA_VM_VRAM=3G
export LLAMA_VM_GGUF=/path/to/model.gguf

llama-server -m /path/to/model.gguf --device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' \
             -ngl 99 --load-mode none --port 8123
```

## CUDA (NVIDIA GPUs)

### Build llama.cpp with CUDA support

```bash
cd /path/to/llama.cpp
cmake -S . -B build-cuda \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --target ggml-cuda -j
```

### Build llama.vm

```bash
cd /path/to/llama.vm
cmake -S . -B build \
  -DLlamaVM_LLAMA_CPP_DIR=/path/to/llama.cpp \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Test

```bash
ctest --test-dir build --output-on-failure
```

### Run a model

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

To use a specific CUDA device, set `LLAMA_VM_GPU=cuda1` (second device), etc.

## MLX (Apple Silicon)

### Prerequisites

Install MLX via Homebrew:
```bash
brew install mlx
```

### MLX Backend

The MLX backend is included in the llama.cpp repository at `ggml/src/ggml-mlx/`.
No patch is needed.

### Build llama.cpp with MLX support

```bash
cd /path/to/llama.cpp
cmake -S . -B build-mlx \
  -DGGML_MLX=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/homebrew:/opt/homebrew/opt/libomp \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp" \
  -DOpenMP_CXX_LIBRARIES="/opt/homebrew/opt/libomp/lib/libomp.dylib"
cmake --build build-mlx --target ggml -j
```

### Build llama.vm

```bash
cd /path/to/llama.vm
cmake -S . -B build \
  -DLlamaVM_LLAMA_CPP_DIR=/path/to/llama.cpp \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Test

```bash
ctest --test-dir build --output-on-failure
```

### Run a model

```bash
export LD_LIBRARY_PATH=$PWD/build:/path/to/llama.cpp/build-mlx/bin:/opt/homebrew/lib
export GGML_BACKEND_PATH=$PWD/build/libggml-llamavm-gpu.dylib
export LLAMA_VM_GPU=mlx
export LLAMA_VM_VRAM=3G
export LLAMA_VM_GGUF=/path/to/model.gguf

llama-server -m /path/to/model.gguf --device LLAMA_VM_GPU -ot '.*=LLAMA_VM_GPU' \
             -ngl 99 --load-mode none --port 8123
```

## Troubleshooting

### Cannot find OpenMP on macOS

Install libomp via Homebrew and specify the path:
```bash
brew install libomp
cmake -S . -B build \
  -DLlamaVM_LLAMA_CPP_DIR=/path/to/llama.cpp \
  -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/libomp \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp" \
  -DOpenMP_CXX_LIBRARIES="/opt/homebrew/opt/libomp/lib/libomp.dylib"
```

### ggml-mlx target not found

The MLX backend is built as part of the main `ggml` target when `-DGGML_MLX=ON` is specified. Do not use `--target ggml-mlx`.

### Device selection

The `--device` flag must be set to `LLAMA_VM_GPU` (the wrapper device name). The physical GPU is chosen by the `LLAMA_VM_GPU` environment variable. Run `llama-server --list-devices` to see available devices.
