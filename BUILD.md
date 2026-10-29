# Building llama.vm

llama.vm supports three compute backends: ROCm (AMD GPUs), CUDA (NVIDIA GPUs), and MLX (Apple Silicon). This document describes how to build for each platform.

## Prerequisites

All platforms require:
- CMake 3.18+
- A C++17 compiler
- llama.cpp source tree with prebuilt ggml libraries

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

### Apply the MLX patch to llama.cpp

```bash
cd /path/to/llama.cpp
git apply /path/to/llama.vm/patches/llama-cpp-mlx-patch.patch
```

### Build llama.cpp with MLX support

```bash
cd /path/to/llama.cpp
cmake -S . -B build-mlx \
  -DGGML_MLX=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/homebrew
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
export GGML_BACKEND_PATH=$PWD/build/libggml-llamavm-gpu.so
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
