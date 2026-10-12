// Standalone dlopen-able backend shim (libggml-llamavm-cpu.so).
//
// This is a thin wrapper so that llama.cpp can load the llama.vm buffer backend
// via the standard ggml backend-loading path (ggml_backend_load / GGML_BACKEND_PATH).
// The actual backend and its state live in libllama_vm.so; this shim only provides
// the ggml_backend_init entry point, so there is exactly one backend instance.

#define GGML_BACKEND_DL
#include "ggml-backend-impl.h"

// provided by libllama_vm.so (see ggml_buffer.cpp)
extern "C" ggml_backend_reg_t llama_vm_ggml_backend_reg(void);

GGML_BACKEND_DL_IMPL(llama_vm_ggml_backend_reg)
