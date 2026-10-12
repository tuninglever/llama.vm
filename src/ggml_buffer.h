#pragma once

// llama.vm ggml buffer: a host ggml buffer_type backed by a llama_vm::pager.
//
// The buffer's memory is the pager's RAM tier (a linear, 32-byte aligned region).
// Tensors placed in it (by ggml's allocator) live at base + offset, and the pager
// maps those offsets onto its page table (each page at base + virtual_offset), so
// direct host access and the pager's read()/write() are consistent.
//
// This is the memory-backend foundation for integrating llama.vm with llama.cpp:
// weights can be routed here via llama_model_params.tensor_buft_overrides.

#include "llama_vm/llama_vm.h"

#include "ggml-backend.h"

namespace llama_vm {

    // the "LLAMA_VM" buffer type; safe to call repeatedly (returns the same object).
    // The buffer's pager page size defaults to 4 MiB and can be overridden with the
    // LLAMA_VM_GGML_PAGE_SIZE environment variable (bytes, power of two >= 4 KiB).
    ggml_backend_buffer_type_t ggml_buft();

    // the backend registry (one device)
    ggml_backend_reg_t ggml_backend_llamavm_reg();

    // register the LLAMA_VM backend with ggml so llama.cpp can see it; idempotent
    void register_backend();

    // access the pager backing a buffer created by this backend (nullptr otherwise);
    // the buffer owns the pager, so do not delete it
    const pager * buffer_pager(ggml_backend_buffer_t buffer);

    // the "LLAMA_VM_GPU" buffer type + wrapper backend (see ggml_buffer_gpu.cpp):
    // weights larger than the operator-dictated VRAM window (LLAMA_VM_VRAM), faulted
    // into the window op by op by the wrapper backend. LLAMA_VM_GPU selects the inner
    // compute backend (auto/hip/cuda/vulkan/cpu); capture mode pages the captured
    // weights through LLAMA_VM_RAM + LLAMA_VM_SWAP, file mode (LLAMA_VM_GGUF) faults
    // them straight from the model file
    ggml_backend_buffer_type_t ggml_gpu_buft();
    ggml_backend_reg_t         ggml_gpu_reg();

    // access the pager backing a LLAMA_VM_GPU buffer (nullptr otherwise)
    const pager * gpu_buffer_pager(ggml_backend_buffer_t buffer);

} // namespace llama_vm

// C entry points for llama.cpp / external consumers
extern "C" {
    ggml_backend_buffer_type_t llama_vm_ggml_buft(void);
    void                       llama_vm_ggml_register_backend(void);
    ggml_backend_reg_t         llama_vm_ggml_backend_reg(void);

    // most recent buffer created by this backend (nullptr if none); for tests/tools
    ggml_backend_buffer_t llama_vm_ggml_last_buffer(void);
    size_t                llama_vm_ggml_buffer_alloc_count(void);
}
