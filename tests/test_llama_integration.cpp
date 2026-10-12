#include "llama_vm/llama_vm.h"
#include "ggml_buffer.h"

#include "test-common.h"

#include "ggml-backend.h"
#include "llama.h"

// Integration test: load a real GGUF model through llama.cpp with every tensor
// routed to the LLAMA_VM buffer (via tensor_buft_overrides). Verifies that the
// buffer backend is accepted by llama.cpp's loader, that a buffer is created for
// the model's tensors, and that the weights actually land in the buffer's host
// memory (the read path the CPU kernels will use).
//
// Auto-skips when no model is available (LLAMA_VM_TEST_MODEL or a bundled one).

using namespace llama_vm;

int main() {
    // ------------------------------ locate model ---------------------------
    const char * env = std::getenv("LLAMA_VM_TEST_MODEL");
    std::string model = (env != nullptr && env[0] != '\0')
        ? env
        : "";
    {
        FILE * f = std::fopen(model.c_str(), "rb");
        if (f == nullptr) {
            std::printf("test_llama_integration: SKIP (no model at %s)\n", model.c_str());
            return 0;
        }
        std::fclose(f);
    }

    // --------------------------- route to our buffer -----------------------
    ggml_backend_buffer_type_t buft = llama_vm_ggml_buft();
    VM_CHECK(buft != nullptr);
    VM_CHECK(std::strcmp(ggml_backend_buft_name(buft), "LLAMA_VM") == 0);

    // match every tensor name -> all weights go to the LLAMA_VM buffer
    llama_model_tensor_buft_override overrides[] = {
        { ".*", buft },
        { nullptr, nullptr },
    };

    llama_model_params params = llama_model_default_params();
    params.tensor_buft_overrides = overrides;
    params.n_gpu_layers = 0; // no device offload; the (host) buffer holds the weights
    // no mmap: with mmap the loader swaps a host buffer_type for the stock CPU buft
    // (to avoid a redundant host copy); we want our pager-backed host buffer instead
    params.load_mode = LLAMA_LOAD_MODE_NONE;

    std::printf("loading %s into the LLAMA_VM buffer...\n", model.c_str());
    llama_model * m = llama_model_load_from_file(model.c_str(), params);
    VM_CHECK(m != nullptr);

    // ------------------------- the buffer was used -------------------------
    ggml_backend_buffer_t buf = llama_vm_ggml_last_buffer();
    VM_CHECK(buf != nullptr);
    const size_t bsize = ggml_backend_buffer_get_size(buf);
    VM_CHECK(bsize > 0);
    VM_CHECK(llama_vm_ggml_buffer_alloc_count() >= 1);
    std::printf("  buffer: %zu bytes (base=%p)\n", bsize, ggml_backend_buffer_get_base(buf));

    // the buffer must be backed by our pager
    const pager * pg = buffer_pager(buf);
    VM_CHECK(pg != nullptr);
    VM_CHECK(pg->find_tensor("buffer") != nullptr);

    // --------------------- the weights landed in memory --------------------
    // for a host buffer the loader writes straight into tensor->data (== base +
    // offset), so the buffer's host memory must now hold non-zero weight bytes
    void * base = ggml_backend_buffer_get_base(buf);
    VM_CHECK(base != nullptr);
    bool any_nonzero = false;
    const unsigned char * p = static_cast<const unsigned char *>(base);
    for (size_t i = 0; i < bsize; i += 256) {
        if (p[i] != 0) { any_nonzero = true; break; }
    }
    VM_CHECK(any_nonzero);

    std::printf("test_llama_integration: OK\n");
    llama_model_free(m);
    return 0;
}
