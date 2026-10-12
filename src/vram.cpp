#include "vram.h"
#include "pager-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace llama_vm {

    static struct ggml_tensor * vram_view(vram_tier & v, size_t off, size_t size) {
        auto * ctx = static_cast<struct ggml_context *>(v.ctx);
        auto * t   = static_cast<struct ggml_tensor *>(v.copy_tensor);
        // the pre-allocated tensor covers the whole buffer, just point it at the slice
        t->data   = static_cast<char *>(v.base) + off;
        (void) ctx;
        (void) size;
        return t;
    }

    bool vram_tier_init(vram_tier & v, const config & cfg) {
        if (cfg.vram_capacity == 0 || cfg.vram_capacity % arena::ALIGN != 0) {
            return false;
        }

        // allow loading backend libraries (e.g. libggml-hip.so) from a separate build tree
        const char * backend_dir = std::getenv("LLAMA_VM_BACKEND_DIR");
        if (backend_dir != nullptr && backend_dir[0] != '\0') {
            ggml_backend_load_all_from_path(backend_dir);
        }

        // ggml_backend_init_by_name matches a *device* name (e.g. "ROCm0"), so enumerate
        // the devices and pick one whose name has the requested prefix. A HIP build
        // reports its device as "ROCm" (GGML_CUDA_NAME); the APU case is an iGPU.
        auto find_backend = [](const char * prefix) -> struct ggml_backend * {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                const char * nm = ggml_backend_dev_name(dev);
                if (nm != nullptr && std::strncmp(nm, prefix, std::strlen(prefix)) == 0) {
                    return ggml_backend_init_by_name(nm, nullptr);
                }
            }
            return nullptr;
        };
        struct ggml_backend * backend = nullptr;
        if (cfg.vram_backend == "auto") {
            backend = ggml_backend_init_best();
        } else if (cfg.vram_backend == "hip") {
            backend = find_backend("ROCm");
            if (backend == nullptr) {
                backend = find_backend("HIP");
            }
        } else if (cfg.vram_backend == "cuda") {
            backend = find_backend("CUDA");
        } else if (cfg.vram_backend == "vulkan") {
            backend = find_backend("Vulkan");
        } else if (cfg.vram_backend == "cpu") {
            backend = find_backend("CPU");
        } else {
            return false;
        }
        if (backend == nullptr) {
            return false;
        }

        struct ggml_backend_buffer * buffer = ggml_backend_alloc_buffer(backend, cfg.vram_capacity);
        if (buffer == nullptr) {
            ggml_backend_free(backend);
            return false;
        }

        void * base = ggml_backend_buffer_get_base(buffer);
        if (base == nullptr) {
            ggml_backend_buffer_free(buffer);
            ggml_backend_free(backend);
            return false;
        }

        // context holding one tensor that spans the whole buffer, used as a copy view
        struct ggml_init_params params = {1024 * 1024, nullptr, true};
        struct ggml_context * ctx = ggml_init(params);
        if (ctx == nullptr) {
            ggml_backend_buffer_free(buffer);
            ggml_backend_free(backend);
            return false;
        }
        const int64_t ne = (static_cast<int64_t>(cfg.vram_capacity) + 3) / 4;
        struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
        if (t == nullptr) {
            ggml_free(ctx);
            ggml_backend_buffer_free(buffer);
            ggml_backend_free(backend);
            return false;
        }
        t->buffer = buffer;
        t->data   = base;

        v.base      = base;
        v.capacity  = cfg.vram_capacity;
        v.is_host   = ggml_backend_buffer_is_host(buffer);
        // an iGPU has no separate device memory: its buffers live in system memory
        // (HMM), so the host can write them directly. LLAMA_VM_VRAM_DIRECT_IO=0 forces
        // the portable scratch path
        v.cpu_accessible = v.is_host;
        if (!v.cpu_accessible) {
            auto buft = ggml_backend_buffer_get_type(buffer);
            if (buft != nullptr) {
                auto dev = ggml_backend_buft_get_device(buft);
                if (dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    v.cpu_accessible = true;
                }
            }
        }
        const char * dio = std::getenv("LLAMA_VM_VRAM_DIRECT_IO");
        if (dio != nullptr && dio[0] == '0') {
            v.cpu_accessible = false;
        }
        v.backend   = backend;
        v.buffer    = buffer;
        v.ctx       = ctx;
        v.copy_tensor = t;
        v.ar.init(base, cfg.vram_capacity);
        return true;
    }

    void vram_tier_free(vram_tier & v) {
        if (v.ctx != nullptr) {
            ggml_free(static_cast<struct ggml_context *>(v.ctx));
            v.ctx = nullptr;
        }
        if (v.buffer != nullptr) {
            ggml_backend_buffer_free(static_cast<struct ggml_backend_buffer *>(v.buffer));
            v.buffer = nullptr;
        }
        if (v.backend != nullptr) {
            ggml_backend_free(static_cast<struct ggml_backend *>(v.backend));
            v.backend = nullptr;
        }
        v.ar.release();
        v.base     = nullptr;
        v.capacity = 0;
        v.copy_tensor = nullptr;
    }

    void vram_copy_in(vram_tier & v, size_t off, size_t size, const void * host) {
        if (off + size > v.capacity) {
            throw std::out_of_range("vram copy in out of range");
        }
        if (v.is_host) {
            std::memcpy(static_cast<char *>(v.base) + off, host, size);
            return;
        }
        auto * t = vram_view(v, off, size);
        ggml_backend_tensor_set(t, host, 0, size);
    }

    void vram_copy_out(vram_tier & v, size_t off, size_t size, void * host) {
        if (off + size > v.capacity) {
            throw std::out_of_range("vram copy out out of range");
        }
        if (v.is_host) {
            std::memcpy(host, static_cast<char *>(v.base) + off, size);
            return;
        }
        auto * t = vram_view(v, off, size);
        ggml_backend_tensor_get(t, host, 0, size);
    }

} // namespace llama_vm
