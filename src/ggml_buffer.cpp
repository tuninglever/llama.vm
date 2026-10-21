#include "ggml_buffer.h"
#include "ggml-backend-impl.h"
#include "vm_env.h"

#include "llama_vm/llama_vm.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    constexpr size_t       LLAMA_VM_BUF_ALIGN = 32; // ggml TENSOR_ALIGNMENT
    constexpr const char * LLAMA_VM_NAME      = "llama_vm";
    constexpr const char * LLAMA_VM_BUFT_NAME = "LLAMA_VM";
    constexpr const char * LLAMA_VM_DESC      = "llama.vm pager-backed host memory";

    // page size for the buffer's pager, overridable via env (bytes, power of 2 >= 4 KiB)
    size_t buffer_page_size() {
        const char * e = std::getenv("LLAMA_VM_GGML_PAGE_SIZE");
        if (e != nullptr && e[0] != '\0') {
            const size_t v = std::strtoull(e, nullptr, 10);
            if (v >= 4096 && (v & (v - 1)) == 0) {
                return v;
            }
        }
        return 4 * 1024 * 1024;
    }

    llama_vm::pager * buf_pager(ggml_backend_buffer_t buffer) {
        return static_cast<llama_vm::pager *>(buffer->context);
    }

    // virtual address of (tensor, offset) within the buffer's linear space
    size_t tensor_vaddr(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, size_t offset) {
        return static_cast<size_t>(static_cast<const char *>(tensor->data) -
                                  static_cast<const char *>(ggml_backend_buffer_get_base(buffer))) + offset;
    }

    static ggml_guid g_guid = { 0x6c, 0x76, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x00,
                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x42 };
    ggml_guid_t llamavm_guid() {
        return &g_guid;
    }

    // ------------------------------ buffer --------------------------------

    const char * buf_get_name(ggml_backend_buffer_type_t buft) {
        (void) buft;
        return LLAMA_VM_BUFT_NAME;
    }

    size_t buf_get_alignment(ggml_backend_buffer_type_t buft) {
        (void) buft;
        return LLAMA_VM_BUF_ALIGN;
    }

    bool buf_is_host(ggml_backend_buffer_type_t buft) {
        (void) buft;
        return true;
    }

    // introspection for tests / tools: the most recent buffer this backend created
    // and how many have been allocated (the backend is a process-wide singleton)
    ggml_backend_buffer_t g_last_buffer = nullptr;
    size_t                g_n_buffers  = 0;

    void buf_free(ggml_backend_buffer_t buffer) {
        if (g_last_buffer == buffer) {
            g_last_buffer = nullptr;
        }
        delete buf_pager(buffer); // the buffer struct itself is freed by ggml
    }

    void * buf_get_base(ggml_backend_buffer_t buffer) {
        return buf_pager(buffer)->tier_base(llama_vm::tier::ram);
    }

    void buf_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                           uint8_t value, size_t offset, size_t size) {
        std::vector<uint8_t> fill(size, value);
        buf_pager(buffer)->write(tensor_vaddr(buffer, tensor, offset), fill.data(), size);
    }

    void buf_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                        const void * data, size_t offset, size_t size) {
        buf_pager(buffer)->write(tensor_vaddr(buffer, tensor, offset), data, size);
    }

    void buf_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                        void * data, size_t offset, size_t size) {
        buf_pager(buffer)->read(tensor_vaddr(buffer, tensor, offset), data, size);
    }

    bool buf_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
        (void) buffer;
        if (ggml_backend_buffer_is_host(src->buffer)) {
            std::memcpy(dst->data, src->data, ggml_nbytes(src));
            return true;
        }
        return false;
    }

    void buf_clear(ggml_backend_buffer_t buffer, uint8_t value) {
        // bulk operation: a direct host fill is valid for a host buffer and avoids a
        // size-byte staging allocation
        std::memset(buf_get_base(buffer), value, buffer->size);
    }

    const struct ggml_backend_buffer_i buf_iface = {
        /* .free_buffer   = */ buf_free,
        /* .get_base      = */ buf_get_base,
        /* .init_tensor   = */ nullptr,
        /* .memset_tensor = */ buf_memset_tensor,
        /* .set_tensor    = */ buf_set_tensor,
        /* .get_tensor    = */ buf_get_tensor,
        /* .set_tensor_2d = */ nullptr,
        /* .get_tensor_2d = */ nullptr,
        /* .cpy_tensor    = */ buf_cpy_tensor,
        /* .clear         = */ buf_clear,
        /* .reset         = */ nullptr,
    };

    ggml_backend_buffer_t buf_alloc(ggml_backend_buffer_type_t buft, size_t size) {
        // a host buffer is never paged, so the RAM tier must cover the whole buffer
        // (round the capacity up to the arena alignment; the buffer size is unchanged).
        // the operator can dictate the budget with LLAMA_VM_RAM: a host buffer cannot
        // page, so a budget below the model size is rejected instead of silently
        // malloc'ing the model
        const size_t need   = llama_vm::align4k(size);
        const size_t budget = llama_vm::env_size("LLAMA_VM_RAM");
        if (budget != 0 && budget < need) {
            std::fprintf(stderr, "llama.vm buffer: LLAMA_VM_RAM budget (%zu bytes) is below the "
                                 "model size (%zu bytes); a host buffer cannot page, use the "
                                 "LLAMA_VM_GPU buffer (with a VRAM window) for a model larger than RAM\n",
                         budget, need);
            return nullptr;
        }

        llama_vm::config cfg;
        cfg.page_size     = buffer_page_size();
        cfg.ram_capacity  = budget != 0 ? std::min(budget, need) : need;
        cfg.vram_capacity = 0;
        cfg.prefetch      = false;

        llama_vm::pager * pg = nullptr;
        try {
            pg = new llama_vm::pager(cfg);
            pg->attach_host_region("buffer", size);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "llama.vm buffer: %s\n", e.what());
            delete pg;
            return nullptr;
        }
        if (pg->tier_base(llama_vm::tier::ram) == nullptr) {
            std::fprintf(stderr, "llama.vm buffer: failed to allocate RAM tier\n");
            delete pg;
            return nullptr;
        }
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, buf_iface, pg, size);
        g_last_buffer = buffer;
        g_n_buffers++;
        return buffer;
    }

    // alloc_buffer_n/get_alloc_size_n exist only in the newer ggml buffer type
    // interface (GGML_BACKEND_API_VERSION >= 3); the initializers are positional
    const struct ggml_backend_buffer_type_i buft_iface = {
        /* .get_name       = */ buf_get_name,
        /* .alloc_buffer   = */ buf_alloc,
#if GGML_BACKEND_API_VERSION >= 3
        /* .alloc_buffer_n = */ nullptr,
#endif
        /* .get_alignment  = */ buf_get_alignment,
        /* .get_max_size   = */ nullptr,
        /* .get_alloc_size = */ nullptr,
#if GGML_BACKEND_API_VERSION >= 3
        /* .get_alloc_size_n = */ nullptr,
#endif
        /* .is_host        = */ buf_is_host,
    };

    // --------------------------- backend (stream) -------------------------
    // memory-only: no compute

    const char * be_get_name(ggml_backend_t backend) {
        (void) backend;
        return LLAMA_VM_NAME;
    }
    void be_free(ggml_backend_t backend) {
        delete backend;
    }
    enum ggml_status be_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
        (void) backend;
        (void) cgraph;
        return GGML_STATUS_FAILED; // memory-only backend
    }

    const struct ggml_backend_i be_iface = {
        /* .get_name       = */ be_get_name,
        /* .free           = */ be_free,
        /* .set_tensor_async   = */ nullptr,
        /* .get_tensor_async   = */ nullptr,
        /* .set_tensor_2d_async= */ nullptr,
        /* .get_tensor_2d_async= */ nullptr,
        /* .cpy_tensor_async   = */ nullptr,
        /* .synchronize        = */ nullptr,
        /* .graph_plan_create  = */ nullptr,
        /* .graph_plan_free    = */ nullptr,
        /* .graph_plan_update  = */ nullptr,
        /* .graph_plan_compute = */ nullptr,
        /* .graph_compute      = */ be_graph_compute,
        /* .event_record       = */ nullptr,
        /* .event_wait         = */ nullptr,
        /* .graph_optimize     = */ nullptr,
    };

    ggml_backend_t be_init(ggml_backend_dev_t dev) {
        return new ggml_backend {
            /* .guid    = */ llamavm_guid(),
            /* .iface   = */ be_iface,
            /* .device  = */ dev,
            /* .context = */ nullptr,
        };
    }

    // ------------------------------ device --------------------------------
    // dev->context carries a pointer to the buffer type (wired in the state ctor)

    const char * dev_get_name(ggml_backend_dev_t dev) {
        (void) dev;
        return LLAMA_VM_NAME;
    }
    const char * dev_get_description(ggml_backend_dev_t dev) {
        (void) dev;
        return LLAMA_VM_DESC;
    }
    void dev_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
        (void) dev;
        if (free)  { *free = 0; }
        if (total) { *total = 0; } // allocated per-buffer, not a fixed pool
    }
    enum ggml_backend_dev_type dev_get_type(ggml_backend_dev_t dev) {
        (void) dev;
        return GGML_BACKEND_DEVICE_TYPE_CPU;
    }
    void dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
        (void) dev;
        props->name         = LLAMA_VM_NAME;
        props->description  = LLAMA_VM_DESC;
        props->memory_free  = 0;
        props->memory_total = 0;
        props->type         = GGML_BACKEND_DEVICE_TYPE_CPU;
        props->device_id    = nullptr;
        props->caps.async                = false;
        props->caps.host_buffer          = true;
        props->caps.buffer_from_host_ptr = false;
        props->caps.events               = false;
        props->caps.mmap_support         = false;
    }
    ggml_backend_t dev_init_backend(ggml_backend_dev_t dev, const char * params) {
        (void) params;
        return be_init(dev);
    }
    ggml_backend_buffer_type_t dev_get_buffer_type(ggml_backend_dev_t dev) {
        return static_cast<ggml_backend_buffer_type_t>(dev->context);
    }
    ggml_backend_buffer_type_t dev_get_host_buffer_type(ggml_backend_dev_t dev) {
        return static_cast<ggml_backend_buffer_type_t>(dev->context);
    }
    bool dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
        (void) dev;
        (void) op;
        return false; // memory-only
    }
    bool dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
        return buft == static_cast<ggml_backend_buffer_type_t>(dev->context);
    }

    const struct ggml_backend_device_i dev_iface = {
        /* .get_name             = */ dev_get_name,
        /* .get_description      = */ dev_get_description,
        /* .get_memory           = */ dev_get_memory,
        /* .get_type             = */ dev_get_type,
        /* .get_props            = */ dev_get_props,
        /* .init_backend         = */ dev_init_backend,
        /* .get_buffer_type      = */ dev_get_buffer_type,
        /* .get_host_buffer_type = */ dev_get_host_buffer_type,
        /* .buffer_from_host_ptr = */ nullptr,
        /* .supports_op          = */ dev_supports_op,
        /* .supports_buft        = */ dev_supports_buft,
        /* .offload_op           = */ nullptr,
        /* .event_new            = */ nullptr,
        /* .event_free           = */ nullptr,
        /* .event_synchronize    = */ nullptr,
    };

    // ------------------------------ registry ------------------------------
    // reg->context carries a pointer to the device (wired in the state ctor)

    const char * reg_get_name(ggml_backend_reg_t reg) {
        (void) reg;
        return LLAMA_VM_NAME;
    }
    size_t reg_get_device_count(ggml_backend_reg_t reg) {
        (void) reg;
        return 1;
    }
    ggml_backend_dev_t reg_get_device(ggml_backend_reg_t reg, size_t index) {
        if (index != 0) {
            return nullptr;
        }
        return static_cast<ggml_backend_dev_t>(reg->context);
    }

    const struct ggml_backend_reg_i reg_iface = {
        /* .get_name         = */ reg_get_name,
        /* .get_device_count = */ reg_get_device_count,
        /* .get_device       = */ reg_get_device,
        /* .get_proc_address = */ nullptr,
    };

    // --------------------------- state (wiring) ---------------------------
    // one buffer type, one device, one registry; cross-references are set in the
    // constructor (avoids static-init-order issues). buft.device -> device,
    // device.context -> buft, reg.context -> device

    struct state {
        ggml_backend_buffer_type buft;
        ggml_backend_device      device;
        ggml_backend_reg         reg;

        state() {
            buft.iface   = buft_iface;
            buft.device  = &device;
            buft.context = nullptr;

            device.iface    = dev_iface;
            device.reg      = &reg;
            device.context  = &buft;

            reg.api_version = GGML_BACKEND_API_VERSION;
            reg.iface       = reg_iface;
            reg.context     = &device;
        }
    };

    state & S() {
        static state s;
        return s;
    }

} // namespace

namespace llama_vm {

    ggml_backend_buffer_type_t ggml_buft() {
        return &S().buft;
    }

    ggml_backend_reg_t ggml_backend_llamavm_reg() {
        return &S().reg;
    }

    void register_backend() {
        ggml_backend_register(&S().reg);
    }

    const pager * buffer_pager(ggml_backend_buffer_t buffer) {
        if (buffer == nullptr || buffer->buft != &S().buft) {
            return nullptr;
        }
        return static_cast<const pager *>(buffer->context);
    }

} // namespace llama_vm

extern "C" {

    ggml_backend_buffer_type_t llama_vm_ggml_buft(void) {
        return llama_vm::ggml_buft();
    }

    void llama_vm_ggml_register_backend(void) {
        llama_vm::register_backend();
    }

    ggml_backend_reg_t llama_vm_ggml_backend_reg(void) {
        return llama_vm::ggml_backend_llamavm_reg();
    }

    // most recent buffer created by this backend (nullptr if none); for tests/tools
    ggml_backend_buffer_t llama_vm_ggml_last_buffer(void) {
        return g_last_buffer;
    }

    // how many buffers this backend has created so far
    size_t llama_vm_ggml_buffer_alloc_count(void) {
        return g_n_buffers;
    }

}

#ifdef GGML_BACKEND_DL
    // allow loading as a standalone backend library (ggml-backend-llamavm.*)
    GGML_BACKEND_DL_IMPL(llama_vm_ggml_backend_reg)
#endif
