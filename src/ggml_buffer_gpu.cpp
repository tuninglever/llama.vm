// llama.vm GPU weight buffer + wrapper backend (LLAMA_VM_GPU).
//
// The LLAMA_VM_GPU buffer_type holds model weights that are larger than the
// operator-dictated VRAM window (LLAMA_VM_VRAM). The buffer's base address is the
// pager's VRAM tier (the window); tensors are placed by ggml at base + offset, but
// their data is not there until the wrapper backend faults it in.
//
// Two data modes:
//   capture (default): weights written by the loader (set_tensor) are captured into
//     the pager's RAM tier (LLAMA_VM_RAM) and swapped out (LLAMA_VM_SWAP) as it
//     fills, so the RAM tier can be far smaller than the model.
//   file (LLAMA_VM_GGUF=<model>): the pager is built from the GGUF file; the
//     loader's writes are ignored and weights fault straight from the file into the
//     VRAM window at compute time (no RAM tier or swap needed for the weights).
//
// The wrapper backend presents the inner compute backend (HIP/CUDA/Vulkan/CPU,
// chosen by LLAMA_VM_GPU) to llama.cpp. Its graph_compute splits the graph into
// batches: the longest run of consecutive ops whose unique weight tensors fit in
// the VRAM window. For each batch it faults those weights into the window (pager
// tensor_data), re-points tensor->data at the contiguous window blocks, and computes
// the batch as one real graph on the inner backend. Weights of an earlier batch may
// then be evicted, so the window only needs to hold one batch's weights (plus nothing
// else: activations live in the inner buft).

#include "ggml-backend-impl.h"
#include "ggml-impl.h" // struct ggml_cgraph + ggml_graph_view (internal, stable layout)
#include "vm_env.h"

#include "llama_vm/llama_vm.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <cstring>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    using namespace llama_vm;

    constexpr const char * GPU_NAME      = "LLAMA_VM_GPU";
    constexpr const char * GPU_BUFT_NAME = "LLAMA_VM_GPU";
    constexpr const char * GPU_DESC      = "llama.vm demand-paged weights";

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

    // inner (compute) backend: "auto" | "hip" | "cuda" | "vulkan" | "cpu"
    std::string gpu_target() {
        const char * e = std::getenv("LLAMA_VM_GPU");
        if (e != nullptr && e[0] != '\0') {
            return e;
        }
        return "auto";
    }

    // the buffer's context: the pager, the mode, and (capture mode) the tensor
    // name -> pager virtual address translation (ggml offsets have padding gaps,
    // the pager's virtual space is packed)
    struct ctx {
        pager * pg = nullptr;
        bool    file_mode = false;
        std::map<std::string, size_t> vaddr; // capture mode only
    };

    // probe whether a buffer from the given backend is writable from the host CPU
    // (an iGPU under the HMM), using a throwaway pager with a minimal window
    bool vram_host_writable(const std::string & backend) {
        config probe;
        probe.page_size     = buffer_page_size();
        probe.vram_capacity = 4 * 1024 * 1024;
        probe.vram_backend  = backend;
        try {
            pager * pp = new pager(probe);
            const bool r = pp->vram_cpu_accessible();
            delete pp;
            return r;
        } catch (const std::exception &) {
            return false;
        }
    }

    ctx * buf_ctx(ggml_backend_buffer_t buffer) {
        return static_cast<ctx *>(buffer->context);
    }

    // the pager backing the buffer (owned by the buffer)
    pager * buf_pager(ggml_backend_buffer_t buffer) {
        return buf_ctx(buffer)->pg;
    }

    // --------------------------- state (wiring) ---------------------------
    // one buffer type, one wrapper device, one registry. the inner compute device
    // is resolved when the registry is probed (all backends are loaded by then);
    // the iface cross-references are wired in wire_state() (called from the probe,
    // once all the static iface objects are initialized)

    struct state {
        ggml_backend_buffer_type buft{};
        ggml_backend_device      device{};      // compute (wrapper) device
        ggml_backend_device      mem_device{};  // memory-only device (buft name resolution)
        ggml_backend_reg         reg{};
        ggml_backend_dev_t       inner_dev = nullptr;
        bool                     wired     = false;
        bool                     loading   = false; // re-entrancy guard: ggml_backend_load_all_from_path
                                                     // can re-enter ggml_gpu_reg() via a shim's init
    };

    state & S() {
        static state s;
        return s;
    }

    void wire_state(); // defined below, after the iface objects

    static ggml_guid g_guid = { 0x6c, 0x76, 0x6d, 0x02, 0x00, 0x00, 0x00, 0x00,
                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x42 };
    ggml_guid_t gpu_guid() {
        return &g_guid;
    }

    // --------------------------- buffer (weights) -------------------------

    const char * buf_get_name(ggml_backend_buffer_type_t buft) {
        (void) buft;
        return GPU_BUFT_NAME;
    }

    size_t buf_get_alignment(ggml_backend_buffer_type_t buft) {
        (void) buft;
        return 32; // ggml TENSOR_ALIGNMENT
    }

    bool buf_is_host(ggml_backend_buffer_type_t buft) {
        (void) buft;
        return false; // the data lives in the VRAM window, faulted in at compute time
    }

    void buf_free(ggml_backend_buffer_t buffer) {
        ctx * c = buf_ctx(buffer);
        delete c->pg;
        delete c; // the buffer struct itself is freed by ggml
    }

    void * buf_get_base(ggml_backend_buffer_t buffer) {
        return buf_pager(buffer)->tier_base(tier::vram);
    }

    // translate (tensor, offset) to a pager virtual address; resolves views through
    // their base tensor (a view shares the base's buffer and region)
    size_t tensor_vaddr(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, size_t offset) {
        ctx * c = buf_ctx(buffer);
        const struct ggml_tensor * base = tensor->view_src != nullptr ? tensor->view_src : tensor;
        const char * name = ggml_get_name(base);
        size_t vaddr;
        if (c->file_mode) {
            const tensor_info * ti = c->pg->find_tensor(name);
            if (ti == nullptr) {
                throw std::runtime_error(std::string("llama.vm GPU buffer: tensor '") + name
                    + "' not found in the LLAMA_VM_GGUF model file");
            }
            vaddr = ti->vaddr;
        } else {
            auto it = c->vaddr.find(name);
            if (it == c->vaddr.end()) {
                throw std::runtime_error(std::string("llama.vm GPU buffer: tensor '") + name
                    + "' was not placed in the buffer");
            }
            vaddr = it->second;
        }
        if (tensor->view_src != nullptr) {
            vaddr += tensor->view_offs;
        }
        return vaddr + offset;
    }

    enum ggml_status buf_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
        ctx * c = buf_ctx(buffer);
        if (tensor->view_src != nullptr) {
            return GGML_STATUS_SUCCESS; // the base tensor carries the region
        }
        if (c->file_mode) {
            if (c->pg->find_tensor(ggml_get_name(tensor)) == nullptr) {
                std::fprintf(stderr, "llama.vm GPU buffer: tensor '%s' not found in the "
                                     "LLAMA_VM_GGUF model file\n", ggml_get_name(tensor));
                return GGML_STATUS_FAILED;
            }
            return GGML_STATUS_SUCCESS;
        }
        const int64_t ne[4] = {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]};
        try {
            c->vaddr[ggml_get_name(tensor)] = c->pg->virtual_size();
            c->pg->attach_captured_region(ggml_get_name(tensor), ggml_nbytes(tensor), ne, tensor->type);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "llama.vm GPU buffer: %s\n", e.what());
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    void buf_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                           uint8_t value, size_t offset, size_t size) {
        std::vector<uint8_t> fill(size, value);
        buf_pager(buffer)->write(tensor_vaddr(buffer, tensor, offset), fill.data(), size);
    }

    void buf_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                        const void * data, size_t offset, size_t size) {
        ctx * c = buf_ctx(buffer);
        if (c->file_mode) {
            return; // the data is in the model file; it is faulted in at compute time
        }
        buf_pager(buffer)->write(tensor_vaddr(buffer, tensor, offset), data, size);
    }

    void buf_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                        void * data, size_t offset, size_t size) {
        buf_pager(buffer)->read(tensor_vaddr(buffer, tensor, offset), data, size);
    }

    const struct ggml_backend_buffer_i buf_iface = {
        /* .free_buffer   = */ buf_free,
        /* .get_base      = */ buf_get_base,
        /* .init_tensor   = */ buf_init_tensor,
        /* .memset_tensor = */ buf_memset_tensor,
        /* .set_tensor    = */ buf_set_tensor,
        /* .get_tensor    = */ buf_get_tensor,
        /* .set_tensor_2d = */ nullptr,
        /* .get_tensor_2d = */ nullptr,
        /* .cpy_tensor    = */ nullptr, // fall back to the generic get+set path
        /* .clear         = */ nullptr,
        /* .reset         = */ nullptr,
    };

    ggml_backend_buffer_t buf_alloc(ggml_backend_buffer_type_t buft, size_t size) {
        const size_t vram_window = env_size("LLAMA_VM_VRAM");
        if (vram_window == 0) {
            std::fprintf(stderr, "llama.vm GPU buffer: LLAMA_VM_VRAM (the VRAM window in bytes, "
                                 "eg 2G) must be set; it is the only GPU memory the pager uses\n");
            return nullptr;
        }

        const char * gguf_e = std::getenv("LLAMA_VM_GGUF");
        const std::string gguf = (gguf_e != nullptr && gguf_e[0] != '\0') ? gguf_e : "";

        config cfg;
        cfg.page_size     = buffer_page_size();
        cfg.vram_capacity = align4k(vram_window);
        cfg.vram_backend  = gpu_target();
        cfg.prefetch      = false;

        pager * pg = nullptr;
        try {
            if (!gguf.empty()) {
                // file mode: fault weights straight from the model file. when the
                // window is host-writable (iGPU HMM) the pager reads the file straight
                // into it (one pass) and no RAM tier is needed; otherwise the RAM tier
                // (default: one window) stages the next batch's weights so a fault is a
                // one-pass RAM -> VRAM copy instead of a file read + copy
                const size_t ram = env_size("LLAMA_VM_RAM");
                if (ram != 0) {
                    cfg.ram_capacity = align4k(ram);
                } else if (vram_host_writable(cfg.vram_backend)) {
                    cfg.ram_capacity = 0;
                } else {
                    cfg.ram_capacity = cfg.vram_capacity;
                }
                cfg.prefetch     = true;
                pg = pager::from_gguf(gguf, cfg).release();
            } else {
                // capture mode: the loader's writes are captured into the RAM tier and
                // swapped out as it fills; the tier defaults to the whole model (no
                // paging) unless the operator dictates a smaller budget
                const size_t ram  = env_size("LLAMA_VM_RAM");
                const char * swap_e = std::getenv("LLAMA_VM_SWAP");
                const std::string swap = (swap_e != nullptr && swap_e[0] != '\0') ? swap_e : "";
                const size_t need = align4k(size);
                cfg.ram_capacity = ram != 0 ? align4k(ram) : need;
                if (cfg.ram_capacity < need && swap.empty()) {
                    std::fprintf(stderr, "llama.vm GPU buffer: the RAM tier (%zu bytes) is below "
                                         "the model size (%zu bytes), so set LLAMA_VM_SWAP to a "
                                         "swap file path (or LLAMA_VM_GGUF to page from the model "
                                         "file instead of capturing)\n", cfg.ram_capacity, need);
                    return nullptr;
                }
                cfg.swap_file = swap;
                pg = new pager(cfg);
                // the per-tensor captured regions are attached in buf_init_tensor
            }
        } catch (const std::exception & e) {
            std::fprintf(stderr, "llama.vm GPU buffer: %s\n", e.what());
            delete pg;
            return nullptr;
        }
        if (pg->tier_base(tier::vram) == nullptr) {
            std::fprintf(stderr, "llama.vm GPU buffer: failed to allocate the VRAM window (%s backend)\n",
                         gpu_target().c_str());
            delete pg;
            return nullptr;
        }

        ctx * c = new ctx;
        c->pg        = pg;
        c->file_mode = !gguf.empty();
        return ggml_backend_buffer_init(buft, buf_iface, c, size);
    }

    const struct ggml_backend_buffer_type_i buft_iface = {
        /* .get_name       = */ buf_get_name,
        /* .alloc_buffer   = */ buf_alloc,
        /* .get_alignment  = */ buf_get_alignment,
        /* .get_max_size   = */ nullptr,
        /* .get_alloc_size = */ nullptr,
        /* .is_host        = */ buf_is_host,
    };

    // --------------------------- wrapper backend --------------------------
    // wraps the inner compute backend; graph_compute faults the weights each op
    // reads into the VRAM window, re-points tensor->data, and delegates the op

    struct wctx {
        ggml_backend_t inner = nullptr;
    };

    const char * wb_get_name(ggml_backend_t backend) {
        (void) backend;
        return GPU_NAME;
    }

    void wb_free(ggml_backend_t backend) {
        wctx * w = static_cast<wctx *>(backend->context);
        if (w->inner != nullptr) {
            ggml_backend_free(w->inner);
        }
        delete w;
        delete backend;
    }

    void wb_synchronize(ggml_backend_t backend) {
        wctx * w = static_cast<wctx *>(backend->context);
        ggml_backend_synchronize(w->inner);
    }

    // debug heartbeat (LLAMA_VM_DEBUG=1): op progress to a separate file
    FILE * dbg_file() {
        static FILE * f = [] -> FILE * {
            const char * e = std::getenv("LLAMA_VM_DEBUG");
            if (e == nullptr || e[0] != '1') {
                return nullptr;
            }
            return std::fopen("/tmp/llamavm_debug.log", "w");
        }();
        return f;
    }

    // view/reshape/permute/transpose/none/empty: no kernel to run (the data is already
    // in place in the inner buft). the inner backend hangs if asked to compute one of
    // these as a single-node graph, so the wrapper skips them
    bool is_noop(const struct ggml_tensor * t) {
        return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
               t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
    }

    // debug context (LLAMA_VM_DEBUG=1): resolve_weights marks each tensor_data
    // ("t <tensor>"), and graph_compute logs one line per batch with the pager
    // stats. the last marker at a stall names the batch and, for a stall in
    // resolve, the tensor being faulted
    struct dbg_ctx {
        FILE * f = nullptr;
        int    n_nodes = 0;
    };

    // fault the minimal range a weight tensor `s` needs and return its data pointer.
    // a view (a MoE expert slice) is faulted as just that slice via tensor_data_range,
    // so only the selected expert's pages become resident; a plain tensor is faulted
    // whole. the returned pointer is the data for `s`.
    void * fault_weight(struct ggml_tensor * s, pager * pg) {
        if (s->view_src != nullptr) {
            return pg->tensor_data_range(ggml_get_name(s->view_src),
                                         (size_t) s->view_offs, (size_t) ggml_nbytes(s), tier::vram);
        }
        return pg->tensor_data(ggml_get_name(s), tier::vram);
    }

    // fault the weight tensors an op reads into the VRAM window and re-point their
    // data pointers at the contiguous window blocks
    void resolve_weights(struct ggml_tensor * node, long op, const dbg_ctx * d) {
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            struct ggml_tensor * s = node->src[j];
            if (s == nullptr || s->buffer == nullptr) {
                continue;
            }
            if (s->buffer->buft != &S().buft) {
                continue;
            }
            struct ggml_tensor * base = s->view_src != nullptr ? s->view_src : s;
            if (d != nullptr) {
                std::fprintf(d->f, "op %ld/%d %s: t %s\n", op, d->n_nodes,
                             ggml_get_name(node), ggml_get_name(base));
                std::fflush(d->f);
            }
            void * p = fault_weight(s, buf_pager(s->buffer));
            if (s->view_src != nullptr) {
                s->data = p; // p already points at the slice (region byte view_offs)
            } else {
                base->data = p;
            }
        }
    }

    // a weight reference: the base tensor name and, for a view, its slice within it
    struct wkey {
        std::string name;
        size_t      off;
        size_t      sz;
        bool operator==(const wkey & o) const { return name == o.name && off == o.off && sz == o.sz; }
    };

    // fault the weights `node` reads, pinning them for this batch only. returns false
    // when a fault could not fit the window (the op should start the next batch).
    // batch_pinned collects the pages to unpin after the batch; batch_faulted dedupes
    // weight references within a batch; batch_page_end tracks the highest faulted page
    // for prefetch staging. a view (a MoE expert slice) is faulted as just that slice,
    // so only the selected expert's pages become resident.
    bool fault_node_weights(struct ggml_tensor * node, pager * pg,
                            std::vector<size_t> & batch_pinned,
                            std::vector<wkey> & batch_faulted,
                            size_t & batch_page_end) {
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            struct ggml_tensor * s = node->src[j];
            if (s == nullptr || s->buffer == nullptr) {
                continue;
            }
            if (s->buffer->buft != &S().buft) {
                continue;
            }
            struct ggml_tensor * base = s->view_src != nullptr ? s->view_src : s;
            const std::string nm = ggml_get_name(base);
            const size_t off = s->view_src != nullptr ? (size_t) s->view_offs : 0;
            const size_t sz  = s->view_src != nullptr ? (size_t) ggml_nbytes(s) : (size_t) ggml_nbytes(base);
            const wkey key{nm, off, sz};
            if (std::find(batch_faulted.begin(), batch_faulted.end(), key) != batch_faulted.end()) {
                continue; // already faulted and pinned for this batch
            }
            const tensor_info * ti = pg->find_tensor(nm);
            if (ti == nullptr) {
                return false;
            }
            try {
                fault_weight(s, pg);
            } catch (const std::exception &) {
                return false; // the union does not fit the window: op starts the next batch
            }
            batch_faulted.push_back(key);
            const size_t p0 = pg->page_at(ti->vaddr + off);
            const size_t p1 = pg->page_at(ti->vaddr + off + sz - 1);
            for (size_t pid = p0; pid <= p1; ++pid) {
                pg->pin(pid);
                batch_pinned.push_back(pid);
            }
            if (p1 + 1 > batch_page_end) {
                batch_page_end = p1 + 1;
            }
        }
        return true;
    }

    // the pager backing the weight buffer (from any weight tensor in the graph)
    pager * graph_pager(struct ggml_cgraph * cgraph) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                const struct ggml_tensor * s = cgraph->nodes[i]->src[j];
                if (s != nullptr && s->buffer != nullptr && s->buffer->buft == &S().buft) {
                    return buf_pager(s->buffer);
                }
            }
        }
        return nullptr;
    }

    enum ggml_status wb_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
        wctx * w = static_cast<wctx *>(backend->context);
        FILE * df = dbg_file();
        static long n_ops = 0;
        dbg_ctx d;
        if (df != nullptr) {
            d.f = df;
            d.n_nodes = cgraph->n_nodes;
        }

        pager * pg = graph_pager(cgraph);
        if (pg == nullptr) {
            // no paged weights in the graph; compute it whole on the inner backend
            const enum ggml_status err = ggml_backend_graph_compute_async(w->inner, cgraph);
            if (err != GGML_STATUS_SUCCESS) {
                return err;
            }
            ggml_backend_synchronize(w->inner);
            return GGML_STATUS_SUCCESS;
        }

        const bool do_prefetch = pg->get_config().prefetch;
        if (do_prefetch) {
            // drop the previous graph's leftover prefetch work (stale ranges)
            pg->cancel_prefetch();
        }

        // batch compute: a batch is the longest run of consecutive ops whose unique
        // weight tensors can be resident in the VRAM window at once. the window holds
        // one batch's weights (activations live in the inner buft), so a weight is
        // resident only while its batch computes, and the pager may evict it as the
        // next batch faults in its weights. batching also removes the per-op stream
        // sync (the dominant per-op cost) and faults a weight shared by several ops
        // once per batch.
        //
        // a batch's weights are pinned while the batch is being faulted in, so a
        // later fault can never evict an earlier member (the arena can fragment even
        // when the union fits); if the window cannot hold the union, the fault throws
        // and the op starts a new batch instead
        std::vector<size_t> pinned;      // expert pages, unpinned after the batch
        std::vector<wkey> batch_faulted; // the batch's unique weight refs, in fault order
        using clock = std::chrono::steady_clock;
        int i = 0;
        while (i < cgraph->n_nodes) {
            pinned.clear();
            batch_faulted.clear();
            size_t batch_page_end = 0; // one past the batch's last weight page
            const clock::time_point t_fault0 = clock::now();
            int j = i;
            for (;;) {
                if (!fault_node_weights(cgraph->nodes[j], pg, pinned, batch_faulted, batch_page_end)) {
                    break; // op j's weights do not fit the window: it starts the next batch
                }
                if (++j == cgraph->n_nodes) {
                    break;
                }
            }
            if (j == i) {
                // op i's weights alone do not fit (a tensor larger than the window is
                // a misconfiguration); compute it on its own so tensor_data throws
                // with a descriptive error
                j = i + 1;
            }

            // re-point the batch's data pointers (the weights are already resident)
            for (int k = i; k < j; ++k) {
                resolve_weights(cgraph->nodes[k], n_ops + 1, df != nullptr ? &d : nullptr);
            }
            const double fault_ms =
                std::chrono::duration<double, std::milli>(clock::now() - t_fault0).count();

            // a single-node graph of a view/noop hangs the inner backend, so skip it;
            // noops inside a multi-node batch are part of a real graph and are fine
            enum ggml_status err = GGML_STATUS_SUCCESS;
            double compute_ms = 0.0;
            if (!((j - i) == 1 && is_noop(cgraph->nodes[i]))) {
                struct ggml_cgraph view = ggml_graph_view(cgraph, i, j);
                const clock::time_point t_comp0 = clock::now();
                err = ggml_backend_graph_compute_async(w->inner, &view);
                if (err == GGML_STATUS_SUCCESS) {
                    // while this batch computes (and as the next batch's faults free
                    // window space), stage the next batch's weights: its tensors follow
                    // this batch's pages in the virtual address space. a host-writable
                    // window (iGPU) is filled straight from the file; otherwise the RAM
                    // tier stages the copies
                    if (do_prefetch && j < cgraph->n_nodes && batch_page_end > 0) {
                        const size_t budget = pg->tier_capacity(tier::vram);
                        const tier pt = pg->vram_cpu_accessible() ? tier::vram : tier::ram;
                        const auto & ts = pg->tensors();
                        size_t staged = 0;
                        for (size_t k = 0; k < ts.size(); ++k) {
                            if (ts[k].page_begin < batch_page_end) {
                                continue; // this batch's (or an earlier) tensor
                            }
                            if (staged >= budget) {
                                break;
                            }
                            pg->prefetch_tensor(k, pt);
                            staged += ts[k].size;
                        }
                    }
                    ggml_backend_synchronize(w->inner);
                }
                compute_ms =
                    std::chrono::duration<double, std::milli>(clock::now() - t_comp0).count();
            }
            for (size_t pid : pinned) {
                pg->unpin(pid);
            }
            if (err != GGML_STATUS_SUCCESS) {
                return err;
            }

            n_ops += (j - i);
            if (df != nullptr) {
                const stats st = pg->get_stats();
                std::fprintf(df, "batch [%d..%d)/%d ops=%d weights=%llu faults=%llu evicts=%llu swap_in=%llu swap_out=%llu tier_copy=%llu fault_ms=%.1f compute_ms=%.1f\n",
                             i, j, cgraph->n_nodes, j - i, (unsigned long long) batch_faulted.size(),
                             (unsigned long long) st.n_fault,
                             (unsigned long long) st.n_evict,
                             (unsigned long long) st.n_swap_in,
                             (unsigned long long) st.n_swap_out,
                             (unsigned long long) st.n_tier_copy,
                             fault_ms, compute_ms);
                std::fflush(df);
            }
            i = j;
        }
        return GGML_STATUS_SUCCESS;
    }

    const struct ggml_backend_i wb_iface = {
        /* .get_name       = */ wb_get_name,
        /* .free           = */ wb_free,
        /* .set_tensor_async   = */ nullptr,
        /* .get_tensor_async   = */ nullptr,
        /* .set_tensor_2d_async= */ nullptr,
        /* .get_tensor_2d_async= */ nullptr,
        /* .cpy_tensor_async   = */ nullptr,
        /* .synchronize        = */ wb_synchronize,
        /* .graph_plan_create  = */ nullptr,
        /* .graph_plan_free    = */ nullptr,
        /* .graph_plan_update  = */ nullptr,
        /* .graph_plan_compute = */ nullptr,
        /* .graph_compute      = */ wb_graph_compute,
        /* .event_record       = */ nullptr,
        /* .event_wait         = */ nullptr,
        /* .graph_optimize     = */ nullptr,
    };

    ggml_backend_t wb_init(ggml_backend_dev_t dev, const char * params) {
        ggml_backend_t inner = ggml_backend_dev_init(S().inner_dev, params);
        if (inner == nullptr) {
            return nullptr;
        }
        wctx * w = new wctx;
        w->inner = inner;
        return new ggml_backend {
            /* .guid    = */ gpu_guid(),
            /* .iface   = */ wb_iface,
            /* .device  = */ dev,
            /* .context = */ w,
        };
    }

    // ------------------------------ device --------------------------------
    // presents the inner device to llama.cpp, plus the LLAMA_VM_GPU buffer type

    const char * dev_get_name(ggml_backend_dev_t dev) {
        (void) dev;
        return GPU_NAME;
    }
    const char * dev_get_description(ggml_backend_dev_t dev) {
        (void) dev;
        return GPU_DESC;
    }
    void dev_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
        (void) dev;
        // the pager uses only the operator-dictated window, not the full device memory
        const size_t window = env_size("LLAMA_VM_VRAM");
        if (free)  { *free = window; }
        if (total) { *total = window; }
    }
    enum ggml_backend_dev_type dev_get_type(ggml_backend_dev_t dev) {
        (void) dev;
        return ggml_backend_dev_type(S().inner_dev);
    }
    void dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
        (void) dev;
        const size_t window = env_size("LLAMA_VM_VRAM");
        props->name         = GPU_NAME;
        props->description  = GPU_DESC;
        props->memory_free  = window;
        props->memory_total = window;
        props->type         = ggml_backend_dev_type(S().inner_dev);
        props->device_id    = "llamavm"; // unique, so it is not deduplicated with the inner device
        props->caps.async                = false; // set_tensor falls back to the sync path
        props->caps.host_buffer          = true;
        props->caps.buffer_from_host_ptr = false;
        props->caps.events               = false;
        props->caps.mmap_support         = false;
    }
    ggml_backend_t dev_init_backend(ggml_backend_dev_t dev, const char * params) {
        return wb_init(dev, params);
    }
    ggml_backend_buffer_type_t dev_get_buffer_type(ggml_backend_dev_t dev) {
        (void) dev;
        return ggml_backend_dev_buffer_type(S().inner_dev); // activations use the inner buft
    }
    ggml_backend_buffer_type_t dev_get_host_buffer_type(ggml_backend_dev_t dev) {
        (void) dev;
        return ggml_backend_dev_host_buffer_type(S().inner_dev);
    }
    bool dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
        (void) dev;
        return ggml_backend_dev_supports_op(S().inner_dev, op);
    }
    bool dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
        (void) dev;
        if (buft == &S().buft) {
            return true; // the paged weight buffer
        }
        return ggml_backend_dev_supports_buft(S().inner_dev, buft);
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

    // ------------------------- memory device ----------------------------
    // a CPU-typed, memory-only device whose default buffer type is the
    // LLAMA_VM_GPU buft. It exists so that llama.cpp's tensor-buffer-override
    // parser (which maps each device's *default* buft name -> buft) can resolve
    // the name "LLAMA_VM_GPU" for -ot. It is never used for compute: CPU type
    // means llama.cpp never selects it as a model device, and the built-in CPU
    // device is registered before this one, so it is not picked as the CPU backend

    const char * memdev_get_name(ggml_backend_dev_t dev) {
        (void) dev;
        return "LLAMA_VM_GPU_MEM";
    }
    const char * memdev_get_description(ggml_backend_dev_t dev) {
        (void) dev;
        return GPU_DESC;
    }
    void memdev_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
        (void) dev;
        const size_t window = env_size("LLAMA_VM_VRAM");
        if (free)  { *free = window; }
        if (total) { *total = window; }
    }
    enum ggml_backend_dev_type memdev_get_type(ggml_backend_dev_t dev) {
        (void) dev;
        return GGML_BACKEND_DEVICE_TYPE_CPU;
    }
    void memdev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
        (void) dev;
        const size_t window = env_size("LLAMA_VM_VRAM");
        props->name         = "LLAMA_VM_GPU_MEM";
        props->description  = GPU_DESC;
        props->memory_free  = window;
        props->memory_total = window;
        props->type         = GGML_BACKEND_DEVICE_TYPE_CPU;
        props->device_id    = "llamavm-mem";
        props->caps.async                = false;
        props->caps.host_buffer          = false;
        props->caps.buffer_from_host_ptr = false;
        props->caps.events               = false;
        props->caps.mmap_support         = false;
    }
    const char * membe_get_name(ggml_backend_t backend) {
        (void) backend;
        return "LLAMA_VM_GPU_MEM";
    }
    void membe_free(ggml_backend_t backend) {
        delete backend;
    }
    enum ggml_status membe_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
        (void) backend;
        (void) cgraph;
        return GGML_STATUS_FAILED; // memory-only backend
    }
    const struct ggml_backend_i membe_iface = {
        /* .get_name       = */ membe_get_name,
        /* .free           = */ membe_free,
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
        /* .graph_compute      = */ membe_graph_compute,
        /* .event_record       = */ nullptr,
        /* .event_wait         = */ nullptr,
        /* .graph_optimize     = */ nullptr,
    };
    ggml_backend_t memdev_init_backend(ggml_backend_dev_t dev, const char * params) {
        (void) params;
        return new ggml_backend {
            /* .guid    = */ gpu_guid(),
            /* .iface   = */ membe_iface,
            /* .device  = */ dev,
            /* .context = */ nullptr,
        };
    }
    ggml_backend_buffer_type_t memdev_get_buffer_type(ggml_backend_dev_t dev) {
        (void) dev;
        return &S().buft; // the paged weight buffer
    }
    ggml_backend_buffer_type_t memdev_get_host_buffer_type(ggml_backend_dev_t dev) {
        (void) dev;
        return &S().buft;
    }
    bool memdev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
        (void) dev;
        (void) op;
        return false; // memory-only
    }
    bool memdev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
        (void) dev;
        return buft == &S().buft;
    }

    const struct ggml_backend_device_i memdev_iface = {
        /* .get_name             = */ memdev_get_name,
        /* .get_description      = */ memdev_get_description,
        /* .get_memory           = */ memdev_get_memory,
        /* .get_type             = */ memdev_get_type,
        /* .get_props            = */ memdev_get_props,
        /* .init_backend         = */ memdev_init_backend,
        /* .get_buffer_type      = */ memdev_get_buffer_type,
        /* .get_host_buffer_type = */ memdev_get_host_buffer_type,
        /* .buffer_from_host_ptr = */ nullptr,
        /* .supports_op          = */ memdev_supports_op,
        /* .supports_buft        = */ memdev_supports_buft,
        /* .offload_op           = */ nullptr,
        /* .event_new            = */ nullptr,
        /* .event_free           = */ nullptr,
        /* .event_synchronize    = */ nullptr,
    };

    // ------------------------------ registry ------------------------------

    const char * reg_get_name(ggml_backend_reg_t reg) {
        (void) reg;
        return GPU_NAME;
    }
    size_t reg_get_device_count(ggml_backend_reg_t reg) {
        (void) reg;
        return S().inner_dev != nullptr ? 2 : 0; // compute (wrapper) + memory-only
    }
    ggml_backend_dev_t reg_get_device(ggml_backend_reg_t reg, size_t index) {
        if (S().inner_dev == nullptr) {
            return nullptr;
        }
        state & s = S();
        if (index == 0) {
            return &s.device; // compute (wrapper) device
        }
        if (index == 1) {
            return &s.mem_device; // memory-only device (buft name resolution)
        }
        return nullptr;
    }

    const struct ggml_backend_reg_i reg_iface = {
        /* .get_name         = */ reg_get_name,
        /* .get_device_count = */ reg_get_device_count,
        /* .get_device       = */ reg_get_device,
        /* .get_proc_address = */ nullptr,
    };

    // --------------------------- wiring + probe ---------------------------

    void wire_state() {
        state & s = S();
        if (s.wired) {
            return;
        }
        s.buft.iface   = buft_iface;
        s.buft.device  = &s.mem_device; // the memory device is the buft's device, so -ot can name it
        s.buft.context = nullptr;

        s.device.iface    = dev_iface;
        s.device.reg      = &s.reg;
        s.device.context  = nullptr;

        s.mem_device.iface    = memdev_iface;
        s.mem_device.reg      = &s.reg;
        s.mem_device.context  = nullptr;

        s.reg.api_version = GGML_BACKEND_API_VERSION;
        s.reg.iface       = reg_iface;
        s.reg.context     = nullptr;
        s.wired           = true;
    }

    // find the inner compute device for the LLAMA_VM_GPU target
    ggml_backend_dev_t find_inner_device() {
        const std::string target = gpu_target();
        const char * prefix = nullptr;
        if (target == "hip") {
            prefix = "ROCm";
        } else if (target == "cuda") {
            prefix = "CUDA";
        } else if (target == "vulkan") {
            prefix = "Vulkan";
        } else if (target == "cpu") {
            prefix = "CPU";
        }
        if (prefix != nullptr) {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (dev == &S().device || dev == &S().mem_device) {
                    continue;
                }
                const char * nm = ggml_backend_dev_name(dev);
                if (nm != nullptr && std::strncmp(nm, prefix, std::strlen(prefix)) == 0) {
                    return dev;
                }
            }
            if (target == "hip") {
                // a HIP build registers its device as "ROCm"; retry with the bare name
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                    if (dev == &S().device || dev == &S().mem_device) {
                        continue;
                    }
                    const char * nm = ggml_backend_dev_name(dev);
                    if (nm != nullptr && std::strncmp(nm, "HIP", 3) == 0) {
                        return dev;
                    }
                }
            }
            return nullptr;
        }
        // auto: the first discrete or integrated GPU
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (dev == &S().device || dev == &S().mem_device) {
                continue;
            }
            const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
            if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                return dev;
            }
        }
        return nullptr;
    }

} // namespace

namespace llama_vm {

    ggml_backend_buffer_type_t ggml_gpu_buft() {
        wire_state();
        return &S().buft;
    }

    ggml_backend_reg_t ggml_gpu_reg() {
        wire_state();
        state & s = S();
        if (s.inner_dev == nullptr) {
            if (s.loading) {
                // re-entered while we are loading the inner backend (ggml_backend_load can
                // call back into ggml_gpu_reg via a shim); nothing to do, the outer frame
                // will finish resolving inner_dev
                return &s.reg;
            }
            s.loading = true;
            // load the inner backend library (eg. libggml-hip.so) from a separate build tree.
            // Load it directly by path rather than ggml_backend_load_all_from_path: that
            // helper re-reads GGML_BACKEND_PATH at its end, which would reload this shim and
            // register the wrapper reg before inner_dev is resolved (0 devices).
            const char * backend_dir = std::getenv("LLAMA_VM_BACKEND_DIR");
            if (backend_dir != nullptr && backend_dir[0] != '\0') {
                const std::string target = gpu_target();
                const char * prefixes[] = { "libggml-hip-", "libggml-cuda-", "libggml-vulkan-", "libggml-cpu-" };
                const char * prefix = nullptr;
                if (target == "hip")    prefix = prefixes[0];
                if (target == "cuda")   prefix = prefixes[1];
                if (target == "vulkan") prefix = prefixes[2];
                if (target == "cpu")    prefix = prefixes[3];
                if (prefix != nullptr) {
                    for (const auto & entry : std::filesystem::directory_iterator(backend_dir)) {
                        const std::string fn = entry.path().filename().string();
                        if (fn.rfind(prefix, 0) == 0 && fn.rfind(".so") == fn.size() - 3) {
                            ggml_backend_load(entry.path().string().c_str());
                            break;
                        }
                    }
                }
            }
            s.inner_dev = find_inner_device();
            s.loading   = false;
            if (s.inner_dev == nullptr) {
                std::fprintf(stderr, "llama.vm GPU backend: no %s device found (set LLAMA_VM_GPU to "
                                     "hip/cuda/vulkan/cpu, and make sure the backend library is loaded)\n",
                                     gpu_target().c_str());
            }
        }
        return &s.reg;
    }

    // access the pager backing a buffer created by this backend (nullptr otherwise);
    // the buffer owns the pager, so do not delete it
    const pager * gpu_buffer_pager(ggml_backend_buffer_t buffer) {
        if (buffer == nullptr || buffer->buft != &S().buft) {
            return nullptr;
        }
        return buf_ctx(buffer)->pg;
    }

} // namespace llama_vm

extern "C" {

    ggml_backend_reg_t llama_vm_ggml_gpu_reg(void) {
        return llama_vm::ggml_gpu_reg();
    }

}
