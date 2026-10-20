#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace llama_vm {

    // memory hierarchy tiers
    enum class tier {
        ram,
        vram,
    };

    struct config {
        // page size in bytes (default 4 MiB)
        size_t page_size = 4 * 1024 * 1024;

        // RAM tier capacity in bytes, 0 = disabled
        size_t ram_capacity = 0;

        // VRAM tier capacity in bytes, 0 = disabled
        size_t vram_capacity = 0;

        // VRAM backend: "auto" | "hip" | "cuda" | "vulkan" | "cpu" (the first device of
        // that type), or an exact device name (eg. "CUDA1") to pick a specific device.
        // "cpu" allocates a plain host buffer, useful for testing paging logic without a GPU
        std::string vram_backend = "auto";

        // swap file path, empty = swap disabled
        // dirty pages are written back here when evicted; clean pages fall back to the source file
        std::string swap_file;

        // prefetch the next sequential page in a background thread when a page faults in
        bool prefetch = false;
    };

    struct tensor_info {
        std::string name;
        std::string file;
        size_t      file_offset = 0;
        size_t      size        = 0;
        int64_t     ne[4]       = {1, 1, 1, 1};
        int32_t     ggml_type   = -1;
        size_t      page_begin  = 0; // [page_begin, page_end) in the global page table
        size_t      page_end    = 0;
        size_t      vaddr       = 0; // start of the region in the virtual address space
    };

    struct page_info {
        size_t vaddr        = 0; // start of the page in the virtual address space
        size_t file_offset  = 0; // offset of the page in its source file
        size_t size         = 0; // bytes covered by this page
        bool   dirty        = false;
        bool   pinned       = false;
        bool   in_ram       = false;
        bool   in_vram      = false;
    };

    struct stats {
        uint64_t n_resolve     = 0; // resolve_page / tensor_data calls
        uint64_t n_fault       = 0; // pages loaded from the source file or swap
        uint64_t n_tier_copy   = 0; // pages loaded from another tier
        uint64_t n_evict       = 0;
        uint64_t n_swap_in     = 0; // pages loaded from the swap file
        uint64_t n_swap_out    = 0; // pages written back to the swap file
        uint64_t bytes_file    = 0; // bytes read from source files
        uint64_t bytes_tier    = 0; // bytes copied between tiers
        uint64_t bytes_swap_in = 0;
        uint64_t bytes_swap_out = 0;
    };

    // the result of sizing the tiers for a model (see plan_capacity)
    struct capacity_plan {
        size_t model_size     = 0; // total model bytes the plan is for
        size_t ram_capacity   = 0; // proposed RAM tier capacity
        size_t vram_capacity  = 0; // proposed VRAM tier capacity
        size_t total_capacity = 0; // ram_capacity + vram_capacity
        bool   fits           = false; // the whole model can be cached in the tiers
    };

    // capacity governor: decide the tier capacities for a model of `model_size` bytes,
    // given the available RAM and VRAM (bytes). RAM is filled first, then VRAM with the
    // remainder of the model. Each capacity is 4 KiB aligned (the arena's alignment) and
    // the total is capped at the capacity the model actually needs: the arena rounds each
    // page's slice up to 4 KiB, so that need is model_size rounded up to 4 KiB (it can
    // slightly exceed model_size for a non-aligned last page). `fits` is true when the
    // whole model can be held in the tiers (no paging required — paging still works if
    // it is not). `page_size` is accepted for API stability but does not change the result.
    inline capacity_plan plan_capacity(size_t model_size, size_t available_ram, size_t available_vram,
                                       size_t page_size = 4 * 1024 * 1024) {
        (void) page_size;
        capacity_plan p;
        p.model_size = model_size;
        const auto u4k = [](size_t v) { return ((v + 4095) / 4096) * 4096; };
        const auto lo  = [](size_t a, size_t b) { return a < b ? a : b; };
        const size_t need = u4k(model_size); // capacity required to cache the whole model
        const size_t ram  = lo(u4k(available_ram), need);
        const size_t left = need - ram;
        const size_t vram = lo(u4k(available_vram), left);
        p.ram_capacity   = ram;
        p.vram_capacity  = vram;
        p.total_capacity = ram + vram;
        p.fits           = (ram + vram) >= need;
        return p;
    }

    class pager {
    public:
        explicit pager(const config & cfg);
        ~pager();

        pager(const pager &)            = delete;
        pager & operator=(const pager &) = delete;

        // create a pager from a GGUF model file, each tensor becomes a region
        static std::unique_ptr<pager> from_gguf(const std::string & path, const config & cfg);

        // model metadata captured by from_gguf
        const std::string & arch() const;
        const std::string & name() const;

        // attach a byte range of an arbitrary file as a named region
        void attach_region(const std::string & name, const std::string & file, size_t file_offset, size_t size,
                            const int64_t * ne = nullptr, int32_t ggml_type = -1);

        // attach a region with no backing file: the data is written in via write()
        // (or the buffer's set_tensor) and read back via read(). the RAM tier capacity
        // must cover the whole region (host regions are never evicted), so this is the
        // basis for a pager-backed host buffer
        void attach_host_region(const std::string & name, size_t size,
                                const int64_t * ne = nullptr, int32_t ggml_type = -1);

        // attach a region with no backing file whose data is captured by write()
        // (eg. model weights written into a GPU buffer at load time). unlike a host
        // region it may exceed the tier capacity: once the data is written in, its
        // pages are paged like file-backed pages (LRU eviction; a dirty page evicted
        // from its last tier is swapped out, so a swap file must be configured for a
        // region larger than the tiers)
        void attach_captured_region(const std::string & name, size_t size,
                                    const int64_t * ne = nullptr, int32_t ggml_type = -1);

        // base address of a tier's backing buffer; nullptr if the tier is disabled
        void * tier_base(tier t) const;

        // true when the VRAM tier's buffer can be written from the host CPU (a host
        // buffer, or an iGPU buffer under the HMM); the pager then reads source files
        // straight into the window in one pass
        bool vram_cpu_accessible() const;

        size_t virtual_size() const;
        size_t n_pages() const;

        // map a virtual address to a page id, throws std::out_of_range
        size_t page_at(size_t vaddr) const;

        const std::vector<tensor_info> & tensors() const;
        const tensor_info * find_tensor(const std::string & name) const;

        page_info get_page_info(size_t page_id) const;

        // demand paging: make sure the page is resident in the given tier and return a pointer
        // to its data inside the tier buffer. treat it as read-only: to modify a page use
        // write(), which also marks it dirty. the pointer stays valid until the page is evicted,
        // which can happen on any later pager operation that needs space, so copy the data out
        // if it must outlive the call; or use read() which is always safe.
        //
        // the pointer is only dereferenceable from the host when the tier's buffer is
        // host-accessible: always for the RAM tier, and for the VRAM tier only when
        // vram_cpu_accessible() is true (a host buffer or an iGPU under the HMM). on a
        // discrete GPU the VRAM tier is device memory: the pointer must not be
        // dereferenced from host code (the data is copied with the backend's tensor
        // set/get, which is what the pager uses internally)
        void * resolve_page(size_t page_id, tier t);

        // contiguous view of a whole tensor in the given tier, faults in all of its pages.
        // the tier must be able to hold the whole tensor at once. same guarantees as resolve_page
        void * tensor_data(const std::string & name, tier t);

        // contiguous view of a byte sub-range [offset, offset+size) of a tensor in the
        // given tier, faulting in only the pages that cover it. this is how a MoE expert
        // slice (a view into a large expert tensor) is faulted without pulling in the
        // whole tensor: only the selected expert's pages become resident. the returned
        // pointer corresponds to the tensor byte at `offset` (not the tensor's start).
        // same guarantees as tensor_data.
        void * tensor_data_range(const std::string & name, size_t offset, size_t size, tier t);

        bool is_resident(size_t page_id, tier t) const;

        // safe copy based access, never invalidated by evictions
        size_t read(size_t vaddr, void * dst, size_t len);
        // modify data and mark the covered pages dirty; a dirty page that is evicted from its
        // last tier is written back to the swap file (a swap file must be configured)
        size_t write(size_t vaddr, const void * src, size_t len);

        // keep a page resident, protected from eviction
        void pin(size_t page_id);
        void unpin(size_t page_id);

        // best effort, only fills free space, never evicts; returns true if the page is resident afterwards
        bool prefetch_page(size_t page_id, tier t);

        // enqueue a range of pages for background prefetch (requires cfg.prefetch, which
        // also starts the prefetch worker). best effort: already-resident pages are
        // skipped; when the tier is full the worker makes room by evicting pages that
        // are also resident in the other tier (pure staging duplicates) and skips the rest
        void prefetch_range(size_t page_begin, size_t page_end, tier t);

        // enqueue a whole tensor for background prefetch (requires cfg.prefetch). the
        // tensor is staged as one contiguous block in the tier, so a later tensor_data
        // of it reuses the block without copying; best effort as prefetch_range
        void prefetch_tensor(size_t tensor_id, tier t);

        // drop all pending prefetch work (the worker finishes whatever page it is on)
        void cancel_prefetch();

        stats get_stats() const;
        void  reset_stats();
        const config & get_config() const;

        // tier bookkeeping
        size_t tier_used(tier t) const;
        size_t tier_capacity(tier t) const;

    public:
        struct impl;

    private:
        std::unique_ptr<impl> p;
    };

} // namespace llama_vm
