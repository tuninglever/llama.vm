#pragma once

#include "llama_vm/llama_vm.h"
#include "arena.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace llama_vm {

    struct page {
        size_t file_offset = 0;
        size_t size        = 0;

        // dirty: data in tiers is newer than the source file and the swap file
        // from_swap: canonical copy lives in the swap file (not the source file)
        // written: a captured page has had its data written in (it then exists in a
        // tier or in the swap file and can be faulted back in)
        bool dirty     = false;
        bool from_swap = false;
        bool written   = false;

        // pin is reference-counted: pin() increments and unpin() decrements, so
        // independent pins (e.g. the same page faulted for two batches) stack
        // without an unpin clearing the other pin. pinned is true iff pin_count > 0.
        uint32_t pin_count = 0;

        // SIZE_MAX = not resident
        size_t ram_off  = SIZE_MAX;
        size_t vram_off = SIZE_MAX;
    };

    struct region {
        std::string name;
        std::string file;
        size_t      file_offset = 0;
        size_t      size        = 0;
        int         fd          = -1;
        // host region: no backing file; the data is written in (eg. a ggml buffer)
        // and must never be evicted, so the tier capacity must cover the whole region
        bool        is_host     = false;
        // captured region: no backing file, but the data is written in (eg. weights
        // captured by a GPU buffer at load time) and the region is paged like a
        // file-backed one: pages may be evicted, a dirty last copy is swapped out
        bool        is_captured = false;
    };

    struct vram_tier {
        void * base      = nullptr;
        size_t capacity  = 0;
        arena  ar;
        bool   is_host   = false;
        // the buffer is writable from the host CPU (a host buffer, or an iGPU buffer
        // under the HMM where device memory is system memory): the pager can read
        // source files straight into it, one pass instead of file -> scratch -> device
        bool   cpu_accessible = false;

        // ggml objects (opaque here, handled in vram.cpp)
        void * backend      = nullptr; // ggml_backend_t
        void * buffer       = nullptr; // ggml_backend_buffer_t
        void * ctx          = nullptr; // ggml_context *
        void * copy_tensor  = nullptr; // ggml_tensor *
    };

    class pager::impl {
    public:
        explicit impl(const config & cfg);
        ~impl();

        // owning copy: callers may pass temporaries
        config cfg;

        // page table
        std::vector<page>     pages;
        std::vector<size_t>   page_starts; // byte offset of each page in the virtual address space
        std::vector<size_t>   page_region; // page id -> region index
        std::vector<region>   regions;
        std::vector<tensor_info> tensors;
        std::map<std::string, size_t> tensor_by_name;
        size_t                virtual_size = 0;

        // metadata from GGUF
        std::string arch_name;
        std::string model_name;

        // RAM tier
        char *     ram_base = nullptr;
        arena      ram_ar;
        std::list<size_t> lru_ram; // MRU at front
        std::vector<std::list<size_t>::iterator> lru_ram_it;

        // VRAM tier
        vram_tier vram;
        std::list<size_t> lru_vram;
        std::vector<std::list<size_t>::iterator> lru_vram_it;

        // bit 0: page is in the RAM LRU list, bit 1: in the VRAM LRU list
        std::vector<uint8_t> lru_flags;

        // swap file layout:
        //   [0)   uint64 magic      [8) uint64 version    [16) uint64 page_size
        //   [24)  uint64 n_pages   [32) bitmap: ceil(n_pages/8) bytes
        //   then n_pages * page_size slots
        // a bitmap bit set means the slot holds the newest copy of that page (it was
        // swapped out and not superseded), so the bit must persist across sessions for
        // crash recovery
        static constexpr uint64_t SWAP_MAGIC    = 0x4C564D31ULL; // "LVM1"
        static constexpr uint64_t SWAP_VERSION  = 2;
        static constexpr uint64_t SWAP_MAX_PAGES = 1u << 24;

        int         swap_fd = -1;
        bool        swap_reused = false; // an existing valid swap file was loaded
        bool        swap_ready  = false; // the swap file layout matches the page table
        uint64_t    swap_file_pages = 0; // n_pages declared by the loaded file
        std::vector<uint8_t> swap_valid;
        size_t swap_bitmap_len() const { return (pages.size() + 7) / 8; }
        // file layout: [32-byte header][slots][validity bitmap]. Slots are at a fixed
        // offset (independent of page count) so attaching regions after the swap file
        // is sized cannot shift them; the bitmap lives after the slots.
        size_t swap_slot_off(size_t page_id) const { return 32 + page_id * cfg.page_size; }
        size_t swap_bitmap_off() const { return 32 + pages.size() * cfg.page_size; }
        bool   swap_bit(size_t page_id) const {
            const size_t idx = page_id / 8;
            return idx < swap_valid.size() && ((swap_valid[idx] >> (page_id % 8)) & 1);
        }
        void   swap_set_bit(size_t page_id) {
            const size_t idx = page_id / 8;
            if (idx >= swap_valid.size()) {
                swap_valid.resize(idx + 1, 0);
            }
            swap_valid[idx] |= 1u << (page_id % 8);
        }
        // size the file to the (now complete) page table and persist the header + bitmap;
        // recovered bits are only kept if the file's page count matches exactly
        void   swap_ensure_ready();
        void   swap_write_header();
        void   swap_sync();

        // scratch for host side copies (page_size bytes)
        char * scratch = nullptr;

        // stats
        stats st;

        // prefetch worker. a task is a page or a whole tensor; tensor tasks are
        // staged as one contiguous arena block so a later gather (tensor_data) finds
        // the pages already contiguous and reuses the block without copying
        struct prefetch_task {
            size_t id; // page id, or tensor id when is_tensor
            tier   t;
            bool   is_tensor = false;
        };
        std::thread           worker;
        std::mutex            qmutex;
        std::condition_variable qcv;
        std::deque<prefetch_task> q;
        bool                  quit = false;

        mutable std::mutex mutex;

        // ---- internal, all with mutex held unless noted ----

        arena &          arena_of(tier t);
        const arena &    arena_of(tier t) const;
        char *           base_of(tier t);
        std::list<size_t> & lru_of(tier t);
        std::vector<std::list<size_t>::iterator> & lru_it_of(tier t);
        tier other(tier t) const;

        bool resident(size_t page_id, tier t) const;
        size_t slice_off(size_t page_id, tier t) const;
        void set_resident(size_t page_id, tier t, size_t off);

        // gather pages [a, b) of tier t into one contiguous arena block; returns the
        // tier offset of the block's first byte (page a's data). every covered page
        // becomes resident in the block (already-resident pages are reused). caller
        // holds the mutex.
        size_t gather_block(size_t a, size_t b, tier t);

        void lru_touch(tier t, size_t page_id);
        void lru_detach(tier t, size_t page_id);

        // load the page into tier t; assumes capacity was ensured; returns pointer to data
        void * load_into(size_t page_id, tier t);
        // load the page data directly at arena offset off (no allocation)
        void load_into_at(size_t page_id, tier t, size_t off);
        void  ensure_capacity(tier t, size_t need);
        void  evict_one(tier t, size_t need);
        // make room in tier t by evicting blocks whose resident pages are all also
        // resident in the other tier (clean staging duplicates, no swap needed);
        // returns false when no such block can satisfy `need`
        bool  evict_redundant(tier t, size_t need);

        bool read_full(int fd, void * dst, size_t off, size_t len);
        bool write_full(int fd, const void * src, size_t off, size_t len);

        void worker_loop();
        void prefetch_enq(size_t id, tier t, bool is_tensor);
        // stage a whole tensor in tier t as one contiguous block (best effort);
        // called with the mutex held
        void prefetch_tensor(size_t tensor_id, tier t);

        int region_fd_for(size_t page_id) const;
    };

} // namespace llama_vm
