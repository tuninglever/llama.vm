#include "pager-impl.h"
#include "vram.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace llama_vm {

    static void require(bool cond, const char * msg) {
        if (!cond) {
            throw std::runtime_error(msg);
        }
    }

    static bool read_full_fd(int fd, void * dst, size_t off, size_t len) {
        auto * out = static_cast<char *>(dst);
        size_t done = 0;
        while (done < len) {
            const ssize_t n = pread(fd, out + done, len - done, static_cast<off_t>(off + done));
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            if (n == 0) {
                return false;
            }
            done += static_cast<size_t>(n);
        }
        return true;
    }

    static bool write_full_fd(int fd, const void * src, size_t off, size_t len) {
        const auto * in = static_cast<const char *>(src);
        size_t done = 0;
        while (done < len) {
            const ssize_t n = pwrite(fd, in + done, len - done, static_cast<off_t>(off + done));
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            done += static_cast<size_t>(n);
        }
        return true;
    }

    // ------------------------------ impl --------------------------------

    pager::impl::impl(const config & cfg) : cfg(cfg) {
        require(cfg.page_size >= arena::ALIGN && (cfg.page_size & (cfg.page_size - 1)) == 0,
                "page_size must be a power of two of at least 4 KiB");
        require(cfg.ram_capacity != 0 || cfg.vram_capacity != 0,
                "at least one of the RAM or VRAM tiers must be enabled");
        if (cfg.ram_capacity != 0) {
            require(cfg.ram_capacity % arena::ALIGN == 0, "ram_capacity must be a multiple of 4096");
            void * base = nullptr;
            require(posix_memalign(&base, arena::ALIGN, cfg.ram_capacity) == 0, "failed to allocate RAM tier");
            require(ram_ar.init(base, cfg.ram_capacity), "failed to init RAM arena");
            ram_base = static_cast<char *>(base);
        }
        if (cfg.vram_capacity != 0) {
            require(vram_tier_init(vram, cfg), "failed to init VRAM tier");
        }
        if (!cfg.swap_file.empty()) {
            swap_fd = open(cfg.swap_file.c_str(), O_RDWR | O_CREAT, 0644);
            require(swap_fd >= 0, "failed to open swap file");
            // if a previous session left a valid swap file, keep its slots (crash recovery)
            uint64_t hdr[4] = {0, 0, 0, 0};
            if (read_full_fd(swap_fd, hdr, 0, sizeof(hdr))
                && hdr[0] == SWAP_MAGIC && hdr[1] == SWAP_VERSION
                && hdr[2] == cfg.page_size && hdr[3] > 0 && hdr[3] <= SWAP_MAX_PAGES) {
                const size_t blen = (hdr[3] + 7) / 8;
                swap_valid.assign(blen, 0);
                if (read_full_fd(swap_fd, swap_valid.data(), 32 + hdr[3] * cfg.page_size, blen)) {
                    swap_reused = true;
                    swap_file_pages = hdr[3];
                } else {
                    swap_valid.assign(blen, 0);
                }
            }
        }
        scratch = static_cast<char *>(std::malloc(cfg.page_size));
        require(scratch != nullptr, "failed to allocate scratch buffer");
        if (cfg.prefetch) {
            worker = std::thread([this] { worker_loop(); });
        }
    }

    pager::impl::~impl() {
        {
            std::lock_guard<std::mutex> lock(qmutex);
            quit = true;
        }
        qcv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        vram_tier_free(vram);
        if (ram_base != nullptr) {
            std::free(ram_base);
        }
        if (scratch != nullptr) {
            std::free(scratch);
        }
        if (swap_fd >= 0) {
            close(swap_fd);
        }
        for (auto & [path, fd] : file_fds) {
            (void) path;
            close(fd);
        }
    }

    arena & pager::impl::arena_of(tier t) {
        return t == tier::ram ? ram_ar : vram.ar;
    }
    const arena & pager::impl::arena_of(tier t) const {
        return t == tier::ram ? ram_ar : vram.ar;
    }
    char * pager::impl::base_of(tier t) {
        return t == tier::ram ? ram_base : static_cast<char *>(vram.base);
    }
    std::list<size_t> & pager::impl::lru_of(tier t) {
        return t == tier::ram ? lru_ram : lru_vram;
    }
    std::vector<std::list<size_t>::iterator> & pager::impl::lru_it_of(tier t) {
        return t == tier::ram ? lru_ram_it : lru_vram_it;
    }
    tier pager::impl::other(tier t) const {
        return t == tier::ram ? tier::vram : tier::ram;
    }

    bool pager::impl::resident(size_t page_id, tier t) const {
        const page & pg = pages[page_id];
        return t == tier::ram ? pg.ram_off != SIZE_MAX : pg.vram_off != SIZE_MAX;
    }

    size_t pager::impl::slice_off(size_t page_id, tier t) const {
        const page & pg = pages[page_id];
        return t == tier::ram ? pg.ram_off : pg.vram_off;
    }

    void pager::impl::set_resident(size_t page_id, tier t, size_t off) {
        page & pg = pages[page_id];
        if (t == tier::ram) {
            pg.ram_off = off;
        } else {
            pg.vram_off = off;
        }
    }

    bool pager::impl::data_exists(size_t page_id) const {
        const page & pg = pages[page_id];
        if (pg.written || resident(page_id, tier::ram) || resident(page_id, tier::vram)) {
            return true;
        }
        // the recovered bitmap is only trustworthy once it is known to describe this
        // exact page table (swap_ensure_ready settles it lazily)
        return swap_fd >= 0 && swap_reused && swap_file_pages == pages.size() && swap_bit(page_id);
    }

    static uint8_t lru_flag(tier t) {
        return t == tier::ram ? 1 : 2;
    }

    void pager::impl::lru_touch(tier t, size_t page_id) {
        auto & lru    = lru_of(t);
        auto & lru_it = lru_it_of(t);
        const uint8_t flag = lru_flag(t);
        if (lru_flags[page_id] & flag) {
            lru.splice(lru.begin(), lru, lru_it[page_id]);
        } else {
            lru.push_front(page_id);
            lru_it[page_id] = lru.begin();
            lru_flags[page_id] |= flag;
        }
    }

    void pager::impl::lru_detach(tier t, size_t page_id) {
        auto & lru    = lru_of(t);
        auto & lru_it = lru_it_of(t);
        const uint8_t flag = lru_flag(t);
        if (!(lru_flags[page_id] & flag)) {
            return;
        }
        lru.erase(lru_it[page_id]);
        lru_flags[page_id] &= ~flag;
    }

    int pager::impl::region_fd_for(size_t page_id) const {
        const size_t rid = page_region[page_id];
        return regions[rid].fd;
    }

    bool pager::impl::read_full(int fd, void * dst, size_t off, size_t len) {
        return read_full_fd(fd, dst, off, len);
    }

    bool pager::impl::write_full(int fd, const void * src, size_t off, size_t len) {
        return write_full_fd(fd, src, off, len);
    }

    void pager::impl::swap_ensure_ready() {
        if (swap_ready) {
            return;
        }
        const size_t blen = swap_bitmap_len();
        // recovered bits are only meaningful if they describe this exact page table
        if (!swap_reused || swap_file_pages != pages.size()) {
            swap_valid.assign(blen, 0);
            swap_reused = false;
        } else if (swap_valid.size() != blen) {
            std::vector<uint8_t> nv(blen, 0);
            const size_t keep = std::min(blen, swap_valid.size());
            std::memcpy(nv.data(), swap_valid.data(), keep);
            swap_valid = std::move(nv);
        }
        swap_write_header();
        const size_t end = swap_bitmap_off() + swap_bitmap_len();
        require(ftruncate(swap_fd, static_cast<off_t>(end)) == 0, "failed to size swap file");
        swap_ready = true;
    }

    void pager::impl::swap_write_header() {
        uint64_t hdr[4] = {SWAP_MAGIC, SWAP_VERSION, cfg.page_size, pages.size()};
        require(write_full_fd(swap_fd, hdr, 0, 32), "failed to write swap header");
        require(write_full_fd(swap_fd, swap_valid.data(), swap_bitmap_off(), swap_valid.size()),
                "failed to write swap bitmap");
    }

    void pager::impl::swap_sync() {
        if (fdatasync(swap_fd) != 0) {
            throw std::runtime_error(std::string("failed to sync swap file: ") + std::strerror(errno));
        }
    }

    void pager::impl::evict_one(tier t, size_t need) {
        auto & lru = lru_of(t);
        auto & ar  = arena_of(t);
        const tier o = other(t);

        if (ar.capacity() < need) {
            throw std::runtime_error("tier capacity is too small for this allocation: need "
                + std::to_string(need) + " bytes, capacity " + std::to_string(ar.capacity()) + " bytes");
        }

        // A gathered tensor's pages share one arena block, so the block is the unit of
        // eviction: freeing it takes every resident page it holds. two passes: first evict
        // blocks that need no swap write-back, then dirty ones
        for (int pass = 0; pass < 2; ++pass) {
            for (auto it = lru.rbegin(); it != lru.rend(); ++it) {
                const size_t v = *it;
                // host pages have no backing file and cannot be re-faulted, so they
                // are never evicted (treat them like pinned pages)
                if (pages[v].pin_count > 0 || regions[page_region[v]].is_host) {
                    continue;
                }
                const size_t base  = ar.alloc_base(slice_off(v, t));
                const size_t bsize = ar.size_of(base);

                bool blocked    = false; // a pinned / host co-tenant in the block
                bool needs_swap = false; // a dirty last-copy page in the block
                for (size_t u = 0; u < pages.size(); ++u) {
                    if (!resident(u, t)) {
                        continue;
                    }
                    const size_t ou = slice_off(u, t);
                    if (ou < base || ou >= base + bsize) {
                        continue;
                    }
                    if (pages[u].pin_count > 0 || regions[page_region[u]].is_host) {
                        blocked = true;
                        break;
                    }
                    if (pages[u].dirty && !resident(u, o)) {
                        needs_swap = true;
                    }
                }
                if (blocked || (pass == 0) == needs_swap) {
                    continue;
                }
                if (needs_swap) {
                    require(swap_fd >= 0, "cannot evict a dirty page: no swap file configured");
                    swap_ensure_ready();
                }

                // write back the dirty last-copy pages of the block; the slots and the
                // validity bitmap must be durable before the in-memory copies drop
                std::vector<size_t> swapped;
                for (size_t u = 0; u < pages.size(); ++u) {
                    if (!resident(u, t) || !(pages[u].dirty && !resident(u, o))) {
                        continue;
                    }
                    const size_t ou = slice_off(u, t);
                    if (ou < base || ou >= base + bsize) {
                        continue;
                    }
                    char * host = scratch;
                    if (t == tier::ram) {
                        host = ram_base + ou;
                    } else {
                        vram_copy_out(vram, ou, pages[u].size, scratch);
                    }
                    if (!write_full(swap_fd, host, swap_slot_off(u), pages[u].size)) {
                        throw std::runtime_error(std::string("swap write-back failed (")
                            + std::strerror(errno) + ") for " + cfg.swap_file);
                    }
                    swap_set_bit(u);
                    swapped.push_back(u);
                    st.n_swap_out++;
                    st.bytes_swap_out += pages[u].size;
                }
                if (!swapped.empty()) {
                    swap_write_header();
                    swap_sync();
                }
                for (const size_t u : swapped) {
                    pages[u].dirty     = false;
                    pages[u].from_swap = true;
                }

                // free the whole block and detach all of its resident pages
                ar.free(base);
                for (size_t u = 0; u < pages.size(); ++u) {
                    if (!resident(u, t)) {
                        continue;
                    }
                    const size_t ou = slice_off(u, t);
                    if (ou < base || ou >= base + bsize) {
                        continue;
                    }
                    set_resident(u, t, SIZE_MAX);
                    lru_detach(t, u);
                    st.n_evict++;
                }
                return;
            }
            if (pass == 0 && swap_fd < 0) {
                require(false, "cannot make space in tier: only dirty pages remain and no swap file is configured");
            }
        }
        require(false, "cannot make space in tier: all pages are pinned");
    }

    void pager::impl::ensure_capacity(tier t, size_t need) {
        auto & ar = arena_of(t);
        // alloc needs one contiguous free block, so evict until the largest free block
        // is big enough (or until nothing is evictable, which makes evict_one throw)
        while (ar.max_free() < need) {
            evict_one(t, need);
        }
    }

    bool pager::impl::evict_redundant(tier t, size_t need) {
        auto & ar = arena_of(t);
        const tier o = other(t);
        for (;;) {
            if (ar.max_free() >= need) {
                return true;
            }
            // one victim per pass, then rescan: the pass detaches co-tenant entries of
            // the same LRU list, so the scan cannot outlive its iterators
            size_t victim = SIZE_MAX;
            for (auto it = lru_of(t).rbegin(); it != lru_of(t).rend(); ++it) {
                const size_t v = *it;
                if (pages[v].pin_count > 0 || pages[v].dirty || regions[page_region[v]].is_host) {
                    continue;
                }
                if (!resident(v, o)) {
                    continue; // the page's only copy; never evicted here
                }
                const size_t base  = ar.alloc_base(slice_off(v, t));
                const size_t bsize = ar.size_of(base);
                // a block sized exactly for its page holds no co-tenants
                if (base == slice_off(v, t) && bsize == pages[v].size) {
                    victim = v;
                    break;
                }
                bool blocked = false; // a pinned / dirty / host / sole-copy co-tenant
                for (size_t u = 0; u < pages.size(); ++u) {
                    if (!resident(u, t)) {
                        continue;
                    }
                    const size_t ou = slice_off(u, t);
                    if (ou < base || ou >= base + bsize) {
                        continue;
                    }
                    if (pages[u].pin_count > 0 || pages[u].dirty || regions[page_region[u]].is_host
                        || !resident(u, o)) {
                        blocked = true;
                        break;
                    }
                }
                if (!blocked) {
                    victim = v;
                    break;
                }
            }
            if (victim == SIZE_MAX) {
                return false;
            }
            const size_t base  = ar.alloc_base(slice_off(victim, t));
            const size_t bsize = ar.size_of(base);
            ar.free(base);
            for (size_t u = 0; u < pages.size(); ++u) {
                if (!resident(u, t)) {
                    continue;
                }
                const size_t ou = slice_off(u, t);
                if (ou < base || ou >= base + bsize) {
                    continue;
                }
                set_resident(u, t, SIZE_MAX);
                lru_detach(t, u);
                st.n_evict++;
            }
        }
    }

    void pager::impl::load_into_at(size_t page_id, tier t, size_t off) {
        page & pg = pages[page_id];
        const tier o = other(t);
        if (swap_fd >= 0) {
            // settle recovered swap bits (adopt or discard) before consulting them
            swap_ensure_ready();
        }
        if (resident(page_id, o)) {
            // one pass: copy straight between the other tier and the destination slice
            if (t == tier::ram) {
                vram_copy_out(vram, pg.vram_off, pg.size, ram_base + off);
            } else {
                vram_copy_in(vram, off, pg.size, ram_base + pg.ram_off);
            }
            st.n_tier_copy++;
            st.bytes_tier += pg.size;
        } else if (resident(page_id, t) && slice_off(page_id, t) != off) {
            // same tier, different location (gather path); the slice holds the newest data
            if (t == tier::ram) {
                std::memcpy(ram_base + off, ram_base + pg.ram_off, pg.size);
            } else {
                vram_copy_out(vram, pg.vram_off, pg.size, scratch);
                vram_copy_in(vram, off, pg.size, scratch);
            }
            st.n_tier_copy++;
            st.bytes_tier += pg.size;
        } else if (pg.from_swap || swap_fd >= 0 && swap_bit(page_id)) {
            require(swap_fd >= 0, "page marked from swap but swap file is gone");
            if (t == tier::ram) {
                require(read_full(swap_fd, ram_base + off, swap_slot_off(page_id), pg.size), "swap read failed");
            } else {
                require(read_full(swap_fd, scratch, swap_slot_off(page_id), pg.size), "swap read failed");
                vram_copy_in(vram, off, pg.size, scratch);
            }
            pg.from_swap = true;
            st.n_swap_in++;
            st.n_fault++;
            st.bytes_swap_in += pg.size;
        } else if (regions[page_region[page_id]].is_host) {
            // host region: no backing file; the arena slot is empty and the caller
            // (write() / the buffer's set_tensor) fills it. nothing to load
            (void) off;
        } else if (regions[page_region[page_id]].is_captured) {
            // captured region: no backing file. a page that has never been written has
            // an empty arena slot that the caller (write()) fills in. a page that has
            // been written must have been loaded by an earlier branch (other tier,
            // same tier, or the swap file), so reaching here means the data is gone
            require(!pg.written, "captured page data is missing: it was written in but no "
                                 "tier or swap file holds it (configure cfg.swap_file to "
                                 "page captured regions larger than the tiers)");
        } else {
            const int fd = region_fd_for(page_id);
            if (t == tier::ram) {
                require(read_full(fd, ram_base + off, pg.file_offset, pg.size), "failed to read page from source file");
            } else if (vram.cpu_accessible) {
                // host-writable window (iGPU HMM): one pass, straight from the file
                require(read_full(fd, static_cast<char *>(vram.base) + off, pg.file_offset, pg.size),
                        "failed to read page from source file");
            } else {
                require(read_full(fd, scratch, pg.file_offset, pg.size), "failed to read page from source file");
                vram_copy_in(vram, off, pg.size, scratch);
            }
            st.n_fault++;
            st.bytes_file += pg.size;
        }
    }

    void * pager::impl::load_into(size_t page_id, tier t) {
        auto & ar = arena_of(t);
        size_t off;
        if (regions[page_region[page_id]].is_host) {
            // host page: fixed linear mapping so that base + virtual_offset is the
            // data location (matches a ggml buffer's tensor placement); no arena
            off = page_starts[page_id];
        } else {
            off = ar.alloc(pages[page_id].size);
            require(off != SIZE_MAX, "out of space in tier");
        }
        load_into_at(page_id, t, off);
        set_resident(page_id, t, off);
        lru_touch(t, page_id);
        return base_of(t) + off;
    }

    void pager::impl::prefetch_enq(size_t id, tier t, bool is_tensor) {
        {
            std::lock_guard<std::mutex> lock(qmutex);
            q.push_back({id, t, is_tensor});
        }
        qcv.notify_one();
    }

    void pager::impl::prefetch_tensor(size_t tensor_id, tier t) {
        if (tensor_id >= tensors.size()) {
            return;
        }
        const tensor_info & ti = tensors[tensor_id];
        // skip when the tensor is already resident and contiguous: the gather's fast
        // path will reuse the block as is
        {
            size_t expect = SIZE_MAX;
            bool ok = true;
            for (size_t pid = ti.page_begin; pid < ti.page_end && ok; ++pid) {
                if (!resident(pid, t)) {
                    ok = false;
                } else if (expect == SIZE_MAX) {
                    expect = slice_off(pid, t);
                } else if (slice_off(pid, t) != expect) {
                    ok = false;
                } else {
                    expect += pages[pid].size;
                }
            }
            if (ok) {
                return;
            }
        }
        size_t total = 0;
        for (size_t pid = ti.page_begin; pid < ti.page_end; ++pid) {
            total += pages[pid].size;
        }
        auto & ar = arena_of(t);
        if (ar.max_free() < total) {
            // the RAM tier doubles as a staging buffer for the other tier, so the
            // worker may make room there by evicting staging duplicates; the VRAM
            // window holds the hot batch and is never evicted for a prefetch
            if (t != tier::ram || !evict_redundant(t, total)) {
                return; // best effort only
            }
        }
        const size_t off = ar.alloc(total);
        if (off == SIZE_MAX) {
            return;
        }
        std::vector<size_t> old_offs;
        old_offs.reserve(ti.page_end - ti.page_begin);
        for (size_t pid = ti.page_begin, acc = 0; pid < ti.page_end; ++pid) {
            const size_t old = resident(pid, t) ? slice_off(pid, t) : SIZE_MAX;
            old_offs.push_back(old);
            if (old != off + acc) {
                load_into_at(pid, t, off + acc);
            }
            set_resident(pid, t, off + acc);
            lru_touch(t, pid);
            acc += pages[pid].size;
        }
        for (const size_t old : old_offs) {
            if (old == SIZE_MAX) {
                continue;
            }
            if (old >= off && old < off + total) {
                continue; // part of the new block
            }
            const size_t base  = ar.alloc_base(old);
            if (base == SIZE_MAX) {
                continue; // already freed
            }
            const size_t bsize = ar.size_of(base);
            bool occupied = false;
            for (size_t u = 0; u < pages.size() && !occupied; ++u) {
                if (!resident(u, t)) {
                    continue;
                }
                const size_t ou = slice_off(u, t);
                if (ou >= base && ou < base + bsize) {
                    occupied = true;
                }
            }
            if (!occupied) {
                ar.free(base);
            }
        }
    }

    void pager::impl::worker_loop() {
        for (;;) {
            prefetch_task task;
            {
                std::unique_lock<std::mutex> lock(qmutex);
                qcv.wait(lock, [this] { return quit || !q.empty(); });
                if (quit && q.empty()) {
                    return;
                }
                task = q.front();
                q.pop_front();
            }
            std::lock_guard<std::mutex> lock(mutex);
            try {
                if (task.is_tensor) {
                    prefetch_tensor(task.id, task.t);
                    continue;
                }
                const size_t page_id = task.id;
                if (page_id >= pages.size() || resident(page_id, task.t)) {
                    continue;
                }
                if (arena_of(task.t).max_free() < pages[page_id].size) {
                    if (task.t != tier::ram || !evict_redundant(tier::ram, pages[page_id].size)) {
                        continue; // best effort only
                    }
                }
                load_into(page_id, task.t);
            } catch (const std::exception &) {
                // best effort: a failed task (eg. an I/O error) is dropped; the page
                // will fault in the normal way when it is actually needed
            }
        }
    }

    // ------------------------------ pager --------------------------------

    pager::pager(const config & cfg) : p(new impl(cfg)) {}
    pager::~pager() = default;

    void pager::attach_region(const std::string & name, const std::string & file, size_t file_offset, size_t size,
                              const int64_t * ne, int32_t ggml_type) {
        std::lock_guard<std::mutex> lock(p->mutex);
        require(size > 0, "empty region");
        require(p->tensor_by_name.count(name) == 0, "region name already in use");
        // share one fd per backing file (see file_fds): a sharded model has one
        // region per tensor, and per-region fds exhaust the fd limit on big models
        int fd;
        auto fft = p->file_fds.find(file);
        if (fft != p->file_fds.end()) {
            fd = fft->second;
        } else {
            fd = open(file.c_str(), O_RDONLY);
            if (fd < 0) {
                std::string msg = "failed to open region file: " + file + " (" + std::strerror(errno) + ")";
                throw std::runtime_error(msg);
            }
            p->file_fds[file] = fd;
        }
        const auto fsz = static_cast<size_t>(lseek(fd, 0, SEEK_END));
        require(file_offset + size <= fsz, "region exceeds file size");

        const size_t rid = p->regions.size();
        region r;
        r.name        = name;
        r.file        = file;
        r.file_offset = file_offset;
        r.size        = size;
        r.fd          = fd;
        p->regions.push_back(std::move(r));

        tensor_info ti;
        ti.name        = name;
        ti.file        = file;
        ti.file_offset = file_offset;
        ti.size        = size;
        if (ne != nullptr) {
            for (int i = 0; i < 4; ++i) {
                ti.ne[i] = ne[i];
            }
        }
        ti.ggml_type  = ggml_type;
        ti.page_begin = p->pages.size();
        ti.vaddr      = p->virtual_size;

        for (size_t b = 0; b < size; b += p->cfg.page_size) {
            page pg;
            pg.file_offset = file_offset + b;
            pg.size        = std::min(p->cfg.page_size, size - b);
            p->page_starts.push_back(p->virtual_size);
            p->virtual_size += pg.size;
            p->pages.push_back(std::move(pg));
            p->page_region.push_back(rid);
            p->lru_ram_it.emplace_back();
            p->lru_vram_it.emplace_back();
            p->lru_flags.push_back(0);
        }
        ti.page_end = p->pages.size();
        const size_t tid = p->tensors.size();
        p->tensors.push_back(std::move(ti));
        p->tensor_by_name[name] = tid;

        // the swap file layout is finalized lazily (swap_ensure_ready) once the page
        // table is complete, so it is not touched here
    }

    void pager::attach_host_region(const std::string & name, size_t size,
                                    const int64_t * ne, int32_t ggml_type) {
        std::lock_guard<std::mutex> lock(p->mutex);
        require(size > 0, "empty region");
        require(p->tensor_by_name.count(name) == 0, "region name already in use");
        // a host region has no file; it must fit in the RAM tier so it is never evicted
        require(p->cfg.ram_capacity >= size,
                "host region exceeds the RAM tier capacity (it cannot be evicted or paged)");

        const size_t rid = p->regions.size();
        region r;
        r.name    = name;
        r.size    = size;
        r.fd      = -1;
        r.is_host = true;
        p->regions.push_back(std::move(r));

        tensor_info ti;
        ti.name       = name;
        ti.size       = size;
        if (ne != nullptr) {
            for (int i = 0; i < 4; ++i) {
                ti.ne[i] = ne[i];
            }
        }
        ti.ggml_type  = ggml_type;
        ti.page_begin = p->pages.size();
        ti.vaddr      = p->virtual_size;

        for (size_t b = 0; b < size; b += p->cfg.page_size) {
            page pg;
            pg.size        = std::min(p->cfg.page_size, size - b);
            p->page_starts.push_back(p->virtual_size);
            p->virtual_size += pg.size;
            p->pages.push_back(std::move(pg));
            p->page_region.push_back(rid);
            p->lru_ram_it.emplace_back();
            p->lru_vram_it.emplace_back();
            p->lru_flags.push_back(0);
        }
        ti.page_end = p->pages.size();
        const size_t tid = p->tensors.size();
        p->tensors.push_back(std::move(ti));
        p->tensor_by_name[name] = tid;
    }

    void pager::attach_captured_region(const std::string & name, size_t size,
                                       const int64_t * ne, int32_t ggml_type) {
        std::lock_guard<std::mutex> lock(p->mutex);
        require(size > 0, "empty region");
        require(p->tensor_by_name.count(name) == 0, "region name already in use");

        const size_t rid = p->regions.size();
        region r;
        r.name        = name;
        r.size        = size;
        r.fd          = -1;
        r.is_captured = true;
        p->regions.push_back(std::move(r));

        tensor_info ti;
        ti.name       = name;
        ti.size       = size;
        if (ne != nullptr) {
            for (int i = 0; i < 4; ++i) {
                ti.ne[i] = ne[i];
            }
        }
        ti.ggml_type  = ggml_type;
        ti.page_begin = p->pages.size();
        ti.vaddr      = p->virtual_size;

        for (size_t b = 0; b < size; b += p->cfg.page_size) {
            page pg;
            pg.size        = std::min(p->cfg.page_size, size - b);
            p->page_starts.push_back(p->virtual_size);
            p->virtual_size += pg.size;
            p->pages.push_back(std::move(pg));
            p->page_region.push_back(rid);
            p->lru_ram_it.emplace_back();
            p->lru_vram_it.emplace_back();
            p->lru_flags.push_back(0);
        }
        ti.page_end = p->pages.size();
        const size_t tid = p->tensors.size();
        p->tensors.push_back(std::move(ti));
        p->tensor_by_name[name] = tid;
    }

    void * pager::tier_base(tier t) const {
        std::lock_guard<std::mutex> lock(p->mutex);
        if (t == tier::ram) {
            return p->ram_base;
        }
        return p->vram.base;
    }

    bool pager::vram_cpu_accessible() const {
        std::lock_guard<std::mutex> lock(p->mutex);
        return p->vram.cpu_accessible;
    }

    size_t pager::virtual_size() const {
        return p->virtual_size;
    }

    size_t pager::n_pages() const {
        return p->pages.size();
    }

    size_t pager::page_at(size_t vaddr) const {
        const auto & starts = p->page_starts;
        require(starts.size() > 0, "no pages attached");
        require(vaddr < p->virtual_size, "virtual address out of range");
        auto it = std::upper_bound(starts.begin(), starts.end(), vaddr);
        --it;
        return static_cast<size_t>(it - starts.begin());
    }

    const std::vector<tensor_info> & pager::tensors() const {
        return p->tensors;
    }

    const tensor_info * pager::find_tensor(const std::string & name) const {
        auto it = p->tensor_by_name.find(name);
        return it == p->tensor_by_name.end() ? nullptr : &p->tensors[it->second];
    }

    page_info pager::get_page_info(size_t page_id) const {
        std::lock_guard<std::mutex> lock(p->mutex);
        require(page_id < p->pages.size(), "page id out of range");
        const page & pg = p->pages[page_id];
        page_info info;
        info.vaddr       = p->page_starts[page_id];
        info.file_offset = pg.file_offset;
        info.size        = pg.size;
        info.dirty       = pg.dirty;
        info.pinned      = (pg.pin_count > 0);
        info.in_ram      = pg.ram_off != SIZE_MAX;
        info.in_vram     = pg.vram_off != SIZE_MAX;
        return info;
    }

    void * pager::resolve_page(size_t page_id, tier t) {
        std::lock_guard<std::mutex> lock(p->mutex);
        require(page_id < p->pages.size(), "page id out of range");
        if (t == tier::vram) {
            require(p->vram.base != nullptr, "VRAM tier not enabled");
        } else {
            require(p->ram_base != nullptr, "RAM tier not enabled");
        }
        p->st.n_resolve++;
        page & pg = p->pages[page_id];
        if (!p->resident(page_id, t)) {
            p->ensure_capacity(t, pg.size);
            void * data = p->load_into(page_id, t);
            if (p->cfg.prefetch && page_id + 1 < p->pages.size()) {
                p->prefetch_enq(page_id + 1, t, false);
            }
            return data;
        }
        p->lru_touch(t, page_id);
        return p->base_of(t) + p->slice_off(page_id, t);
    }

    size_t pager::impl::gather_block(size_t a, size_t b, tier t) {
        size_t total = 0;
        for (size_t pid = a; pid < b; ++pid) {
            total += pages[pid].size;
        }

        // fast path: all pages resident and already contiguous
        {
            size_t expect = SIZE_MAX;
            bool contiguous = true;
            for (size_t pid = a; pid < b; ++pid) {
                if (!resident(pid, t)) {
                    contiguous = false;
                    break;
                }
                const size_t off = slice_off(pid, t);
                if (expect == SIZE_MAX) {
                    expect = off;
                } else if (off != expect) {
                    contiguous = false;
                    break;
                }
                expect += pages[pid].size;
            }
            if (contiguous) {
                for (size_t pid = a; pid < b; ++pid) {
                    lru_touch(t, pid);
                }
                return slice_off(a, t);
            }
        }

        // gather into one contiguous block
        auto & ar = arena_of(t);
        ensure_capacity(t, total);
        const size_t off = ar.alloc(total);
        require(off != SIZE_MAX, "out of space in tier");
        std::vector<size_t> old_offs;
        old_offs.reserve(b - a);
        for (size_t pid = a, acc = 0; pid < b; ++pid) {
            const size_t old = resident(pid, t) ? slice_off(pid, t) : SIZE_MAX;
            old_offs.push_back(old);
            if (old != off + acc) {
                load_into_at(pid, t, off + acc);
            }
            set_resident(pid, t, off + acc);
            lru_touch(t, pid);
            acc += pages[pid].size;
        }
        for (const size_t old : old_offs) {
            if (old == SIZE_MAX) {
                continue;
            }
            // the old slice may coincide with the new block, in which case it is part of it
            if (old >= off && old < off + total) {
                continue;
            }
            // a gathered block's old slices share one arena block; only free the block
            // once no live page remains in it
            const size_t base  = ar.alloc_base(old);
            if (base == SIZE_MAX) {
                continue; // already freed
            }
            const size_t bsize = ar.size_of(base);
            bool occupied = false;
            for (size_t u = 0; u < pages.size() && !occupied; ++u) {
                if (!resident(u, t)) {
                    continue;
                }
                const size_t ou = slice_off(u, t);
                if (ou >= base && ou < base + bsize) {
                    occupied = true;
                }
            }
            if (!occupied) {
                ar.free(base);
            }
        }
        return off;
    }

    void * pager::tensor_data(const std::string & name, tier t) {
        std::lock_guard<std::mutex> lock(p->mutex);
        const tensor_info * ti = find_tensor(name);
        require(ti != nullptr, "tensor not found");
        if (t == tier::vram) {
            require(p->vram.base != nullptr, "VRAM tier not enabled");
        } else {
            require(p->ram_base != nullptr, "RAM tier not enabled");
        }
        p->st.n_resolve++;

        const region & r = p->regions[p->page_region[ti->page_begin]];
        if (r.is_captured) {
            for (size_t pid = ti->page_begin; pid < ti->page_end; ++pid) {
                require(p->data_exists(pid),
                        "tensor_data of a captured tensor before it was written in");
            }
        }

        const size_t off = p->gather_block(ti->page_begin, ti->page_end, t);
        return p->base_of(t) + off;
    }

    void * pager::tensor_data_range(const std::string & name, size_t offset, size_t size, tier t) {
        std::lock_guard<std::mutex> lock(p->mutex);
        const tensor_info * ti = find_tensor(name);
        require(ti != nullptr, "tensor not found");
        if (t == tier::vram) {
            require(p->vram.base != nullptr, "VRAM tier not enabled");
        } else {
            require(p->ram_base != nullptr, "RAM tier not enabled");
        }
        require(size > 0, "empty range");
        require(offset + size <= ti->size, "range out of bounds");
        p->st.n_resolve++;

        const region & r = p->regions[p->page_region[ti->page_begin]];
        if (r.is_captured) {
            for (size_t pid = ti->page_begin; pid < ti->page_end; ++pid) {
                require(p->data_exists(pid),
                        "tensor_data_range of a captured tensor before it was written in");
            }
        }

        // the pages covering [offset, offset+size); rel is the byte offset of `offset`
        // within the first page, so the returned pointer lands exactly on the range
        const size_t pid_begin = page_at(ti->vaddr + offset);
        const size_t pid_end   = page_at(ti->vaddr + offset + size - 1);
        const size_t rel = offset - (p->page_starts[pid_begin] - ti->vaddr);
        const size_t off = p->gather_block(pid_begin, pid_end + 1, t);
        return p->base_of(t) + off + rel;
    }

    bool pager::is_resident(size_t page_id, tier t) const {
        std::lock_guard<std::mutex> lock(p->mutex);
        require(page_id < p->pages.size(), "page id out of range");
        return p->resident(page_id, t);
    }

    size_t pager::read(size_t vaddr, void * dst, size_t len) {
        std::lock_guard<std::mutex> lock(p->mutex);
        require(vaddr + len <= p->virtual_size, "read out of range");
        char * out = static_cast<char *>(dst);
        size_t done = 0;
        while (done < len) {
            const size_t v   = vaddr + done;
            const size_t pid = page_at(v);
            const size_t in_off = v - p->page_starts[pid];
            const size_t n = std::min(p->pages[pid].size - in_off, len - done);

            char * src = nullptr;
            if (p->resident(pid, tier::ram)) {
                src = p->ram_base + p->pages[pid].ram_off + in_off;
                p->lru_touch(tier::ram, pid);
            } else if (p->resident(pid, tier::vram)) {
                vram_copy_out(p->vram, p->pages[pid].vram_off + in_off, n, p->scratch);
                src = p->scratch;
                p->lru_touch(tier::vram, pid);
            } else {
                const region & r = p->regions[p->page_region[pid]];
                if (r.is_captured && !p->data_exists(pid)) {
                    throw std::runtime_error("read of a captured page before its data was "
                                             "written in (write() the region first)");
                }
                const tier t = p->ram_base != nullptr ? tier::ram : tier::vram;
                p->ensure_capacity(t, p->pages[pid].size);
                p->load_into(pid, t);
                if (t == tier::ram) {
                    src = p->ram_base + p->pages[pid].ram_off + in_off;
                } else {
                    vram_copy_out(p->vram, p->pages[pid].vram_off + in_off, n, p->scratch);
                    src = p->scratch;
                }
            }
            std::memcpy(out + done, src, n);
            done += n;
        }
        return len;
    }

    size_t pager::write(size_t vaddr, const void * src, size_t len) {
        std::lock_guard<std::mutex> lock(p->mutex);
        require(vaddr + len <= p->virtual_size, "write out of range");
        const tier t = p->ram_base != nullptr ? tier::ram : tier::vram;
        const char * in = static_cast<const char *>(src);
        size_t done = 0;
        while (done < len) {
            const size_t v   = vaddr + done;
            const size_t pid = page_at(v);
            const size_t in_off = v - p->page_starts[pid];
            const size_t n = std::min(p->pages[pid].size - in_off, len - done);

            page & pg = p->pages[pid];
            if (!p->resident(pid, t)) {
                p->ensure_capacity(t, pg.size);
                p->load_into(pid, t);
            }
            if (t == tier::ram) {
                std::memcpy(p->ram_base + pg.ram_off + in_off, in + done, n);
            } else {
                vram_copy_in(p->vram, pg.vram_off + in_off, n, in + done);
            }
            // dirty is per page, not per tier: a co-resident copy in the other tier
            // would go stale and be served with old bytes (the gather fast path
            // returns tier-resident data without re-checking, and a stale last copy
            // would be swapped out as if it were the newest), so keep it consistent
            const tier o = t == tier::ram ? tier::vram : tier::ram;
            if (p->resident(pid, o)) {
                if (t == tier::ram) {
                    vram_copy_in(p->vram, p->pages[pid].vram_off + in_off, n,
                                 p->ram_base + pg.ram_off + in_off);
                } else {
                    std::memcpy(p->ram_base + pg.ram_off + in_off, in + done, n);
                }
            }
            p->lru_touch(t, pid);
            pg.dirty   = true;
            pg.written = true;
            done += n;
        }
        return len;
    }

    void pager::pin(size_t page_id) {
        std::lock_guard<std::mutex> lock(p->mutex);
        require(page_id < p->pages.size(), "page id out of range");
        page & pg = p->pages[page_id];
        pg.pin_count++;
    }

    void pager::unpin(size_t page_id) {
        std::lock_guard<std::mutex> lock(p->mutex);
        require(page_id < p->pages.size(), "page id out of range");
        page & pg = p->pages[page_id];
        require(pg.pin_count > 0, "unpin of an unpinned page");
        pg.pin_count--;
    }

    bool pager::prefetch_page(size_t page_id, tier t) {
        std::lock_guard<std::mutex> lock(p->mutex);
        require(page_id < p->pages.size(), "page id out of range");
        if (t == tier::vram) {
            require(p->vram.base != nullptr, "VRAM tier not enabled");
        } else {
            require(p->ram_base != nullptr, "RAM tier not enabled");
        }
        if (p->resident(page_id, t)) {
            return true;
        }
        if (p->arena_of(t).max_free() < p->pages[page_id].size) {
            return false;
        }
        p->load_into(page_id, t);
        return true;
    }

    void pager::prefetch_range(size_t page_begin, size_t page_end, tier t) {
        std::lock_guard<std::mutex> lock(p->mutex);
        require(p->cfg.prefetch, "prefetch_range requires cfg.prefetch (it starts the prefetch worker)");
        require(page_begin <= page_end && page_end <= p->pages.size(), "page range out of bounds");
        if (t == tier::vram) {
            require(p->vram.base != nullptr, "VRAM tier not enabled");
        } else {
            require(p->ram_base != nullptr, "RAM tier not enabled");
        }
        for (size_t pid = page_begin; pid < page_end; ++pid) {
            p->prefetch_enq(pid, t, false);
        }
    }

    void pager::prefetch_tensor(size_t tensor_id, tier t) {
        std::lock_guard<std::mutex> lock(p->mutex);
        require(p->cfg.prefetch, "prefetch_tensor requires cfg.prefetch (it starts the prefetch worker)");
        require(tensor_id < p->tensors.size(), "tensor id out of range");
        if (t == tier::vram) {
            require(p->vram.base != nullptr, "VRAM tier not enabled");
        } else {
            require(p->ram_base != nullptr, "RAM tier not enabled");
        }
        p->prefetch_enq(tensor_id, t, true);
    }

    void pager::cancel_prefetch() {
        std::lock_guard<std::mutex> lock(p->qmutex);
        p->q.clear();
    }

    stats pager::get_stats() const {
        std::lock_guard<std::mutex> lock(p->mutex);
        return p->st;
    }

    void pager::reset_stats() {
        std::lock_guard<std::mutex> lock(p->mutex);
        p->st = stats();
    }

    const config & pager::get_config() const {
        return p->cfg;
    }

    size_t pager::tier_used(tier t) const {
        std::lock_guard<std::mutex> lock(p->mutex);
        return p->arena_of(t).used();
    }

    size_t pager::tier_capacity(tier t) const {
        std::lock_guard<std::mutex> lock(p->mutex);
        return p->arena_of(t).capacity();
    }

    const std::string & pager::arch() const {
        return p->arch_name;
    }

    const std::string & pager::name() const {
        return p->model_name;
    }

} // namespace llama_vm
