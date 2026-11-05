#include "llama_vm/llama_vm.h"

#include "test-common.h"

#include <fcntl.h>
#include <unistd.h>

#include <thread>
#include <chrono>

using namespace llama_vm;

static const size_t PAGE = 4096;
static const int SEED = 7;

int main() {
    const std::string dir = "/tmp/llama-vm-test-";
    const std::string model = dir + "model.bin";
    const std::string swap  = dir + "swap";
    std::remove(model.c_str());
    std::remove(swap.c_str());

    // 8 pages worth of data, one 5-page tensor and one 3-page tensor
    const size_t model_size = 8 * PAGE + 1024; // last page of t2 is partial
    vm_make_pattern_file(model, model_size, SEED);

    // ---------------- basic faults and LRU eviction ----------------
    {
        std::remove(swap.c_str());
        config cfg;
        cfg.page_size    = PAGE;
        cfg.ram_capacity = 4 * PAGE;
        cfg.vram_capacity = 0;
        cfg.swap_file    = swap;

        pager pg(cfg);
        pg.attach_region("t1", model, 0, 5 * PAGE);
        pg.attach_region("t2", model, 5 * PAGE, 3 * PAGE + 1024);

        VM_CHECK(pg.n_pages() == 9);
        VM_CHECK(pg.virtual_size() == model_size);

        // fault in pages 0..3, fills the whole cache
        for (size_t i = 0; i < 4; ++i) {
            void * d = pg.resolve_page(i, tier::ram);
            VM_CHECK(d != nullptr);
            VM_CHECK(vm_check_pattern(d, i * PAGE, PAGE, SEED));
        }
        {
            auto st = pg.get_stats();
            VM_CHECK(st.n_fault == 4);
            VM_CHECK(st.n_evict == 0);
        }

        // fault page 4: LRU victim is page 0
        void * d4 = pg.resolve_page(4, tier::ram);
        VM_CHECK(vm_check_pattern(d4, 4 * PAGE, PAGE, SEED));
        VM_CHECK(!pg.is_resident(0, tier::ram));
        VM_CHECK(pg.is_resident(1, tier::ram));
        VM_CHECK(pg.is_resident(4, tier::ram));
        {
            auto st = pg.get_stats();
            VM_CHECK(st.n_fault == 5);
            VM_CHECK(st.n_evict == 1);
            VM_CHECK(st.n_swap_out == 0); // page 0 was clean
        }

        // re-fault page 0, data must still be correct (from file)
        void * d0 = pg.resolve_page(0, tier::ram);
        VM_CHECK(vm_check_pattern(d0, 0, PAGE, SEED));
    }

    // ---------------- dirty pages and swap write-back ----------------
    {
        std::remove(swap.c_str());
        config cfg;
        cfg.page_size    = PAGE;
        cfg.ram_capacity = 2 * PAGE;
        cfg.vram_capacity = 0;
        cfg.swap_file    = swap;

        pager pg(cfg);
        pg.attach_region("t1", model, 0, 4 * PAGE);

        // load pages 0 and 1, then modify both via write() which marks them dirty
        char buf[PAGE];
        std::memset(buf, 0xAB, sizeof(buf));
        pg.write(0, buf, PAGE);
        std::memset(buf, 0xCD, sizeof(buf));
        pg.write(PAGE, buf, PAGE);
        {
            VM_CHECK(pg.get_page_info(0).dirty);
            VM_CHECK(pg.get_page_info(1).dirty);
        }
        auto st0 = pg.get_stats();
        VM_CHECK(st0.n_swap_out == 0);

        // both resident pages are dirty, so resolving page 2 forces a write-back
        // of the LRU dirty page (page 0) to the swap file
        pg.resolve_page(2, tier::ram);
        VM_CHECK(!pg.is_resident(0, tier::ram));
        {
            auto st = pg.get_stats();
            VM_CHECK(st.n_swap_out == 1);
            VM_CHECK(st.bytes_swap_out == PAGE);
        }

        // re-fault: must come from the swap file with the modified data
        void * d0b = pg.resolve_page(0, tier::ram);
        VM_CHECK(d0b != nullptr);
        const auto * b = static_cast<const uint8_t *>(d0b);
        bool all_ab = true;
        for (size_t i = 0; i < PAGE; ++i) {
            if (b[i] != 0xAB) {
                all_ab = false;
                break;
            }
        }
        VM_CHECK(all_ab);
        {
            auto st = pg.get_stats();
            VM_CHECK(st.n_swap_in == 1);
            VM_CHECK(st.bytes_swap_in == PAGE);
        }
    }

    // ---------------- swap recovery across a restart ----------------
    {
        const std::string swap2 = dir + "swap-recover";
        std::remove(swap2.c_str());

        config cfg;
        cfg.page_size    = PAGE;
        cfg.ram_capacity = 2 * PAGE;
        cfg.vram_capacity = 0;
        cfg.swap_file    = swap2;

        // session 1: modify a page and force it out to the swap file
        {
            pager pg(cfg);
            pg.attach_region("t1", model, 0, 4 * PAGE);
            char buf[PAGE];
            std::memset(buf, 0x5A, sizeof(buf));
            pg.write(0, buf, PAGE);          // page 0 dirty (0x5A)
            std::memset(buf, 0xCD, sizeof(buf));
            pg.write(PAGE, buf, PAGE);       // page 1 dirty too, so page 0 has no clean co-victim
            pg.resolve_page(2, tier::ram);   // cache full: evicts dirty page 0 -> swap write-back
            VM_CHECK(pg.get_stats().n_swap_out == 1);
        } // pager destroyed: simulate a crash; the swap file must survive

        // the source file still holds the original (pre-modification) data
        {
            int fd = open(model.c_str(), O_RDONLY);
            char src[PAGE];
            VM_CHECK(pread(fd, src, PAGE, 0) == PAGE);
            char expect[PAGE];
            std::memset(expect, 0x5A, PAGE);
            VM_CHECK(std::memcmp(src, expect, PAGE) != 0); // src is NOT 0x5A
            close(fd);
        }

        // session 2: re-open with the same model + swap file; page 0 must be
        // recovered from the swap file, not re-read from the source
        {
            pager pg(cfg);
            pg.attach_region("t1", model, 0, 4 * PAGE);
            void * d = pg.resolve_page(0, tier::ram);
            const auto * b = static_cast<const uint8_t *>(d);
            bool all_5a = true;
            for (size_t i = 0; i < PAGE; ++i) {
                if (b[i] != 0x5A) {
                    all_5a = false;
                    break;
                }
            }
            VM_CHECK(all_5a);
            auto st = pg.get_stats();
            VM_CHECK(st.n_swap_in >= 1);
        }
        std::remove(swap2.c_str());
    }

    // ---------------- dirty eviction without swap must fail ----------------
    {
        config cfg;
        cfg.page_size    = PAGE;
        cfg.ram_capacity = 2 * PAGE;
        cfg.vram_capacity = 0;
        // no swap file
        pager pg(cfg);
        pg.attach_region("t1", model, 0, 4 * PAGE);

        // make every resident page dirty so nothing can be safely evicted
        char buf[PAGE];
        std::memset(buf, 0xCD, sizeof(buf));
        pg.write(0, buf, PAGE);
        pg.write(PAGE, buf, PAGE);

        // no clean victim left and no swap file: the fault must fail
        VM_EXPECT_THROW(pg.resolve_page(2, tier::ram));
    }

    // ---------------- pinning ----------------
    {
        std::remove(swap.c_str());
        config cfg;
        cfg.page_size    = PAGE;
        cfg.ram_capacity = 3 * PAGE;
        cfg.vram_capacity = 0;
        cfg.swap_file    = swap;

        pager pg(cfg);
        pg.attach_region("t1", model, 0, 6 * PAGE);

        void * d0 = pg.resolve_page(0, tier::ram);
        VM_CHECK(vm_check_pattern(d0, 0, PAGE, SEED));
        pg.pin(0);

        // fill the cache past capacity; page 0 must survive
        for (size_t i = 1; i < 6; ++i) {
            pg.resolve_page(i, tier::ram);
        }
        VM_CHECK(pg.is_resident(0, tier::ram));
        void * d0b = pg.resolve_page(0, tier::ram);
        VM_CHECK(vm_check_pattern(d0b, 0, PAGE, SEED));
        pg.unpin(0);
    }

    // ---------------- read/write crossing page boundaries ----------------
    {
        std::remove(swap.c_str());
        config cfg;
        cfg.page_size    = PAGE;
        cfg.ram_capacity = 2 * PAGE;
        cfg.vram_capacity = 0;
        cfg.swap_file    = swap;

        pager pg(cfg);
        pg.attach_region("t1", model, 0, 4 * PAGE);

        // read across the page 1/2 boundary
        char buf[PAGE * 2 + 8];
        pg.read(PAGE - 4, buf, sizeof(buf));
        VM_CHECK(vm_check_pattern(buf, PAGE - 4, sizeof(buf), SEED));

        // write across a boundary, then verify from a cold cache
        const char pattern[] = "llama-vm-write";
        pg.write(2 * PAGE - 8, pattern, sizeof(pattern) - 1);

        // evict everything by loading other pages
        pg.resolve_page(3, tier::ram);
        VM_CHECK(!pg.is_resident(1, tier::ram) || !pg.is_resident(2, tier::ram));

        char buf2[64] = {};
        pg.read(2 * PAGE - 8, buf2, sizeof(pattern) - 1);
        VM_CHECK(std::memcmp(buf2, pattern, sizeof(pattern) - 1) == 0);

        // stats: the write went through the RAM tier, the re-read came from swap
        auto st = pg.get_stats();
        VM_CHECK(st.n_swap_out >= 1);
        VM_CHECK(st.n_swap_in >= 1);
    }

    // ---------------- tensor_data contiguity ----------------
    {
        std::remove(swap.c_str());
        config cfg;
        cfg.page_size    = PAGE;
        cfg.ram_capacity = 5 * PAGE; // exactly the tensor size
        cfg.vram_capacity = 0;
        cfg.swap_file    = swap;

        pager pg(cfg);
        pg.attach_region("big", model, 0, 5 * PAGE);
        const tensor_info * ti = pg.find_tensor("big");
        VM_CHECK(ti != nullptr);
        VM_CHECK(ti->page_begin == 0 && ti->page_end == 5);

        // one page resident, tensor not fully resident and not contiguous
        pg.resolve_page(0, tier::ram);

        void * d = pg.tensor_data("big", tier::ram);
        VM_CHECK(d != nullptr);
        // whole tensor must be readable through one contiguous pointer
        VM_CHECK(vm_check_pattern(d, 0, 5 * PAGE, SEED));

        // second call hits the contiguous fast path
        void * d2 = pg.tensor_data("big", tier::ram);
        VM_CHECK(d2 == d);
        VM_CHECK(vm_check_pattern(d2, 0, 5 * PAGE, SEED));
    }

    // ---------------- tensor_data_range (fault only a sub-range) ----------------
    {
        std::remove(swap.c_str());
        config cfg;
        cfg.page_size    = PAGE;
        cfg.ram_capacity = 2 * PAGE; // only the sub-range's pages fit
        cfg.vram_capacity = 0;
        cfg.swap_file    = swap;

        pager pg(cfg);
        pg.attach_region("big", model, 0, 5 * PAGE);

        // fault only bytes [PAGE, 3*PAGE): pages 1 and 2 become resident, nothing else
        void * d = pg.tensor_data_range("big", PAGE, 2 * PAGE, tier::ram);
        VM_CHECK(d != nullptr);
        VM_CHECK(pg.is_resident(1, tier::ram));
        VM_CHECK(pg.is_resident(2, tier::ram));
        VM_CHECK(!pg.is_resident(0, tier::ram));
        VM_CHECK(!pg.is_resident(3, tier::ram));
        VM_CHECK(!pg.is_resident(4, tier::ram));
        // the pointer lands on tensor byte PAGE and reads the sub-range correctly
        VM_CHECK(vm_check_pattern(d, PAGE, 2 * PAGE, SEED));

        // second call hits the contiguous fast path (same pointer)
        void * d2 = pg.tensor_data_range("big", PAGE, 2 * PAGE, tier::ram);
        VM_CHECK(d2 == d);

        // a range straddling a page boundary from a mid-page offset
        void * d3 = pg.tensor_data_range("big", PAGE + 100, PAGE - 100, tier::ram);
        VM_CHECK(d3 != nullptr);
        VM_CHECK(vm_check_pattern(d3, PAGE + 100, PAGE - 100, SEED));

        // out-of-bounds and empty ranges are rejected
        VM_EXPECT_THROW(pg.tensor_data_range("big", 4 * PAGE, 2 * PAGE, tier::ram));
        VM_EXPECT_THROW(pg.tensor_data_range("big", 0, 0, tier::ram));
    }

    // ---------------- partial page handling ----------------
    {
        std::remove(swap.c_str());
        config cfg;
        cfg.page_size    = PAGE;
        cfg.ram_capacity = 4 * PAGE;
        cfg.vram_capacity = 0;
        cfg.swap_file    = swap;

        pager pg(cfg);
        pg.attach_region("t1", model, 0, 5 * PAGE);
        pg.attach_region("t2", model, 5 * PAGE, 3 * PAGE + 1024);

        // t2 is 3*PAGE+1024 bytes = 3 full pages + one partial 1024-byte page
        VM_CHECK(pg.n_pages() == 9);
        const size_t last = pg.n_pages() - 1;
        page_info pi = pg.get_page_info(last);
        VM_CHECK(pi.size == 1024);
        VM_CHECK(pi.vaddr == 8 * PAGE);

        void * d = pg.resolve_page(last, tier::ram);
        VM_CHECK(vm_check_pattern(d, pi.vaddr, pi.size, SEED));
    }

    // ---------------- page_at mapping ----------------
    {
        config cfg;
        cfg.page_size    = PAGE;
        cfg.ram_capacity = PAGE;
        cfg.vram_capacity = 0;
        pager pg(cfg);
        pg.attach_region("t1", model, 0, 5 * PAGE);
        pg.attach_region("t2", model, 5 * PAGE, 3 * PAGE + 1024);

        VM_CHECK(pg.page_at(0) == 0);
        VM_CHECK(pg.page_at(PAGE - 1) == 0);
        VM_CHECK(pg.page_at(PAGE) == 1);
        VM_CHECK(pg.page_at(5 * PAGE) == 5);
        VM_CHECK(pg.page_at(7 * PAGE + 512) == 7);
        VM_EXPECT_THROW(pg.page_at(pg.virtual_size()));
    }

    // ---------------- prefetch ----------------
    {
        std::remove(swap.c_str());
        config cfg;
        cfg.page_size    = PAGE;
        cfg.ram_capacity = 4 * PAGE;
        cfg.vram_capacity = 0;
        cfg.swap_file    = swap;
        cfg.prefetch     = true;

        pager pg(cfg);
        pg.attach_region("t1", model, 0, 6 * PAGE);

        pg.resolve_page(0, tier::ram);
        // give the prefetch worker a moment
        for (int i = 0; i < 100 && !pg.is_resident(1, tier::ram); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        VM_CHECK(pg.is_resident(1, tier::ram));
    }

    // ---------------- tensor prefetch (contiguous staging) ----------------
    {
        std::remove(swap.c_str());
        config cfg;
        cfg.page_size    = PAGE;
        cfg.ram_capacity = 8 * PAGE;
        cfg.vram_capacity = 0;
        cfg.prefetch     = true;

        pager pg(cfg);
        pg.attach_region("t1", model, 0, 5 * PAGE);
        pg.attach_region("t2", model, 5 * PAGE, 3 * PAGE + 1024);

        const stats before = pg.get_stats();
        pg.prefetch_tensor(1, tier::ram); // t2: pages 5..8 (last one partial)
        for (int i = 0; i < 300 && !pg.is_resident(8, tier::ram); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        VM_CHECK(pg.is_resident(5, tier::ram));
        VM_CHECK(pg.is_resident(8, tier::ram));
        // the tensor was staged as one contiguous block: tensor_data takes the fast
        // path (exactly the staged pages faulted, no same-tier copies)
        void * d = pg.tensor_data("t2", tier::ram);
        VM_CHECK(vm_check_pattern(d, 5 * PAGE, 3 * PAGE + 1024, SEED));
        const stats after = pg.get_stats();
        VM_CHECK(after.n_fault == before.n_fault + 4);
        VM_CHECK(after.n_tier_copy == before.n_tier_copy);
        pg.cancel_prefetch();
    }

    // ---------------- worker evicts staging duplicates ----------------
    {
        std::remove(swap.c_str());
        config cfg;
        cfg.page_size    = PAGE;
        cfg.ram_capacity = 4 * PAGE;
        cfg.vram_capacity = 4 * PAGE;
        cfg.vram_backend = "cpu";
        cfg.prefetch     = true;

        pager pg(cfg);
        pg.attach_region("t1", model, 0, 4 * PAGE);
        pg.attach_region("t2", model, 4 * PAGE, 4 * PAGE);

        // t1 resident in both tiers; the RAM tier is full of redundant copies
        pg.tensor_data("t1", tier::ram);
        pg.tensor_data("t1", tier::vram);
        VM_CHECK(pg.is_resident(0, tier::ram));
        VM_CHECK(pg.is_resident(0, tier::vram));

        // staging t2 needs the worker to evict t1's RAM copies (the VRAM copies
        // remain, so no data is lost and no swap is needed)
        pg.prefetch_tensor(1, tier::ram);
        for (int i = 0; i < 300 && !pg.is_resident(7, tier::ram); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        VM_CHECK(pg.is_resident(4, tier::ram));
        VM_CHECK(pg.is_resident(7, tier::ram));
        VM_CHECK(!pg.is_resident(0, tier::ram));
        VM_CHECK(pg.is_resident(0, tier::vram));
        const auto st = pg.get_stats();
        VM_CHECK(st.n_swap_out == 0);
    }

    // ---------------- capacity governor ----------------
    {
        const size_t MiB = 1024 * 1024;

        // model fits in RAM: RAM is capped at the model size, no VRAM
        {
            capacity_plan p = plan_capacity(10 * MiB, 16 * MiB, 0);
            VM_CHECK(p.ram_capacity == 10 * MiB);
            VM_CHECK(p.vram_capacity == 0);
            VM_CHECK(p.total_capacity == 10 * MiB);
            VM_CHECK(p.fits);
        }

        // model exceeds RAM but fits in RAM + VRAM
        {
            capacity_plan p = plan_capacity(16 * MiB, 8 * MiB, 8 * MiB);
            VM_CHECK(p.ram_capacity == 8 * MiB);
            VM_CHECK(p.vram_capacity == 8 * MiB);
            VM_CHECK(p.fits);
        }

        // model exceeds RAM + VRAM: both tiers are used up, it will page
        {
            capacity_plan p = plan_capacity(16 * MiB, 4 * MiB, 4 * MiB);
            VM_CHECK(p.ram_capacity == 4 * MiB);
            VM_CHECK(p.vram_capacity == 4 * MiB);
            VM_CHECK(p.total_capacity == 8 * MiB);
            VM_CHECK(!p.fits);
        }

        // VRAM only fills the remainder left by RAM
        {
            capacity_plan p = plan_capacity(20 * MiB, 8 * MiB, 40 * MiB);
            VM_CHECK(p.ram_capacity == 8 * MiB);
            VM_CHECK(p.vram_capacity == 12 * MiB); // capped at the model, not available
            VM_CHECK(p.fits);
        }

        // a non-page-aligned model: capacity is 4 KiB aligned and enough to hold it
        {
            const size_t model = 1000 * 1024 + 123; // not 4 KiB aligned
            capacity_plan p = plan_capacity(model, 4 * MiB, 4 * MiB);
            VM_CHECK(p.ram_capacity % 4096 == 0);
            VM_CHECK(p.vram_capacity % 4096 == 0);
            VM_CHECK(p.ram_capacity <= ((model + 4095) / 4096) * 4096); // capped at the model's need
            VM_CHECK(p.total_capacity >= model);                        // available covers the model
            VM_CHECK(p.fits);
        }

        // nothing available -> no tiers, does not fit
        {
            capacity_plan p = plan_capacity(10 * MiB, 0, 0);
            VM_CHECK(p.ram_capacity == 0);
            VM_CHECK(p.vram_capacity == 0);
            VM_CHECK(!p.fits);
        }
    }

    std::remove(model.c_str());
    std::remove(swap.c_str());
    std::printf("test_pager: OK\n");
    return 0;
}
