#include "llama_vm/llama_vm.h"

#include "test-common.h"

using namespace llama_vm;

static const size_t PAGE = 4096;
static const int SEED = 11;

int main() {
    if (std::getenv("LLAMA_VM_TEST_VRAM") == nullptr) {
        vm_skip("set LLAMA_VM_TEST_VRAM=1 to run VRAM tier tests");
    }
    const char * backend = std::getenv("LLAMA_VM_VRAM_BACKEND");
    const std::string backend_name = backend != nullptr ? backend : "cpu";
    std::printf("vram backend: %s\n", backend_name.c_str());

    const std::string dir = "/tmp/llama-vm-vram-test-";
    const std::string model = dir + "model.bin";
    const std::string swap  = dir + "swap";
    std::remove(model.c_str());
    std::remove(swap.c_str());

    const size_t model_size = 12 * PAGE + 1024;
    vm_make_pattern_file(model, model_size, SEED);

    // ---------------- basic VRAM residency ----------------
    {
        std::remove(swap.c_str());
        config cfg;
        cfg.page_size     = PAGE;
        cfg.ram_capacity  = 2 * PAGE;
        cfg.vram_capacity = 6 * PAGE;
        cfg.vram_backend  = backend_name;
        cfg.swap_file     = swap;

        pager pg(cfg);
        pg.attach_region("t1", model, 0, model_size);

        void * d = pg.resolve_page(3, tier::vram);
        VM_CHECK(d != nullptr);
        VM_CHECK(vm_check_pattern(d, 3 * PAGE, PAGE, SEED));
        VM_CHECK(pg.is_resident(3, tier::vram));

        auto st = pg.get_stats();
        VM_CHECK(st.n_fault == 1);
    }

    // ---------------- tier to tier copy ----------------
    {
        std::remove(swap.c_str());
        config cfg;
        cfg.page_size     = PAGE;
        cfg.ram_capacity  = 2 * PAGE;
        cfg.vram_capacity = 2 * PAGE;
        cfg.vram_backend  = backend_name;
        cfg.swap_file     = swap;

        pager pg(cfg);
        pg.attach_region("t1", model, 0, model_size);

        // page into RAM
        pg.resolve_page(0, tier::ram);
        // same page into VRAM: should be copied from RAM, not re-read from disk
        void * dv = pg.resolve_page(0, tier::vram);
        VM_CHECK(vm_check_pattern(dv, 0, PAGE, SEED));
        {
            auto st = pg.get_stats();
            VM_CHECK(st.n_tier_copy == 1);
            VM_CHECK(st.n_fault == 1);
        }
    }

    // ---------------- VRAM eviction and LRU ----------------
    {
        std::remove(swap.c_str());
        config cfg;
        cfg.page_size     = PAGE;
        cfg.ram_capacity  = 0;
        cfg.vram_capacity = 3 * PAGE;
        cfg.vram_backend  = backend_name;
        cfg.swap_file     = swap;

        pager pg(cfg);
        pg.attach_region("t1", model, 0, model_size);

        for (size_t i = 0; i < 4; ++i) {
            void * d = pg.resolve_page(i, tier::vram);
            VM_CHECK(vm_check_pattern(d, i * PAGE, PAGE, SEED));
        }
        VM_CHECK(!pg.is_resident(0, tier::vram));
        VM_CHECK(pg.is_resident(3, tier::vram));
        auto st = pg.get_stats();
        VM_CHECK(st.n_evict == 1);
        VM_CHECK(st.n_fault == 4);

        // re-fault from the source file
        void * d0 = pg.resolve_page(0, tier::vram);
        VM_CHECK(vm_check_pattern(d0, 0, PAGE, SEED));
    }

    // ---------------- dirty page through the VRAM tier ----------------
    {
        std::remove(swap.c_str());
        config cfg;
        cfg.page_size     = PAGE;
        cfg.ram_capacity  = 0;
        cfg.vram_capacity = PAGE;
        cfg.vram_backend  = backend_name;
        cfg.swap_file     = swap;

        pager pg(cfg);
        pg.attach_region("t1", model, 0, model_size);

        // load into VRAM, modify via write(), evict to swap, reload into VRAM
        pg.resolve_page(5, tier::vram);
        char buf[PAGE];
        std::memset(buf, 0x5A, sizeof(buf));
        pg.write(5 * PAGE, buf, sizeof(buf));
        {
            auto pi = pg.get_page_info(5);
            VM_CHECK(pi.dirty);
            VM_CHECK(pi.in_vram);
        }

        // filling the 1-page VRAM cache forces a write-back of the dirty page
        pg.resolve_page(6, tier::vram);
        {
            auto st = pg.get_stats();
            VM_CHECK(st.n_swap_out == 1);
            VM_CHECK(!pg.is_resident(5, tier::vram));
        }

        // reload from the swap file into VRAM
        void * d2 = pg.resolve_page(5, tier::vram);
        const auto * b = static_cast<const uint8_t *>(d2);
        bool ok = true;
        for (size_t i = 0; i < PAGE; ++i) {
            if (b[i] != 0x5A) {
                ok = false;
                break;
            }
        }
        VM_CHECK(ok);
        auto st = pg.get_stats();
        VM_CHECK(st.n_swap_in == 1);
    }

    // ---------------- tensor_data in VRAM ----------------
    {
        std::remove(swap.c_str());
        config cfg;
        cfg.page_size     = PAGE;
        cfg.ram_capacity  = 2 * PAGE;
        cfg.vram_capacity = 13 * PAGE; // model is 12*PAGE+1024, i.e. 13 pages
        cfg.vram_backend  = backend_name;
        cfg.swap_file     = swap;

        pager pg(cfg);
        pg.attach_region("t1", model, 0, model_size);

        // touch a page in RAM so the gather path has a mixed state
        pg.resolve_page(1, tier::ram);

        void * d = pg.tensor_data("t1", tier::vram);
        VM_CHECK(vm_check_pattern(d, 0, model_size, SEED));

        // second call hits the contiguous fast path
        void * d2 = pg.tensor_data("t1", tier::vram);
        VM_CHECK(d2 == d);
        VM_CHECK(vm_check_pattern(d2, 0, model_size, SEED));
    }

    std::remove(model.c_str());
    std::remove(swap.c_str());
    std::printf("test_vram: OK\n");
    return 0;
}
