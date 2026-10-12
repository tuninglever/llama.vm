#include "llama_vm/llama_vm.h"

#include "test-common.h"

// Captured regions: data written in via write() (as a GPU buffer captures weights at
// load time), paged like file-backed regions — LRU eviction across the tiers, swap
// write-back for dirty last copies, and contiguous tensor_data gathers.

using namespace llama_vm;

int main() {
    const size_t PAGE = 16 * 1024;
    const size_t RAM  = 128 * 1024; // 8 pages
    const size_t VRAM = 512 * 1024; // holds the whole tensor
    const size_t SIZE = 512 * 1024; // 32 pages, 4x the RAM tier

    const std::string swap = "/tmp/llama_vm_test_captured.swap";
    std::remove(swap.c_str());

    config cfg;
    cfg.page_size    = PAGE;
    cfg.ram_capacity = RAM;
    cfg.vram_capacity = VRAM;
    cfg.vram_backend = "cpu"; // exercise the paging logic without a GPU
    cfg.swap_file    = swap;

    pager pg(cfg);
    pg.attach_captured_region("w", SIZE);

    VM_CHECK(pg.n_pages() == SIZE / PAGE);
    VM_CHECK(pg.virtual_size() == SIZE);
    const tensor_info * ti = pg.find_tensor("w");
    VM_CHECK(ti != nullptr);
    VM_CHECK(ti->vaddr == 0);

    // reading a captured page before it is written in must fail
    VM_EXPECT_THROW(pg.read(0, std::vector<char>(PAGE).data(), PAGE));

    // write the whole region (pattern data); the RAM tier (128 KiB) overflows and the
    // oldest dirty pages are swapped out as the write proceeds
    std::vector<uint8_t> in(SIZE);
    for (size_t i = 0; i < SIZE; ++i) {
        in[i] = vm_pat_byte(i, 3);
    }
    pg.write(0, in.data(), SIZE);

    const stats after_write = pg.get_stats();
    VM_CHECK(after_write.n_evict > 0);
    VM_CHECK(after_write.n_swap_out > 0);

    // read it all back: pages come from the RAM tier or the swap file
    std::vector<uint8_t> out(SIZE);
    pg.read(0, out.data(), SIZE);
    VM_CHECK(vm_check_pattern(out.data(), 0, SIZE, 3));

    // contiguous gather into the VRAM tier: the window (512 KiB) holds the whole
    // tensor, so every page must be faulted in (RAM or swap) and packed
    void * v = pg.tensor_data("w", tier::vram);
    VM_CHECK(v != nullptr);
    VM_CHECK(vm_check_pattern(v, 0, SIZE, 3));

    // a second gather is the fast path (all pages resident and contiguous in VRAM)
    const size_t v2 = reinterpret_cast<size_t>(pg.tensor_data("w", tier::vram));
    VM_CHECK(v2 == reinterpret_cast<size_t>(v));

    // partial read of a middle slice
    std::vector<uint8_t> slice(4096);
    pg.read(100 * 1024, slice.data(), 4096);
    VM_CHECK(vm_check_pattern(slice.data(), 100 * 1024, 4096, 3));

    // rewrite part of the region; the change must survive paging
    std::vector<uint8_t> patch(8192, 0xAB);
    pg.write(200 * 1024, patch.data(), patch.size());
    std::vector<uint8_t> patched_buf(patch.size());
    pg.read(200 * 1024, patched_buf.data(), patch.size());
    bool patched = true;
    for (size_t i = 0; i < patch.size(); ++i) {
        if (patched_buf[i] != 0xAB) { patched = false; break; }
    }
    VM_CHECK(patched);
    // the untouched neighbours are intact
    std::vector<uint8_t> before(4096), after(4096);
    pg.read(200 * 1024 - 4096, before.data(), 4096);
    pg.read(200 * 1024 + patch.size(), after.data(), 4096);
    VM_CHECK(vm_check_pattern(before.data(), 200 * 1024 - 4096, 4096, 3));
    VM_CHECK(vm_check_pattern(after.data(), 200 * 1024 + patch.size(), 4096, 3));

    const stats st = pg.get_stats();
    std::printf("  captured pager stats: resolve=%llu faults=%llu tier_copy=%llu "
                "evicts=%llu swap_in=%llu swap_out=%llu\n",
                (unsigned long long) st.n_resolve,
                (unsigned long long) st.n_fault,
                (unsigned long long) st.n_tier_copy,
                (unsigned long long) st.n_evict,
                (unsigned long long) st.n_swap_in,
                (unsigned long long) st.n_swap_out);
    VM_CHECK(st.n_swap_in > 0);

    std::remove(swap.c_str());
    std::printf("test_captured: OK\n");
    return 0;
}
