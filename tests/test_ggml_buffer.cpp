#include "llama_vm/llama_vm.h"
#include "ggml_buffer.h"

#include "test-common.h"

#include "ggml.h"
#include "ggml-backend.h"

// Integration test: the llama.vm ggml buffer, exercised through the real ggml C
// API. Validates the buffer contract (type props, alloc, tensor placement,
// set/get/cpy/clear, host flag, pager backing) against a live ggml context.

using namespace llama_vm;

int main() {
    // ------------------------- buffer type -------------------------------
    ggml_backend_buffer_type_t buft = ggml_buft();
    VM_CHECK(buft != nullptr);
    VM_CHECK(std::strcmp(ggml_backend_buft_name(buft), "LLAMA_VM") == 0);
    VM_CHECK(ggml_backend_buft_get_alignment(buft) == 32);
    VM_CHECK(ggml_backend_buft_is_host(buft));

    // --------------------------- registry --------------------------------
    ggml_backend_reg_t reg = ggml_backend_llamavm_reg();
    VM_CHECK(reg != nullptr);
    VM_CHECK(std::strcmp(ggml_backend_reg_name(reg), "llama_vm") == 0);
    VM_CHECK(ggml_backend_reg_dev_count(reg) == 1);
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    VM_CHECK(dev != nullptr);
    VM_CHECK(std::strcmp(ggml_backend_dev_name(dev), "llama_vm") == 0);
    VM_CHECK(ggml_backend_dev_buffer_type(dev) == buft);
    VM_CHECK(ggml_backend_dev_host_buffer_type(dev) == buft);
    VM_CHECK(ggml_backend_dev_supports_buft(dev, buft));

    // --------------------------- buffer ----------------------------------
    // 8 MiB buffer, 4 MiB pages -> 2 pages (set a small page for finer checks)
    const size_t PAGE  = 4 * 1024 * 1024;
    const size_t SIZE  = 2 * PAGE;
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, SIZE);
    VM_CHECK(buf != nullptr);
    void * base = ggml_backend_buffer_get_base(buf);
    VM_CHECK(base != nullptr);
    VM_CHECK(reinterpret_cast<uintptr_t>(base) % 32 == 0);
    VM_CHECK(ggml_backend_buffer_is_host(buf));
    VM_CHECK(ggml_backend_buffer_get_size(buf) == SIZE);

    // the buffer is backed by a pager
    const pager * pg = buffer_pager(buf);
    VM_CHECK(pg != nullptr);
    VM_CHECK(pg->virtual_size() == SIZE);
    VM_CHECK(pg->n_pages() == SIZE / PAGE);
    VM_CHECK(pg->find_tensor("buffer") != nullptr);

    // ----------------------- tensor in the buffer ------------------------
    struct ggml_init_params params = {1024 * 1024, nullptr, true};
    struct ggml_context * ctx = ggml_init(params);
    VM_CHECK(ctx != nullptr);

    // a 1 MiB f32 tensor placed at a 32-byte aligned offset
    const size_t n_elem  = (1 * 1024 * 1024) / 4;
    const size_t nbytes  = n_elem * 4;
    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elem);
    VM_CHECK(t != nullptr);
    const size_t off = 64; // a small, aligned, non-zero offset
    void * addr = static_cast<char *>(base) + off;
    enum ggml_status st = ggml_backend_tensor_alloc(buf, t, addr);
    VM_CHECK(st == GGML_STATUS_SUCCESS);
    VM_CHECK(t->data == addr);

    // write a deterministic pattern, read it back through the buffer
    std::vector<uint8_t> write(nbytes), readback(nbytes);
    for (size_t i = 0; i < nbytes; ++i) {
        write[i] = vm_pat_byte(i, 7);
    }
    ggml_backend_tensor_set(t, write.data(), 0, nbytes);
    ggml_backend_tensor_get(t, readback.data(), 0, nbytes);
    VM_CHECK(vm_check_pattern(readback.data(), 0, nbytes, 7));

    // partial get (middle slice)
    std::vector<uint8_t> slice(4096);
    const size_t soff = 1000;
    ggml_backend_tensor_get(t, slice.data(), soff, 4096);
    VM_CHECK(vm_check_pattern(slice.data(), soff, 4096, 7));

    // set a partial region and verify the rest is unchanged
    std::vector<uint8_t> patch(256);
    for (size_t i = 0; i < 256; ++i) {
        patch[i] = static_cast<uint8_t>(0xA0 + (i & 7));
    }
    ggml_backend_tensor_set(t, patch.data(), 5000, 256);
    std::vector<uint8_t> verify(nbytes);
    ggml_backend_tensor_get(t, verify.data(), 0, nbytes);
    bool ok = true;
    for (size_t i = 0; i < nbytes; ++i) {
        const uint8_t expect = (i >= 5000 && i < 5000 + 256)
            ? patch[i - 5000]
            : vm_pat_byte(i, 7);
        if (verify[i] != expect) { ok = false; break; }
    }
    VM_CHECK(ok);

    // --------------------------- copy ------------------------------------
    // t2 lives in the second page (offset PAGE), still inside the 2-page buffer
    struct ggml_tensor * t2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elem);
    VM_CHECK(t2 != nullptr);
    st = ggml_backend_tensor_alloc(buf, t2, static_cast<char *>(base) + PAGE);
    VM_CHECK(st == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_copy(t, t2);
    std::vector<uint8_t> t2data(nbytes);
    ggml_backend_tensor_get(t2, t2data.data(), 0, nbytes);
    bool ok2 = true;
    for (size_t i = 0; i < nbytes; ++i) {
        const uint8_t expect = (i >= 5000 && i < 5000 + 256)
            ? patch[i - 5000]
            : vm_pat_byte(i, 7);
        if (t2data[i] != expect) { ok2 = false; break; }
    }
    VM_CHECK(ok2);

    // --------------------------- clear -----------------------------------
    ggml_backend_buffer_clear(buf, 0);
    ggml_backend_tensor_get(t, verify.data(), 0, nbytes);
    bool zeroed = true;
    for (size_t i = 0; i < nbytes; ++i) {
        if (verify[i] != 0) { zeroed = false; break; }
    }
    VM_CHECK(zeroed);

    // re-write after clear works
    ggml_backend_tensor_set(t, write.data(), 0, nbytes);
    ggml_backend_tensor_get(t, readback.data(), 0, nbytes);
    VM_CHECK(vm_check_pattern(readback.data(), 0, nbytes, 7));

    // --------------------------- stats -----------------------------------
    const stats stt = pg->get_stats();
    std::printf("  buffer pager stats: resolve=%llu faults=%llu evicts=%llu\n",
                (unsigned long long) stt.n_resolve,
                (unsigned long long) stt.n_fault,
                (unsigned long long) stt.n_evict);

    // --------------------------- teardown --------------------------------
    ggml_free(ctx);
    ggml_backend_buffer_free(buf);

    // the accessor is null-safe and rejects non-LLAMA_VM buffers
    VM_CHECK(buffer_pager(nullptr) == nullptr);

    // ------------------------ operator budget ----------------------------
    // LLAMA_VM_RAM dictates the RAM tier; a host buffer cannot page, so a budget
    // below the model size must be rejected (NULL), never silently malloc'ed
    setenv("LLAMA_VM_RAM", "1M", 1);
    VM_CHECK(ggml_backend_buft_alloc_buffer(buft, SIZE) == nullptr); // 1M < 8M

    setenv("LLAMA_VM_RAM", "16M", 1);
    ggml_backend_buffer_t cap = ggml_backend_buft_alloc_buffer(buft, SIZE); // 16M > 8M
    VM_CHECK(cap != nullptr);
    const pager * cpg = buffer_pager(cap);
    VM_CHECK(cpg != nullptr);
    VM_CHECK(cpg->tier_capacity(tier::ram) == SIZE); // capped at the model size
    ggml_backend_buffer_free(cap);

    unsetenv("LLAMA_VM_RAM");
    ggml_backend_buffer_t dflt = ggml_backend_buft_alloc_buffer(buft, SIZE);
    VM_CHECK(dflt != nullptr);
    VM_CHECK(buffer_pager(dflt)->tier_capacity(tier::ram) == SIZE);
    ggml_backend_buffer_free(dflt);

    std::printf("test_ggml_buffer: OK\n");
    return 0;
}
