#include "llama_vm/llama_vm.h"
#include "ggml_buffer.h"

#include "test-common.h"

#include "ggml.h"
#include "ggml-backend.h"

// End-to-end test of the LLAMA_VM_GPU weight buffer + wrapper backend, with the
// inner compute backend set to CPU (LLAMA_VM_GPU=cpu) so it runs without a GPU:
// weights larger than the VRAM window are captured at load time into a small RAM
// tier (+ swap), then faulted into the window op by op while a real graph computes.
//
// Part 1 (below) exercises the whole-tensor fault path. Part 2 exercises the
// MoE expert-slice path: a view into a weight tensor that is *larger* than the
// VRAM window computes because the wrapper faults only the view's pages via
// tensor_data_range, and the pages outside the slice never become resident.

using namespace llama_vm;

int main() {
    const size_t PAGE  = 16 * 1024;
    const size_t WSIZE = 64 * 1024; // each weight: 4 pages
    const size_t N_W   = 4;

    setenv("LLAMA_VM_GPU", "cpu", 1);
    setenv("LLAMA_VM_VRAM", "128K", 1);   // window: holds two weights, not all four
    setenv("LLAMA_VM_RAM", "64K", 1);     // capture tier: holds one weight
    setenv("LLAMA_VM_GGML_PAGE_SIZE", "16384", 1);
    const std::string swap = "/tmp/llama_vm_test_gpu.swap";
    setenv("LLAMA_VM_SWAP", swap.c_str(), 1);
    std::remove(swap.c_str());

    // --------------------------- backend / device -------------------------
    ggml_backend_reg_t reg = ggml_gpu_reg();
    VM_CHECK(reg != nullptr);
    VM_CHECK(ggml_backend_reg_dev_count(reg) == 2); // compute (wrapper) + memory-only; the CPU device was found
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    VM_CHECK(dev != nullptr);
    VM_CHECK(std::strcmp(ggml_backend_dev_name(dev), "LLAMA_VM_GPU") == 0);

    ggml_backend_buffer_type_t buft = ggml_gpu_buft();
    VM_CHECK(std::strcmp(ggml_backend_buft_name(buft), "LLAMA_VM_GPU") == 0);
    VM_CHECK(!ggml_backend_buft_is_host(buft));
    VM_CHECK(ggml_backend_dev_supports_buft(dev, buft));
    VM_CHECK(ggml_backend_dev_buffer_type(dev) != buft); // activations use the inner buft

    // the memory-only device's default buft is the paged weight buft, which is how
    // llama.cpp's -ot parser resolves the name "LLAMA_VM_GPU" to a buffer type
    ggml_backend_dev_t mem = ggml_backend_reg_dev_get(reg, 1);
    VM_CHECK(mem != nullptr);
    VM_CHECK(ggml_backend_dev_type(mem) == GGML_BACKEND_DEVICE_TYPE_CPU);
    VM_CHECK(ggml_backend_dev_buffer_type(mem) == buft);
    VM_CHECK(std::strcmp(ggml_backend_buft_name(ggml_backend_dev_buffer_type(mem)), "LLAMA_VM_GPU") == 0);
    VM_CHECK(!ggml_backend_dev_supports_op(mem, nullptr));

    // ------------------------------ buffer --------------------------------
    // place the weights where ggml's allocator would: padded, 32-byte aligned
    struct ggml_init_params params = {1024 * 1024, nullptr, true};
    struct ggml_context * ctx = ggml_init(params);
    VM_CHECK(ctx != nullptr);

    const int64_t n_elem = WSIZE / 4; // f32
    std::vector<struct ggml_tensor *> w(N_W);
    size_t total = 0;
    for (size_t i = 0; i < N_W; ++i) {
        w[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elem);
        VM_CHECK(w[i] != nullptr);
        std::snprintf(w[i]->name, sizeof(w[i]->name), "w%zu", i);
        total += (ggml_nbytes(w[i]) + 31) & ~(size_t) 31;
    }

    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, total);
    VM_CHECK(buf != nullptr);
    VM_CHECK(!ggml_backend_buffer_is_host(buf));
    VM_CHECK(ggml_backend_buffer_get_size(buf) == total);
    void * base = ggml_backend_buffer_get_base(buf);
    VM_CHECK(base != nullptr);

    const pager * pg = gpu_buffer_pager(buf);
    VM_CHECK(pg != nullptr);

    size_t off = 0;
    std::vector<std::vector<uint8_t>> wdata(N_W);
    for (size_t i = 0; i < N_W; ++i) {
        off = (off + 31) & ~(size_t) 31;
        enum ggml_status st = ggml_backend_tensor_alloc(buf, w[i], static_cast<char *>(base) + off);
        VM_CHECK(st == GGML_STATUS_SUCCESS);
        VM_CHECK(w[i]->data == static_cast<char *>(base) + off);
        off += (ggml_nbytes(w[i]) + 31) & ~(size_t) 31;

        // capture the weight: the RAM tier (64 KiB) overflows, pages go to swap
        wdata[i].resize(ggml_nbytes(w[i]));
        float * wf = reinterpret_cast<float *>(wdata[i].data());
        for (size_t j = 0; j < n_elem; ++j) {
            wf[j] = 0.001f * static_cast<float>((j * 7 + i * 131) % 997);
        }
        ggml_backend_tensor_set(w[i], wdata[i].data(), 0, wdata[i].size());
    }
    VM_CHECK(pg->find_tensor("w0") != nullptr);
    VM_CHECK(pg->find_tensor("w3") != nullptr);

    const stats after_load = pg->get_stats();
    VM_CHECK(after_load.n_evict > 0);    // the RAM tier recycled during capture
    VM_CHECK(after_load.n_swap_out > 0); // dirty pages were swapped out

    // ------------------------------ graph ---------------------------------
    // t1  = w0 + w1
    // t1b = t1 + w0      reuses w0 within the batch (the batcher must dedupe it)
    // t2  = t1b + w2     new weight; the window (two weights) is full -> new batch
    // y   = t2 + w3      new weight, fits by evicting an unpinned weight -> same batch
    // y2  = y + w0       w0 again; w2/w3 are pinned -> new batch
    // the window holds two weights, so this graph spans three batches
    struct ggml_tensor * t1  = ggml_add(ctx, w[0], w[1]);
    struct ggml_tensor * t1b = ggml_add(ctx, t1,  w[0]);
    struct ggml_tensor * t2  = ggml_add(ctx, t1b, w[2]);
    struct ggml_tensor * y   = ggml_add(ctx, t2,  w[3]);
    struct ggml_tensor * y2  = ggml_add(ctx, y,   w[0]);
    VM_CHECK(t1 && t1b && t2 && y && y2);

    // allocate the intermediates on the plain CPU backend (the weights are already
    // placed in the LLAMA_VM_GPU buffer)
    ggml_backend_t cpu = ggml_backend_dev_init(ggml_backend_dev_by_name("CPU"), nullptr);
    VM_CHECK(cpu != nullptr);
    ggml_backend_buffer_t abuf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    VM_CHECK(abuf != nullptr);
    VM_CHECK(t1->data != nullptr && y->data != nullptr);

    struct ggml_cgraph * g = ggml_new_graph(ctx);
    VM_CHECK(g != nullptr);
    // ggml_graph_add_node does not mark ops for compute (the graph builder does);
    // the backends' compute loops skip nodes without the flag
    t1->flags  |= GGML_TENSOR_FLAG_COMPUTE;
    t1b->flags |= GGML_TENSOR_FLAG_COMPUTE;
    t2->flags  |= GGML_TENSOR_FLAG_COMPUTE;
    y->flags   |= GGML_TENSOR_FLAG_COMPUTE;
    y2->flags  |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_graph_add_node(g, t1);
    ggml_graph_add_node(g, t1b);
    ggml_graph_add_node(g, t2);
    ggml_graph_add_node(g, y);
    ggml_graph_add_node(g, y2);

    // --------------------------- compute + verify -------------------------
    ggml_backend_t wb = ggml_backend_dev_init(dev, nullptr);
    VM_CHECK(wb != nullptr);
    enum ggml_status st = ggml_backend_graph_compute(wb, g);
    VM_CHECK(st == GGML_STATUS_SUCCESS);

    // expected values in the graph's exact addition order (float is not
    // associative; the comparison is bit-exact)
    const int order[5] = {0, 1, 0, 2, 3}; // t1, t1b, t2, y: w0 + w1 + w0 + w2 + w3
    std::vector<uint8_t> out(ggml_nbytes(y));
    ggml_backend_tensor_get(y, out.data(), 0, out.size());
    const float * yf = reinterpret_cast<const float *>(out.data());
    bool ok = true;
    for (size_t j = 0; j < n_elem; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < 5; ++k) {
            sum += reinterpret_cast<const float *>(wdata[order[k]].data())[j];
        }
        if (yf[j] != sum) {
            ok = false;
        }
    }
    VM_CHECK(ok);

    // y2 = y + w0: one addition per element, bit-exact against y
    std::vector<uint8_t> out2(ggml_nbytes(y2));
    ggml_backend_tensor_get(y2, out2.data(), 0, out2.size());
    const float * y2f = reinterpret_cast<const float *>(out2.data());
    ok = true;
    for (size_t j = 0; j < n_elem && ok; ++j) {
        if (y2f[j] != yf[j] + reinterpret_cast<const float *>(wdata[0].data())[j]) {
            ok = false;
        }
    }
    VM_CHECK(ok);

    // ------------------------------ stats ---------------------------------
    const stats stt = pg->get_stats();
    std::printf("  GPU buffer pager stats: resolve=%llu faults=%llu tier_copy=%llu "
                "evicts=%llu swap_in=%llu swap_out=%llu\n",
                (unsigned long long) stt.n_resolve,
                (unsigned long long) stt.n_fault,
                (unsigned long long) stt.n_tier_copy,
                (unsigned long long) stt.n_evict,
                (unsigned long long) stt.n_swap_in,
                (unsigned long long) stt.n_swap_out);
    VM_CHECK(stt.n_swap_in > 0); // weights were re-faulted from the swap file

    // ------------------------------ teardown ------------------------------
    ggml_backend_free(wb);
    ggml_backend_buffer_free(abuf);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf);

    std::remove(swap.c_str());
    unsetenv("LLAMA_VM_GPU");
    unsetenv("LLAMA_VM_VRAM");
    unsetenv("LLAMA_VM_RAM");
    unsetenv("LLAMA_VM_SWAP");
    unsetenv("LLAMA_VM_GGML_PAGE_SIZE");

    // ------------------- part 2: MoE expert slice (view) path -------------------
    // a weight tensor bigger than the VRAM window, faulted as just a slice: the
    // wrapper must fault only the view's pages via tensor_data_range, so a graph
    // that reads a slice of an over-window tensor computes, and the pages outside
    // the slice never become resident in the window.
    const size_t PAGE2  = 16 * 1024;
    const size_t BIGSZ  = 3 * PAGE2; // 3 pages = 48 KiB, larger than the window
    const size_t SLICEOFF = PAGE2;   // the slice is the last two pages [16K, 48K)
    const size_t SLICESZ  = 2 * PAGE2;

    setenv("LLAMA_VM_GPU", "cpu", 1);
    setenv("LLAMA_VM_VRAM", "32K", 1);      // window: 2 pages, holds the slice, not the tensor
    setenv("LLAMA_VM_RAM", "64K", 1);       // capture tier holds the whole tensor (no swap)
    setenv("LLAMA_VM_GGML_PAGE_SIZE", "16384", 1);
    unsetenv("LLAMA_VM_SWAP");

    struct ggml_init_params p2 = {1024 * 1024, nullptr, true};
    struct ggml_context * ctx2 = ggml_init(p2);
    VM_CHECK(ctx2 != nullptr);

    struct ggml_tensor * big = ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, BIGSZ / 4);
    VM_CHECK(big != nullptr);
    std::snprintf(big->name, sizeof(big->name), "big");

    ggml_backend_buffer_t buf2 = ggml_backend_buft_alloc_buffer(buft, BIGSZ);
    VM_CHECK(buf2 != nullptr);
    const pager * pg2 = gpu_buffer_pager(buf2);
    VM_CHECK(pg2 != nullptr);
    enum ggml_status st2 = ggml_backend_tensor_alloc(buf2, big, ggml_backend_buffer_get_base(buf2));
    VM_CHECK(st2 == GGML_STATUS_SUCCESS);

    // capture the whole tensor
    std::vector<uint8_t> bigdata(BIGSZ);
    float * bf = reinterpret_cast<float *>(bigdata.data());
    for (size_t j = 0; j < BIGSZ / 4; ++j) {
        bf[j] = 0.001f * static_cast<float>((j * 13) % 991);
    }
    ggml_backend_tensor_set(big, bigdata.data(), 0, BIGSZ);
    VM_CHECK(pg2->find_tensor("big") != nullptr);

    // a view over the last two pages; set its buffer so the wrapper recognizes it
    // (ggml leaves view->buffer NULL until ggml-alloc assigns the base's buffer)
    struct ggml_tensor * view = ggml_view_1d(ctx2, big, SLICESZ / 4, SLICEOFF);
    VM_CHECK(view != nullptr);
    VM_CHECK(view->view_src == big && view->view_offs == SLICEOFF);
    view->buffer = buf2;
    view->data   = static_cast<char *>(ggml_backend_buffer_get_base(buf2)) + SLICEOFF;

    struct ggml_tensor * ys = ggml_add(ctx2, view, view);
    VM_CHECK(ys != nullptr);
    ys->flags |= GGML_TENSOR_FLAG_COMPUTE;
    struct ggml_cgraph * g2 = ggml_new_graph(ctx2);
    VM_CHECK(g2 != nullptr);
    ggml_graph_add_node(g2, ys);

    ggml_backend_t cpu2 = ggml_backend_dev_init(ggml_backend_dev_by_name("CPU"), nullptr);
    VM_CHECK(cpu2 != nullptr);
    ggml_backend_buffer_t abuf2 = ggml_backend_alloc_ctx_tensors(ctx2, cpu2);
    VM_CHECK(abuf2 != nullptr);

    ggml_backend_t wb2 = ggml_backend_dev_init(dev, nullptr);
    VM_CHECK(wb2 != nullptr);
    enum ggml_status st3 = ggml_backend_graph_compute(wb2, g2);
    VM_CHECK(st3 == GGML_STATUS_SUCCESS); // the slice fits the window though the tensor does not

    // the slice's pages are resident in the window; the page outside it is not
    const tensor_info * ti = pg2->find_tensor("big");
    VM_CHECK(ti != nullptr);
    const size_t p_out = pg2->page_at(ti->vaddr);              // page 0, outside the slice
    const size_t p_in1 = pg2->page_at(ti->vaddr + SLICEOFF);   // page 1
    const size_t p_in2 = pg2->page_at(ti->vaddr + SLICESZ);    // page 2
    VM_CHECK(!pg2->is_resident(p_out, tier::vram));
    VM_CHECK(pg2->is_resident(p_in1, tier::vram));
    VM_CHECK(pg2->is_resident(p_in2, tier::vram));

    // y2 = view + view = 2 * slice, bit-exact
    std::vector<uint8_t> out3(ggml_nbytes(ys));
    ggml_backend_tensor_get(ys, out3.data(), 0, out3.size());
    const float * o3 = reinterpret_cast<const float *>(out3.data());
    const float * sl = reinterpret_cast<const float *>(bigdata.data() + SLICEOFF);
    bool ok3 = true;
    for (size_t j = 0; j < SLICESZ / 4 && ok3; ++j) {
        if (o3[j] != 2.0f * sl[j]) {
            ok3 = false;
        }
    }
    VM_CHECK(ok3);

    const stats stt3 = pg2->get_stats();
    std::printf("  GPU buffer slice pager stats: faults=%llu tier_copy=%llu bytes_tier=%llu\n",
                (unsigned long long) stt3.n_fault,
                (unsigned long long) stt3.n_tier_copy,
                (unsigned long long) stt3.bytes_tier);

    ggml_backend_free(wb2);
    ggml_backend_buffer_free(abuf2);
    ggml_backend_free(cpu2);
    ggml_free(ctx2);
    ggml_backend_buffer_free(buf2);

    unsetenv("LLAMA_VM_GPU");
    unsetenv("LLAMA_VM_VRAM");
    unsetenv("LLAMA_VM_RAM");
    unsetenv("LLAMA_VM_GGML_PAGE_SIZE");

    std::printf("test_gpu_buffer: OK\n");
    return 0;
}
