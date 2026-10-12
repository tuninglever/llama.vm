#include "llama_vm/llama_vm.h"

#include "gguf.h"

#include "test-common.h"

#include <fcntl.h>
#include <map>
#include <unistd.h>

using namespace llama_vm;

static std::string find_model() {
    if (const char * env = std::getenv("LLAMA_VM_TEST_MODEL")) {
        return env;
    }
    return "";
}

// lazily open (and cache) a read-only fd per source file, so a sharded model's
// tensors can each be verified against their own shard
struct file_cache {
    std::map<std::string, int> fds;
    int get(const std::string & path) {
        auto it = fds.find(path);
        if (it != fds.end()) {
            return it->second;
        }
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::fprintf(stderr, "failed to open %s\n", path.c_str());
        }
        fds.emplace(path, fd);
        return fd;
    }
    ~file_cache() {
        for (auto & kv : fds) {
            close(kv.second);
        }
    }
};

// largest tensor across all shards of a (possibly sharded) model; the cache must
// hold the whole largest tensor for tensor_data(), and shard 0 alone may only list
// the small tensors
static size_t largest_tensor(const std::string & model) {
    struct gguf_init_params params = {true, nullptr};
    struct gguf_context * ctx0 = gguf_init_from_file(model.c_str(), params);
    if (ctx0 == nullptr) {
        return 0;
    }
    auto max_in = [](struct gguf_context * c, size_t m) {
        for (int64_t i = 0; i < gguf_get_n_tensors(c); ++i) {
            m = std::max(m, gguf_get_tensor_size(c, i));
        }
        return m;
    };
    size_t largest = max_in(ctx0, 0);
    const int64_t id = gguf_find_key(ctx0, "split.count");
    uint16_t n_split = (id >= 0 && gguf_get_kv_type(ctx0, id) == GGUF_TYPE_UINT16)
                        ? gguf_get_val_u16(ctx0, id) : 1;
    if (n_split > 1) {
        char post[64];
        std::snprintf(post, sizeof(post), "-%05d-of-%05d.gguf", 1, n_split);
        const std::string suffix(post);
        if (model.size() > suffix.size() &&
            model.compare(model.size() - suffix.size(), suffix.size(), suffix) == 0) {
            const std::string prefix = model.substr(0, model.size() - suffix.size());
            for (uint16_t i = 1; i < n_split; ++i) {
                char p2[64];
                std::snprintf(p2, sizeof(p2), "-%05d-of-%05d.gguf", i + 1, n_split);
                struct gguf_context * c = gguf_init_from_file((prefix + p2).c_str(), params);
                if (c != nullptr) {
                    largest = max_in(c, largest);
                    gguf_free(c);
                }
            }
        }
    }
    gguf_free(ctx0);
    return largest;
}

// find the tensor that owns a page (tensors store their page ranges)
static const tensor_info * tensor_of_page(const pager & pg, size_t page_id) {
    for (const tensor_info & ti : pg.tensors()) {
        if (page_id >= ti.page_begin && page_id < ti.page_end) {
            return &ti;
        }
    }
    return nullptr;
}

int main() {
    const std::string model = find_model();
    if (model.empty()) {
        vm_skip("no GGUF model found (set LLAMA_VM_TEST_MODEL)");
    }
    std::printf("model: %s\n", model.c_str());

    const size_t page_size = 4 * 1024 * 1024;

    // tensor_data needs a contiguous block per tensor, so the cache must fit the largest tensor;
    // keep it as small as possible to force lots of paging
    size_t largest = largest_tensor(model);
    largest = (largest + page_size - 1) / page_size * page_size;
    largest = std::max(largest, 8 * page_size);
    std::printf("largest tensor: %zu MiB\n", largest >> 20);

    config cfg;
    cfg.page_size     = page_size;
    cfg.ram_capacity  = largest;
    cfg.vram_capacity = 0;
    cfg.swap_file     = "/tmp/llama-vm-test-gguf-swap";
    std::remove(cfg.swap_file.c_str());

    auto pg = pager::from_gguf(model, cfg);
    VM_CHECK(pg->n_pages() > 0);
    std::printf("arch: %s, name: %s, tensors: %zu, virtual size: %zu MiB, pages: %zu, cache: %zu MiB\n",
                pg->arch().c_str(), pg->name().c_str(),
                pg->tensors().size(), pg->virtual_size() >> 20, pg->n_pages(), largest >> 20);

    file_cache files;
    char buf[1024 * 1024];
    size_t n_ok = 0;
    for (const tensor_info & ti : pg->tensors()) {
        const int fd = files.get(ti.file);
        VM_CHECK(fd >= 0);
        // fault the whole tensor into the (much smaller) cache
        void * d = pg->tensor_data(ti.name, tier::ram);
        VM_CHECK(d != nullptr);
        // verify every chunk against a direct read of the tensor's own source file
        size_t done = 0;
        bool ok = true;
        while (done < ti.size) {
            const size_t n = std::min(sizeof(buf), ti.size - done);
            if (pread(fd, buf, n, static_cast<off_t>(ti.file_offset + done)) != static_cast<ssize_t>(n)) {
                ok = false;
                break;
            }
            if (std::memcmp(buf, static_cast<char *>(d) + done, n) != 0) {
                ok = false;
                break;
            }
            done += n;
        }
        if (!ok) {
            std::fprintf(stderr, "MISMATCH in tensor %s (file %s) at byte %zu\n",
                         ti.name.c_str(), ti.file.c_str(), done);
            return 1;
        }
        ++n_ok;
    }
    std::printf("verified %zu tensors through the pager\n", n_ok);

    // spot check with the safe read() API at random positions inside full pages;
    // compare against the page's own file offset (tensors are aligned in the file,
    // so the virtual address space does not map 1:1 onto file offsets)
    int n_spot = 0;
    for (int i = 0; i < 1000 && n_spot < 100; ++i) {
        const size_t pid = (static_cast<size_t>(i) * 2654435761u) % pg->n_pages();
        const page_info pi = pg->get_page_info(pid);
        if (pi.size < 4096) {
            continue; // partial page, keep the read within a single page
        }
        const tensor_info * ti = tensor_of_page(*pg, pid);
        VM_CHECK(ti != nullptr);
        const int fd = files.get(ti->file);
        VM_CHECK(fd >= 0);
        const size_t in_off = (static_cast<size_t>(i) * 2654435761u) % (pi.size - 4096 + 1);
        char a[4096], b[4096];
        pg->read(pi.vaddr + in_off, a, sizeof(a));
        VM_CHECK(pread(fd, b, sizeof(b), static_cast<off_t>(pi.file_offset + in_off)) == sizeof(b));
        VM_CHECK(std::memcmp(a, b, sizeof(a)) == 0);
        ++n_spot;
    }
    VM_CHECK(n_spot == 100);

    auto st = pg->get_stats();
    std::printf("stats: faults=%llu evictions=%llu tier_copies=%llu swap_in=%llu swap_out=%llu file_bytes=%.2f MiB\n",
                (unsigned long long) st.n_fault, (unsigned long long) st.n_evict,
                (unsigned long long) st.n_tier_copy, (unsigned long long) st.n_swap_in,
                (unsigned long long) st.n_swap_out,
                st.bytes_file / 1048576.0);
    VM_CHECK(st.n_fault > 0);
    VM_CHECK(st.n_evict > 0); // cache is smaller than the model, so evictions are expected

    std::remove(cfg.swap_file.c_str());
    std::printf("test_gguf: OK\n");
    return 0;
}
