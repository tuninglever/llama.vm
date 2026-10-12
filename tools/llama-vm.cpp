#include "llama_vm/llama_vm.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace llama_vm;

static void usage() {
    std::printf(
        "llama-vm - demand paging, LRU caching and swapping for LLM weights\n"
        "\n"
        "usage: llama-vm <command> [options]\n"
        "\n"
        "commands:\n"
        "  info MODEL       show model and tensor information\n"
        "  verify MODEL     page the whole model through the pager and verify the data\n"
        "  bench MODEL      measure paging throughput\n"
        "  pages MODEL      show the page table\n"
        "  plan MODEL       compute the tier capacity plan (here --ram/--vram are the\n"
        "                   *available* memory; the governor sizes the tiers from the model)\n"
        "  tiered MODEL     exercise RAM + VRAM + file together: fault pages into both\n"
        "                   tiers (needs --ram and --vram) and verify the bytes against\n"
        "                   the source file; set capacities below the model size to force\n"
        "                   LRU evictions and re-faults\n"
        "  demo             run a self-contained demo (no model needed)\n"
        "\n"
        "options:\n"
        "  --ram SIZE       RAM tier capacity (default 64M), 0 disables the tier\n"
        "  --vram SIZE      VRAM tier capacity (default 0 = disabled)\n"
        "  --backend NAME   vram backend: auto | hip | cuda | vulkan | cpu (default auto)\n"
        "  --page SIZE      page size (default 4M)\n"
        "  --swap FILE      swap file path (default /tmp/llama-vm-swap)\n"
        "  --no-swap        disable swapping\n"
        "  --prefetch       prefetch sequential pages in the background\n"
        "  --tensor NAME    restrict to one tensor (verify/bench)\n"
        "  --pattern NAME   bench pattern: seq | rand (default seq)\n"
        "  --pass N         bench passes (default 1)\n");
}

static size_t parse_size(const std::string & s, const char * what) {
    size_t mult = 1;
    size_t val = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (c >= '0' && c <= '9') {
            val = val * 10 + (c - '0');
        } else if (i == s.size() - 1) {
            switch (c) {
                case 'K': case 'k': mult = 1024; break;
                case 'M': case 'm': mult = 1024 * 1024; break;
                case 'G': case 'g': mult = 1024 * 1024 * 1024; break;
                default:
                    std::fprintf(stderr, "bad %s size: %s\n", what, s.c_str());
                    std::exit(2);
            }
        } else {
            std::fprintf(stderr, "bad %s size: %s\n", what, s.c_str());
            std::exit(2);
        }
    }
    return val * mult;
}

struct args {
    std::string model;
    std::string cmd;
    size_t ram = 64 * 1024 * 1024;
    size_t vram = 0;
    std::string backend = "auto";
    size_t page = 4 * 1024 * 1024;
    std::string swap = "/tmp/llama-vm-swap";
    bool no_swap = false;
    bool prefetch = false;
    std::string tensor;
    std::string pattern = "seq";
    int pass = 1;
};

static args parse_args(int argc, char ** argv) {
    if (argc < 2) {
        usage();
        std::exit(2);
    }
    args a;
    a.cmd = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&] {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", k.c_str());
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (k == "--ram") {
            a.ram = parse_size(next(), "ram");
        } else if (k == "--vram") {
            a.vram = parse_size(next(), "vram");
        } else if (k == "--backend") {
            a.backend = next();
        } else if (k == "--page") {
            a.page = parse_size(next(), "page");
        } else if (k == "--swap") {
            a.swap = next();
        } else if (k == "--no-swap") {
            a.no_swap = true;
        } else if (k == "--prefetch") {
            a.prefetch = true;
        } else if (k == "--tensor") {
            a.tensor = next();
        } else if (k == "--pattern") {
            a.pattern = next();
        } else if (k == "--pass") {
            a.pass = std::atoi(next().c_str());
        } else if (k == "--help" || k == "-h") {
            usage();
            std::exit(0);
        } else {
            a.model = k;
        }
    }
    if (a.ram == 0 && a.vram == 0) {
        std::fprintf(stderr, "at least one tier must be enabled\n");
        std::exit(2);
    }
    return a;
}

static config make_config(const args & a) {
    config cfg;
    cfg.page_size     = a.page;
    cfg.ram_capacity  = a.ram;
    cfg.vram_capacity = a.vram;
    cfg.vram_backend  = a.backend;
    if (!a.no_swap) {
        cfg.swap_file = a.swap;
    }
    cfg.prefetch = a.prefetch;
    return cfg;
}

static std::unique_ptr<pager> make_pager(const args & a) {
    if (a.cmd == "demo") {
        return nullptr;
    }
    if (a.model.empty()) {
        std::fprintf(stderr, "no model given\n");
        std::exit(2);
    }
    try {
        return pager::from_gguf(a.model, make_config(a));
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        std::exit(1);
    }
}

static void print_stats(const stats & st, double secs) {
    std::printf("  page faults:   %llu\n", (unsigned long long) st.n_fault);
    std::printf("  tier copies:   %llu\n", (unsigned long long) st.n_tier_copy);
    std::printf("  evictions:     %llu\n", (unsigned long long) st.n_evict);
    std::printf("  swap in/out:   %llu / %llu\n", (unsigned long long) st.n_swap_in, (unsigned long long) st.n_swap_out);
    std::printf("  bytes:         file %.2f MiB, tier %.2f MiB, swap %.2f / %.2f MiB\n",
                st.bytes_file / 1048576.0, st.bytes_tier / 1048576.0,
                st.bytes_swap_in / 1048576.0, st.bytes_swap_out / 1048576.0);
    if (secs > 0) {
        std::printf("  throughput:    %.2f MiB/s (%llu bytes in %.3f s)\n",
                    (st.bytes_file + st.bytes_tier + st.bytes_swap_in) / 1048576.0 / secs,
                    (unsigned long long) (st.bytes_file + st.bytes_tier + st.bytes_swap_in), secs);
    }
}

// ------------------------------ info ------------------------------------

static int cmd_info(const args & a) {
    auto pg = make_pager(a);
    std::printf("model:      %s\n", a.model.c_str());
    std::printf("arch:       %s\n", pg->arch().c_str());
    std::printf("name:       %s\n", pg->name().c_str());
    std::printf("tensors:    %zu\n", pg->tensors().size());
    std::printf("size:       %.2f MiB (%zu bytes)\n", pg->virtual_size() / 1048576.0, pg->virtual_size());
    std::printf("page size:  %zu bytes, pages: %zu\n", pg->get_config().page_size, pg->n_pages());
    std::printf("\n%-40s %-8s %12s\n", "tensor", "type", "size");
    size_t total = 0;
    for (const tensor_info & ti : pg->tensors()) {
        std::printf("%-40s %8s %12zu\n", ti.name.c_str(), "-", ti.size);
        total += ti.size;
    }
    std::printf("%-40s %8s %12zu\n", "total", "-", total);
    return 0;
}

// find the tensor that owns a page (tensors store their page ranges); a page's
// file_offset is relative to that tensor's source file (which may be a shard)
static const tensor_info * tensor_of_page(const pager & pg, size_t page_id) {
    for (const tensor_info & ti : pg.tensors()) {
        if (page_id >= ti.page_begin && page_id < ti.page_end) {
            return &ti;
        }
    }
    return nullptr;
}

// ------------------------------ verify ----------------------------------

static int cmd_verify(const args & a) {
    auto pg = make_pager(a);
    const tier t = a.vram > 0 ? tier::vram : tier::ram;

    std::printf("verifying %s through %s tier (ram=%zu MiB, vram=%zu MiB, page=%zu KiB)\n",
                a.model.c_str(), t == tier::vram ? "VRAM" : "RAM",
                a.ram >> 20, a.vram >> 20, a.page >> 10);

    // open each tensor's own source file (a sharded model's tensors live in
    // different files), caching one FILE* per path
    std::map<std::string, FILE *> files;
    auto open_file = [&](const std::string & path) -> FILE * {
        auto it = files.find(path);
        if (it != files.end()) {
            return it->second;
        }
        FILE * f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            std::fprintf(stderr, "cannot open %s for direct reads\n", path.c_str());
        }
        files.emplace(path, f);
        return f;
    };

    char buf[1 << 20];
    size_t n_ok = 0, n_fail = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (const tensor_info & ti : pg->tensors()) {
        if (!a.tensor.empty() && ti.name != a.tensor) {
            continue;
        }
        FILE * f = open_file(ti.file);
        if (f == nullptr) {
            ++n_fail;
            continue;
        }
        void * d = pg->tensor_data(ti.name, t);
        if (d == nullptr) {
            std::fprintf(stderr, "FAIL: could not materialize %s\n", ti.name.c_str());
            ++n_fail;
            continue;
        }
        size_t done = 0;
        bool ok = true;
        while (done < ti.size) {
            const size_t n = std::min(sizeof(buf), ti.size - done);
            if (std::fseek(f, (long) (ti.file_offset + done), SEEK_SET) != 0) {
                ok = false;
                break;
            }
            if (std::fread(buf, 1, n, f) != n) {
                ok = false;
                break;
            }
            if (std::memcmp(buf, static_cast<char *>(d) + done, n) != 0) {
                ok = false;
                break;
            }
            done += n;
        }
        if (ok) {
            ++n_ok;
        } else {
            ++n_fail;
            std::fprintf(stderr, "MISMATCH: %s at byte %zu of %zu\n", ti.name.c_str(), done, ti.size);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    for (auto & kv : files) {
        std::fclose(kv.second);
    }

    std::printf("tensors: %zu ok, %zu failed\n", n_ok, n_fail);
    print_stats(pg->get_stats(), secs);
    return n_fail == 0 ? 0 : 1;
}

// ------------------------------ bench -----------------------------------

static int cmd_bench(const args & a) {
    auto pg = make_pager(a);
    const tier t = a.vram > 0 ? tier::vram : tier::ram;
    const size_t n = pg->n_pages();
    if (n == 0) {
        std::fprintf(stderr, "no pages\n");
        return 1;
    }

    std::printf("bench %s: %zu pages of %zu KiB, tier=%s, pattern=%s, pass=%d\n",
                a.model.c_str(), n, a.page >> 10,
                t == tier::vram ? "vram" : "ram", a.pattern.c_str(), a.pass);

    std::mt19937_64 rng(1234);
    auto t0 = std::chrono::steady_clock::now();
    for (int pass = 0; pass < a.pass; ++pass) {
        for (size_t i = 0; i < n; ++i) {
            const size_t pid = a.pattern == "rand" ? rng() % n : i;
            void * d = pg->resolve_page(pid, t);
            // touch a few bytes so nothing gets optimized away
            volatile char c = static_cast<char *>(d)[0];
            (void) c;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    print_stats(pg->get_stats(), secs);
    std::printf("tier used: %zu / %zu MiB\n", pg->tier_used(t) >> 20, pg->tier_capacity(t) >> 20);
    return 0;
}

// ------------------------------ pages -----------------------------------

static int cmd_pages(const args & a) {
    auto pg = make_pager(a);
    std::printf("%8s %16s %12s %6s %6s %5s %5s %5s\n",
                "page", "vaddr", "file_off", "size", "dirty", "pin", "ram", "vram");
    const size_t n = std::min<size_t>(pg->n_pages(), 64);
    for (size_t i = 0; i < n; ++i) {
        page_info pi = pg->get_page_info(i);
        std::printf("%8zu %16zu %12zu %6zu %5d %5d %5d %5d\n",
                    i, pi.vaddr, pi.file_offset, pi.size,
                    pi.dirty ? 1 : 0, pi.pinned ? 1 : 0, pi.in_ram ? 1 : 0, pi.in_vram ? 1 : 0);
    }
    if (pg->n_pages() > n) {
        std::printf("... (%zu more pages)\n", pg->n_pages() - n);
    }
    return 0;
}

// ------------------------------ plan ------------------------------------

static int cmd_plan(const args & a) {
    // only the model size is needed, so use a minimal tier (no pages are faulted)
    config cfg;
    cfg.page_size     = a.page;
    cfg.ram_capacity  = 4096;
    cfg.vram_capacity = 0;
    auto pg = pager::from_gguf(a.model, cfg);
    const size_t model_size = pg->virtual_size();

    // here --ram/--vram are the *available* memory; the governor sizes the tiers
    const capacity_plan p = plan_capacity(model_size, a.ram, a.vram, a.page);
    std::printf("model:      %s\n", a.model.c_str());
    std::printf("arch:       %s\n", pg->arch().c_str());
    std::printf("model size: %.2f MiB (%zu bytes), %zu tensors\n",
                model_size / 1048576.0, model_size, pg->tensors().size());
    std::printf("available:  ram=%.2f MiB, vram=%.2f MiB\n",
                a.ram / 1048576.0, a.vram / 1048576.0);
    std::printf("plan:       ram=%.2f MiB, vram=%.2f MiB (total %.2f MiB)\n",
                p.ram_capacity / 1048576.0, p.vram_capacity / 1048576.0, p.total_capacity / 1048576.0);
    std::printf("fits:       %s\n",
                p.fits ? "yes — the whole model fits in the tiers (no paging)"
                       : "no — the excess pages out to the file/swap");
    return 0;
}

// ------------------------------ tiered ----------------------------------

static int cmd_tiered(const args & a) {
    if (a.ram == 0 || a.vram == 0) {
        std::fprintf(stderr, "tiered needs both --ram and --vram enabled\n");
        return 2;
    }
    auto pg = make_pager(a);
    const size_t n = pg->n_pages();
    if (n == 0) {
        std::fprintf(stderr, "no pages\n");
        return 1;
    }
    const size_t model = pg->virtual_size();
    std::printf("tiered %s\n", a.model.c_str());
    std::printf("  model  : %zu MiB (%zu pages of %zu MiB)\n", model >> 20, n, a.page >> 20);
    std::printf("  ram    : %zu MiB\n", a.ram >> 20);
    std::printf("  vram   : %zu MiB (%s)\n", a.vram >> 20, a.backend.c_str());
    std::printf("  paging : %s\n", model > a.ram + a.vram ? "yes (model > ram+vram)" : "NO (model fits in the tiers)");

    // Phase 1: fault every page into RAM. When the model exceeds the RAM capacity this
    // forces LRU evictions back to the source file (the pages stream through the RAM tier).
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n; ++i) {
        void * d = pg->resolve_page(i, tier::ram);
        volatile char c = static_cast<char *>(d)[0];
        (void) c;
    }
    const double s1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Phase 2: re-fault every page into VRAM. A page still resident in RAM is copied
    // across (RAM -> VRAM tier copy); one that was evicted is loaded from the file into
    // VRAM. This exercises the tier-copy path plus the GPU load path.
    auto t1 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n; ++i) {
        void * d = pg->resolve_page(i, tier::vram);
        volatile char c = static_cast<char *>(d)[0];
        (void) c;
    }
    const double s2 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

    // Phase 3: re-read every page through the safe read path (RAM, then VRAM, then
    // file/swap) and verify the bytes against the source file (each page's own file,
    // which for a sharded model is its shard).
    std::map<std::string, FILE *> files;
    auto open_file = [&](const std::string & path) -> FILE * {
        auto it = files.find(path);
        if (it != files.end()) {
            return it->second;
        }
        FILE * f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            std::fprintf(stderr, "cannot open %s for verification\n", path.c_str());
        }
        files.emplace(path, f);
        return f;
    };
    char pbuf[1 << 20], fbuf[1 << 20];
    size_t n_bad = 0;
    auto t2 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n && n_bad == 0; ++i) {
        const page_info pi = pg->get_page_info(i);
        const tensor_info * ti = tensor_of_page(*pg, i);
        if (ti == nullptr) {
            ++n_bad;
            break;
        }
        FILE * f = open_file(ti->file);
        if (f == nullptr) {
            ++n_bad;
            break;
        }
        for (size_t off = 0; off < pi.size; off += sizeof(pbuf)) {
            const size_t m = std::min(sizeof(pbuf), pi.size - off);
            pg->read(pi.vaddr + off, pbuf, m);
            if (std::fseek(f, (long) (pi.file_offset + off), SEEK_SET) != 0 || std::fread(fbuf, 1, m, f) != m) {
                ++n_bad;
                break;
            }
            if (std::memcmp(pbuf, fbuf, m) != 0) {
                ++n_bad;
                break;
            }
        }
    }
    const double s3 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
    for (auto & kv : files) {
        std::fclose(kv.second);
    }

    std::printf("phase1 (fault into both tiers)   : %.2f s\n", s1);
    std::printf("phase2 (re-fault opposite tier)  : %.2f s\n", s2);
    std::printf("phase3 (verify %zu pages)        : %.2f s  %s\n", n, s3, n_bad ? "MISMATCH" : "OK");
    print_stats(pg->get_stats(), 0.0);
    std::printf("tier used: ram=%zu/%zu MiB, vram=%zu/%zu MiB\n",
                pg->tier_used(tier::ram) >> 20, pg->tier_capacity(tier::ram) >> 20,
                pg->tier_used(tier::vram) >> 20, pg->tier_capacity(tier::vram) >> 20);
    return n_bad ? 1 : 0;
}

// ------------------------------ demo ------------------------------------

static int cmd_demo(const args & a) {
    (void) a;
    std::printf("demo: demand paging with LRU and swap\n");

    // build a synthetic "model" with a known pattern
    const std::string model = "/tmp/llama-vm-demo-model.bin";
    const std::string swap  = "/tmp/llama-vm-demo-swap";
    const size_t size = 16 * 1024 * 1024;
    {
        FILE * f = std::fopen(model.c_str(), "wb");
        if (f == nullptr) {
            std::fprintf(stderr, "cannot create demo model\n");
            return 1;
        }
        char buf[4096];
        for (size_t off = 0; off < size; off += sizeof(buf)) {
            for (size_t i = 0; i < sizeof(buf); ++i) {
                buf[i] = static_cast<char>((off + i) & 0xff);
            }
            const size_t n = std::min(sizeof(buf), size - off);
            std::fwrite(buf, 1, n, f);
        }
        std::fclose(f);
    }
    std::remove(swap.c_str());

    config cfg;
    cfg.page_size     = 4 * 1024 * 1024;
    cfg.ram_capacity  = 4 * 1024 * 1024; // cache holds 1 page of the 4-page model
    cfg.vram_capacity = 0;
    cfg.swap_file     = swap;
    cfg.prefetch      = true;

    pager pg(cfg);
    pg.attach_region("model", model, 0, size);
    std::printf("model: 4 pages of 4 MiB, cache: 4 MiB, swap: %s\n", swap.c_str());

    // walk through the whole model, the pager must page it in and evict
    char buf[4096];
    for (size_t off = 0; off < size; off += 4096) {
        pg.read(off, buf, sizeof(buf));
        for (size_t i = 0; i < sizeof(buf); ++i) {
            if (buf[i] != static_cast<char>((off + i) & 0xff)) {
                std::fprintf(stderr, "demo: data mismatch at %zu\n", off + i);
                return 1;
            }
        }
    }
    std::printf("full walk: OK\n");

    // write, evict, and read back through the swap file
    char pat[256];
    std::memset(pat, 0x7E, sizeof(pat));
    pg.write(5 * 1024 * 1024, pat, sizeof(pat));
    // evict page 1 by loading page 3
    pg.read(3 * 1024 * 1024, buf, sizeof(buf));
    char back[256] = {};
    pg.read(5 * 1024 * 1024, back, sizeof(back));
    if (std::memcmp(pat, back, sizeof(pat)) != 0) {
        std::fprintf(stderr, "demo: dirty data lost through swap\n");
        return 1;
    }
    std::printf("dirty swap round-trip: OK\n");

    auto st = pg.get_stats();
    print_stats(st, 0.0);

    std::remove(model.c_str());
    std::remove(swap.c_str());
    std::printf("demo: OK\n");
    return 0;
}

int main(int argc, char ** argv) {
    args a = parse_args(argc, argv);
    if (a.cmd == "info") {
        return cmd_info(a);
    }
    if (a.cmd == "verify") {
        return cmd_verify(a);
    }
    if (a.cmd == "bench") {
        return cmd_bench(a);
    }
    if (a.cmd == "pages") {
        return cmd_pages(a);
    }
    if (a.cmd == "plan") {
        return cmd_plan(a);
    }
    if (a.cmd == "tiered") {
        return cmd_tiered(a);
    }
    if (a.cmd == "demo") {
        return cmd_demo(a);
    }
    usage();
    return 2;
}
