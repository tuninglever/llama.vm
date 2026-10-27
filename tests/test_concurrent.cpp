#include "llama_vm/llama_vm.h"

#include "test-common.h"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace llama_vm;

// Concurrency stress test: several readers and one writer share a single
// pager instance (the real deployment model). The cache is small enough that
// every access faults and evicts, and the writer's dirty pages are swapped out
// and re-faulted while readers run.
//
// Safety notes:
//  - read()/write() are copy-based and hold the pager lock for the whole call,
//    so a page is always observed in a consistent state (fully pre- or fully
//    post-write).
//  - tensor_data() pointers are deliberately NOT dereferenced while other
//    threads are running (they may be evicted under us); the call itself still
//    exercises the gather + block-eviction path.
//
// Data model: the file has 8 pages, page i filled with byte (0x10 + i).
// The writer repeatedly overwrites pages 0 and 4 with 0xEE. So:
//   - pages 0 and 4 must always be observed uniformly as their file byte or
//     0xEE (the writer's value) -- never mixed, never a third value;
//   - every other page must always match its file fill byte.
// A torn page, a freed/reused arena slice, or a lost swap write-back would
// produce a mixed or wrong byte and fail the check.

static const size_t PAGE = 1024 * 1024;
static const int NREADERS = 3;
static const int ITERS = 12;

static const int FILE_BYTE[8] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17};
static const int WRITER_BYTE  = 0xEE;
static const int CONTESTED[8] = {1, 0, 0, 0, 1, 0, 0, 0}; // pages 0 and 4

struct error_log {
    std::mutex m;
    std::string first;
    void record(const std::string & msg) {
        std::lock_guard<std::mutex> lock(m);
        if (first.empty()) {
            first = msg;
        }
    }
};

static void make_fill_file(const std::string & path, size_t n_pages) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot create %s\n", path.c_str());
        std::exit(1);
    }
    std::vector<char> buf(PAGE);
    for (size_t p = 0; p < n_pages; ++p) {
        std::memset(buf.data(), static_cast<char>(FILE_BYTE[p]), PAGE);
        if (std::fwrite(buf.data(), 1, PAGE, f) != PAGE) {
            std::fclose(f);
            std::fprintf(stderr, "write failed for %s\n", path.c_str());
            std::exit(1);
        }
    }
    std::fclose(f);
}

// a page is valid if it is uniformly one of its allowed bytes
static bool check_page(const void * data, int page) {
    const auto * in = static_cast<const uint8_t *>(data);
    const int allowed0 = FILE_BYTE[page];
    const int allowed1 = CONTESTED[page] ? WRITER_BYTE : -1;
    const int first = in[0];
    if (first != allowed0 && first != allowed1) {
        return false;
    }
    for (size_t i = 0; i < PAGE; ++i) {
        if (in[i] != static_cast<uint8_t>(first)) {
            return false;
        }
    }
    return true;
}

int main() {
    const std::string dir = "/tmp/llama-vm-conc-test-";
    const std::string model = dir + "model.bin";
    const std::string swap  = dir + "swap";
    std::remove(model.c_str());
    std::remove(swap.c_str());

    make_fill_file(model, 8);

    // cache holds exactly one 2-page tensor block: constant fault + eviction
    config cfg;
    cfg.page_size    = PAGE;
    cfg.ram_capacity = 2 * PAGE;
    cfg.vram_capacity = 0;
    cfg.swap_file    = swap;

    pager shared(cfg);
    shared.attach_region("t0", model, 0, 2 * PAGE);
    shared.attach_region("t1", model, 2 * PAGE, 2 * PAGE);
    shared.attach_region("t2", model, 4 * PAGE, 2 * PAGE);
    shared.attach_region("t3", model, 6 * PAGE, 2 * PAGE);

    error_log errs;

    std::vector<std::thread> readers;
    for (int r = 0; r < NREADERS; ++r) {
        readers.emplace_back([&, r] {
            try {
                std::vector<char> buf(PAGE);
                for (int it = 0; it < ITERS; ++it) {
                    for (int k = 0; k < 4; ++k) {
                        const std::string name = std::string("t") + static_cast<char>('0' + k);
                        // exercise gather + block eviction; do not use the pointer
                        // while other threads run (it may be evicted under us)
                        shared.tensor_data(name, tier::ram);

                        // safe, locked verification of both pages of the tensor
                        if (!shared.read(k * 2 * PAGE, buf.data(), PAGE) || !check_page(buf.data(), 2 * k)) {
                            errs.record("reader " + std::to_string(r) + ": page " + std::to_string(2 * k) + " invalid");
                            return;
                        }
                        if (!shared.read(k * 2 * PAGE + PAGE, buf.data(), PAGE) || !check_page(buf.data(), 2 * k + 1)) {
                            errs.record("reader " + std::to_string(r) + ": page " + std::to_string(2 * k + 1) + " invalid");
                            return;
                        }
                    }
                }
            } catch (const std::exception & e) {
                errs.record("reader " + std::to_string(r) + ": " + e.what());
            } catch (...) {
                errs.record("reader " + std::to_string(r) + ": unknown exception");
            }
        });
    }

    std::thread writer([&] {
        try {
            std::vector<char> buf(PAGE, WRITER_BYTE);
            for (int it = 0; it < ITERS; ++it) {
                // dirty both contested pages, then fault a third so the cache
                // (2 pages) must evict a dirty page -> swap write-back + re-fault
                shared.write(0, buf.data(), PAGE);
                shared.write(4 * PAGE, buf.data(), PAGE);
                shared.resolve_page(2, tier::ram);
                shared.resolve_page(6, tier::ram);
            }
        } catch (const std::exception & e) {
            errs.record(std::string("writer: ") + e.what());
        } catch (...) {
            errs.record("writer: unknown exception");
        }
    });

    for (auto & t : readers) {
        t.join();
    }
    writer.join();

    if (!errs.first.empty()) {
        std::fprintf(stderr, "FAIL concurrent: %s\n", errs.first.c_str());
        std::remove(model.c_str());
        std::remove(swap.c_str());
        return 1;
    }

    // quiescent final verification: every page must be uniformly one of its
    // allowed bytes (file byte, or the writer's 0xEE for the contested pages)
    {
        std::vector<char> buf(PAGE);
        for (int p = 0; p < 8; ++p) {
            shared.read(p * PAGE, buf.data(), PAGE);
            if (!check_page(buf.data(), p)) {
                std::fprintf(stderr, "FAIL concurrent: final page %d invalid\n", p);
                return 1;
            }
        }
        // tensor_data contiguity + correctness in the quiescent state
        void * d = shared.tensor_data("t2", tier::ram);
        const auto * t2 = static_cast<const uint8_t *>(d);
        if (!check_page(t2, 4) || !check_page(t2 + PAGE, 5)) {
            std::fprintf(stderr, "FAIL concurrent: tensor_data(t2) data invalid\n");
            return 1;
        }
    }

    auto st = shared.get_stats();
    std::printf("test_concurrent: OK (faults=%llu evicts=%llu swap_out=%llu swap_in=%llu)\n",
        (unsigned long long) st.n_fault, (unsigned long long) st.n_evict,
        (unsigned long long) st.n_swap_out, (unsigned long long) st.n_swap_in);
    std::remove(model.c_str());
    std::remove(swap.c_str());
    return 0;
}
