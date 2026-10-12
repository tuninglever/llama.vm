#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define VM_CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1); \
        } \
    } while (0)

#define VM_EXPECT_THROW(stmt) \
    do { \
        bool vm_threw_ = false; \
        try { \
            stmt; \
        } catch (const std::exception &) { \
            vm_threw_ = true; \
        } \
        if (!vm_threw_) { \
            std::fprintf(stderr, "FAIL %s:%d: expected exception from: %s\n", __FILE__, __LINE__, #stmt); \
            std::exit(1); \
        } \
    } while (0)

// deterministic byte pattern for synthetic files
static inline uint8_t vm_pat_byte(size_t i, int seed) {
    return static_cast<uint8_t>((i * 2654435761u + static_cast<uint64_t>(seed)) & 0xff);
}

static inline std::string vm_make_pattern_file(const std::string & path, size_t size, int seed) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot create %s\n", path.c_str());
        std::exit(1);
    }
    char buf[65536];
    size_t done = 0;
    while (done < size) {
        const size_t n = std::min(sizeof(buf), size - done);
        for (size_t i = 0; i < n; ++i) {
            buf[i] = static_cast<char>(vm_pat_byte(done + i, seed));
        }
        if (std::fwrite(buf, 1, n, f) != n) {
            std::fclose(f);
            std::fprintf(stderr, "write failed for %s\n", path.c_str());
            std::exit(1);
        }
        done += n;
    }
    std::fclose(f);
    return path;
}

static inline bool vm_check_pattern(const void * data, size_t vaddr, size_t len, int seed) {
    const auto * in = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < len; ++i) {
        if (in[i] != vm_pat_byte(vaddr + i, seed)) {
            return false;
        }
    }
    return true;
}

static inline void vm_skip(const char * why) {
    std::printf("SKIP: %s\n", why);
    std::exit(0);
}
