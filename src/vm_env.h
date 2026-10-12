#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace llama_vm {

    // parse a size with optional K/M/G suffix (binary, 1024-based), eg "64M".
    // returns 0 for an empty or unparseable string
    inline size_t parse_size(const char * s) {
        if (s == nullptr || s[0] == '\0') {
            return 0;
        }
        char * end = nullptr;
        const unsigned long long v = std::strtoull(s, &end, 10);
        if (end == s) {
            return 0;
        }
        size_t mult = 1;
        switch (*end) {
            case 'K': case 'k': mult = 1ULL << 10;  break;
            case 'M': case 'm': mult = 1ULL << 20;  break;
            case 'G': case 'g': mult = 1ULL << 30;  break;
            case '\0':          mult = 1;           break;
            default:            return 0;
        }
        return static_cast<size_t>(v * mult);
    }

    // a size from an environment variable with K/M/G suffix support; 0 when unset
    inline size_t env_size(const char * name) {
        const char * e = std::getenv(name);
        return parse_size(e);
    }

    // round up to the arena alignment (4 KiB)
    inline size_t align4k(size_t v) {
        return (v + 4095) & ~(size_t) 4095;
    }

} // namespace llama_vm
