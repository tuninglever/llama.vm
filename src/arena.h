#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>

namespace llama_vm {

    // byte pool with page-aligned first-fit allocation and free block coalescing
    class arena {
    public:
        static constexpr size_t ALIGN = 4096;

        arena() = default;

        bool init(void * base, size_t capacity) {
            if (base == nullptr || capacity == 0 || capacity % ALIGN != 0) {
                return false;
            }
            base_ = static_cast<char *>(base);
            capacity_ = capacity;
            free_(0, capacity);
            return true;
        }

        void release() {
            allocs_.clear();
            frees_.clear();
            base_ = nullptr;
            capacity_ = 0;
            used_ = 0;
        }

        size_t alloc(size_t size) {
            size = (size + ALIGN - 1) & ~(ALIGN - 1);
            if (size > capacity_) {
                return SIZE_MAX;
            }
            for (auto it = frees_.begin(); it != frees_.end(); ++it) {
                if (it->second >= size) {
                    const size_t off  = it->first;
                    const size_t rest = it->second - size;
                    frees_.erase(it);
                    allocs_[off] = size;
                    if (rest != 0) {
                        free_(off + size, rest);
                    }
                    used_ += size;
                    return off;
                }
            }
            return SIZE_MAX;
        }

        bool free(size_t off) {
            auto it = allocs_.find(off);
            if (it == allocs_.end()) {
                return false;
            }
            const size_t size = it->second;
            allocs_.erase(it);
            used_ -= size;
            free_(off, size);
            return true;
        }

        bool   contains(size_t off) const { return allocs_.count(off) != 0; }
        size_t size_of(size_t off) const {
            auto it = allocs_.find(off);
            return it == allocs_.end() ? 0 : it->second;
        }
        // start offset of the allocation containing off (SIZE_MAX if none)
        size_t alloc_base(size_t off) const {
            auto it = allocs_.upper_bound(off);
            if (it == allocs_.begin()) {
                return SIZE_MAX;
            }
            --it;
            if (off >= it->first + it->second) {
                return SIZE_MAX;
            }
            return it->first;
        }

        size_t free_bytes() const { return capacity_ - used_; }
        size_t max_free() const {
            size_t m = 0;
            for (const auto & kv : frees_) {
                m = std::max(m, kv.second);
            }
            return m;
        }
        size_t capacity() const { return capacity_; }
        size_t used() const { return used_; }
        void * base() const { return base_; }
        char * base() { return base_; }

    private:
        // insert [off, off+size) into frees_, merging with adjacent blocks
        void free_(size_t off, size_t size) {
            auto it = frees_.lower_bound(off);
            if (it != frees_.end() && it->first == off + size) {
                size += it->second;
                it = frees_.erase(it);
            }
            if (it != frees_.begin()) {
                auto prev = it;
                --prev;
                if (prev->first + prev->second == off) {
                    off = prev->first;
                    size += prev->second;
                    it = frees_.erase(prev);
                }
            }
            frees_[off] = size;
        }

        char * base_ = nullptr;
        size_t capacity_ = 0;
        size_t used_ = 0;
        std::map<size_t, size_t> allocs_;
        std::map<size_t, size_t> frees_;
    };

} // namespace llama_vm
