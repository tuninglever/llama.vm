#pragma once

#include <cstddef>

namespace llama_vm {

    struct vram_tier;
    struct config;

    // init a VRAM tier over a ggml backend buffer, returns false on failure
    bool vram_tier_init(vram_tier & v, const config & cfg);
    void vram_tier_free(vram_tier & v);

    // host -> device copy into [off, off+size)
    void vram_copy_in(vram_tier & v, size_t off, size_t size, const void * host);
    // device -> host copy out of [off, off+size)
    void vram_copy_out(vram_tier & v, size_t off, size_t size, void * host);

} // namespace llama_vm
