#include "pager-impl.h"

#include "gguf.h"

#include <cstdio>
#include <stdexcept>

namespace llama_vm {

    static void require_gguf(bool cond, const char * msg) {
        if (!cond) {
            throw std::runtime_error(msg);
        }
    }

    // GGUF split naming: "<prefix>-NNNNN-of-MMMMM.gguf" (1-indexed, 5 digits),
    // the same convention llama.cpp uses (llama_split_path / llama_split_prefix)
    static std::string split_path(const std::string & prefix, int split_no, int split_count) {
        char post[64];
        std::snprintf(post, sizeof(post), "-%05d-of-%05d.gguf", split_no + 1, split_count);
        return prefix + post;
    }

    // recover the prefix (everything before "-NNNNN-of-MMMMM.gguf") from a split path
    static std::string split_prefix(const std::string & path, int split_no, int split_count) {
        char post[64];
        std::snprintf(post, sizeof(post), "-%05d-of-%05d.gguf", split_no + 1, split_count);
        const std::string suffix(post);
        if (path.size() <= suffix.size()) {
            return "";
        }
        if (path.compare(path.size() - suffix.size(), suffix.size(), suffix) != 0) {
            return "";
        }
        return path.substr(0, path.size() - suffix.size());
    }

    // read a uint16 split key, returning def when absent or mistyped
    static uint16_t split_key_u16(struct gguf_context * ctx, const char * key, uint16_t def) {
        const int64_t id = gguf_find_key(ctx, key);
        if (id >= 0 && gguf_get_kv_type(ctx, id) == GGUF_TYPE_UINT16) {
            return gguf_get_val_u16(ctx, id);
        }
        return def;
    }

    // attach every tensor listed in `ctx` as a region backed by `file`; the tensor
    // offsets are relative to this file's own data section
    static void attach_tensors(pager * p, struct gguf_context * ctx, const std::string & file) {
        const size_t data_off = gguf_get_data_offset(ctx);
        const int64_t n_tensors = gguf_get_n_tensors(ctx);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            require_gguf(name != nullptr, "null tensor name");
            const size_t off  = gguf_get_tensor_offset(ctx, i);
            const size_t size = gguf_get_tensor_size(ctx, i);
            const int64_t * ne = gguf_get_tensor_ne(ctx, i);
            const int32_t type = static_cast<int32_t>(gguf_get_tensor_type(ctx, i));
            p->attach_region(name, file, data_off + off, size, ne, type);
        }
    }

    std::unique_ptr<pager> pager::from_gguf(const std::string & path, const config & cfg) {
        struct gguf_init_params params = {true, nullptr};
        struct gguf_context * ctx0 = gguf_init_from_file(path.c_str(), params);
        if (ctx0 == nullptr) {
            throw std::runtime_error("failed to load GGUF file: " + path);
        }

        auto p = std::unique_ptr<pager>(new pager(cfg));

        // model metadata (lives in the first split)
        const int64_t n_kv = gguf_get_n_kv(ctx0);
        for (int64_t i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(ctx0, i);
            if (key == nullptr) {
                continue;
            }
            if (std::string(key) == "general.architecture") {
                const int64_t id = gguf_find_key(ctx0, key);
                if (gguf_get_kv_type(ctx0, id) == GGUF_TYPE_STRING) {
                    p->p->arch_name = gguf_get_val_str(ctx0, id);
                }
            } else if (std::string(key) == "general.name") {
                const int64_t id = gguf_find_key(ctx0, key);
                if (gguf_get_kv_type(ctx0, id) == GGUF_TYPE_STRING) {
                    p->p->model_name = gguf_get_val_str(ctx0, id);
                }
            }
        }

        const uint16_t split_count = split_key_u16(ctx0, "split.count", 1);
        const uint16_t split_no    = split_key_u16(ctx0, "split.no", 0);

        if (split_count > 1) {
            if (split_no != 0) {
                gguf_free(ctx0);
                throw std::runtime_error("sharded model must be loaded with the first split "
                                         "(got split.no=" + std::to_string(split_no) + " in " + path + ")");
            }
            const std::string prefix = split_prefix(path, 0, split_count);
            if (prefix.empty()) {
                gguf_free(ctx0);
                throw std::runtime_error("failed to derive split prefix from: " + path);
            }
            for (uint16_t i = 0; i < split_count; ++i) {
                const std::string shard = split_path(prefix, i, split_count);
                struct gguf_context * ctx = ctx0;
                bool owns = false;
                if (i != 0) {
                    ctx = gguf_init_from_file(shard.c_str(), params);
                    owns = true;
                    if (ctx == nullptr) {
                        gguf_free(ctx0);
                        throw std::runtime_error("failed to load GGUF split: " + shard);
                    }
                    if (split_key_u16(ctx, "split.no", 0) != i) {
                        gguf_free(ctx);
                        gguf_free(ctx0);
                        throw std::runtime_error("invalid split file idx in " + shard);
                    }
                }
                attach_tensors(p.get(), ctx, shard);
                if (owns) {
                    gguf_free(ctx);
                }
            }
            gguf_free(ctx0);
        } else {
            attach_tensors(p.get(), ctx0, path);
            gguf_free(ctx0);
        }

        return p;
    }

} // namespace llama_vm
