#include "arena.h"

#include "test-common.h"

using namespace llama_vm;

int main() {
    // basic alloc
    {
        char mem[1 << 20];
        arena ar;
        VM_CHECK(ar.init(mem, sizeof(mem)));
        VM_CHECK(ar.free_bytes() == sizeof(mem));

        const size_t a = ar.alloc(4096);
        VM_CHECK(a == 0);
        const size_t b = ar.alloc(8192);
        VM_CHECK(b == 4096);
        VM_CHECK(ar.free_bytes() == sizeof(mem) - 12288);
        VM_CHECK(ar.contains(a) && ar.contains(b));
        VM_CHECK(ar.size_of(b) == 8192);

        VM_CHECK(ar.free(a));
        VM_CHECK(!ar.contains(a));
        VM_CHECK(ar.free_bytes() == sizeof(mem) - 8192);

        // first fit reuses the freed block
        const size_t c = ar.alloc(4096);
        VM_CHECK(c == 0);
        VM_CHECK(ar.free(c));
    }

    // coalescing
    {
        char mem[64 * 1024];
        arena ar;
        VM_CHECK(ar.init(mem, sizeof(mem)));
        const size_t a = ar.alloc(8192);   // [0, 8K)
        const size_t b = ar.alloc(16384);  // [8K, 24K)
        const size_t c = ar.alloc(8192);   // [24K, 32K)
        VM_CHECK(a == 0 && b == 8192 && c == 24576);

        ar.free(b);
        ar.free(a);
        ar.free(c);
        // the three free blocks coalesced into one 32K block at the front
        const size_t d = ar.alloc(32768);
        VM_CHECK(d == 0);
        VM_CHECK(ar.free_bytes() == 32768);
        ar.free(d);
        VM_CHECK(ar.free_bytes() == sizeof(mem));
    }

    // fragmentation and large alloc failure
    {
        char mem[16 * 1024];
        arena ar;
        VM_CHECK(ar.init(mem, sizeof(mem)));
        const size_t a = ar.alloc(4096); // [0,4K)
        const size_t b = ar.alloc(4096); // [4K,8K)
        const size_t c = ar.alloc(4096); // [8K,12K)
        const size_t d = ar.alloc(4096); // [12K,16K)
        VM_CHECK(ar.alloc(1) == SIZE_MAX); // fully used
        ar.free(a);
        ar.free(c);
        // free: [0,4K) and [8K,12K); b and d still allocated -> no 16K contiguous
        VM_CHECK(ar.alloc(16384) == SIZE_MAX);
        ar.free(b);
        ar.free(d);
        // everything coalesced back into [0,16K)
        VM_CHECK(ar.alloc(16384) == 0);
    }

    // alignment
    {
        char mem[32 * 1024];
        arena ar;
        VM_CHECK(ar.init(mem, sizeof(mem)));
        for (size_t i = 1; i <= 7; ++i) {
            const size_t off = ar.alloc(i);
            VM_CHECK(off % 4096 == 0);
        }
        // size rounds up to 4096
        VM_CHECK(ar.alloc(1) == 7 * 4096);
    }

    // oversize
    {
        char mem[8192];
        arena ar;
        VM_CHECK(ar.init(mem, sizeof(mem)));
        VM_CHECK(ar.alloc(8193) == SIZE_MAX);
        VM_CHECK(ar.alloc(0) == 0); // 0 rounds up to 4096
    }

    std::printf("test_arena: OK\n");
    return 0;
}
