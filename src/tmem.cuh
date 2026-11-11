#pragma once

#include <cuda_runtime.h>

// 32x32b.xN moves N 32-bit registers per lane, across a whole warp.
template<int N>
__device__ __forceinline__ void tmem_load(unsigned (&values)[N], unsigned address);
template<int N>
__device__ __forceinline__ void tmem_store(unsigned address, const unsigned (&values)[N]);

#define VALUES_1(M) M(0)
#define VALUES_8(M) VALUES_1(M), M(1), M(2), M(3), M(4), M(5), M(6), M(7)
#define VALUES_16(M) VALUES_8(M), M(8), M(9), M(10), M(11), M(12), M(13), M(14), M(15)
#define VALUES_32(M) VALUES_16(M), M(16), M(17), M(18), M(19), M(20), M(21), M(22), M(23), \
    M(24), M(25), M(26), M(27), M(28), M(29), M(30), M(31)
#define OUTPUT(I) "=r"(values[I])
#define INPUT(I) "r"(values[I])
#define TRANSFER(N, REGISTERS) \
    template<> \
    __device__ __forceinline__ void tmem_load<N>(unsigned (&values)[N], unsigned address) { \
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x" #N ".b32 {" REGISTERS "}, [%" #N "];" \
            : VALUES_##N(OUTPUT) : "r"(address) : "memory"); \
    } \
    template<> \
    __device__ __forceinline__ void tmem_store<N>(unsigned address, const unsigned (&values)[N]) { \
        asm volatile("tcgen05.st.sync.aligned.32x32b.x" #N ".b32 [%" #N "], {" REGISTERS "};" \
            : : VALUES_##N(INPUT), "r"(address) : "memory"); \
    }

#define REGISTERS_1 "%0"
#define REGISTERS_8 REGISTERS_1 ", %1, %2, %3, %4, %5, %6, %7"
#define REGISTERS_16 REGISTERS_8 ", %8, %9, %10, %11, %12, %13, %14, %15"
#define REGISTERS_32 REGISTERS_16 ", %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31"
TRANSFER(1, REGISTERS_1)
TRANSFER(8, REGISTERS_8)
TRANSFER(16, REGISTERS_16)
TRANSFER(32, REGISTERS_32)

#undef TRANSFER
#undef INPUT
#undef OUTPUT
#undef VALUES_1
#undef VALUES_8
#undef VALUES_16
#undef VALUES_32
#undef REGISTERS_1
#undef REGISTERS_8
#undef REGISTERS_16
#undef REGISTERS_32

__device__ __forceinline__ void tmem_wait_load() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_wait_store() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
