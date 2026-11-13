#include "tmem.cuh"
#include <cuda_fp8.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>

namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Result { unsigned long long clocks, ns; unsigned errors[128]; };
__device__ unsigned shared_address(const void* p) { return unsigned(__cvta_generic_to_shared(p)); }
__device__ unsigned long long timer(bool ns) {
    unsigned long long t;
    if (ns) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    else asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}
// K-major, unswizzled: eight rows of 16-byte vectors, then the next K vector.
__device__ int offset(int row, int k) { return row / 8 * 1024 + k / 16 * 128 + row % 8 * 16 + k % 16; }
__device__ unsigned long long descriptor(const void* p) {
    return (shared_address(p) >> 4) | (8ull << 16) | (64ull << 32) | (1ull << 46);
}
__device__ void commit_wait(unsigned bar, unsigned phase) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(bar) : "memory");
    asm volatile("{ .reg .pred p; wait_loop: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; "
                 "@!p bra wait_loop; }" :: "r"(bar), "r"(phase) : "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ void mma(unsigned d, unsigned long long a, unsigned long long b, bool accumulate) {
    // FP32 output, E4M3 A/B, M=128, N=128; native K=32.
    constexpr unsigned idesc = (1u << 4) | (16u << 17) | (8u << 24);
    asm volatile("{ .reg .pred p; setp.ne.u32 p, %4, 0; "
                 "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p; }"
                 :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(unsigned(accumulate)) : "memory");
}

template<int Atoms, bool WaitEach>
__global__ void measure(Result* result, int repeats, unsigned clock_mask) {
    __shared__ __align__(128) unsigned char a[16384], b[16384];
    __shared__ unsigned base;
    __shared__ unsigned long long barrier;
    const int tid = threadIdx.x;
    for (int i = tid; i < 16384; i += 128) {
        const int row = i / 128, k = i % 128;
        a[offset(row, k)] = __nv_fp8_e4m3(float((row % 3 + 1) * (k % 2 + 1))).__x;
        b[offset(row, k)] = __nv_fp8_e4m3(float((row % 5 + 1) * (k % 2 ? 1 : -1))).__x;
    }
    const unsigned bar = shared_address(&barrier);
    if (tid == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
    if (tid < 32) {
        const unsigned holding = shared_address(&base);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;" :: "r"(holding) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        const auto da = descriptor(a), db = descriptor(b);
        const auto start_ns = timer(true), start = timer(false);
        const unsigned destination = base ^ (unsigned(start ^ start_ns) & clock_mask);
        #pragma unroll 1
        for (int iteration = 0; iteration < repeats; ++iteration) {
            #pragma unroll
            for (int atom = 0; atom < Atoms; ++atom)
                mma(destination, da + atom * 16, db + atom * 16, iteration != 0 || atom != 0);
            if constexpr (WaitEach) commit_wait(bar, iteration & 1);
        }
        if constexpr (!WaitEach) commit_wait(bar, 0);
        const auto end = timer(false), end_ns = timer(true);
        result->clocks = end - start;
        result->ns = end_ns - start_ns;
    }
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    unsigned errors = 0;
    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        unsigned words[32];
        tmem_load(words, base + ((tid / 32 * 32) << 16) + chunk * 32);
        tmem_wait_load();
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const float expected = float(16 * Atoms * repeats * (tid % 3 + 1) * ((chunk * 32 + i) % 5 + 1));
            errors += __uint_as_float(words[i]) != expected;
        }
    }
    result->errors[tid] = errors;
    __syncthreads();
    if (tid < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" :: "r"(base) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
}
struct Timing { double clocks, ns; };
template<int Atoms, bool WaitEach>
Timing bench(Result* device, int repeats) {
    std::array<double, 31> clocks{}, ns{};
    for (int i = -5; i < int(clocks.size()); ++i) {
        measure<Atoms, WaitEach><<<1, 128>>>(device, repeats, 0);
        check(cudaGetLastError());
        Result r{};
        check(cudaMemcpy(&r, device, sizeof(r), cudaMemcpyDeviceToHost));
        for (auto e : r.errors) if (e) throw std::runtime_error("MMA output validation failed");
        if (i >= 0) { clocks[i] = r.clocks; ns[i] = r.ns; }
    }
    std::sort(clocks.begin(), clocks.end()); std::sort(ns.begin(), ns.end());
    return {clocks[15], ns[15]};
}

template<int Atoms, bool WaitEach>
void report(Result* device) {
    // Small runs check first-iteration initialization and barrier phases too.
    bench<Atoms, WaitEach>(device, 1); bench<Atoms, WaitEach>(device, 2);
    const auto a = bench<Atoms, WaitEach>(device, 256), b = bench<Atoms, WaitEach>(device, 512);
    const double clocks = (b.clocks - a.clocks) / 256;
    const double ns = (b.ns - a.ns) / 256;
    if (clocks <= 0 || ns <= 0) throw std::runtime_error("Invalid timing slope");
    std::printf("128x128x%-3d %-15s %8.2f %9.2f %12.2f\n", Atoms * 32,
                WaitEach ? "wait each" : "queue all", clocks, ns,
                (2.0 * 128 * 128 * Atoms * 32) / ns / 1000);
}
} // namespace
int main() {
    try {
        cudaDeviceProp prop; check(cudaGetDeviceProperties(&prop, 0));
        std::printf("%s: FP8 E4M3 x E4M3 -> FP32, SMEM/SMEM, one CTA\n", prop.name);
        std::puts("Shape        Mode              Clocks        ns   TFLOP/s/SM");
        Result* device; check(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(Result)));
        report<1, true>(device); report<1, false>(device);
        report<4, true>(device); report<4, false>(device);
        check(cudaFree(device));
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
