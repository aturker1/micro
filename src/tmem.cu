#include "tmem.cuh"

#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>

namespace {
constexpr int iterations = 4096;
struct Result { unsigned long long clocks; unsigned errors; };

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

__device__ __forceinline__ unsigned long long clock_now() {
    unsigned long long clock;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(clock) : : "memory");
    return clock;
}

template<int N, bool Store>
__global__ void measure(Result* results, unsigned clock_mask) {
    // One warp, one CTA: no other warps to hide transfer completion time.
    __shared__ unsigned base;
    const unsigned shared = unsigned(__cvta_generic_to_shared(&base));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
                 : : "r"(shared) : "memory");
    __syncwarp();
    unsigned address = base;
    unsigned values[N];
    #pragma unroll
    for (int i = 0; i < N; ++i) values[i] = threadIdx.x * 64 + i;
    values[0] = address; // Uniform pointer chase: each load feeds the next load's address.

    tmem_store(address, values);
    tmem_wait_store();
    tmem_load(values, address);
    tmem_wait_load();

    const auto start = clock_now();
    // Runtime zero creates a dependency on the timestamp for the first transfer.
    address ^= unsigned(start) & clock_mask;
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        if constexpr (Store) {
            tmem_store(address, values);
            tmem_wait_store();
        } else {
            tmem_load(values, address);
            tmem_wait_load();
            address = values[0];
        }
    }
    const auto end = clock_now();

    // Check every register in every lane, outside the timed region.
    if constexpr (Store) {
        tmem_load(values, address);
        tmem_wait_load();
    }
    unsigned errors = 0;
    #pragma unroll
    for (int i = 0; i < N; ++i)
        errors += values[i] != (i == 0 ? base : threadIdx.x * 64 + i);
    results[threadIdx.x] = {end - start, errors};
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"
                 : : "r"(base) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}

template<int N, bool Store>
float bench(Result* device) {
    std::array<double, 9> samples{};
    for (int sample = -1; sample < int(samples.size()); ++sample) {
        measure<N, Store><<<1, 32>>>(device, 0);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        std::array<Result, 32> results{};
        check(cudaMemcpy(results.data(), device, sizeof(results), cudaMemcpyDeviceToHost));
        unsigned long long clocks = 0;
        for (const auto& result : results) {
            if (result.errors) throw std::runtime_error("TMEM round-trip validation failed");
            clocks = std::max(clocks, result.clocks);
        }
        if (sample >= 0) samples[sample] = double(clocks) / iterations;
    }
    std::sort(samples.begin(), samples.end());
    return float(samples[samples.size() / 2]);
}

template<int N>
void print(Result* device) {
    const float load = bench<N, false>(device);
    const float store = bench<N, true>(device);
    std::printf("%-12d %-12d %-16.2f %.2f\n", N, 32 * N * 4, load, store);
}

#include "tmem_tile.cuh"
} // namespace

int main() {
    Result* device = nullptr;
    try {
        check(cudaSetDevice(0));
        cudaDeviceProp gpu{};
        check(cudaGetDeviceProperties(&gpu, 0));
        check(cudaMalloc(reinterpret_cast<void**>(&device), 32 * sizeof(Result)));
        std::printf("%s: serialized 32x32b transfers, one warp\n", gpu.name);
        std::puts("Clocks/transfer include completion wait and loop overhead; allocation excluded.");
        std::puts("Regs/thread  Bytes/warp   TMEM -> regs     Regs -> TMEM");
        print<1>(device);
        print<8>(device);
        print<16>(device);
        print<32>(device);
        check(cudaFree(device));
        device = nullptr;
        print_tile();
    } catch (const std::exception& error) {
        if (device) cudaFree(device);
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
