#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace benchbed {
namespace detail {
constexpr int threads = 512, chains = 16, iterations = 4096;

inline void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

__device__ __forceinline__ unsigned long long clock_after(int ready) {
    unsigned long long clock;
    asm volatile("{ .reg .pred p; setp.ne.s32 p, %1, 0; "
                 "mov.u64 %0, 0; @p mov.u64 %0, %%clock64; }"
                 : "=l"(clock) : "r"(ready) : "memory");
    return clock;
}

template<class Function>
__global__ void kernel(Function function, unsigned long long* cycles, unsigned clock_mask) {
    extern __shared__ unsigned long long timer[];
    float values[chains], sum = 0;
    #pragma unroll
    for (int chain = 0; chain < chains; ++chain) {
        values[chain] = 0.125f + float((threadIdx.x + chain * threads) % 4096) / 8192.0f;
        sum += values[chain];
    }
    int ready = __syncthreads_and(isfinite(sum));
    if (threadIdx.x == 0) timer[0] = clock_after(ready);
    __syncthreads();
    const unsigned long long start = timer[0];
    const unsigned dependency = unsigned(start) & clock_mask;
    #pragma unroll
    for (int chain = 0; chain < chains; ++chain)
        values[chain] = __uint_as_float(__float_as_uint(values[chain]) ^ dependency);

    #pragma unroll 8
    for (int iteration = 0; iteration < iterations; ++iteration) {
        #pragma unroll
        for (int chain = 0; chain < chains; ++chain)
            values[chain] = function(values[chain]);
    }
    sum = 0;
    #pragma unroll
    for (int chain = 0; chain < chains; ++chain) sum += values[chain];
    int done = __syncthreads_and(isfinite(sum));
    if (threadIdx.x == 0) {
        const unsigned long long end = clock_after(done);
        cycles[blockIdx.x] = ready && done && end > start ? end - start : 0;
    }
}
} // namespace detail

// Device callable float(float), inlined into independent register chains.
// Assumes its recurrence stays finite. Returns achieved calls / SM clock.
template<class Function>
float run(Function function) {
    using detail::check;
    check(cudaSetDevice(0));
    cudaDeviceProp gpu{};
    check(cudaGetDeviceProperties(&gpu, 0));
    const auto kernel = detail::kernel<Function>;
    const size_t shared = gpu.sharedMemPerMultiprocessor / 2 + 256;
    check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)));
    check(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    int resident = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, detail::threads, shared));
    if (resident != 1) throw std::runtime_error("Expected one resident block/SM");

    const int blocks = gpu.multiProcessorCount * 4, samples = 5;
    std::vector<unsigned long long> cycles(blocks);
    unsigned long long* device_cycles = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device_cycles), blocks * sizeof(*device_cycles)));
    std::unique_ptr<unsigned long long, decltype(&cudaFree)> allocation(device_cycles, cudaFree);
    auto launch = [&] {
        kernel<<<blocks, detail::threads, shared>>>(function, device_cycles, 0);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
    };
    launch(); // Warm up.
    unsigned long long total_cycles = 0;
    for (int sample = 0; sample < samples; ++sample) {
        launch();
        check(cudaMemcpy(cycles.data(), device_cycles, blocks * sizeof(*device_cycles), cudaMemcpyDeviceToHost));
        for (auto elapsed : cycles) {
            if (!elapsed) throw std::runtime_error("Invalid timing or nonfinite result");
            total_cycles += elapsed;
        }
    }
    const double calls = double(detail::threads) * detail::chains * detail::iterations * blocks * samples;
    return float(calls / total_cycles);
}
} // namespace benchbed
