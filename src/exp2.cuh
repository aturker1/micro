#include <cuda_runtime.h>

constexpr int kThreads = 256;
constexpr int kChains = 16;

struct BlockResult {
    unsigned sm;
    unsigned long long cycles;
    bool valid;
};

// Read the ID of the SM running this thread.
__device__ __forceinline__ unsigned read_sm_id() {
    unsigned id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

// Make the clock read wait for the preceding block-wide reduction.
__device__ __forceinline__ unsigned long long read_clock_after(int ready) {
    unsigned long long clock;
    asm volatile("{ .reg .pred p; setp.ne.s32 p, %1, 0; "
                 "mov.u64 %0, 0; @p mov.u64 %0, %%clock64; }"
                 : "=l"(clock) : "r"(ready) : "memory");
    return clock;
}

// Native approximate exp2(-x). Values in [0,1] stay in [0.5,1].
__device__ __forceinline__ float exp2_negative(float x) {
    float result;
    asm volatile("{ .reg .f32 negative; neg.f32 negative, %1; "
                 "ex2.approx.ftz.f32 %0, negative; }"
                 : "=f"(result) : "f"(x));
    return result;
}

// Input/output layout: [block][chain][thread]. Each block writes one result.
__global__ void exp2_kernel(const float* input, float* output, BlockResult* results,
                            int iterations, unsigned clock_mask) {
    extern __shared__ unsigned long long start_clock[];
    const int thread = threadIdx.x;
    const size_t base = size_t(blockIdx.x) * kThreads * kChains + thread;
    float values[kChains];
    float sum = 0;

    #pragma unroll
    for (int chain = 0; chain < kChains; ++chain) {
        values[chain] = input[base + chain * kThreads];
        sum += values[chain];
    }

    // Finish all input loads, then share one start timestamp across the block.
    int ready = __syncthreads_and(sum > 0 && isfinite(sum));
    unsigned sm = read_sm_id();
    if (thread == 0) start_clock[0] = read_clock_after(ready);
    __syncthreads();
    unsigned long long start = start_clock[0];

    // The host passes zero. This dependency prevents exp2 from moving before start.
    unsigned dependency = unsigned(start) & clock_mask;
    #pragma unroll
    for (int chain = 0; chain < kChains; ++chain)
        values[chain] = __uint_as_float(__float_as_uint(values[chain]) ^ dependency);

    // The measured work: independent register chains, no memory accesses.
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        #pragma unroll
        for (int chain = 0; chain < kChains; ++chain)
            values[chain] = exp2_negative(values[chain]);
    }

    sum = 0;
    #pragma unroll
    for (int chain = 0; chain < kChains; ++chain) sum += values[chain];
    int done = __syncthreads_and(sum > 0 && isfinite(sum));
    if (thread == 0) {
        unsigned long long end = read_clock_after(done);
        results[blockIdx.x] = {sm, end - start,
                              bool(ready && done && end > start && sm == read_sm_id())};
    }

    // Wait for the end timestamp before any thread stores its final values.
    __syncthreads();
    #pragma unroll
    for (int chain = 0; chain < kChains; ++chain)
        output[base + chain * kThreads] = values[chain];
}
