// Included by tmem.cu inside its anonymous namespace.
struct TileResult {
    unsigned long long start, end;
    unsigned errors;
};

__device__ __forceinline__ unsigned long long tile_clock_after(unsigned ready) {
    unsigned long long value;
    asm volatile("{ .reg .pred p; setp.ne.u32 p, %1, 0; mov.u64 %0, 0; "
                 "@p mov.u64 %0, %%clock64; }"
                 : "=l"(value) : "r"(ready) : "memory");
    return value;
}

// Mode: 0 = timer baseline, 1 = load, 2 = store.
template<int Mode, bool WaitEach = false>
__global__ void measure_tile(TileResult* results, unsigned clock_mask, int repeats) {
    __shared__ unsigned base;
    if (threadIdx.x < 32) {
        const unsigned shared = unsigned(__cvta_generic_to_shared(&base));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;"
                     : : "r"(shared) : "memory");
    }
    __syncthreads();
    // Warp 0..3 owns TMEM rows 0..31, 32..63, 64..95, 96..127.
    const unsigned row_base = base + ((threadIdx.x / 32 * 32) << 16);
    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        unsigned initial[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) initial[i] = threadIdx.x * 128 + chunk * 32 + i;
        initial[0] = row_base; // Uniform pointer anchor in every chunk and lane.
        tmem_store(row_base + chunk * 32, initial);
        tmem_wait_store();
    }

    unsigned values[4][32];
    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk)
        tmem_load(values[chunk], row_base + chunk * 32);
    tmem_wait_load();
    __syncthreads();

    const auto start = tile_clock_after(values[3][0] == row_base);
    unsigned address = row_base ^ (unsigned(start) & clock_mask);
    // Same x32 load atom as FA4: four chunks cover 128 columns per warp.
    // Four distinct register arrays keep queued loads independent and live.
    #pragma unroll 1
    for (int iteration = 0; iteration < repeats; ++iteration) {
        #pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
            if constexpr (Mode == 1) tmem_load(values[chunk], address + chunk * 32);
            if constexpr (Mode == 2) tmem_store(address + chunk * 32, values[chunk]);
            if constexpr (WaitEach && Mode == 1) tmem_wait_load();
            if constexpr (WaitEach && Mode == 2) tmem_wait_store();
            if constexpr (WaitEach && Mode == 1) address = values[chunk][0];
        }
        if constexpr (!WaitEach && Mode == 1) tmem_wait_load();
        if constexpr (!WaitEach && Mode == 2) tmem_wait_store();
        if constexpr (!WaitEach && Mode == 1)
            address = values[0][0] ^ values[1][0] ^ values[2][0] ^ values[3][0] ^ row_base;
    }
    // Consume the final loaded pointer before reading the timestamp as well.
    const auto end = tile_clock_after((Mode == 1 ? address : values[3][0]) == row_base);

    if constexpr (Mode == 2) {
        #pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk)
            tmem_load(values[chunk], row_base + chunk * 32);
        tmem_wait_load();
    }
    // Timestamp dependency keeps expected-value generation outside the timing.
    const unsigned seed = threadIdx.x * 128 + (unsigned(end) & clock_mask);
    unsigned errors = 0;
    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        #pragma unroll
        for (int i = 0; i < 32; ++i)
            errors += values[chunk][i] != (i == 0 ? row_base : seed + chunk * 32 + i);
    }
    results[threadIdx.x] = {start, end, errors};
    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;"
                     : : "r"(base) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
}

template<int Mode, bool WaitEach = false>
float bench_tile(TileResult* device, int repeats = 1) {
    std::array<unsigned long long, 101> samples{};
    for (int sample = -10; sample < int(samples.size()); ++sample) {
        measure_tile<Mode, WaitEach><<<1, 128>>>(device, 0, repeats);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        std::array<TileResult, 128> results{};
        check(cudaMemcpy(results.data(), device, sizeof(results), cudaMemcpyDeviceToHost));
        auto first = results[0].start, last = results[0].end;
        for (const auto& result : results) {
            if (result.errors) throw std::runtime_error("128x128 TMEM validation failed");
            first = std::min(first, result.start);
            last = std::max(last, result.end);
        }
        if (sample >= 0) samples[sample] = last - first;
    }
    std::sort(samples.begin(), samples.end());
    return float(samples[samples.size() / 2]);
}

template<int Mode, bool WaitEach>
float tile_rate(TileResult* device) {
    const float short_run = bench_tile<Mode, WaitEach>(device, 256);
    const float result = (bench_tile<Mode, WaitEach>(device, 512) - short_run) / 256;
    if (result <= 0) throw std::runtime_error("Nonpositive TMEM timing slope");
    return result;
}

template<bool WaitEach>
void print_tile_rate(TileResult* device) {
    const auto load = tile_rate<1, WaitEach>(device);
    const auto store = tile_rate<2, WaitEach>(device);
    std::printf("%-19s %-17.2f %.2f\n",
                WaitEach ? "Wait each x32" : "Queue 4 x32",
                load, store);
}

void print_tile() {
    TileResult* device = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device), 128 * sizeof(TileResult)));
    try {
        const auto baseline = bench_tile<0>(device);
        const auto load = bench_tile<1>(device);
        const auto store = bench_tile<2>(device);
        std::puts("\n128x128 FP32 tile: 4 warps, 64 KiB, four x32 instructions/warp");
        std::puts("Clocks/tile: earliest warp start to latest completion, wait included.");
        std::printf("Timer baseline: %.0f clocks\nTMEM -> regs: %.0f clocks\nRegs -> TMEM: %.0f clocks\n",
                    baseline, load, store);
        std::puts("\n128x128 tile, 256/512-repeat slopes, waits included:");
        std::puts("Mode                Load clocks/tile  Store clocks/tile");
        print_tile_rate<true>(device);
        print_tile_rate<false>(device);
        check(cudaFree(device));
    } catch (...) {
        cudaFree(device);
        throw;
    }
}
