// Build and run from micro/ with: make run
#include "exp2.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

constexpr int kIterations = 4096;
constexpr int kSamples = 5;
constexpr int kWaves = 4;

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

static void benchmark() {
    check(cudaSetDevice(0));
    cudaDeviceProp gpu{};
    check(cudaGetDeviceProperties(&gpu, 0));
    if (gpu.major < 7) throw std::runtime_error("Requires compute capability >= 7.0");

    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, exp2_kernel));
    if (attributes.localSizeBytes != 0)
        throw std::runtime_error("Kernel uses local memory; inspect the ptxas spill report");
    if (kThreads > attributes.maxThreadsPerBlock)
        throw std::runtime_error("Block exceeds kernel thread limit");

    // Reserve more than half an SM's shared memory to allow only one block/SM.
    size_t shared_bytes = (gpu.sharedMemPerMultiprocessor / 2 / 256 + 1) * 256;
    size_t shared_limit = gpu.sharedMemPerBlockOptin ? gpu.sharedMemPerBlockOptin
                                                  : gpu.sharedMemPerBlock;
    if (shared_bytes + attributes.sharedSizeBytes > shared_limit)
        throw std::runtime_error("Not enough shared memory to isolate one block/SM");
    check(cudaFuncSetAttribute(exp2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               int(shared_bytes)));
    check(cudaFuncSetAttribute(exp2_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    int resident_blocks = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_blocks, exp2_kernel, kThreads, shared_bytes));
    if (resident_blocks != 1) throw std::runtime_error("Expected one resident block/SM");

    const int blocks = gpu.multiProcessorCount * kWaves;
    const size_t count = size_t(blocks) * kThreads * kChains;
    std::vector<float> input(count);
    for (size_t i = 0; i < count; ++i) input[i] = 0.125f + float(i % 4096) / 8192.0f;
    DeviceBuffer<float> device_input(count), device_output(count);
    DeviceBuffer<BlockResult> device_results(blocks);
    check(cudaMemcpy(device_input.data, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    auto launch = [&](int iterations, int grid) {
        exp2_kernel<<<grid, kThreads, shared_bytes>>>(
            device_input.data, device_output.data, device_results.data, iterations, 0);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
    };

    // Check short recurrences before convergence hides missing iterations.
    const int validation_blocks = std::min(blocks, 2);
    const size_t checked = size_t(validation_blocks) * kThreads * kChains;
    std::vector<float> actual(checked);
    for (int iterations : {0, 1, 2}) {
        launch(iterations, validation_blocks);
        check(cudaMemcpy(actual.data(), device_output.data, checked * sizeof(float),
                         cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < checked; ++i) {
            float expected = input[i];
            for (int j = 0; j < iterations; ++j) expected = std::exp2(-expected);
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected) > 2e-6f)
                throw std::runtime_error("Numerical validation failed");
        }
    }
    launch(kIterations, blocks); // Warm up before collecting samples.

    std::ofstream csv("exp2_samples.csv");
    if (!csv) throw std::runtime_error("Cannot open exp2_samples.csv");
    csv << "sample,block,sm,cycles,scalar_exp2_ops,ops_per_sm_clock\n";
    csv << std::setprecision(12);
    const uint64_t operations = uint64_t(kThreads) * kChains * kIterations;
    struct Totals { uint64_t operations = 0, cycles = 0; };
    std::map<unsigned, Totals> per_sm; // SM IDs need not be contiguous.
    std::vector<BlockResult> results(blocks);
    for (int sample = 0; sample < kSamples; ++sample) {
        launch(kIterations, blocks);
        check(cudaMemcpy(results.data(), device_results.data, blocks * sizeof(BlockResult),
                         cudaMemcpyDeviceToHost));
        for (int block = 0; block < blocks; ++block) {
            const auto& result = results[block];
            if (!result.valid || result.cycles == 0)
                throw std::runtime_error("Invalid timer, nonfinite recurrence, or changed SM ID");
            auto& totals = per_sm[result.sm];
            totals.operations += operations;
            totals.cycles += result.cycles;
            csv << sample << ',' << block << ',' << result.sm << ',' << result.cycles
                << ',' << operations << ',' << double(operations) / result.cycles << '\n';
        }
    }
    csv.close();
    if (!csv) throw std::runtime_error("Failed to write samples");

    std::cout << gpu.name << ": " << gpu.multiProcessorCount << " SMs\n"
              << kThreads << " threads/block, " << kChains << " chains/thread, "
              << kIterations << " iterations, " << kSamples << " samples\n"
              << attributes.numRegs << " registers/thread, " << shared_bytes << " shared bytes/block\n"
              << "SM    exp2 ops/clock\n" << std::fixed << std::setprecision(3);
    Totals all;
    for (const auto& [sm, totals] : per_sm) {
        std::cout << std::setw(3) << sm << "   " << double(totals.operations) / totals.cycles << '\n';
        all.operations += totals.operations;
        all.cycles += totals.cycles;
    }
    std::cout << "Combined ops/SM clock: " << double(all.operations) / all.cycles << '\n'
              << "Observed SMs: " << per_sm.size() << '/' << gpu.multiProcessorCount << '\n'
              << "Raw samples: exp2_samples.csv\n";
    if (per_sm.size() != size_t(gpu.multiProcessorCount))
        std::cout << "NOTE: incomplete SM coverage.\n";
}

int main() {
    try {
        benchmark();
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
