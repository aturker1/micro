#include "exp2.cuh"
#include "alu.cuh"
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int kThreads = 512, kChains = 16;
constexpr int kIterations = 4096, kSamples = 5, kWaves = 4;

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

template<class Op>
static double benchmark() {
    // Reserve enough shared memory for exactly one block per SM.
    cudaDeviceProp gpu{};
    check(cudaGetDeviceProperties(&gpu, 0));
    const auto kernel = exp2_kernel<Op, kThreads, kChains>;
    const size_t shared = gpu.sharedMemPerMultiprocessor / 2 + 256;
    check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)));
    check(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    int resident = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, kThreads, shared));
    if (resident != 1) throw std::runtime_error("Expected one resident block/SM");
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, kernel));
    if (attributes.localSizeBytes) throw std::runtime_error("Kernel uses local memory");

    // Allocate inputs, outputs, and one timing record per block.
    const int blocks = gpu.multiProcessorCount * kWaves;
    const size_t count = size_t(blocks) * kThreads * kChains;
    std::vector<float> input(count);
    for (size_t i = 0; i < count; ++i) input[i] = 0.125f + float(i % 4096) / 8192.0f;
    DeviceBuffer<float> x(count), y(count);
    DeviceBuffer<BlockResult> timings(blocks);
    check(cudaMemcpy(x.data, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    auto launch = [&](int iterations, int grid) {
        kernel<<<grid, kThreads, shared>>>(
            x.data, y.data, timings.data, iterations, 0);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
    };

    // Validate short runs before convergence can hide missing iterations.
    std::vector<float> actual(2 * kThreads * kChains);
    for (int iterations : {0, 1, 2}) {
        launch(iterations, 2);
        check(cudaMemcpy(actual.data(), y.data, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < actual.size(); ++i) {
            float expected = input[i];
            for (int j = 0; j < iterations; ++j) expected = Op::reference(expected);
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected) > Op::tolerance)
                throw std::runtime_error(std::string(Op::name) + ": numerical validation failed");
        }
    }

    // Warm up, then accumulate operations and cycles for each actual SM ID.
    launch(kIterations, blocks);
    constexpr uint64_t operations = uint64_t(kThreads) * kChains * kIterations;
    struct Totals { uint64_t operations = 0, cycles = 0; };
    std::map<unsigned, Totals> per_sm;
    std::vector<BlockResult> results(blocks);
    Totals all;
    for (int sample = 0; sample < kSamples; ++sample) {
        launch(kIterations, blocks);
        check(cudaMemcpy(results.data(), timings.data, blocks * sizeof(BlockResult), cudaMemcpyDeviceToHost));
        for (const auto& result : results) {
            if (!result.valid || !result.cycles) throw std::runtime_error("Invalid timing record");
            auto& sm = per_sm[result.sm];
            sm.operations += operations;
            sm.cycles += result.cycles;
            all.operations += operations;
            all.cycles += result.cycles;
        }
    }

    std::cout << "== " << Op::name << '\n'
              << gpu.name << ": " << gpu.multiProcessorCount << " SMs\n"
              << kThreads << " threads/block, " << kChains << " chains/thread, "
              << kIterations << " iterations, " << kSamples << " samples\n"
              << attributes.numRegs << " registers/thread, " << shared << " shared bytes/block\n"
              << "SM    scalar results/clock\n" << std::fixed << std::setprecision(3);
    for (const auto& [sm, totals] : per_sm)
        std::cout << std::setw(3) << sm << "   " << double(totals.operations) / totals.cycles << '\n';
    const double rate = double(all.operations) / all.cycles;
    std::cout << "Combined results/SM clock: " << rate << '\n'
              << "Observed SMs: " << per_sm.size() << '/' << gpu.multiProcessorCount << '\n';
    if (per_sm.size() != size_t(gpu.multiProcessorCount)) std::cout << "NOTE: incomplete SM coverage.\n";
    std::cout << '\n';
    return rate;
}

int main() {
    try {
        check(cudaSetDevice(0));
        struct Row { const char* name; double rate; int flops; };
        const Row rows[] = {
            {Ex2Op::name, benchmark<Ex2Op>(), 0},
            {AddOp::name, benchmark<AddOp>(), 1},
            {SubOp::name, benchmark<SubOp>(), 1},
            {MulOp::name, benchmark<MulOp>(), 1},
            {FmaOp::name, benchmark<FmaOp>(), 2},
        };
        std::cout << "Summary (scalar results and FLOPs per SM clock)\n";
        for (const auto& row : rows) {
            std::cout << row.name << ": " << row.rate << " results";
            if (row.flops) std::cout << ", " << row.rate * row.flops << " FLOPs";
            std::cout << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
