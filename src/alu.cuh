#pragma once
#include "benchbed.cuh"
#include <array>

inline std::array<float, 4> bench_alu() {
    return {{
        benchbed::run([] __device__ (float x) {
            float y;
            asm volatile("add.rn.f32 %0, %1, 0f37800000;" : "=f"(y) : "f"(x));
            return y;
        }),
        benchbed::run([] __device__ (float x) {
            float y;
            asm volatile("sub.rn.f32 %0, 0f3F800000, %1;" : "=f"(y) : "f"(x));
            return y;
        }),
        benchbed::run([] __device__ (float x) {
            float y;
            asm volatile("mul.rn.f32 %0, %1, 0f3F7FF000;" : "=f"(y) : "f"(x));
            return y;
        }),
        benchbed::run([] __device__ (float x) {
            float y;
            asm volatile("fma.rn.f32 %0, %1, %1, 0f3E000000;" : "=f"(y) : "f"(x));
            return y;
        }),
    }};
}
