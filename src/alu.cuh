#pragma once
#include <cuda_runtime.h>
#include <cmath>

// One explicit FP32 instruction per chain update. Values stay positive and finite.
struct AddOp {
    static constexpr const char* name = "FP32 add";
    static constexpr float tolerance = 0.0f;
    __device__ __forceinline__ static float apply(float x) {
        float y;
        asm volatile("add.rn.f32 %0, %1, 0f37800000;" : "=f"(y) : "f"(x));
        return y;
    }
    static float reference(float x) { return x + 0x1p-16f; }
};

struct SubOp {
    static constexpr const char* name = "FP32 subtract";
    static constexpr float tolerance = 0.0f;
    __device__ __forceinline__ static float apply(float x) {
        float y;
        asm volatile("sub.rn.f32 %0, 0f3F800000, %1;" : "=f"(y) : "f"(x));
        return y;
    }
    static float reference(float x) { return 1.0f - x; }
};

struct MulOp {
    static constexpr const char* name = "FP32 multiply";
    static constexpr float tolerance = 0.0f;
    __device__ __forceinline__ static float apply(float x) {
        float y;
        asm volatile("mul.rn.f32 %0, %1, 0f3F7FF000;" : "=f"(y) : "f"(x));
        return y;
    }
    static float reference(float x) { return x * 0x1.ffep-1f; }
};

struct FmaOp {
    static constexpr const char* name = "FP32 FMA";
    static constexpr float tolerance = 0.0f;
    __device__ __forceinline__ static float apply(float x) {
        float y;
        asm volatile("fma.rn.f32 %0, %1, %1, 0f3E000000;" : "=f"(y) : "f"(x));
        return y;
    }
    static float reference(float x) { return std::fmaf(x, x, 0.125f); }
};
