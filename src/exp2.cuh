#pragma once
#include "benchbed.cuh"

inline float bench_exp2() {
    return benchbed::run([] __device__ (float x) {
        float result;
        asm volatile("{ .reg .f32 negative; neg.f32 negative, %1; "
                     "ex2.approx.ftz.f32 %0, negative; }"
                     : "=f"(result) : "f"(x));
        return result;
    });
}
