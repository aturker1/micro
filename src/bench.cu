#include "exp2.cuh"
#include "alu.cuh"
#include <iomanip>
#include <iostream>

int main() {
    try {
        const float exp2 = bench_exp2();
        const auto alu = bench_alu();
        std::cout << std::fixed << std::setprecision(3)
                  << "Calls per SM clock\n"
                  << "exp2: " << exp2 << '\n'
                  << "FP32 add: " << alu[0] << '\n'
                  << "FP32 subtract: " << alu[1] << '\n'
                  << "FP32 multiply: " << alu[2] << '\n'
                  << "FP32 FMA: " << alu[3] << " (" << 2 * alu[3] << " FLOPs)\n";
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
