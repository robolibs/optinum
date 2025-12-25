// =============================================================================
// examples/float_precision_benchmark.cpp
// Benchmark for FLOAT precision SIMD math functions (32-bit)
// =============================================================================

#include <chrono>
#include <cmath>
#include <datapod/matrix.hpp>
#include <iostream>
#include <optinum/simd/algo/transform.hpp>
#include <optinum/simd/bridge.hpp>

namespace on = optinum;

volatile float g_sink = 0.0f;

void escape(void *p) { asm volatile("" : : "g"(p) : "memory"); }
void clobber() { asm volatile("" : : : "memory"); }

int main() {
    constexpr size_t N = 1024;
    constexpr size_t ITERATIONS = 10000;

    alignas(32) datapod::mat::vector<float, N> input;
    alignas(32) datapod::mat::vector<float, N> output;
    alignas(32) datapod::mat::vector<float, N> scalar_output;

    std::cout << "Float Precision (32-bit) SIMD Math Benchmark\n";
    std::cout << "Array size: " << N << " floats\n";
    std::cout << "Iterations: " << ITERATIONS << "\n";
    std::cout << "SIMD width: 8 floats (AVX) vs 4 doubles (AVX)\n";
    std::cout << "Total operations per test: " << (N * ITERATIONS) << "\n\n";

    // sinh() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = -3.0f + (6.0f * i) / N;
        }

        std::cout << "=== sinh() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vy = on::simd::view<8>(output);
            on::simd::sinh(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::sinh(input[i]);
            }
            clobber();
        }
        end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "SIMD:   " << simd_time << " ms\n";
        std::cout << "Scalar: " << scalar_time << " ms\n";
        std::cout << "Speedup: " << (scalar_time / simd_time) << "x\n\n";

        g_sink = output[0] + scalar_output[0];
    }

    // cosh() benchmark
    {
        std::cout << "=== cosh() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vy = on::simd::view<8>(output);
            on::simd::cosh(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::cosh(input[i]);
            }
            clobber();
        }
        end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "SIMD:   " << simd_time << " ms\n";
        std::cout << "Scalar: " << scalar_time << " ms\n";
        std::cout << "Speedup: " << (scalar_time / simd_time) << "x\n\n";

        g_sink = output[0] + scalar_output[0];
    }

    // exp2() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = -5.0f + (15.0f * i) / N;
        }

        std::cout << "=== exp2() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vy = on::simd::view<8>(output);
            on::simd::exp2(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::exp2(input[i]);
            }
            clobber();
        }
        end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "SIMD:   " << simd_time << " ms\n";
        std::cout << "Scalar: " << scalar_time << " ms\n";
        std::cout << "Speedup: " << (scalar_time / simd_time) << "x\n\n";

        g_sink = output[0] + scalar_output[0];
    }

    // log2() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = 0.1f + (1000.0f * i) / N;
        }

        std::cout << "=== log2() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vy = on::simd::view<8>(output);
            on::simd::log2(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::log2(input[i]);
            }
            clobber();
        }
        end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "SIMD:   " << simd_time << " ms\n";
        std::cout << "Scalar: " << scalar_time << " ms\n";
        std::cout << "Speedup: " << (scalar_time / simd_time) << "x\n\n";

        g_sink = output[0] + scalar_output[0];
    }

    // log10() benchmark
    {
        std::cout << "=== log10() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vy = on::simd::view<8>(output);
            on::simd::log10(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::log10(input[i]);
            }
            clobber();
        }
        end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "SIMD:   " << simd_time << " ms\n";
        std::cout << "Scalar: " << scalar_time << " ms\n";
        std::cout << "Speedup: " << (scalar_time / simd_time) << "x\n\n";

        g_sink = output[0] + scalar_output[0];
    }

    return 0;
}
