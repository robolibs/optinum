// =============================================================================
// examples/double_precision_benchmark.cpp
// Benchmark for double precision SIMD math functions
// =============================================================================

#include <chrono>
#include <cmath>
#include <datapod/matrix.hpp>
#include <iostream>
#include <optinum/simd/algo/transform.hpp>
#include <optinum/simd/bridge.hpp>

namespace on = optinum;

// Prevent compiler from optimizing away results
volatile double g_sink = 0.0;

void escape(void *p) { asm volatile("" : : "g"(p) : "memory"); }

void clobber() { asm volatile("" : : : "memory"); }

int main() {
    constexpr size_t N = 1024;
    constexpr size_t ITERATIONS = 10000;

    // Allocate aligned memory
    alignas(32) datapod::mat::vector<double, N> input;
    alignas(32) datapod::mat::vector<double, N> output;
    alignas(32) datapod::mat::vector<double, N> scalar_output;

    std::cout << "Double Precision SIMD Math Benchmark\n";
    std::cout << "Array size: " << N << " doubles\n";
    std::cout << "Iterations: " << ITERATIONS << "\n";
    std::cout << "Total operations per test: " << (N * ITERATIONS) << "\n\n";

    // exp() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = -5.0 + (10.0 * i) / N;
        }

        std::cout << "=== exp() ===\n";

        // SIMD version
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
            on::simd::exp(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        // Scalar version
        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::exp(input[i]);
            }
            clobber();
        }
        end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "SIMD:   " << simd_time << " ms\n";
        std::cout << "Scalar: " << scalar_time << " ms\n";
        std::cout << "Speedup: " << (scalar_time / simd_time) << "x\n\n";

        // Check results match
        g_sink = output[0] + scalar_output[0];
    }

    // log() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = 0.1 + (10.0 * i) / N;
        }

        std::cout << "=== log() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
            on::simd::log(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::log(input[i]);
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

    // sin() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = -3.14159 + (6.28318 * i) / N;
        }

        std::cout << "=== sin() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
            on::simd::sin(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::sin(input[i]);
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

    // cos() benchmark
    {
        std::cout << "=== cos() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
            on::simd::cos(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::cos(input[i]);
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

    // tan() benchmark
    {
        std::cout << "=== tan() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
            on::simd::tan(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::tan(input[i]);
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

    // tanh() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = -5.0 + (10.0 * i) / N;
        }

        std::cout << "=== tanh() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
            on::simd::tanh(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::tanh(input[i]);
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

    // sqrt() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = (100.0 * i) / N;
        }

        std::cout << "=== sqrt() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
            on::simd::sqrt(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::sqrt(input[i]);
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
