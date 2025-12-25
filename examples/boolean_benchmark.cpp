// =============================================================================
// examples/boolean_benchmark.cpp
// Benchmark for SIMD boolean test functions (isinf, isnan, isfinite)
// =============================================================================

#include <chrono>
#include <cmath>
#include <datapod/matrix.hpp>
#include <iostream>
#include <optinum/simd/bridge.hpp>
#include <optinum/simd/math/isfinite.hpp>
#include <optinum/simd/math/isinf.hpp>
#include <optinum/simd/math/isnan.hpp>

namespace on = optinum;

volatile int g_sink = 0;

void escape(void *p) { asm volatile("" : : "g"(p) : "memory"); }
void clobber() { asm volatile("" : : : "memory"); }

int main() {
    constexpr size_t N = 1024;
    constexpr size_t ITERATIONS = 100000;

    alignas(32) datapod::mat::vector<float, N> input_f;
    alignas(32) datapod::mat::vector<double, N> input_d;

    std::cout << "Boolean Functions SIMD Benchmark\n";
    std::cout << "Array size: " << N << " elements\n";
    std::cout << "Iterations: " << ITERATIONS << "\n";
    std::cout << "Total operations per test: " << (N * ITERATIONS) << "\n\n";

    // Initialize test data with mixed values: normal, inf, -inf, nan
    for (size_t i = 0; i < N; ++i) {
        if (i % 10 == 0)
            input_f[i] = INFINITY;
        else if (i % 10 == 1)
            input_f[i] = -INFINITY;
        else if (i % 10 == 2)
            input_f[i] = NAN;
        else
            input_f[i] = -5.0f + (10.0f * i) / N;

        input_d[i] = static_cast<double>(input_f[i]);
    }

    std::cout << "Test data composition (per " << N << " elements):\n";
    std::cout << "  Infinities: ~" << (N / 5) << " (" << (100.0 / 5) << "%)\n";
    std::cout << "  NaNs:       ~" << (N / 10) << " (" << (100.0 / 10) << "%)\n";
    std::cout << "  Finite:     ~" << (N * 7 / 10) << " (" << (100.0 * 7 / 10) << "%)\n\n";

    // ==========================================================================
    // FLOAT (32-bit) benchmarks
    // ==========================================================================

    std::cout << "========================================\n";
    std::cout << "FLOAT (32-bit) - AVX (8-wide)\n";
    std::cout << "========================================\n\n";

    // isinf() - float
    {
        std::cout << "=== isinf() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        int simd_count = 0;
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; i += 8) {
                auto pack = on::simd::pack<float, 8>::loadu(&input_f[i]);
                auto mask = on::simd::isinf(pack);
                simd_count += mask.popcount();
            }
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        int scalar_count = 0;
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                if (std::isinf(input_f[i]))
                    scalar_count++;
            }
            clobber();
        }
        end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "SIMD:   " << simd_time << " ms (found: " << simd_count / ITERATIONS << " per iteration)\n";
        std::cout << "Scalar: " << scalar_time << " ms (found: " << scalar_count / ITERATIONS << " per iteration)\n";
        std::cout << "Speedup: " << (scalar_time / simd_time) << "x\n\n";

        g_sink += simd_count + scalar_count;
    }

    // isnan() - float
    {
        std::cout << "=== isnan() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        int simd_count = 0;
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; i += 8) {
                auto pack = on::simd::pack<float, 8>::loadu(&input_f[i]);
                auto mask = on::simd::isnan(pack);
                simd_count += mask.popcount();
            }
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        int scalar_count = 0;
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                if (std::isnan(input_f[i]))
                    scalar_count++;
            }
            clobber();
        }
        end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "SIMD:   " << simd_time << " ms (found: " << simd_count / ITERATIONS << " per iteration)\n";
        std::cout << "Scalar: " << scalar_time << " ms (found: " << scalar_count / ITERATIONS << " per iteration)\n";
        std::cout << "Speedup: " << (scalar_time / simd_time) << "x\n\n";

        g_sink += simd_count + scalar_count;
    }

    // isfinite() - float
    {
        std::cout << "=== isfinite() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        int simd_count = 0;
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; i += 8) {
                auto pack = on::simd::pack<float, 8>::loadu(&input_f[i]);
                auto mask = on::simd::isfinite(pack);
                simd_count += mask.popcount();
            }
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        int scalar_count = 0;
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                if (std::isfinite(input_f[i]))
                    scalar_count++;
            }
            clobber();
        }
        end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "SIMD:   " << simd_time << " ms (found: " << simd_count / ITERATIONS << " per iteration)\n";
        std::cout << "Scalar: " << scalar_time << " ms (found: " << scalar_count / ITERATIONS << " per iteration)\n";
        std::cout << "Speedup: " << (scalar_time / simd_time) << "x\n\n";

        g_sink += simd_count + scalar_count;
    }

    // ==========================================================================
    // DOUBLE (64-bit) benchmarks
    // ==========================================================================

    std::cout << "========================================\n";
    std::cout << "DOUBLE (64-bit) - AVX (4-wide)\n";
    std::cout << "========================================\n\n";

    // isinf() - double
    {
        std::cout << "=== isinf() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        int simd_count = 0;
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; i += 4) {
                auto pack = on::simd::pack<double, 4>::loadu(&input_d[i]);
                auto mask = on::simd::isinf(pack);
                simd_count += mask.popcount();
            }
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        int scalar_count = 0;
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                if (std::isinf(input_d[i]))
                    scalar_count++;
            }
            clobber();
        }
        end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "SIMD:   " << simd_time << " ms (found: " << simd_count / ITERATIONS << " per iteration)\n";
        std::cout << "Scalar: " << scalar_time << " ms (found: " << scalar_count / ITERATIONS << " per iteration)\n";
        std::cout << "Speedup: " << (scalar_time / simd_time) << "x\n\n";

        g_sink += simd_count + scalar_count;
    }

    // isnan() - double
    {
        std::cout << "=== isnan() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        int simd_count = 0;
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; i += 4) {
                auto pack = on::simd::pack<double, 4>::loadu(&input_d[i]);
                auto mask = on::simd::isnan(pack);
                simd_count += mask.popcount();
            }
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        int scalar_count = 0;
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                if (std::isnan(input_d[i]))
                    scalar_count++;
            }
            clobber();
        }
        end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "SIMD:   " << simd_time << " ms (found: " << simd_count / ITERATIONS << " per iteration)\n";
        std::cout << "Scalar: " << scalar_time << " ms (found: " << scalar_count / ITERATIONS << " per iteration)\n";
        std::cout << "Speedup: " << (scalar_time / simd_time) << "x\n\n";

        g_sink += simd_count + scalar_count;
    }

    // isfinite() - double
    {
        std::cout << "=== isfinite() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        int simd_count = 0;
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; i += 4) {
                auto pack = on::simd::pack<double, 4>::loadu(&input_d[i]);
                auto mask = on::simd::isfinite(pack);
                simd_count += mask.popcount();
            }
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        int scalar_count = 0;
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                if (std::isfinite(input_d[i]))
                    scalar_count++;
            }
            clobber();
        }
        end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "SIMD:   " << simd_time << " ms (found: " << simd_count / ITERATIONS << " per iteration)\n";
        std::cout << "Scalar: " << scalar_time << " ms (found: " << scalar_count / ITERATIONS << " per iteration)\n";
        std::cout << "Speedup: " << (scalar_time / simd_time) << "x\n\n";

        g_sink += simd_count + scalar_count;
    }

    std::cout << "========================================\n";
    std::cout << "Summary\n";
    std::cout << "========================================\n";
    std::cout << "Boolean functions are simple bit pattern checks,\n";
    std::cout << "so they should show good SIMD speedups.\n";
    std::cout << "Expected speedup: ~4-8x (matching SIMD width)\n";

    return 0;
}
