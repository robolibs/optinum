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

    // atan() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = -10.0f + (20.0f * i) / N;
        }

        std::cout << "=== atan() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vy = on::simd::view<8>(output);
            on::simd::atan(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::atan(input[i]);
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

    // atan2() benchmark
    {
        alignas(32) datapod::mat::vector<float, N> y_vals;
        for (size_t i = 0; i < N; ++i) {
            input[i] = -5.0f + (10.0f * i) / N;  // x values
            y_vals[i] = -5.0f + (10.0f * i) / N; // y values
        }

        std::cout << "=== atan2() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vy = on::simd::view<8>(y_vals);
            auto vx = on::simd::view<8>(input);
            auto vz = on::simd::view<8>(output);
            on::simd::atan2(vy, vx, vz);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::atan2(y_vals[i], input[i]);
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

    // asin() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = -1.0f + (2.0f * i) / N;
        }

        std::cout << "=== asin() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vy = on::simd::view<8>(output);
            on::simd::asin(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::asin(input[i]);
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

    // acos() benchmark
    {
        std::cout << "=== acos() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vy = on::simd::view<8>(output);
            on::simd::acos(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::acos(input[i]);
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

    // asinh() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = -100.0f + (200.0f * i) / N;
        }

        std::cout << "=== asinh() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vy = on::simd::view<8>(output);
            on::simd::asinh(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::asinh(input[i]);
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

    // acosh() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = 1.0f + (100.0f * i) / N;
        }

        std::cout << "=== acosh() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vy = on::simd::view<8>(output);
            on::simd::acosh(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::acosh(input[i]);
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

    // atanh() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = -0.99f + (1.98f * i) / N;
        }

        std::cout << "=== atanh() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vy = on::simd::view<8>(output);
            on::simd::atanh(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::atanh(input[i]);
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

    // expm1() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = -2.0f + (4.0f * i) / N;
        }

        std::cout << "=== expm1() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vy = on::simd::view<8>(output);
            on::simd::expm1(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::expm1(input[i]);
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

    // log1p() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = -0.5f + (10.0f * i) / N;
        }

        std::cout << "=== log1p() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vy = on::simd::view<8>(output);
            on::simd::log1p(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::log1p(input[i]);
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

    // abs() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = -10.0f + (20.0f * i) / N;
        }

        std::cout << "=== abs() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vy = on::simd::view<8>(output);
            using on::simd::abs;
            abs(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::abs(input[i]);
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

    // cbrt() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = -100.0f + (200.0f * i) / N;
        }

        std::cout << "=== cbrt() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vy = on::simd::view<8>(output);
            using on::simd::cbrt;
            cbrt(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::cbrt(input[i]);
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

    // clamp() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = -10.0f + (20.0f * i) / N;
        }

        alignas(32) datapod::mat::vector<float, N> lo_vals;
        alignas(32) datapod::mat::vector<float, N> hi_vals;
        for (size_t i = 0; i < N; ++i) {
            lo_vals[i] = -5.0f;
            hi_vals[i] = 5.0f;
        }

        std::cout << "=== clamp() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vlo = on::simd::view<8>(lo_vals);
            auto vhi = on::simd::view<8>(hi_vals);
            auto vy = on::simd::view<8>(output);
            using on::simd::clamp;
            clamp(vx, vlo, vhi, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::clamp(input[i], -5.0f, 5.0f);
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

    // hypot() benchmark
    {
        alignas(32) datapod::mat::vector<float, N> input2;
        for (size_t i = 0; i < N; ++i) {
            input[i] = -10.0f + (20.0f * i) / N;
            input2[i] = 5.0f + (10.0f * i) / N;
        }

        std::cout << "=== hypot() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<8>(input);
            auto vy = on::simd::view<8>(input2);
            auto vz = on::simd::view<8>(output);
            using on::simd::hypot;
            hypot(vx, vy, vz);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::hypot(input[i], input2[i]);
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
