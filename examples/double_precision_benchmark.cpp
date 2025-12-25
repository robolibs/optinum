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

    // ceil() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = -10.0 + (20.0 * i) / N;
        }

        std::cout << "=== ceil() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
            on::simd::ceil(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::ceil(input[i]);
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

    // floor() benchmark
    {
        std::cout << "=== floor() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
            on::simd::floor(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::floor(input[i]);
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

    // round() benchmark
    {
        std::cout << "=== round() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
            on::simd::round(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::round(input[i]);
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

    // trunc() benchmark
    {
        std::cout << "=== trunc() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
            on::simd::trunc(vx, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::trunc(input[i]);
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

    // pow() benchmark
    {
        alignas(32) datapod::mat::vector<double, N> exponent;
        for (size_t i = 0; i < N; ++i) {
            input[i] = 1.0 + (10.0 * i) / N;
            exponent[i] = 0.5 + (2.0 * i) / N;
        }

        std::cout << "=== pow() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto ve = on::simd::view<4>(exponent);
            auto vy = on::simd::view<4>(output);
            on::simd::pow(vx, ve, vy);
            clobber();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                scalar_output[i] = std::pow(input[i], exponent[i]);
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

    // sinh() benchmark
    {
        for (size_t i = 0; i < N; ++i) {
            input[i] = -3.0 + (6.0 * i) / N;
        }

        std::cout << "=== sinh() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
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
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
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
            input[i] = -5.0 + (15.0 * i) / N;
        }

        std::cout << "=== exp2() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
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
            input[i] = 0.1 + (1000.0 * i) / N;
        }

        std::cout << "=== log2() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
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
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
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
            input[i] = -10.0 + (20.0 * i) / N;
        }

        std::cout << "=== atan() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
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
        alignas(32) datapod::mat::vector<double, N> y_vals;
        for (size_t i = 0; i < N; ++i) {
            input[i] = -5.0 + (10.0 * i) / N;  // x values
            y_vals[i] = -5.0 + (10.0 * i) / N; // y values
        }

        std::cout << "=== atan2() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vy = on::simd::view<4>(y_vals);
            auto vx = on::simd::view<4>(input);
            auto vz = on::simd::view<4>(output);
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
            input[i] = -1.0 + (2.0 * i) / N;
        }

        std::cout << "=== asin() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
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
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
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
            input[i] = -100.0 + (200.0 * i) / N;
        }

        std::cout << "=== asinh() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
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
            input[i] = 1.0 + (100.0 * i) / N;
        }

        std::cout << "=== acosh() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
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
            input[i] = -0.99 + (1.98 * i) / N;
        }

        std::cout << "=== atanh() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
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
            input[i] = -2.0 + (4.0 * i) / N;
        }

        std::cout << "=== expm1() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
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
            input[i] = -0.5 + (10.0 * i) / N;
        }

        std::cout << "=== log1p() ===\n";

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            auto vx = on::simd::view<4>(input);
            auto vy = on::simd::view<4>(output);
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

    return 0;
}
