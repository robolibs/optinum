// Complete SIMD Math Functions Benchmark
// Benchmarks ALL missing math operations: trig, inverse trig, hyperbolic, power, rounding, etc.

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <optinum/simd/math/simd_math.hpp>
#include <optinum/simd/pack/pack.hpp>

constexpr std::size_t SIMD_WIDTH_F = optinum::simd::arch::SIMD_WIDTH_FLOAT;
constexpr std::size_t NUM_ELEMENTS = 1024 * 1024;
constexpr std::size_t NUM_ITERATIONS = 100;

template <typename T> std::vector<T> generate_random_data(std::size_t n, T min_val, T max_val) {
    std::vector<T> data(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(min_val, max_val);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

class Timer {
  public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return static_cast<double>(duration.count()) / 1000.0;
    }

  private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// Inverse Trigonometric Functions
// ============================================================================

template <std::size_t W> double benchmark_simd_asin(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::asin(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_asin(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::asin(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_acos(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::acos(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_acos(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::acos(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_atan(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::atan(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_atan(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::atan(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W>
double benchmark_simd_atan2(const std::vector<float> &y, const std::vector<float> &x, std::vector<float> &output) {
    const std::size_t vec_count = y.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto vy = optinum::simd::pack<float, W>::loadu(&y[i * W]);
            auto vx = optinum::simd::pack<float, W>::loadu(&x[i * W]);
            auto r = optinum::simd::atan2(vy, vx);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_atan2(const std::vector<float> &y, const std::vector<float> &x, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < y.size(); ++i) {
            output[i] = std::atan2(y[i], x[i]);
        }
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Hyperbolic Functions
// ============================================================================

template <std::size_t W> double benchmark_simd_sinh(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::sinh(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_sinh(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::sinh(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_cosh(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::cosh(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_cosh(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::cosh(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_asinh(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::asinh(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_asinh(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::asinh(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_acosh(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::acosh(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_acosh(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::acosh(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_atanh(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::atanh(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_atanh(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::atanh(input[i]);
        }
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Power and Root Functions
// ============================================================================

template <std::size_t W> double benchmark_simd_tan(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::tan(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_tan(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::tan(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W>
double benchmark_simd_pow(const std::vector<float> &base, const std::vector<float> &exp, std::vector<float> &output) {
    const std::size_t vec_count = base.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto vb = optinum::simd::pack<float, W>::loadu(&base[i * W]);
            auto ve = optinum::simd::pack<float, W>::loadu(&exp[i * W]);
            auto r = optinum::simd::pow(vb, ve);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_pow(const std::vector<float> &base, const std::vector<float> &exp, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < base.size(); ++i) {
            output[i] = std::pow(base[i], exp[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_cbrt(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::cbrt(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_cbrt(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::cbrt(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W>
double benchmark_simd_hypot(const std::vector<float> &x, const std::vector<float> &y, std::vector<float> &output) {
    const std::size_t vec_count = x.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto vx = optinum::simd::pack<float, W>::loadu(&x[i * W]);
            auto vy = optinum::simd::pack<float, W>::loadu(&y[i * W]);
            auto r = optinum::simd::hypot(vx, vy);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_hypot(const std::vector<float> &x, const std::vector<float> &y, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < x.size(); ++i) {
            output[i] = std::hypot(x[i], y[i]);
        }
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Rounding Functions
// ============================================================================

template <std::size_t W> double benchmark_simd_ceil(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::ceil(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_ceil(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::ceil(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_floor(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::floor(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_floor(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::floor(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_round(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::round(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_round(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::round(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_trunc(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::trunc(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_trunc(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::trunc(input[i]);
        }
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Logarithm Variants
// ============================================================================

template <std::size_t W> double benchmark_simd_log2(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::log2(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_log2(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::log2(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_log10(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::log10(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_log10(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::log10(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_log1p(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::log1p(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_log1p(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::log1p(input[i]);
        }
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Exponential Variants
// ============================================================================

template <std::size_t W> double benchmark_simd_exp2(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::exp2(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_exp2(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::exp2(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_expm1(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::expm1(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_expm1(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::expm1(input[i]);
        }
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Utility Functions
// ============================================================================

template <std::size_t W> double benchmark_simd_abs(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::abs(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_abs(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::abs(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_clamp(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    auto lo = optinum::simd::pack<float, W>(-1.0f);
    auto hi = optinum::simd::pack<float, W>(1.0f);
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::clamp(v, lo, hi);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_clamp(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::clamp(input[i], -1.0f, 1.0f);
        }
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Special Functions
// ============================================================================

template <std::size_t W> double benchmark_simd_erf(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::erf(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_erf(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::erf(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_tgamma(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::tgamma(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_tgamma(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::tgamma(input[i]);
        }
    }
    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_lgamma(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t vec_count = input.size() / W;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::lgamma(v);
            r.storeu(&output[i * W]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_lgamma(const std::vector<float> &input, std::vector<float> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = std::lgamma(input[i]);
        }
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

void print_result(const char *name, double simd_time, double scalar_time) {
    double speedup = scalar_time / simd_time;
    std::cout << std::setw(20) << std::left << name << std::setw(12) << std::right << std::fixed << std::setprecision(2)
              << simd_time << " ms" << std::setw(12) << scalar_time << " ms" << std::setw(10) << speedup << "x\n";
}

int main() {
    std::cout << "\n=== Complete SIMD Math Functions Benchmark ===\n";
    std::cout << "Elements: " << NUM_ELEMENTS << ", Iterations: " << NUM_ITERATIONS << "\n";
    std::cout << "SIMD Width (float): " << SIMD_WIDTH_F << "\n\n";

    std::vector<float> output(NUM_ELEMENTS);

    // Inverse Trigonometric
    std::cout << "--- Inverse Trigonometric Functions ---\n";
    std::cout << std::setw(20) << std::left << "Function" << std::setw(12) << std::right << "SIMD Time" << std::setw(12)
              << "Scalar Time" << std::setw(10) << "Speedup\n";
    std::cout << std::string(54, '-') << "\n";

    auto input_trig = generate_random_data<float>(NUM_ELEMENTS, -0.99f, 0.99f);
    print_result("asin", benchmark_simd_asin<SIMD_WIDTH_F>(input_trig, output),
                 benchmark_scalar_asin(input_trig, output));
    print_result("acos", benchmark_simd_acos<SIMD_WIDTH_F>(input_trig, output),
                 benchmark_scalar_acos(input_trig, output));

    auto input_atan = generate_random_data<float>(NUM_ELEMENTS, -10.0f, 10.0f);
    print_result("atan", benchmark_simd_atan<SIMD_WIDTH_F>(input_atan, output),
                 benchmark_scalar_atan(input_atan, output));

    auto y_data = generate_random_data<float>(NUM_ELEMENTS, -10.0f, 10.0f);
    auto x_data = generate_random_data<float>(NUM_ELEMENTS, -10.0f, 10.0f);
    print_result("atan2", benchmark_simd_atan2<SIMD_WIDTH_F>(y_data, x_data, output),
                 benchmark_scalar_atan2(y_data, x_data, output));

    // Hyperbolic Functions
    std::cout << "\n--- Hyperbolic Functions ---\n";
    std::cout << std::setw(20) << std::left << "Function" << std::setw(12) << std::right << "SIMD Time" << std::setw(12)
              << "Scalar Time" << std::setw(10) << "Speedup\n";
    std::cout << std::string(54, '-') << "\n";

    auto input_hyp = generate_random_data<float>(NUM_ELEMENTS, -2.0f, 2.0f);
    print_result("sinh", benchmark_simd_sinh<SIMD_WIDTH_F>(input_hyp, output),
                 benchmark_scalar_sinh(input_hyp, output));
    print_result("cosh", benchmark_simd_cosh<SIMD_WIDTH_F>(input_hyp, output),
                 benchmark_scalar_cosh(input_hyp, output));

    auto input_asinh = generate_random_data<float>(NUM_ELEMENTS, -10.0f, 10.0f);
    print_result("asinh", benchmark_simd_asinh<SIMD_WIDTH_F>(input_asinh, output),
                 benchmark_scalar_asinh(input_asinh, output));

    auto input_acosh = generate_random_data<float>(NUM_ELEMENTS, 1.01f, 10.0f);
    print_result("acosh", benchmark_simd_acosh<SIMD_WIDTH_F>(input_acosh, output),
                 benchmark_scalar_acosh(input_acosh, output));

    auto input_atanh = generate_random_data<float>(NUM_ELEMENTS, -0.99f, 0.99f);
    print_result("atanh", benchmark_simd_atanh<SIMD_WIDTH_F>(input_atanh, output),
                 benchmark_scalar_atanh(input_atanh, output));

    // Trigonometric - tan
    std::cout << "\n--- Additional Trigonometric ---\n";
    std::cout << std::setw(20) << std::left << "Function" << std::setw(12) << std::right << "SIMD Time" << std::setw(12)
              << "Scalar Time" << std::setw(10) << "Speedup\n";
    std::cout << std::string(54, '-') << "\n";

    auto input_tan = generate_random_data<float>(NUM_ELEMENTS, -1.5f, 1.5f);
    print_result("tan", benchmark_simd_tan<SIMD_WIDTH_F>(input_tan, output), benchmark_scalar_tan(input_tan, output));

    // Power and Root
    std::cout << "\n--- Power and Root Functions ---\n";
    std::cout << std::setw(20) << std::left << "Function" << std::setw(12) << std::right << "SIMD Time" << std::setw(12)
              << "Scalar Time" << std::setw(10) << "Speedup\n";
    std::cout << std::string(54, '-') << "\n";

    auto base_data = generate_random_data<float>(NUM_ELEMENTS, 0.1f, 10.0f);
    auto exp_data = generate_random_data<float>(NUM_ELEMENTS, -2.0f, 2.0f);
    print_result("pow", benchmark_simd_pow<SIMD_WIDTH_F>(base_data, exp_data, output),
                 benchmark_scalar_pow(base_data, exp_data, output));

    auto input_cbrt = generate_random_data<float>(NUM_ELEMENTS, -10.0f, 10.0f);
    print_result("cbrt", benchmark_simd_cbrt<SIMD_WIDTH_F>(input_cbrt, output),
                 benchmark_scalar_cbrt(input_cbrt, output));

    print_result("hypot", benchmark_simd_hypot<SIMD_WIDTH_F>(x_data, y_data, output),
                 benchmark_scalar_hypot(x_data, y_data, output));

    // Rounding
    std::cout << "\n--- Rounding Functions ---\n";
    std::cout << std::setw(20) << std::left << "Function" << std::setw(12) << std::right << "SIMD Time" << std::setw(12)
              << "Scalar Time" << std::setw(10) << "Speedup\n";
    std::cout << std::string(54, '-') << "\n";

    auto input_round = generate_random_data<float>(NUM_ELEMENTS, -100.5f, 100.5f);
    print_result("ceil", benchmark_simd_ceil<SIMD_WIDTH_F>(input_round, output),
                 benchmark_scalar_ceil(input_round, output));
    print_result("floor", benchmark_simd_floor<SIMD_WIDTH_F>(input_round, output),
                 benchmark_scalar_floor(input_round, output));
    print_result("round", benchmark_simd_round<SIMD_WIDTH_F>(input_round, output),
                 benchmark_scalar_round(input_round, output));
    print_result("trunc", benchmark_simd_trunc<SIMD_WIDTH_F>(input_round, output),
                 benchmark_scalar_trunc(input_round, output));

    // Logarithms
    std::cout << "\n--- Logarithm Variants ---\n";
    std::cout << std::setw(20) << std::left << "Function" << std::setw(12) << std::right << "SIMD Time" << std::setw(12)
              << "Scalar Time" << std::setw(10) << "Speedup\n";
    std::cout << std::string(54, '-') << "\n";

    auto input_log = generate_random_data<float>(NUM_ELEMENTS, 0.1f, 10.0f);
    print_result("log2", benchmark_simd_log2<SIMD_WIDTH_F>(input_log, output),
                 benchmark_scalar_log2(input_log, output));
    print_result("log10", benchmark_simd_log10<SIMD_WIDTH_F>(input_log, output),
                 benchmark_scalar_log10(input_log, output));

    auto input_log1p = generate_random_data<float>(NUM_ELEMENTS, -0.9f, 10.0f);
    print_result("log1p", benchmark_simd_log1p<SIMD_WIDTH_F>(input_log1p, output),
                 benchmark_scalar_log1p(input_log1p, output));

    // Exponentials
    std::cout << "\n--- Exponential Variants ---\n";
    std::cout << std::setw(20) << std::left << "Function" << std::setw(12) << std::right << "SIMD Time" << std::setw(12)
              << "Scalar Time" << std::setw(10) << "Speedup\n";
    std::cout << std::string(54, '-') << "\n";

    auto input_exp = generate_random_data<float>(NUM_ELEMENTS, -5.0f, 5.0f);
    print_result("exp2", benchmark_simd_exp2<SIMD_WIDTH_F>(input_exp, output),
                 benchmark_scalar_exp2(input_exp, output));

    auto input_expm1 = generate_random_data<float>(NUM_ELEMENTS, -2.0f, 2.0f);
    print_result("expm1", benchmark_simd_expm1<SIMD_WIDTH_F>(input_expm1, output),
                 benchmark_scalar_expm1(input_expm1, output));

    // Utility
    std::cout << "\n--- Utility Functions ---\n";
    std::cout << std::setw(20) << std::left << "Function" << std::setw(12) << std::right << "SIMD Time" << std::setw(12)
              << "Scalar Time" << std::setw(10) << "Speedup\n";
    std::cout << std::string(54, '-') << "\n";

    auto input_abs = generate_random_data<float>(NUM_ELEMENTS, -10.0f, 10.0f);
    print_result("abs", benchmark_simd_abs<SIMD_WIDTH_F>(input_abs, output), benchmark_scalar_abs(input_abs, output));

    auto input_clamp = generate_random_data<float>(NUM_ELEMENTS, -5.0f, 5.0f);
    print_result("clamp", benchmark_simd_clamp<SIMD_WIDTH_F>(input_clamp, output),
                 benchmark_scalar_clamp(input_clamp, output));

    // Special Functions
    std::cout << "\n--- Special Functions ---\n";
    std::cout << std::setw(20) << std::left << "Function" << std::setw(12) << std::right << "SIMD Time" << std::setw(12)
              << "Scalar Time" << std::setw(10) << "Speedup\n";
    std::cout << std::string(54, '-') << "\n";

    auto input_erf = generate_random_data<float>(NUM_ELEMENTS, -3.0f, 3.0f);
    print_result("erf", benchmark_simd_erf<SIMD_WIDTH_F>(input_erf, output), benchmark_scalar_erf(input_erf, output));

    auto input_gamma = generate_random_data<float>(NUM_ELEMENTS, 0.5f, 5.0f);
    print_result("tgamma", benchmark_simd_tgamma<SIMD_WIDTH_F>(input_gamma, output),
                 benchmark_scalar_tgamma(input_gamma, output));
    print_result("lgamma", benchmark_simd_lgamma<SIMD_WIDTH_F>(input_gamma, output),
                 benchmark_scalar_lgamma(input_gamma, output));

    std::cout << "\n=== Benchmark Complete ===\n\n";
    return 0;
}
