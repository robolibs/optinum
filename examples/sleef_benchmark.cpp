// SLEEF SIMD Math Benchmark
// Compares performance of SIMD math functions with and without SLEEF

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Include SIMD math after chrono to avoid operator conflicts
#include <optinum/simd/math/fast_exp.hpp>
#include <optinum/simd/math/simd_math.hpp>

// Use architecture constants for SIMD width
constexpr std::size_t SIMD_WIDTH_F = optinum::simd::arch::SIMD_WIDTH_FLOAT;
constexpr std::size_t SIMD_WIDTH_D = optinum::simd::arch::SIMD_WIDTH_DOUBLE;

// Benchmark configuration
constexpr std::size_t NUM_ELEMENTS = 1024 * 1024; // 1M elements
constexpr std::size_t NUM_ITERATIONS = 100;

// Helper to generate random data
template <typename T> std::vector<T> generate_random_data(std::size_t n, T min_val, T max_val) {
    std::vector<T> data(n);
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<T> dist(min_val, max_val);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

// Timer helper class
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
// SIMD benchmarks (uses SLEEF when OPTINUM_USE_SLEEF is defined)
// ============================================================================

template <typename T, std::size_t W> double benchmark_simd_exp(const std::vector<T> &input, std::vector<T> &output) {
    const std::size_t n = input.size();
    const std::size_t vec_count = n / W;

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::SIMDVec<T, W>::loadu(&input[i * W]);
            auto r = optinum::simd::exp(v);
            r.storeu(&output[i * W]);
        }
    }

    return timer.elapsed_ms();
}

// Benchmark our own fast_exp implementation
template <std::size_t W> double benchmark_fast_exp(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t n = input.size();
    const std::size_t vec_count = n / W;

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::SIMDVec<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::fast_exp(v);
            r.storeu(&output[i * W]);
        }
    }

    return timer.elapsed_ms();
}

template <typename T, std::size_t W> double benchmark_simd_log(const std::vector<T> &input, std::vector<T> &output) {
    const std::size_t n = input.size();
    const std::size_t vec_count = n / W;

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::SIMDVec<T, W>::loadu(&input[i * W]);
            auto r = optinum::simd::log(v);
            r.storeu(&output[i * W]);
        }
    }

    return timer.elapsed_ms();
}

template <typename T, std::size_t W> double benchmark_simd_sin(const std::vector<T> &input, std::vector<T> &output) {
    const std::size_t n = input.size();
    const std::size_t vec_count = n / W;

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::SIMDVec<T, W>::loadu(&input[i * W]);
            auto r = optinum::simd::sin(v);
            r.storeu(&output[i * W]);
        }
    }

    return timer.elapsed_ms();
}

template <typename T, std::size_t W> double benchmark_simd_cos(const std::vector<T> &input, std::vector<T> &output) {
    const std::size_t n = input.size();
    const std::size_t vec_count = n / W;

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::SIMDVec<T, W>::loadu(&input[i * W]);
            auto r = optinum::simd::cos(v);
            r.storeu(&output[i * W]);
        }
    }

    return timer.elapsed_ms();
}

template <typename T, std::size_t W>
double benchmark_simd_pow(const std::vector<T> &input, const std::vector<T> &exp_vals, std::vector<T> &output) {
    const std::size_t n = input.size();
    const std::size_t vec_count = n / W;

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::SIMDVec<T, W>::loadu(&input[i * W]);
            auto e = optinum::simd::SIMDVec<T, W>::loadu(&exp_vals[i * W]);
            auto r = optinum::simd::pow(v, e);
            r.storeu(&output[i * W]);
        }
    }

    return timer.elapsed_ms();
}

template <typename T, std::size_t W> double benchmark_simd_tanh(const std::vector<T> &input, std::vector<T> &output) {
    const std::size_t n = input.size();
    const std::size_t vec_count = n / W;

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::SIMDVec<T, W>::loadu(&input[i * W]);
            auto r = optinum::simd::tanh(v);
            r.storeu(&output[i * W]);
        }
    }

    return timer.elapsed_ms();
}

// ============================================================================
// Scalar benchmarks (always uses std:: functions)
// ============================================================================

template <typename T> double benchmark_scalar_exp(const std::vector<T> &input, std::vector<T> &output) {
    const std::size_t n = input.size();

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = std::exp(input[i]);
        }
    }

    return timer.elapsed_ms();
}

template <typename T> double benchmark_scalar_log(const std::vector<T> &input, std::vector<T> &output) {
    const std::size_t n = input.size();

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = std::log(input[i]);
        }
    }

    return timer.elapsed_ms();
}

template <typename T> double benchmark_scalar_sin(const std::vector<T> &input, std::vector<T> &output) {
    const std::size_t n = input.size();

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = std::sin(input[i]);
        }
    }

    return timer.elapsed_ms();
}

template <typename T> double benchmark_scalar_cos(const std::vector<T> &input, std::vector<T> &output) {
    const std::size_t n = input.size();

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = std::cos(input[i]);
        }
    }

    return timer.elapsed_ms();
}

template <typename T>
double benchmark_scalar_pow(const std::vector<T> &input, const std::vector<T> &exp_vals, std::vector<T> &output) {
    const std::size_t n = input.size();

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = std::pow(input[i], exp_vals[i]);
        }
    }

    return timer.elapsed_ms();
}

template <typename T> double benchmark_scalar_tanh(const std::vector<T> &input, std::vector<T> &output) {
    const std::size_t n = input.size();

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = std::tanh(input[i]);
        }
    }

    return timer.elapsed_ms();
}

// ============================================================================
// Print utilities
// ============================================================================

void print_separator() { std::cout << "+----------+--------+----+------------+------------+----------+\n"; }

void print_header_row() { std::cout << "| Function |  Type  | W  |  SIMD (ms) | Scalar(ms) |  Speedup |\n"; }

void print_result(const char *func_name, const char *type_name, std::size_t width, double simd_ms, double scalar_ms) {
    double speedup = scalar_ms / simd_ms;
    std::cout << "| " << std::setw(8) << func_name << " | " << std::setw(6) << type_name << " | " << std::setw(2)
              << width << " | " << std::setw(10) << std::fixed << std::setprecision(2) << simd_ms << " | "
              << std::setw(10) << scalar_ms << " | " << std::setw(7) << std::setprecision(2) << speedup << "x |\n";
}

int main() {
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "           SLEEF SIMD Math Benchmark\n";
    std::cout << "================================================================\n";
#ifdef OPTINUM_USE_SLEEF
    std::cout << "  Backend: SLEEF (vectorized intrinsics)\n";
#else
    std::cout << "  Backend: std:: (lane-by-lane scalar fallback)\n";
#endif
    std::cout << "  SIMD Level: " << optinum::simd::arch::simd_level() << "-bit\n";
    std::cout << "  Float width: " << SIMD_WIDTH_F << " elements\n";
    std::cout << "  Double width: " << SIMD_WIDTH_D << " elements\n";
    std::cout << "  Elements: " << NUM_ELEMENTS << " (" << (NUM_ELEMENTS / 1024 / 1024) << "M)\n";
    std::cout << "  Iterations: " << NUM_ITERATIONS << "\n";
    std::cout << "  Total ops per function: " << (NUM_ELEMENTS * NUM_ITERATIONS / 1000000) << "M\n";
    std::cout << "================================================================\n\n";

    print_separator();
    print_header_row();
    print_separator();

    // ========================================================================
    // exp() benchmark
    // ========================================================================
    {
        auto input_f = generate_random_data<float>(NUM_ELEMENTS, -10.0f, 10.0f);
        auto input_d = generate_random_data<double>(NUM_ELEMENTS, -10.0, 10.0);
        std::vector<float> output_f(NUM_ELEMENTS);
        std::vector<double> output_d(NUM_ELEMENTS);

        double scalar_f = benchmark_scalar_exp(input_f, output_f);
        double scalar_d = benchmark_scalar_exp(input_d, output_d);

        double simd_f = benchmark_simd_exp<float, SIMD_WIDTH_F>(input_f, output_f);
        double simd_d = benchmark_simd_exp<double, SIMD_WIDTH_D>(input_d, output_d);
        print_result("exp", "float", SIMD_WIDTH_F, simd_f, scalar_f);
        print_result("exp", "double", SIMD_WIDTH_D, simd_d, scalar_d);

        // Also benchmark our own fast_exp
        double fast_f = benchmark_fast_exp<SIMD_WIDTH_F>(input_f, output_f);
        print_result("fast_exp", "float", SIMD_WIDTH_F, fast_f, scalar_f);
    }

    // ========================================================================
    // log() benchmark
    // ========================================================================
    {
        auto input_f = generate_random_data<float>(NUM_ELEMENTS, 0.001f, 1000.0f);
        auto input_d = generate_random_data<double>(NUM_ELEMENTS, 0.001, 1000.0);
        std::vector<float> output_f(NUM_ELEMENTS);
        std::vector<double> output_d(NUM_ELEMENTS);

        double scalar_f = benchmark_scalar_log(input_f, output_f);
        double scalar_d = benchmark_scalar_log(input_d, output_d);

        double simd_f = benchmark_simd_log<float, SIMD_WIDTH_F>(input_f, output_f);
        double simd_d = benchmark_simd_log<double, SIMD_WIDTH_D>(input_d, output_d);
        print_result("log", "float", SIMD_WIDTH_F, simd_f, scalar_f);
        print_result("log", "double", SIMD_WIDTH_D, simd_d, scalar_d);
    }

    // ========================================================================
    // sin() benchmark
    // ========================================================================
    {
        auto input_f = generate_random_data<float>(NUM_ELEMENTS, -3.14159f, 3.14159f);
        auto input_d = generate_random_data<double>(NUM_ELEMENTS, -3.14159, 3.14159);
        std::vector<float> output_f(NUM_ELEMENTS);
        std::vector<double> output_d(NUM_ELEMENTS);

        double scalar_f = benchmark_scalar_sin(input_f, output_f);
        double scalar_d = benchmark_scalar_sin(input_d, output_d);

        double simd_f = benchmark_simd_sin<float, SIMD_WIDTH_F>(input_f, output_f);
        double simd_d = benchmark_simd_sin<double, SIMD_WIDTH_D>(input_d, output_d);
        print_result("sin", "float", SIMD_WIDTH_F, simd_f, scalar_f);
        print_result("sin", "double", SIMD_WIDTH_D, simd_d, scalar_d);
    }

    // ========================================================================
    // cos() benchmark
    // ========================================================================
    {
        auto input_f = generate_random_data<float>(NUM_ELEMENTS, -3.14159f, 3.14159f);
        auto input_d = generate_random_data<double>(NUM_ELEMENTS, -3.14159, 3.14159);
        std::vector<float> output_f(NUM_ELEMENTS);
        std::vector<double> output_d(NUM_ELEMENTS);

        double scalar_f = benchmark_scalar_cos(input_f, output_f);
        double scalar_d = benchmark_scalar_cos(input_d, output_d);

        double simd_f = benchmark_simd_cos<float, SIMD_WIDTH_F>(input_f, output_f);
        double simd_d = benchmark_simd_cos<double, SIMD_WIDTH_D>(input_d, output_d);
        print_result("cos", "float", SIMD_WIDTH_F, simd_f, scalar_f);
        print_result("cos", "double", SIMD_WIDTH_D, simd_d, scalar_d);
    }

    // ========================================================================
    // tanh() benchmark (important for ML/neural networks)
    // ========================================================================
    {
        auto input_f = generate_random_data<float>(NUM_ELEMENTS, -5.0f, 5.0f);
        auto input_d = generate_random_data<double>(NUM_ELEMENTS, -5.0, 5.0);
        std::vector<float> output_f(NUM_ELEMENTS);
        std::vector<double> output_d(NUM_ELEMENTS);

        double scalar_f = benchmark_scalar_tanh(input_f, output_f);
        double scalar_d = benchmark_scalar_tanh(input_d, output_d);

        double simd_f = benchmark_simd_tanh<float, SIMD_WIDTH_F>(input_f, output_f);
        double simd_d = benchmark_simd_tanh<double, SIMD_WIDTH_D>(input_d, output_d);
        print_result("tanh", "float", SIMD_WIDTH_F, simd_f, scalar_f);
        print_result("tanh", "double", SIMD_WIDTH_D, simd_d, scalar_d);
    }

    // ========================================================================
    // pow() benchmark
    // ========================================================================
    {
        auto input_f = generate_random_data<float>(NUM_ELEMENTS, 0.1f, 10.0f);
        auto exp_f = generate_random_data<float>(NUM_ELEMENTS, 0.5f, 3.0f);
        auto input_d = generate_random_data<double>(NUM_ELEMENTS, 0.1, 10.0);
        auto exp_d = generate_random_data<double>(NUM_ELEMENTS, 0.5, 3.0);
        std::vector<float> output_f(NUM_ELEMENTS);
        std::vector<double> output_d(NUM_ELEMENTS);

        double scalar_f = benchmark_scalar_pow(input_f, exp_f, output_f);
        double scalar_d = benchmark_scalar_pow(input_d, exp_d, output_d);

        double simd_f = benchmark_simd_pow<float, SIMD_WIDTH_F>(input_f, exp_f, output_f);
        double simd_d = benchmark_simd_pow<double, SIMD_WIDTH_D>(input_d, exp_d, output_d);
        print_result("pow", "float", SIMD_WIDTH_F, simd_f, scalar_f);
        print_result("pow", "double", SIMD_WIDTH_D, simd_d, scalar_d);
    }

    print_separator();

    std::cout << "\n";
    std::cout << "Legend:\n";
    std::cout << "  W        = SIMD width (elements processed in parallel)\n";
    std::cout << "  SIMD     = Time using SIMD operations";
#ifdef OPTINUM_USE_SLEEF
    std::cout << " (SLEEF vectorized)\n";
#else
    std::cout << " (scalar fallback)\n";
#endif
    std::cout << "  Scalar   = Time using std:: scalar operations\n";
    std::cout << "  Speedup  = Scalar time / SIMD time (higher is better)\n";
    std::cout << "\n";

#ifdef OPTINUM_USE_SLEEF
    std::cout << "SLEEF is ENABLED - you should see significant speedups\n";
    std::cout << "(typically 4-8x for float, 2-4x for double on AVX).\n";
#else
    std::cout << "SLEEF is DISABLED - SIMD falls back to lane-by-lane std::\n";
    std::cout << "calls, so speedup will be ~1x. Enable with: xmake f --sleef=y\n";
#endif
    std::cout << "\n";

    return 0;
}
