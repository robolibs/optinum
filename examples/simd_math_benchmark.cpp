// Native SIMD Math Benchmark
// Benchmarks our SIMD math implementations against scalar std:: functions
// Uses the new pack<T,W> API

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Include SIMD math
#include <optinum/simd/math/simd_math.hpp>
#include <optinum/simd/pack/pack.hpp>

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
// SIMD benchmarks using pack<T,W> API
// ============================================================================

template <std::size_t W> double benchmark_simd_exp(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t n = input.size();
    const std::size_t vec_count = n / W;

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::exp(v);
            r.storeu(&output[i * W]);
        }
    }

    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_log(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t n = input.size();
    const std::size_t vec_count = n / W;

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::log(v);
            r.storeu(&output[i * W]);
        }
    }

    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_sin(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t n = input.size();
    const std::size_t vec_count = n / W;

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::sin(v);
            r.storeu(&output[i * W]);
        }
    }

    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_cos(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t n = input.size();
    const std::size_t vec_count = n / W;

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::cos(v);
            r.storeu(&output[i * W]);
        }
    }

    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_tanh(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t n = input.size();
    const std::size_t vec_count = n / W;

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::tanh(v);
            r.storeu(&output[i * W]);
        }
    }

    return timer.elapsed_ms();
}

template <std::size_t W> double benchmark_simd_sqrt(const std::vector<float> &input, std::vector<float> &output) {
    const std::size_t n = input.size();
    const std::size_t vec_count = n / W;

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < vec_count; ++i) {
            auto v = optinum::simd::pack<float, W>::loadu(&input[i * W]);
            auto r = optinum::simd::sqrt(v);
            r.storeu(&output[i * W]);
        }
    }

    return timer.elapsed_ms();
}

// ============================================================================
// Scalar benchmarks (uses std:: functions)
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

template <typename T> double benchmark_scalar_sqrt(const std::vector<T> &input, std::vector<T> &output) {
    const std::size_t n = input.size();

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = std::sqrt(input[i]);
        }
    }

    return timer.elapsed_ms();
}

// ============================================================================
// Print utilities
// ============================================================================

void print_separator() { std::cout << "+-----------+--------+----+------------+------------+----------+\n"; }

void print_header_row() { std::cout << "|  Function |  Type  | W  |  SIMD (ms) | Scalar(ms) |  Speedup |\n"; }

void print_result(const char *func_name, const char *type_name, std::size_t width, double simd_ms, double scalar_ms) {
    double speedup = scalar_ms / simd_ms;
    std::cout << "| " << std::setw(9) << func_name << " | " << std::setw(6) << type_name << " | " << std::setw(2)
              << width << " | " << std::setw(10) << std::fixed << std::setprecision(2) << simd_ms << " | "
              << std::setw(10) << scalar_ms << " | " << std::setw(7) << std::setprecision(2) << speedup << "x |\n";
}

int main() {
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "           Native SIMD Math Benchmark\n";
    std::cout << "================================================================\n";
    std::cout << "  Backend: optinum pack<T,W> (native SIMD, no external deps)\n";
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
        std::vector<float> output_f(NUM_ELEMENTS);

        double scalar_f = benchmark_scalar_exp(input_f, output_f);
        double simd_f = benchmark_simd_exp<SIMD_WIDTH_F>(input_f, output_f);
        print_result("exp", "float", SIMD_WIDTH_F, simd_f, scalar_f);
    }

    // ========================================================================
    // log() benchmark
    // ========================================================================
    {
        auto input_f = generate_random_data<float>(NUM_ELEMENTS, 0.001f, 1000.0f);
        std::vector<float> output_f(NUM_ELEMENTS);

        double scalar_f = benchmark_scalar_log(input_f, output_f);
        double simd_f = benchmark_simd_log<SIMD_WIDTH_F>(input_f, output_f);
        print_result("log", "float", SIMD_WIDTH_F, simd_f, scalar_f);
    }

    // ========================================================================
    // sin() benchmark
    // ========================================================================
    {
        auto input_f = generate_random_data<float>(NUM_ELEMENTS, -3.14159f, 3.14159f);
        std::vector<float> output_f(NUM_ELEMENTS);

        double scalar_f = benchmark_scalar_sin(input_f, output_f);
        double simd_f = benchmark_simd_sin<SIMD_WIDTH_F>(input_f, output_f);
        print_result("sin", "float", SIMD_WIDTH_F, simd_f, scalar_f);
    }

    // ========================================================================
    // cos() benchmark
    // ========================================================================
    {
        auto input_f = generate_random_data<float>(NUM_ELEMENTS, -3.14159f, 3.14159f);
        std::vector<float> output_f(NUM_ELEMENTS);

        double scalar_f = benchmark_scalar_cos(input_f, output_f);
        double simd_f = benchmark_simd_cos<SIMD_WIDTH_F>(input_f, output_f);
        print_result("cos", "float", SIMD_WIDTH_F, simd_f, scalar_f);
    }

    // ========================================================================
    // tanh() benchmark (important for ML/neural networks)
    // ========================================================================
    {
        auto input_f = generate_random_data<float>(NUM_ELEMENTS, -5.0f, 5.0f);
        std::vector<float> output_f(NUM_ELEMENTS);

        double scalar_f = benchmark_scalar_tanh(input_f, output_f);
        double simd_f = benchmark_simd_tanh<SIMD_WIDTH_F>(input_f, output_f);
        print_result("tanh", "float", SIMD_WIDTH_F, simd_f, scalar_f);
    }

    // ========================================================================
    // sqrt() benchmark
    // ========================================================================
    {
        auto input_f = generate_random_data<float>(NUM_ELEMENTS, 0.001f, 1000.0f);
        std::vector<float> output_f(NUM_ELEMENTS);

        double scalar_f = benchmark_scalar_sqrt(input_f, output_f);
        double simd_f = benchmark_simd_sqrt<SIMD_WIDTH_F>(input_f, output_f);
        print_result("sqrt", "float", SIMD_WIDTH_F, simd_f, scalar_f);
    }

    print_separator();

    std::cout << "\n";
    std::cout << "Legend:\n";
    std::cout << "  W        = SIMD width (elements processed in parallel)\n";
    std::cout << "  SIMD     = Time using native SIMD functions with pack<T,W>\n";
    std::cout << "  Scalar   = Time using std:: scalar operations\n";
    std::cout << "  Speedup  = Scalar time / SIMD time (higher is better)\n";
    std::cout << "\n";
    std::cout << "All math functions use native AVX/SSE intrinsics with ~3-5 ULP accuracy.\n";
    std::cout << "No external dependencies (SLEEF, SVML, etc.) required.\n";
    std::cout << "\n";

    return 0;
}
