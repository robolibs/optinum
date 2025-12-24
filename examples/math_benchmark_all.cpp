// =============================================================================
// Comprehensive Math Benchmark - All 6 Functions
// Compares new pack<T,W> API against scalar implementations
// Uses datapod 0.0.8 heap-allocated vectors for large data
// =============================================================================

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

// New API
#include <datapod/matrix.hpp>
#include <optinum/simd/math/cos.hpp>
#include <optinum/simd/math/exp.hpp>
#include <optinum/simd/math/log.hpp>
#include <optinum/simd/math/sin.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/math/tanh.hpp>

// Benchmark configuration
constexpr std::size_t NUM_ELEMENTS = 1024 * 1024; // 1M elements
constexpr std::size_t NUM_ITERATIONS = 100;

// Timer helper
class Timer {
  public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count() / 1000.0;
    }

  private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Generate random data
void fill_random(datapod::mat::vector<float, NUM_ELEMENTS> &vec, float min_val, float max_val) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
        vec[i] = dist(gen);
    }
}

// =============================================================================
// Scalar Benchmarks
// =============================================================================

double benchmark_scalar_exp(const datapod::mat::vector<float, NUM_ELEMENTS> &input,
                            datapod::mat::vector<float, NUM_ELEMENTS> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
            output[i] = std::exp(input[i]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_log(const datapod::mat::vector<float, NUM_ELEMENTS> &input,
                            datapod::mat::vector<float, NUM_ELEMENTS> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
            output[i] = std::log(input[i]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_sin(const datapod::mat::vector<float, NUM_ELEMENTS> &input,
                            datapod::mat::vector<float, NUM_ELEMENTS> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
            output[i] = std::sin(input[i]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_cos(const datapod::mat::vector<float, NUM_ELEMENTS> &input,
                            datapod::mat::vector<float, NUM_ELEMENTS> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
            output[i] = std::cos(input[i]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_tanh(const datapod::mat::vector<float, NUM_ELEMENTS> &input,
                             datapod::mat::vector<float, NUM_ELEMENTS> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
            output[i] = std::tanh(input[i]);
        }
    }
    return timer.elapsed_ms();
}

double benchmark_scalar_sqrt(const datapod::mat::vector<float, NUM_ELEMENTS> &input,
                             datapod::mat::vector<float, NUM_ELEMENTS> &output) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < NUM_ELEMENTS; ++i) {
            output[i] = std::sqrt(input[i]);
        }
    }
    return timer.elapsed_ms();
}

// =============================================================================
// SIMD Benchmarks (new pack<T,W> API)
// =============================================================================

template <std::size_t W>
double benchmark_simd_exp(const datapod::mat::vector<float, NUM_ELEMENTS> &input,
                          datapod::mat::vector<float, NUM_ELEMENTS> &output) {
    const std::size_t vec_count = NUM_ELEMENTS / W;
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

template <std::size_t W>
double benchmark_simd_log(const datapod::mat::vector<float, NUM_ELEMENTS> &input,
                          datapod::mat::vector<float, NUM_ELEMENTS> &output) {
    const std::size_t vec_count = NUM_ELEMENTS / W;
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

template <std::size_t W>
double benchmark_simd_sin(const datapod::mat::vector<float, NUM_ELEMENTS> &input,
                          datapod::mat::vector<float, NUM_ELEMENTS> &output) {
    const std::size_t vec_count = NUM_ELEMENTS / W;
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

template <std::size_t W>
double benchmark_simd_cos(const datapod::mat::vector<float, NUM_ELEMENTS> &input,
                          datapod::mat::vector<float, NUM_ELEMENTS> &output) {
    const std::size_t vec_count = NUM_ELEMENTS / W;
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

template <std::size_t W>
double benchmark_simd_tanh(const datapod::mat::vector<float, NUM_ELEMENTS> &input,
                           datapod::mat::vector<float, NUM_ELEMENTS> &output) {
    const std::size_t vec_count = NUM_ELEMENTS / W;
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

template <std::size_t W>
double benchmark_simd_sqrt(const datapod::mat::vector<float, NUM_ELEMENTS> &input,
                           datapod::mat::vector<float, NUM_ELEMENTS> &output) {
    const std::size_t vec_count = NUM_ELEMENTS / W;
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

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "   Comprehensive Math Benchmark - New pack<T,W> API\n";
    std::cout << "================================================================\n";
    std::cout << "  Elements: " << NUM_ELEMENTS << " (1M)\n";
    std::cout << "  Iterations: " << NUM_ITERATIONS << "\n";
    std::cout << "  Total ops: " << (NUM_ELEMENTS * NUM_ITERATIONS / 1000000) << "M\n";
    std::cout << "================================================================\n\n";

    // Datapod 0.0.8 heap-allocated vectors
    datapod::mat::vector<float, NUM_ELEMENTS> input_exp;
    datapod::mat::vector<float, NUM_ELEMENTS> input_log;
    datapod::mat::vector<float, NUM_ELEMENTS> input_trig;
    datapod::mat::vector<float, NUM_ELEMENTS> input_tanh;
    datapod::mat::vector<float, NUM_ELEMENTS> input_sqrt;
    datapod::mat::vector<float, NUM_ELEMENTS> output_scalar;
    datapod::mat::vector<float, NUM_ELEMENTS> output_simd;

    std::cout << "Datapod vector info:\n";
    std::cout << "  uses_heap: " << std::boolalpha << input_exp.uses_heap << "\n";
    std::cout << "  is_pod: " << input_exp.is_pod << "\n";
    std::cout << "  data ptr: " << (void *)input_exp.data() << "\n\n";

    // Fill with appropriate ranges for each function
    fill_random(input_exp, -5.0f, 5.0f);    // exp: [-5, 5]
    fill_random(input_log, 0.1f, 10.0f);    // log: (0, 10]
    fill_random(input_trig, -3.14f, 3.14f); // sin/cos: [-π, π]
    fill_random(input_tanh, -5.0f, 5.0f);   // tanh: [-5, 5]
    fill_random(input_sqrt, 0.1f, 100.0f);  // sqrt: [0.1, 100]

    std::cout << "+-----------+--------+----+------------+------------+----------+\n";
    std::cout << "| Function  |  Type  | W  | SIMD (ms)  | Scalar(ms) | Speedup  |\n";
    std::cout << "+-----------+--------+----+------------+------------+----------+\n";

    // exp()
    {
        double time_scalar = benchmark_scalar_exp(input_exp, output_scalar);
        double time_simd = benchmark_simd_exp<8>(input_exp, output_simd);
        std::cout << "|  exp      |  float |  8 | " << std::setw(10) << std::fixed << std::setprecision(2) << time_simd
                  << " | " << std::setw(10) << time_scalar << " | " << std::setw(7) << std::setprecision(2)
                  << (time_scalar / time_simd) << "x |\n";
    }

    // log()
    {
        double time_scalar = benchmark_scalar_log(input_log, output_scalar);
        double time_simd = benchmark_simd_log<8>(input_log, output_simd);
        std::cout << "|  log      |  float |  8 | " << std::setw(10) << std::fixed << std::setprecision(2) << time_simd
                  << " | " << std::setw(10) << time_scalar << " | " << std::setw(7) << std::setprecision(2)
                  << (time_scalar / time_simd) << "x |\n";
    }

    // sin()
    {
        double time_scalar = benchmark_scalar_sin(input_trig, output_scalar);
        double time_simd = benchmark_simd_sin<8>(input_trig, output_simd);
        std::cout << "|  sin      |  float |  8 | " << std::setw(10) << std::fixed << std::setprecision(2) << time_simd
                  << " | " << std::setw(10) << time_scalar << " | " << std::setw(7) << std::setprecision(2)
                  << (time_scalar / time_simd) << "x |\n";
    }

    // cos()
    {
        double time_scalar = benchmark_scalar_cos(input_trig, output_scalar);
        double time_simd = benchmark_simd_cos<8>(input_trig, output_simd);
        std::cout << "|  cos      |  float |  8 | " << std::setw(10) << std::fixed << std::setprecision(2) << time_simd
                  << " | " << std::setw(10) << time_scalar << " | " << std::setw(7) << std::setprecision(2)
                  << (time_scalar / time_simd) << "x |\n";
    }

    // tanh()
    {
        double time_scalar = benchmark_scalar_tanh(input_tanh, output_scalar);
        double time_simd = benchmark_simd_tanh<8>(input_tanh, output_simd);
        std::cout << "|  tanh     |  float |  8 | " << std::setw(10) << std::fixed << std::setprecision(2) << time_simd
                  << " | " << std::setw(10) << time_scalar << " | " << std::setw(7) << std::setprecision(2)
                  << (time_scalar / time_simd) << "x |\n";
    }

    // sqrt()
    {
        double time_scalar = benchmark_scalar_sqrt(input_sqrt, output_scalar);
        double time_simd = benchmark_simd_sqrt<8>(input_sqrt, output_simd);
        std::cout << "|  sqrt     |  float |  8 | " << std::setw(10) << std::fixed << std::setprecision(2) << time_simd
                  << " | " << std::setw(10) << time_scalar << " | " << std::setw(7) << std::setprecision(2)
                  << (time_scalar / time_simd) << "x |\n";
    }

    std::cout << "+-----------+--------+----+------------+------------+----------+\n";
    std::cout << "\nNotes:\n";
    std::cout << "  - All functions use pack<float, 8> (AVX - 256-bit)\n";
    std::cout << "  - Input ranges optimized for each function\n";
    std::cout << "  - Datapod 0.0.8 heap allocation for large vectors\n";

    return 0;
}
