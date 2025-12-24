// =============================================================================
// Fast Math Benchmark - New API (pack<T,W>) with datapod mat::vector
// Uses datapod 0.0.8 heap-allocated vectors for large data
// =============================================================================

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

// New API
#include <datapod/matrix.hpp>
#include <optinum/simd/math/exp.hpp>

// Benchmark configuration - MATCH OLD BENCHMARK EXACTLY
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

// Scalar baseline
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

// NEW API - pack<float,W>
template <std::size_t W>
double benchmark_new_fast_exp(const datapod::mat::vector<float, NUM_ELEMENTS> &input,
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

int main() {
    std::cout << "================================================================\n";
    std::cout << "     NEW API Fast Exp Benchmark (pack<T,W> + datapod 0.0.8)\n";
    std::cout << "================================================================\n";
    std::cout << "  Elements: " << NUM_ELEMENTS << " (1M)\n";
    std::cout << "  Iterations: " << NUM_ITERATIONS << "\n";
    std::cout << "  Total ops: " << (NUM_ELEMENTS * NUM_ITERATIONS / 1000000) << "M\n";
    std::cout << "================================================================\n\n";

    // Datapod 0.0.8 heap-allocated vectors!
    datapod::mat::vector<float, NUM_ELEMENTS> input;
    datapod::mat::vector<float, NUM_ELEMENTS> output_scalar;
    datapod::mat::vector<float, NUM_ELEMENTS> output_simd;

    std::cout << "Datapod vector info:\n";
    std::cout << "  uses_heap: " << std::boolalpha << input.uses_heap << "\n";
    std::cout << "  is_pod: " << input.is_pod << "\n";
    std::cout << "  data ptr: " << (void *)input.data() << "\n\n";

    fill_random(input, -5.0f, 5.0f);

    std::cout << "+-----------+--------+----+------------+------------+----------+\n";
    std::cout << "|  Function |  Type  | W  |  SIMD (ms) | Scalar(ms) |  Speedup |\n";
    std::cout << "+-----------+--------+----+------------+------------+----------+\n";

    // Scalar
    double time_scalar = benchmark_scalar_exp(input, output_scalar);

    // NEW API W=8
    double time_simd_w8 = benchmark_new_fast_exp<8>(input, output_simd);

    std::cout << "|  fast_exp |  float |  8 | " << std::setw(10) << std::fixed << std::setprecision(2) << time_simd_w8
              << " | " << std::setw(10) << time_scalar << " | " << std::setw(7) << std::setprecision(2)
              << (time_scalar / time_simd_w8) << "x |\n";

    std::cout << "+-----------+--------+----+------------+------------+----------+\n";

    return 0;
}
