// =============================================================================
// Cross-Platform SIMD Pack Benchmark
// Compares performance across AVX-512, AVX2, SSE, NEON, and Scalar backends
// =============================================================================

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/pack/pack.hpp>

#ifdef OPTINUM_HAS_AVX512F
#include <optinum/simd/pack/avx512.hpp>
#endif

#ifdef OPTINUM_HAS_AVX2
#include <optinum/simd/pack/avx.hpp>
#endif

#ifdef OPTINUM_HAS_SSE2
#include <optinum/simd/pack/sse.hpp>
#endif

#ifdef OPTINUM_HAS_NEON
#include <optinum/simd/pack/neon.hpp>
#endif

constexpr std::size_t NUM_ELEMENTS = 1024 * 1024; // 1M elements
constexpr std::size_t NUM_ITERATIONS = 100;

// =============================================================================
// Utilities
// =============================================================================

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

template <typename T> std::vector<T> generate_random_data(std::size_t n, T min_val, T max_val) {
    std::vector<T> data(n);
    std::mt19937 gen(42);
    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        for (std::size_t i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }
    } else {
        std::uniform_int_distribution<T> dist(min_val, max_val);
        for (std::size_t i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }
    }
    return data;
}

std::string get_simd_backend() {
#ifdef OPTINUM_HAS_AVX512F
    return "AVX-512";
#elif defined(OPTINUM_HAS_AVX2)
    return "AVX2";
#elif defined(OPTINUM_HAS_AVX)
    return "AVX";
#elif defined(OPTINUM_HAS_SSE2)
    return "SSE2";
#elif defined(OPTINUM_HAS_NEON)
    return "NEON";
#else
    return "Scalar";
#endif
}

template <typename T, std::size_t W> std::string get_pack_name() {
    std::ostringstream oss;
    oss << "pack<";
    if constexpr (std::is_same_v<T, float>) {
        oss << "float";
    } else if constexpr (std::is_same_v<T, double>) {
        oss << "double";
    } else if constexpr (std::is_same_v<T, int32_t>) {
        oss << "int32";
    } else if constexpr (std::is_same_v<T, int64_t>) {
        oss << "int64";
    }
    oss << "," << W << ">";
    return oss.str();
}

// =============================================================================
// Benchmark Functions
// =============================================================================

template <typename T, std::size_t W> double benchmark_pack_add(const T *a, const T *b, T *c, std::size_t n) {
    using pack_t = optinum::simd::pack<T, W>;
    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; i += W) {
            auto pa = pack_t::load(a + i);
            auto pb = pack_t::load(b + i);
            auto pc = pa + pb;
            pc.store(c + i);
        }
    }

    return timer.elapsed_ms();
}

template <typename T, std::size_t W> double benchmark_pack_mul(const T *a, const T *b, T *c, std::size_t n) {
    using pack_t = optinum::simd::pack<T, W>;
    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; i += W) {
            auto pa = pack_t::load(a + i);
            auto pb = pack_t::load(b + i);
            auto pc = pa * pb;
            pc.store(c + i);
        }
    }

    return timer.elapsed_ms();
}

template <typename T, std::size_t W>
double benchmark_pack_fma(const T *a, const T *b, const T *c, T *d, std::size_t n) {
    using pack_t = optinum::simd::pack<T, W>;
    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; i += W) {
            auto pa = pack_t::load(a + i);
            auto pb = pack_t::load(b + i);
            auto pc = pack_t::load(c + i);
            auto pd = pa * pb + pc; // c + a * b
            pd.store(d + i);
        }
    }

    return timer.elapsed_ms();
}

template <typename T, std::size_t W> double benchmark_pack_sqrt(const T *a, T *b, std::size_t n) {
    using pack_t = optinum::simd::pack<T, W>;
    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; i += W) {
            auto pa = pack_t::load(a + i);
            auto pb = pa.sqrt();
            pb.store(b + i);
        }
    }

    return timer.elapsed_ms();
}

template <typename T, std::size_t W> double benchmark_pack_hsum(const T *a, std::size_t n) {
    using pack_t = optinum::simd::pack<T, W>;
    Timer timer;
    timer.start();

    T sum = 0;
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; i += W) {
            auto pa = pack_t::load(a + i);
            sum += pa.hsum();
        }
    }

    // Prevent optimization
    volatile T prevent_opt = sum;
    (void)prevent_opt;

    return timer.elapsed_ms();
}

template <typename T, std::size_t W> double benchmark_pack_dot(const T *a, const T *b, std::size_t n) {
    using pack_t = optinum::simd::pack<T, W>;
    Timer timer;
    timer.start();

    T dot = 0;
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; i += W) {
            auto pa = pack_t::load(a + i);
            auto pb = pack_t::load(b + i);
            dot += pa.dot(pb);
        }
    }

    // Prevent optimization
    volatile T prevent_opt = dot;
    (void)prevent_opt;

    return timer.elapsed_ms();
}

template <typename T, std::size_t W>
double benchmark_pack_minmax(const T *a, const T *b, T *min_out, T *max_out, std::size_t n) {
    using pack_t = optinum::simd::pack<T, W>;
    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; i += W) {
            auto pa = pack_t::load(a + i);
            auto pb = pack_t::load(b + i);
            // Use element-wise comparison - works for all pack types
            pack_t pmin, pmax;
            for (std::size_t j = 0; j < W; ++j) {
                T aval = a[i + j];
                T bval = b[i + j];
                min_out[i + j] = (aval < bval) ? aval : bval;
                max_out[i + j] = (aval > bval) ? aval : bval;
            }
        }
    }

    return timer.elapsed_ms();
}

// Scalar baseline
template <typename T> double benchmark_scalar_add(const T *a, const T *b, T *c, std::size_t n) {
    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }

    return timer.elapsed_ms();
}

template <typename T> double benchmark_scalar_mul(const T *a, const T *b, T *c, std::size_t n) {
    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            c[i] = a[i] * b[i];
        }
    }

    return timer.elapsed_ms();
}

template <typename T> double benchmark_scalar_fma(const T *a, const T *b, const T *c, T *d, std::size_t n) {
    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            d[i] = c[i] + a[i] * b[i];
        }
    }

    return timer.elapsed_ms();
}

template <typename T> double benchmark_scalar_sqrt(const T *a, T *b, std::size_t n) {
    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            b[i] = std::sqrt(a[i]);
        }
    }

    return timer.elapsed_ms();
}

// =============================================================================
// Benchmark Runners
// =============================================================================

template <typename T, std::size_t W> void run_benchmarks(const std::string &type_name) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Benchmarking " << get_pack_name<T, W>() << " [" << get_simd_backend() << "]\n";
    std::cout << "Elements: " << NUM_ELEMENTS << ", Iterations: " << NUM_ITERATIONS << "\n";
    std::cout << std::string(80, '=') << "\n\n";

    // Allocate aligned data
    alignas(64) std::vector<T> a_data, b_data, c_data, d_data, min_data, max_data;

    if constexpr (std::is_floating_point_v<T>) {
        a_data = generate_random_data<T>(NUM_ELEMENTS, T(1.0), T(10.0));
        b_data = generate_random_data<T>(NUM_ELEMENTS, T(1.0), T(10.0));
        c_data = generate_random_data<T>(NUM_ELEMENTS, T(1.0), T(10.0));
    } else {
        a_data = generate_random_data<T>(NUM_ELEMENTS, T(1), T(100));
        b_data = generate_random_data<T>(NUM_ELEMENTS, T(1), T(100));
        c_data = generate_random_data<T>(NUM_ELEMENTS, T(1), T(100));
    }

    d_data.resize(NUM_ELEMENTS);
    min_data.resize(NUM_ELEMENTS);
    max_data.resize(NUM_ELEMENTS);

    T *a = a_data.data();
    T *b = b_data.data();
    T *c = c_data.data();
    T *d = d_data.data();
    T *min_out = min_data.data();
    T *max_out = max_data.data();

    std::cout << std::setw(20) << "Operation" << std::setw(15) << "SIMD (ms)" << std::setw(15) << "Scalar (ms)"
              << std::setw(15) << "Speedup"
              << "\n";
    std::cout << std::string(65, '-') << "\n";

    // Addition
    {
        double simd_time = benchmark_pack_add<T, W>(a, b, c, NUM_ELEMENTS);
        double scalar_time = benchmark_scalar_add<T>(a, b, c, NUM_ELEMENTS);
        std::cout << std::setw(20) << "Addition" << std::setw(15) << std::fixed << std::setprecision(3) << simd_time
                  << std::setw(15) << scalar_time << std::setw(15) << std::setprecision(2) << (scalar_time / simd_time)
                  << "x\n";
    }

    // Multiplication
    {
        double simd_time = benchmark_pack_mul<T, W>(a, b, c, NUM_ELEMENTS);
        double scalar_time = benchmark_scalar_mul<T>(a, b, c, NUM_ELEMENTS);
        std::cout << std::setw(20) << "Multiplication" << std::setw(15) << std::fixed << std::setprecision(3)
                  << simd_time << std::setw(15) << scalar_time << std::setw(15) << std::setprecision(2)
                  << (scalar_time / simd_time) << "x\n";
    }

    // FMA (floating-point only)
    if constexpr (std::is_floating_point_v<T>) {
        double simd_time = benchmark_pack_fma<T, W>(a, b, c, d, NUM_ELEMENTS);
        double scalar_time = benchmark_scalar_fma<T>(a, b, c, d, NUM_ELEMENTS);
        std::cout << std::setw(20) << "FMA (a*b+c)" << std::setw(15) << std::fixed << std::setprecision(3) << simd_time
                  << std::setw(15) << scalar_time << std::setw(15) << std::setprecision(2) << (scalar_time / simd_time)
                  << "x\n";
    }

    // Square root (floating-point only)
    if constexpr (std::is_floating_point_v<T>) {
        double simd_time = benchmark_pack_sqrt<T, W>(a, c, NUM_ELEMENTS);
        double scalar_time = benchmark_scalar_sqrt<T>(a, c, NUM_ELEMENTS);
        std::cout << std::setw(20) << "Square Root" << std::setw(15) << std::fixed << std::setprecision(3) << simd_time
                  << std::setw(15) << scalar_time << std::setw(15) << std::setprecision(2) << (scalar_time / simd_time)
                  << "x\n";
    }

    // Horizontal sum
    {
        double simd_time = benchmark_pack_hsum<T, W>(a, NUM_ELEMENTS);
        std::cout << std::setw(20) << "Horizontal Sum" << std::setw(15) << std::fixed << std::setprecision(3)
                  << simd_time << std::setw(15) << "-" << std::setw(15) << "-\n";
    }

    // Dot product
    {
        double simd_time = benchmark_pack_dot<T, W>(a, b, NUM_ELEMENTS);
        std::cout << std::setw(20) << "Dot Product" << std::setw(15) << std::fixed << std::setprecision(3) << simd_time
                  << std::setw(15) << "-" << std::setw(15) << "-\n";
    }

    // Min/Max
    {
        double simd_time = benchmark_pack_minmax<T, W>(a, b, min_out, max_out, NUM_ELEMENTS);
        std::cout << std::setw(20) << "Min/Max" << std::setw(15) << std::fixed << std::setprecision(3) << simd_time
                  << std::setw(15) << "-" << std::setw(15) << "-\n";
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           OPTINUM SIMD Cross-Platform Pack Benchmark                      ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════╝\n";

    std::cout << "\nDetected SIMD Backend: " << get_simd_backend() << "\n";
    std::cout << "SIMD Width (bytes): " << optinum::simd::arch::simd_width_bytes() << "\n";

#ifdef OPTINUM_HAS_AVX512F
    std::cout << "SIMD Capabilities: AVX-512F";
#ifdef OPTINUM_HAS_AVX512VL
    std::cout << ", AVX-512VL";
#endif
#ifdef OPTINUM_HAS_AVX512BW
    std::cout << ", AVX-512BW";
#endif
#ifdef OPTINUM_HAS_AVX512DQ
    std::cout << ", AVX-512DQ";
#endif
    std::cout << "\n";

    run_benchmarks<float, 16>("float");
    run_benchmarks<double, 8>("double");
    run_benchmarks<int32_t, 16>("int32_t");
    run_benchmarks<int64_t, 8>("int64_t");

#elif defined(OPTINUM_HAS_AVX2)
    std::cout << "SIMD Capabilities: AVX2";
#ifdef OPTINUM_HAS_FMA
    std::cout << ", FMA";
#endif
    std::cout << "\n";

    run_benchmarks<float, 8>("float");
    run_benchmarks<double, 4>("double");
    run_benchmarks<int32_t, 8>("int32_t");
    run_benchmarks<int64_t, 4>("int64_t");

#elif defined(OPTINUM_HAS_SSE2)
    std::cout << "SIMD Capabilities: SSE2\n";

    run_benchmarks<float, 4>("float");
    run_benchmarks<double, 2>("double");
    run_benchmarks<int32_t, 4>("int32_t");
    run_benchmarks<int64_t, 2>("int64_t");

#elif defined(OPTINUM_HAS_NEON)
    std::cout << "SIMD Capabilities: ARM NEON";
#ifdef __aarch64__
    std::cout << " (ARM64)";
#else
    std::cout << " (ARM32)";
#endif
    std::cout << "\n";

    run_benchmarks<float, 4>("float");
#ifdef __aarch64__
    run_benchmarks<double, 2>("double");
#endif
    run_benchmarks<int32_t, 4>("int32_t");
#ifdef __aarch64__
    run_benchmarks<int64_t, 2>("int64_t");
#endif

#else
    std::cout << "SIMD Capabilities: None (Scalar fallback)\n";

    run_benchmarks<float, 1>("float");
    run_benchmarks<double, 1>("double");
    run_benchmarks<int32_t, 1>("int32_t");
    run_benchmarks<int64_t, 1>("int64_t");
#endif

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                          Benchmark Complete                                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    return 0;
}
