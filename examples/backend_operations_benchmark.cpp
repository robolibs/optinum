// Backend Operations Benchmark
// Benchmarks SIMD backend: elementwise, reduce, dot, matmul, transpose, norm

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <optinum/simd/backend/backend.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/backend/matmul.hpp>
#include <optinum/simd/backend/norm.hpp>
#include <optinum/simd/backend/reduce.hpp>
#include <optinum/simd/backend/transpose.hpp>

constexpr std::size_t NUM_ELEMENTS = 1024 * 1024;
constexpr std::size_t NUM_ITERATIONS = 100;

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
// Elementwise Operations
// ============================================================================

template <typename T, std::size_t N> double benchmark_simd_fill(T *dst, T value) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        optinum::simd::backend::fill<T, N>(dst, value);
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_scalar_fill(T *dst, T value) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < N; ++i) {
            dst[i] = value;
        }
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_simd_iota(T *dst, T start, T step) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        optinum::simd::backend::iota<T, N>(dst, start, step);
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_scalar_iota(T *dst, T start, T step) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < N; ++i) {
            dst[i] = start + static_cast<T>(i) * step;
        }
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_simd_reverse(T *data) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        optinum::simd::backend::reverse<T, N>(data);
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_scalar_reverse(T *data) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < N / 2; ++i) {
            std::swap(data[i], data[N - 1 - i]);
        }
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_simd_add(T *dst, const T *lhs, const T *rhs) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        optinum::simd::backend::add<T, N>(dst, lhs, rhs);
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_scalar_add(T *dst, const T *lhs, const T *rhs) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < N; ++i) {
            dst[i] = lhs[i] + rhs[i];
        }
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_simd_sub(T *dst, const T *lhs, const T *rhs) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        optinum::simd::backend::sub<T, N>(dst, lhs, rhs);
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_scalar_sub(T *dst, const T *lhs, const T *rhs) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < N; ++i) {
            dst[i] = lhs[i] - rhs[i];
        }
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_simd_mul(T *dst, const T *lhs, const T *rhs) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        optinum::simd::backend::mul<T, N>(dst, lhs, rhs);
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_scalar_mul(T *dst, const T *lhs, const T *rhs) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < N; ++i) {
            dst[i] = lhs[i] * rhs[i];
        }
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_simd_div(T *dst, const T *lhs, const T *rhs) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        optinum::simd::backend::div<T, N>(dst, lhs, rhs);
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_scalar_div(T *dst, const T *lhs, const T *rhs) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < N; ++i) {
            dst[i] = lhs[i] / rhs[i];
        }
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_simd_mul_scalar(T *dst, const T *src, T scalar) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        optinum::simd::backend::mul_scalar<T, N>(dst, src, scalar);
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_scalar_mul_scalar(T *dst, const T *src, T scalar) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < N; ++i) {
            dst[i] = src[i] * scalar;
        }
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Reduction Operations
// ============================================================================

template <typename T, std::size_t N> double benchmark_simd_reduce_sum(const T *data, T &result) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        result = optinum::simd::backend::reduce_sum<T, N>(data);
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_scalar_reduce_sum(const T *data, T &result) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        T sum{};
        for (std::size_t i = 0; i < N; ++i) {
            sum += data[i];
        }
        result = sum;
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_simd_reduce_min(const T *data, T &result) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        result = optinum::simd::backend::reduce_min<T, N>(data);
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_scalar_reduce_min(const T *data, T &result) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        T min_val = data[0];
        for (std::size_t i = 1; i < N; ++i) {
            if (data[i] < min_val)
                min_val = data[i];
        }
        result = min_val;
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_simd_reduce_max(const T *data, T &result) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        result = optinum::simd::backend::reduce_max<T, N>(data);
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_scalar_reduce_max(const T *data, T &result) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        T max_val = data[0];
        for (std::size_t i = 1; i < N; ++i) {
            if (data[i] > max_val)
                max_val = data[i];
        }
        result = max_val;
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_simd_dot(const T *lhs, const T *rhs, T &result) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        result = optinum::simd::backend::dot<T, N>(lhs, rhs);
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_scalar_dot(const T *lhs, const T *rhs, T &result) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        T sum{};
        for (std::size_t i = 0; i < N; ++i) {
            sum += lhs[i] * rhs[i];
        }
        result = sum;
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Matrix Operations
// ============================================================================

template <typename T, std::size_t M, std::size_t N> double benchmark_simd_transpose(T *dst, const T *src) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        optinum::simd::backend::transpose<T, M, N>(dst, src);
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t M, std::size_t N> double benchmark_scalar_transpose(T *dst, const T *src) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                dst[i + j * M] = src[j + i * N];
            }
        }
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t M, std::size_t K, std::size_t N>
double benchmark_simd_matmul(T *dst, const T *lhs, const T *rhs) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        optinum::simd::backend::matmul<T, M, K, N>(dst, lhs, rhs);
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t M, std::size_t K, std::size_t N>
double benchmark_scalar_matmul(T *dst, const T *lhs, const T *rhs) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // Column-major: dst(i,j) = sum_k lhs(i,k) * rhs(k,j)
        for (std::size_t j = 0; j < N; ++j) {
            for (std::size_t i = 0; i < M; ++i) {
                T sum{};
                for (std::size_t k = 0; k < K; ++k) {
                    sum += lhs[i + k * M] * rhs[k + j * K];
                }
                dst[i + j * M] = sum;
            }
        }
    }
    return timer.elapsed_ms();
}
template <typename T, std::size_t N> double benchmark_simd_norm(const T *data, T &result) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        result = optinum::simd::backend::norm_l2<T, N>(data);
    }
    return timer.elapsed_ms();
}

template <typename T, std::size_t N> double benchmark_scalar_norm(const T *data, T &result) {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        T sum{};
        for (std::size_t i = 0; i < N; ++i) {
            sum += data[i] * data[i];
        }
        result = std::sqrt(sum);
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

void print_result(const char *name, double simd_time, double scalar_time) {
    double speedup = scalar_time / simd_time;
    std::cout << std::setw(25) << std::left << name << std::setw(12) << std::right << std::fixed << std::setprecision(2)
              << simd_time << " ms" << std::setw(12) << scalar_time << " ms" << std::setw(10) << speedup << "x\n";
}

int main() {
    std::cout << "\n=== Backend Operations Benchmark ===\n";
    std::cout << "Elements: " << NUM_ELEMENTS << ", Iterations: " << NUM_ITERATIONS << "\n\n";

    // Allocate buffers
    std::vector<float> data1 = generate_random_data<float>(NUM_ELEMENTS, -10.0f, 10.0f);
    std::vector<float> data2 = generate_random_data<float>(NUM_ELEMENTS, -10.0f, 10.0f);
    std::vector<float> result(NUM_ELEMENTS);
    float scalar_result = 0.0f;

    // Elementwise Operations
    std::cout << "--- Elementwise Operations (float) ---\n";
    std::cout << std::setw(25) << std::left << "Operation" << std::setw(12) << std::right << "SIMD Time"
              << std::setw(12) << "Scalar Time" << std::setw(10) << "Speedup\n";
    std::cout << std::string(59, '-') << "\n";

    print_result("fill", benchmark_simd_fill<float, NUM_ELEMENTS>(result.data(), 3.14f),
                 benchmark_scalar_fill<float, NUM_ELEMENTS>(result.data(), 3.14f));

    print_result("iota", benchmark_simd_iota<float, NUM_ELEMENTS>(result.data(), 0.0f, 1.0f),
                 benchmark_scalar_iota<float, NUM_ELEMENTS>(result.data(), 0.0f, 1.0f));

    print_result("reverse", benchmark_simd_reverse<float, NUM_ELEMENTS>(data1.data()),
                 benchmark_scalar_reverse<float, NUM_ELEMENTS>(data1.data()));

    print_result("add", benchmark_simd_add<float, NUM_ELEMENTS>(result.data(), data1.data(), data2.data()),
                 benchmark_scalar_add<float, NUM_ELEMENTS>(result.data(), data1.data(), data2.data()));

    print_result("sub", benchmark_simd_sub<float, NUM_ELEMENTS>(result.data(), data1.data(), data2.data()),
                 benchmark_scalar_sub<float, NUM_ELEMENTS>(result.data(), data1.data(), data2.data()));

    print_result("mul", benchmark_simd_mul<float, NUM_ELEMENTS>(result.data(), data1.data(), data2.data()),
                 benchmark_scalar_mul<float, NUM_ELEMENTS>(result.data(), data1.data(), data2.data()));

    print_result("div", benchmark_simd_div<float, NUM_ELEMENTS>(result.data(), data1.data(), data2.data()),
                 benchmark_scalar_div<float, NUM_ELEMENTS>(result.data(), data1.data(), data2.data()));

    print_result("mul_scalar", benchmark_simd_mul_scalar<float, NUM_ELEMENTS>(result.data(), data1.data(), 2.5f),
                 benchmark_scalar_mul_scalar<float, NUM_ELEMENTS>(result.data(), data1.data(), 2.5f));

    // Reduction Operations
    std::cout << "\n--- Reduction Operations (float) ---\n";
    std::cout << std::setw(25) << std::left << "Operation" << std::setw(12) << std::right << "SIMD Time"
              << std::setw(12) << "Scalar Time" << std::setw(10) << "Speedup\n";
    std::cout << std::string(59, '-') << "\n";

    print_result("reduce_sum", benchmark_simd_reduce_sum<float, NUM_ELEMENTS>(data1.data(), scalar_result),
                 benchmark_scalar_reduce_sum<float, NUM_ELEMENTS>(data1.data(), scalar_result));

    print_result("reduce_min", benchmark_simd_reduce_min<float, NUM_ELEMENTS>(data1.data(), scalar_result),
                 benchmark_scalar_reduce_min<float, NUM_ELEMENTS>(data1.data(), scalar_result));

    print_result("reduce_max", benchmark_simd_reduce_max<float, NUM_ELEMENTS>(data1.data(), scalar_result),
                 benchmark_scalar_reduce_max<float, NUM_ELEMENTS>(data1.data(), scalar_result));

    print_result("dot", benchmark_simd_dot<float, NUM_ELEMENTS>(data1.data(), data2.data(), scalar_result),
                 benchmark_scalar_dot<float, NUM_ELEMENTS>(data1.data(), data2.data(), scalar_result));

    print_result("norm", benchmark_simd_norm<float, NUM_ELEMENTS>(data1.data(), scalar_result),
                 benchmark_scalar_norm<float, NUM_ELEMENTS>(data1.data(), scalar_result));

    // Matrix Operations (smaller sizes for matmul)
    std::cout << "\n--- Matrix Operations (float) ---\n";
    std::cout << std::setw(25) << std::left << "Operation" << std::setw(12) << std::right << "SIMD Time"
              << std::setw(12) << "Scalar Time" << std::setw(10) << "Speedup\n";
    std::cout << std::string(59, '-') << "\n";

    // 64x64 matrices
    constexpr std::size_t MAT_SIZE = 64;
    constexpr std::size_t MAT_ELEMENTS = MAT_SIZE * MAT_SIZE;
    std::vector<float> mat1 = generate_random_data<float>(MAT_ELEMENTS, -1.0f, 1.0f);
    std::vector<float> mat2 = generate_random_data<float>(MAT_ELEMENTS, -1.0f, 1.0f);
    std::vector<float> mat_result(MAT_ELEMENTS);

    print_result("transpose 64x64", benchmark_simd_transpose<float, MAT_SIZE, MAT_SIZE>(mat_result.data(), mat1.data()),
                 benchmark_scalar_transpose<float, MAT_SIZE, MAT_SIZE>(mat_result.data(), mat1.data()));

    print_result(
        "matmul 64x64x64",
        benchmark_simd_matmul<float, MAT_SIZE, MAT_SIZE, MAT_SIZE>(mat_result.data(), mat1.data(), mat2.data()),
        benchmark_scalar_matmul<float, MAT_SIZE, MAT_SIZE, MAT_SIZE>(mat_result.data(), mat1.data(), mat2.data()));

    std::cout << "\n=== Benchmark Complete ===\n\n";
    return 0;
}
