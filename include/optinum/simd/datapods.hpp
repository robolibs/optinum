#pragma once

// =============================================================================
// optinum/simd/datapods.hpp
// Factory functions and SIMD convenience operations for datapod types
//
// This file provides:
// 1. Factory functions: zeros<T,N>(), ones<T,N>(), identity<T,R,C>(), arange<T,N>()
// 2. SIMD convenience free functions: norm(), dot(), sum(), normalize(), etc.
// 3. I/O operators (operator<<) for dp::mat::vector and dp::mat::matrix
//
// All factory functions return dp::mat::* types (ownership in datapod).
// All SIMD operations work via simd::view() internally.
// =============================================================================

#include <datapod/datapod.hpp>
#include <iomanip>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/backend/norm.hpp>
#include <optinum/simd/backend/reduce.hpp>
#include <optinum/simd/bridge.hpp>
#include <ostream>
#include <random>
#include <sstream>
#include <type_traits>

namespace optinum::simd {

    // Import Dynamic constant from datapod
    using dp::mat::Dynamic;

    // =============================================================================
    // Vector Factory Functions
    // =============================================================================

    /// Create a zero-initialized vector
    /// @tparam T Element type (float, double, int, etc.)
    /// @tparam N Vector size (compile-time constant or Dynamic)
    /// @return dp::mat::Vector<T, N> filled with zeros
    template <typename T, std::size_t N> constexpr dp::mat::Vector<T, N> zeros() noexcept {
        static_assert(N != Dynamic, "Use zeros(size) for dynamic vectors");
        dp::mat::Vector<T, N> v{};
        for (std::size_t i = 0; i < N; ++i) {
            v[i] = T{0};
        }
        return v;
    }

    /// Create a zero-initialized dynamic vector
    /// @param size Runtime size
    /// @return dp::mat::Vector<T, Dynamic> filled with zeros
    template <typename T> dp::mat::Vector<T, Dynamic> zeros(std::size_t size) {
        dp::mat::Vector<T, Dynamic> v(size);
        for (std::size_t i = 0; i < size; ++i) {
            v[i] = T{0};
        }
        return v;
    }

    /// Create a vector filled with ones
    template <typename T, std::size_t N> constexpr dp::mat::Vector<T, N> ones() noexcept {
        static_assert(N != Dynamic, "Use ones(size) for dynamic vectors");
        dp::mat::Vector<T, N> v{};
        for (std::size_t i = 0; i < N; ++i) {
            v[i] = T{1};
        }
        return v;
    }

    /// Create a dynamic vector filled with ones
    template <typename T> dp::mat::Vector<T, Dynamic> ones(std::size_t size) {
        dp::mat::Vector<T, Dynamic> v(size);
        for (std::size_t i = 0; i < size; ++i) {
            v[i] = T{1};
        }
        return v;
    }

    /// Create a vector with sequential values [0, 1, 2, ...]
    template <typename T, std::size_t N> constexpr dp::mat::Vector<T, N> arange() noexcept {
        static_assert(N != Dynamic, "Use arange(size) for dynamic vectors");
        dp::mat::Vector<T, N> v{};
        for (std::size_t i = 0; i < N; ++i) {
            v[i] = static_cast<T>(i);
        }
        return v;
    }

    /// Create a vector with sequential values [start, start+1, start+2, ...]
    template <typename T, std::size_t N> constexpr dp::mat::Vector<T, N> arange(T start) noexcept {
        static_assert(N != Dynamic, "Use arange(size, start) for dynamic vectors");
        dp::mat::Vector<T, N> v{};
        for (std::size_t i = 0; i < N; ++i) {
            v[i] = start + static_cast<T>(i);
        }
        return v;
    }

    /// Create a vector with sequential values [start, start+step, start+2*step, ...]
    template <typename T, std::size_t N> constexpr dp::mat::Vector<T, N> arange(T start, T step) noexcept {
        static_assert(N != Dynamic, "Use arange(size, start, step) for dynamic vectors");
        dp::mat::Vector<T, N> v{};
        for (std::size_t i = 0; i < N; ++i) {
            v[i] = start + static_cast<T>(i) * step;
        }
        return v;
    }

    /// Create a dynamic vector with sequential values
    template <typename T> dp::mat::Vector<T, Dynamic> arange(std::size_t size) {
        dp::mat::Vector<T, Dynamic> v(size);
        for (std::size_t i = 0; i < size; ++i) {
            v[i] = static_cast<T>(i);
        }
        return v;
    }

    /// Create a dynamic vector with sequential values from start
    template <typename T> dp::mat::Vector<T, Dynamic> arange(std::size_t size, T start) {
        dp::mat::Vector<T, Dynamic> v(size);
        for (std::size_t i = 0; i < size; ++i) {
            v[i] = start + static_cast<T>(i);
        }
        return v;
    }

    /// Create a dynamic vector with sequential values with custom start and step
    template <typename T> dp::mat::Vector<T, Dynamic> arange(std::size_t size, T start, T step) {
        dp::mat::Vector<T, Dynamic> v(size);
        for (std::size_t i = 0; i < size; ++i) {
            v[i] = start + static_cast<T>(i) * step;
        }
        return v;
    }

    /// Create a vector filled with a specific value
    template <typename T, std::size_t N> constexpr dp::mat::Vector<T, N> filled(T value) noexcept {
        static_assert(N != Dynamic, "Use filled(size, value) for dynamic vectors");
        dp::mat::Vector<T, N> v{};
        for (std::size_t i = 0; i < N; ++i) {
            v[i] = value;
        }
        return v;
    }

    /// Create a dynamic vector filled with a specific value
    template <typename T> dp::mat::Vector<T, Dynamic> filled(std::size_t size, T value) {
        dp::mat::Vector<T, Dynamic> v(size);
        for (std::size_t i = 0; i < size; ++i) {
            v[i] = value;
        }
        return v;
    }

    /// Create a vector with random values in [0, 1) for floating point
    template <typename T, std::size_t N> dp::mat::Vector<T, N> random() {
        static_assert(N != Dynamic, "Use random(size) for dynamic vectors");
        static std::random_device rd;
        static std::mt19937 gen(rd());

        dp::mat::Vector<T, N> v{};
        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> dis(T{0}, T{1});
            for (std::size_t i = 0; i < N; ++i) {
                v[i] = dis(gen);
            }
        } else {
            std::uniform_int_distribution<T> dis(T{0}, std::numeric_limits<T>::max());
            for (std::size_t i = 0; i < N; ++i) {
                v[i] = dis(gen);
            }
        }
        return v;
    }

    /// Create a dynamic vector with random values
    template <typename T> dp::mat::Vector<T, Dynamic> random(std::size_t size) {
        static std::random_device rd;
        static std::mt19937 gen(rd());

        dp::mat::Vector<T, Dynamic> v(size);
        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> dis(T{0}, T{1});
            for (std::size_t i = 0; i < size; ++i) {
                v[i] = dis(gen);
            }
        } else {
            std::uniform_int_distribution<T> dis(T{0}, std::numeric_limits<T>::max());
            for (std::size_t i = 0; i < size; ++i) {
                v[i] = dis(gen);
            }
        }
        return v;
    }

    // =============================================================================
    // Matrix Factory Functions
    // =============================================================================

    /// Create a zero-initialized matrix
    template <typename T, std::size_t R, std::size_t C> constexpr dp::mat::Matrix<T, R, C> zeros_matrix() noexcept {
        static_assert(R != Dynamic && C != Dynamic, "Use zeros_matrix(rows, cols) for dynamic matrices");
        dp::mat::Matrix<T, R, C> m{};
        for (std::size_t i = 0; i < R * C; ++i) {
            m[i] = T{0};
        }
        return m;
    }

    /// Create a dynamic zero-initialized matrix
    template <typename T> dp::mat::Matrix<T, Dynamic, Dynamic> zeros_matrix(std::size_t rows, std::size_t cols) {
        dp::mat::Matrix<T, Dynamic, Dynamic> m(rows, cols);
        for (std::size_t i = 0; i < rows * cols; ++i) {
            m[i] = T{0};
        }
        return m;
    }

    /// Create a matrix filled with ones
    template <typename T, std::size_t R, std::size_t C> constexpr dp::mat::Matrix<T, R, C> ones_matrix() noexcept {
        static_assert(R != Dynamic && C != Dynamic, "Use ones_matrix(rows, cols) for dynamic matrices");
        dp::mat::Matrix<T, R, C> m{};
        for (std::size_t i = 0; i < R * C; ++i) {
            m[i] = T{1};
        }
        return m;
    }

    /// Create a dynamic matrix filled with ones
    template <typename T> dp::mat::Matrix<T, Dynamic, Dynamic> ones_matrix(std::size_t rows, std::size_t cols) {
        dp::mat::Matrix<T, Dynamic, Dynamic> m(rows, cols);
        for (std::size_t i = 0; i < rows * cols; ++i) {
            m[i] = T{1};
        }
        return m;
    }

    /// Create an identity matrix
    template <typename T, std::size_t N> constexpr dp::mat::Matrix<T, N, N> identity() noexcept {
        static_assert(N != Dynamic, "Use identity(size) for dynamic matrices");
        dp::mat::Matrix<T, N, N> m{};
        for (std::size_t i = 0; i < N * N; ++i) {
            m[i] = T{0};
        }
        for (std::size_t i = 0; i < N; ++i) {
            m(i, i) = T{1};
        }
        return m;
    }

    /// Create a dynamic identity matrix
    template <typename T> dp::mat::Matrix<T, Dynamic, Dynamic> identity(std::size_t size) {
        dp::mat::Matrix<T, Dynamic, Dynamic> m(size, size);
        for (std::size_t i = 0; i < size * size; ++i) {
            m[i] = T{0};
        }
        for (std::size_t i = 0; i < size; ++i) {
            m(i, i) = T{1};
        }
        return m;
    }

    /// Create a matrix filled with a specific value
    template <typename T, std::size_t R, std::size_t C>
    constexpr dp::mat::Matrix<T, R, C> filled_matrix(T value) noexcept {
        static_assert(R != Dynamic && C != Dynamic, "Use filled_matrix(rows, cols, value) for dynamic matrices");
        dp::mat::Matrix<T, R, C> m{};
        for (std::size_t i = 0; i < R * C; ++i) {
            m[i] = value;
        }
        return m;
    }

    /// Create a dynamic matrix filled with a specific value
    template <typename T>
    dp::mat::Matrix<T, Dynamic, Dynamic> filled_matrix(std::size_t rows, std::size_t cols, T value) {
        dp::mat::Matrix<T, Dynamic, Dynamic> m(rows, cols);
        for (std::size_t i = 0; i < rows * cols; ++i) {
            m[i] = value;
        }
        return m;
    }

    // =============================================================================
    // SIMD Convenience Free Functions for Vectors
    // These work on dp::mat::vector and use SIMD internally via views
    // =============================================================================

    /// Compute L2 norm of a vector (SIMD-accelerated)
    template <typename T, std::size_t N> T norm(const dp::mat::Vector<T, N> &v) noexcept {
        if constexpr (N == Dynamic) {
            return backend::norm_l2_runtime<T>(v.data(), v.size());
        } else {
            return backend::norm_l2<T, N>(v.data());
        }
    }

    /// Compute dot product of two vectors (SIMD-accelerated)
    template <typename T, std::size_t N>
    T dot(const dp::mat::Vector<T, N> &a, const dp::mat::Vector<T, N> &b) noexcept {
        if constexpr (N == Dynamic) {
            return backend::dot_runtime<T>(a.data(), b.data(), a.size());
        } else {
            return backend::dot<T, N>(a.data(), b.data());
        }
    }

    /// Compute sum of all elements (SIMD-accelerated)
    template <typename T, std::size_t N> T sum(const dp::mat::Vector<T, N> &v) noexcept {
        if constexpr (N == Dynamic) {
            return backend::reduce_sum_runtime<T>(v.data(), v.size());
        } else {
            return backend::reduce_sum<T, N>(v.data());
        }
    }

    /// Normalize a vector in-place (SIMD-accelerated)
    template <typename T, std::size_t N> void normalize(dp::mat::Vector<T, N> &v) noexcept {
        T n = norm(v);
        if (n > T{0}) {
            if constexpr (N == Dynamic) {
                backend::div_scalar_runtime<T>(v.data(), v.data(), n, v.size());
            } else {
                backend::div_scalar<T, N>(v.data(), v.data(), n);
            }
        }
    }

    /// Return a normalized copy of a vector (SIMD-accelerated)
    template <typename T, std::size_t N> dp::mat::Vector<T, N> normalized(const dp::mat::Vector<T, N> &v) noexcept {
        dp::mat::Vector<T, N> result;
        if constexpr (N == Dynamic) {
            result.resize(v.size());
        }
        backend::normalize<T, N>(result.data(), v.data());
        return result;
    }

    /// Fill a vector with a value (SIMD-accelerated)
    template <typename T, std::size_t N> void fill(dp::mat::Vector<T, N> &v, T value) noexcept {
        if constexpr (N == Dynamic) {
            backend::fill_runtime(v.data(), v.size(), value);
        } else {
            backend::fill<T, N>(v.data(), value);
        }
    }

    /// Add two vectors element-wise, store result in out (SIMD-accelerated)
    template <typename T, std::size_t N>
    void add(dp::mat::Vector<T, N> &out, const dp::mat::Vector<T, N> &a, const dp::mat::Vector<T, N> &b) noexcept {
        if constexpr (N == Dynamic) {
            backend::add_runtime<T>(out.data(), a.data(), b.data(), a.size());
        } else {
            backend::add<T, N>(out.data(), a.data(), b.data());
        }
    }

    /// Subtract two vectors element-wise, store result in out (SIMD-accelerated)
    template <typename T, std::size_t N>
    void sub(dp::mat::Vector<T, N> &out, const dp::mat::Vector<T, N> &a, const dp::mat::Vector<T, N> &b) noexcept {
        if constexpr (N == Dynamic) {
            backend::sub_runtime<T>(out.data(), a.data(), b.data(), a.size());
        } else {
            backend::sub<T, N>(out.data(), a.data(), b.data());
        }
    }

    /// Multiply two vectors element-wise, store result in out (SIMD-accelerated)
    template <typename T, std::size_t N>
    void mul(dp::mat::Vector<T, N> &out, const dp::mat::Vector<T, N> &a, const dp::mat::Vector<T, N> &b) noexcept {
        if constexpr (N == Dynamic) {
            backend::mul_runtime<T>(out.data(), a.data(), b.data(), a.size());
        } else {
            backend::mul<T, N>(out.data(), a.data(), b.data());
        }
    }

    /// Divide two vectors element-wise, store result in out (SIMD-accelerated)
    template <typename T, std::size_t N>
    void div(dp::mat::Vector<T, N> &out, const dp::mat::Vector<T, N> &a, const dp::mat::Vector<T, N> &b) noexcept {
        if constexpr (N == Dynamic) {
            backend::div_runtime<T>(out.data(), a.data(), b.data(), a.size());
        } else {
            backend::div<T, N>(out.data(), a.data(), b.data());
        }
    }

    /// Scale a vector by a scalar, store result in out (SIMD-accelerated)
    template <typename T, std::size_t N>
    void scale(dp::mat::Vector<T, N> &out, const dp::mat::Vector<T, N> &v, T scalar) noexcept {
        if constexpr (N == Dynamic) {
            backend::mul_scalar_runtime<T>(out.data(), v.data(), scalar, v.size());
        } else {
            backend::mul_scalar<T, N>(out.data(), v.data(), scalar);
        }
    }

    /// Scale a vector by a scalar in-place (SIMD-accelerated)
    template <typename T, std::size_t N> void scale(dp::mat::Vector<T, N> &v, T scalar) noexcept {
        if constexpr (N == Dynamic) {
            backend::mul_scalar_runtime<T>(v.data(), v.data(), scalar, v.size());
        } else {
            backend::mul_scalar<T, N>(v.data(), v.data(), scalar);
        }
    }

    // =============================================================================
    // Compound Assignment Operators for dp::mat::vector
    // These allow v += u, v -= u, v *= scalar, v /= scalar syntax
    // =============================================================================

    /// operator+= for dp::mat::vector
    template <typename T, std::size_t N>
    dp::mat::Vector<T, N> &operator+=(dp::mat::Vector<T, N> &a, const dp::mat::Vector<T, N> &b) noexcept {
        if constexpr (N == Dynamic) {
            backend::add_runtime<T>(a.data(), a.data(), b.data(), a.size());
        } else {
            backend::add<T, N>(a.data(), a.data(), b.data());
        }
        return a;
    }

    /// operator-= for dp::mat::vector
    template <typename T, std::size_t N>
    dp::mat::Vector<T, N> &operator-=(dp::mat::Vector<T, N> &a, const dp::mat::Vector<T, N> &b) noexcept {
        if constexpr (N == Dynamic) {
            backend::sub_runtime<T>(a.data(), a.data(), b.data(), a.size());
        } else {
            backend::sub<T, N>(a.data(), a.data(), b.data());
        }
        return a;
    }

    /// operator*= (scalar) for dp::mat::vector
    template <typename T, std::size_t N>
    dp::mat::Vector<T, N> &operator*=(dp::mat::Vector<T, N> &v, T scalar) noexcept {
        if constexpr (N == Dynamic) {
            backend::mul_scalar_runtime<T>(v.data(), v.data(), scalar, v.size());
        } else {
            backend::mul_scalar<T, N>(v.data(), v.data(), scalar);
        }
        return v;
    }

    /// operator/= (scalar) for dp::mat::vector
    template <typename T, std::size_t N>
    dp::mat::Vector<T, N> &operator/=(dp::mat::Vector<T, N> &v, T scalar) noexcept {
        if constexpr (N == Dynamic) {
            backend::div_scalar_runtime<T>(v.data(), v.data(), scalar, v.size());
        } else {
            backend::div_scalar<T, N>(v.data(), v.data(), scalar);
        }
        return v;
    }

    // =============================================================================
    // SIMD Convenience Free Functions for Matrices
    // =============================================================================

    /// Compute Frobenius norm of a matrix (SIMD-accelerated)
    template <typename T, std::size_t R, std::size_t C> T frobenius_norm(const dp::mat::Matrix<T, R, C> &m) noexcept {
        constexpr std::size_t total = R * C;
        if constexpr (R == Dynamic || C == Dynamic) {
            return backend::norm_l2_runtime<T>(m.data(), m.size());
        } else {
            return backend::norm_l2<T, total>(m.data());
        }
    }

    /// Compute trace of a square matrix
    template <typename T, std::size_t N> T trace(const dp::mat::Matrix<T, N, N> &m) noexcept {
        static_assert(N != Dynamic, "Use trace with fixed-size square matrices");
        T result{0};
        for (std::size_t i = 0; i < N; ++i) {
            result += m(i, i);
        }
        return result;
    }

    /// Fill a matrix with a value (SIMD-accelerated)
    template <typename T, std::size_t R, std::size_t C> void fill(dp::mat::Matrix<T, R, C> &m, T value) noexcept {
        constexpr std::size_t total = R * C;
        if constexpr (R == Dynamic || C == Dynamic) {
            backend::fill_runtime(m.data(), m.size(), value);
        } else {
            backend::fill<T, total>(m.data(), value);
        }
    }

    // =============================================================================
    // I/O Operators for dp::mat::vector and dp::mat::matrix
    // =============================================================================

    /// Stream output for dp::mat::vector
    template <typename T, std::size_t N> std::ostream &operator<<(std::ostream &os, const dp::mat::Vector<T, N> &v) {
        os << "[";
        const std::size_t size = (N == Dynamic) ? v.size() : N;
        for (std::size_t i = 0; i < size; ++i) {
            if (i > 0)
                os << ", ";
            os << v[i];
        }
        os << "]";
        return os;
    }

    /// Stream output for dp::mat::matrix
    template <typename T, std::size_t R, std::size_t C>
    std::ostream &operator<<(std::ostream &os, const dp::mat::Matrix<T, R, C> &m) {
        const std::size_t rows = (R == Dynamic) ? m.rows() : R;
        const std::size_t cols = (C == Dynamic) ? m.cols() : C;

        // Find max width for alignment
        std::size_t max_width = 1;
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                std::ostringstream oss;
                oss << m(i, j);
                max_width = std::max(max_width, oss.str().size());
            }
        }

        for (std::size_t i = 0; i < rows; ++i) {
            os << (i == 0 ? "[" : " ");
            os << "[";
            for (std::size_t j = 0; j < cols; ++j) {
                if (j > 0)
                    os << ", ";
                os << std::setw(static_cast<int>(max_width)) << m(i, j);
            }
            os << "]";
            if (i < rows - 1)
                os << "\n";
        }
        os << "]";
        return os;
    }

} // namespace optinum::simd
