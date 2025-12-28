#pragma once

// =============================================================================
// optinum/lina/solve/lstsq_dynamic.hpp
// Least squares solver for dynamic/runtime-sized matrices
// =============================================================================

#include <datapod/adapters/error.hpp>
#include <datapod/adapters/result.hpp>
#include <optinum/lina/decompose/qr_dynamic.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>
#include <cstddef>
#include <limits>

namespace optinum::lina {

    namespace dp = ::datapod;

    /**
     * @brief Solve least squares problem min ||Ax - b||_2 for dynamic-size matrices
     *
     * Uses QR decomposition to solve the overdetermined system.
     * Requires m >= n (more equations than unknowns).
     *
     * @tparam T Scalar type (float, double)
     * @param a Coefficient matrix (m x n)
     * @param b Right-hand side vector (m elements)
     * @return Result containing solution vector x (n elements), or error
     */
    template <typename T>
    [[nodiscard]] inline dp::Result<simd::Vector<T, simd::Dynamic>, dp::Error>
    try_lstsq_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a,
                      const simd::Vector<T, simd::Dynamic> &b) noexcept {
        static_assert(std::is_floating_point_v<T>, "lstsq_dynamic() requires floating-point type");

        const std::size_t m = a.rows();
        const std::size_t n = a.cols();

        if (m < n) {
            return dp::Result<simd::Vector<T, simd::Dynamic>, dp::Error>::err(
                dp::Error::invalid_argument("lstsq requires m >= n (overdetermined system)"));
        }

        const auto f = qr_dynamic(a);

        // Compute y = Q^T * b
        // y[i] = sum_j Q[j, i] * b[j] = sum_j Q^T[i, j] * b[j]
        simd::Vector<T, simd::Dynamic> y(m);
        for (std::size_t i = 0; i < m; ++i) {
            T sum{};
            for (std::size_t j = 0; j < m; ++j) {
                // Q^T[i,j] = Q[j,i]
                sum += f.q(j, i) * b[j];
            }
            y[i] = sum;
        }

        // Solve R(0:n-1, 0:n-1) * x = y(0:n-1) via back substitution
        simd::Vector<T, simd::Dynamic> x(n);
        for (std::size_t ii = 0; ii < n; ++ii) {
            const std::size_t i = n - 1 - ii;
            T sum = y[i];
            for (std::size_t j = i + 1; j < n; ++j) {
                sum -= f.r(i, j) * x[j];
            }
            const T diag = f.r(i, i);
            if (std::abs(diag) < std::numeric_limits<T>::epsilon() * T{100}) {
                return dp::Result<simd::Vector<T, simd::Dynamic>, dp::Error>::err(
                    dp::Error::invalid_argument("rank deficient R"));
            }
            x[i] = sum / diag;
        }

        return dp::Result<simd::Vector<T, simd::Dynamic>, dp::Error>::ok(x);
    }

    /**
     * @brief Solve least squares problem min ||Ax - b||_2 for dynamic-size matrices
     *
     * Returns zero vector on error.
     *
     * @tparam T Scalar type
     * @param a Coefficient matrix (m x n)
     * @param b Right-hand side vector (m elements)
     * @return Solution vector x (n elements), or zeros on error
     */
    template <typename T>
    [[nodiscard]] inline simd::Vector<T, simd::Dynamic>
    lstsq_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a,
                  const simd::Vector<T, simd::Dynamic> &b) noexcept {
        auto r = try_lstsq_dynamic(a, b);
        if (r.is_ok()) {
            return r.value();
        }
        simd::Vector<T, simd::Dynamic> result(a.cols());
        result.fill(T{});
        return result;
    }

    /**
     * @brief Solve multiple least squares problems: min ||AX - B||_F
     *
     * @tparam T Scalar type
     * @param a Coefficient matrix (m x n)
     * @param b Right-hand side matrix (m x k)
     * @return Result containing solution matrix X (n x k), or error
     */
    template <typename T>
    [[nodiscard]] inline dp::Result<simd::Matrix<T, simd::Dynamic, simd::Dynamic>, dp::Error>
    try_lstsq_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a,
                      const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &b) noexcept {
        static_assert(std::is_floating_point_v<T>, "lstsq_dynamic() requires floating-point type");

        const std::size_t m = a.rows();
        const std::size_t n = a.cols();
        const std::size_t k = b.cols();

        if (m < n) {
            return dp::Result<simd::Matrix<T, simd::Dynamic, simd::Dynamic>, dp::Error>::err(
                dp::Error::invalid_argument("lstsq requires m >= n (overdetermined system)"));
        }

        const auto f = qr_dynamic(a);

        simd::Matrix<T, simd::Dynamic, simd::Dynamic> x(n, k);

        // Solve for each column of B
        for (std::size_t col = 0; col < k; ++col) {
            // Extract column from B
            simd::Vector<T, simd::Dynamic> b_col(m);
            for (std::size_t i = 0; i < m; ++i) {
                b_col[i] = b(i, col);
            }

            // Compute y = Q^T * b_col
            simd::Vector<T, simd::Dynamic> y(m);
            for (std::size_t i = 0; i < m; ++i) {
                T sum{};
                for (std::size_t j = 0; j < m; ++j) {
                    sum += f.q(j, i) * b_col[j];
                }
                y[i] = sum;
            }

            // Back substitution
            simd::Vector<T, simd::Dynamic> x_col(n);
            for (std::size_t ii = 0; ii < n; ++ii) {
                const std::size_t i = n - 1 - ii;
                T sum = y[i];
                for (std::size_t j = i + 1; j < n; ++j) {
                    sum -= f.r(i, j) * x_col[j];
                }
                const T diag = f.r(i, i);
                if (std::abs(diag) < std::numeric_limits<T>::epsilon() * T{100}) {
                    return dp::Result<simd::Matrix<T, simd::Dynamic, simd::Dynamic>, dp::Error>::err(
                        dp::Error::invalid_argument("rank deficient R"));
                }
                x_col[i] = sum / diag;
            }

            // Store in X
            for (std::size_t i = 0; i < n; ++i) {
                x(i, col) = x_col[i];
            }
        }

        return dp::Result<simd::Matrix<T, simd::Dynamic, simd::Dynamic>, dp::Error>::ok(x);
    }

    /**
     * @brief Solve multiple least squares problems
     *
     * Returns zero matrix on error.
     */
    template <typename T>
    [[nodiscard]] inline simd::Matrix<T, simd::Dynamic, simd::Dynamic>
    lstsq_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a,
                  const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &b) noexcept {
        auto r = try_lstsq_dynamic(a, b);
        if (r.is_ok()) {
            return r.value();
        }
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> result(a.cols(), b.cols());
        result.fill(T{});
        return result;
    }

    /**
     * @brief Compute residual ||Ax - b||_2 for least squares solution
     *
     * @tparam T Scalar type
     * @param a Coefficient matrix (m x n)
     * @param x Solution vector (n elements)
     * @param b Right-hand side vector (m elements)
     * @return Residual norm
     */
    template <typename T>
    [[nodiscard]] inline T lstsq_residual_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a,
                                                  const simd::Vector<T, simd::Dynamic> &x,
                                                  const simd::Vector<T, simd::Dynamic> &b) noexcept {
        const std::size_t m = a.rows();
        const std::size_t n = a.cols();

        T sum_sq{};
        for (std::size_t i = 0; i < m; ++i) {
            T ax_i{};
            for (std::size_t j = 0; j < n; ++j) {
                ax_i += a(i, j) * x[j];
            }
            const T diff = ax_i - b[i];
            sum_sq += diff * diff;
        }
        return std::sqrt(sum_sq);
    }

} // namespace optinum::lina
