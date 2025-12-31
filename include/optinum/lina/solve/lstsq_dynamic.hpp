#pragma once

// =============================================================================
// optinum/lina/solve/lstsq_dynamic.hpp
// Least squares solver for dynamic/runtime-sized matrices
// Uses SIMD for matrix-vector operations.
// =============================================================================

#include <datapod/adapters/error.hpp>
#include <datapod/adapters/result.hpp>
#include <optinum/lina/decompose/qr_dynamic.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/pack/pack.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>
#include <cstddef>
#include <limits>

namespace optinum::lina {

    namespace dp = ::datapod;

    namespace lstsq_dynamic_detail {

        // SIMD matrix-vector product for column-major matrix: y = A^T * x
        // A is m x m, x is m x 1, y is m x 1
        // y[i] = sum_j A[j,i] * x[j] = sum_j A^T[i,j] * x[j]
        // A[:,i] is contiguous at A.data() + i*m
        template <typename T>
        inline void mat_transpose_vec_simd(T *y, const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a, const T *x,
                                           std::size_t m) noexcept {
            for (std::size_t i = 0; i < m; ++i) {
                // y[i] = dot(A[:,i], x) = dot(A.data() + i*m, x, m)
                y[i] = simd::backend::dot_runtime<T>(a.data() + i * m, x, m);
            }
        }

    } // namespace lstsq_dynamic_detail

    /**
     * @brief Solve least squares problem min ||Ax - b||_2 for dynamic-size matrices
     *
     * Uses QR decomposition with SIMD-accelerated operations.
     * Requires m >= n (more equations than unknowns).
     *
     * @tparam T Scalar type (float, double)
     * @param a Coefficient matrix (m x n)
     * @param b Right-hand side vector (m elements)
     * @return Result containing solution vector x (n elements), or error
     */
    template <typename T>
    [[nodiscard]] inline dp::Result<dp::mat::vector<T, dp::mat::Dynamic>, dp::Error>
    try_lstsq_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a,
                      const simd::Vector<T, simd::Dynamic> &b) noexcept {
        static_assert(std::is_floating_point_v<T>, "lstsq_dynamic() requires floating-point type");

        const std::size_t m = a.rows();
        const std::size_t n = a.cols();

        if (m < n) {
            return dp::Result<dp::mat::vector<T, dp::mat::Dynamic>, dp::Error>::err(
                dp::Error::invalid_argument("lstsq requires m >= n (overdetermined system)"));
        }

        const auto f = qr_dynamic(a);

        // Compute y = Q^T * b using SIMD
        // Q is m x m column-major, so Q[:,i] is contiguous
        // y[i] = sum_j Q[j,i] * b[j] = dot(Q[:,i], b)
        dp::mat::vector<T, dp::mat::Dynamic> y_storage(m);
        simd::Vector<T, simd::Dynamic> y(y_storage);
        lstsq_dynamic_detail::mat_transpose_vec_simd(y.data(), f.q, b.data(), m);

        // Solve R(0:n-1, 0:n-1) * x = y(0:n-1) via back substitution
        dp::mat::vector<T, dp::mat::Dynamic> x_storage(n);
        simd::Vector<T, simd::Dynamic> x(x_storage);
        for (std::size_t ii = 0; ii < n; ++ii) {
            const std::size_t i = n - 1 - ii;
            T sum = y[i];
            for (std::size_t j = i + 1; j < n; ++j) {
                sum -= f.r(i, j) * x[j];
            }
            const T diag = f.r(i, i);
            if (std::abs(diag) < std::numeric_limits<T>::epsilon() * T{100}) {
                return dp::Result<dp::mat::vector<T, dp::mat::Dynamic>, dp::Error>::err(
                    dp::Error::invalid_argument("rank deficient R"));
            }
            x[i] = sum / diag;
        }

        return dp::Result<dp::mat::vector<T, dp::mat::Dynamic>, dp::Error>::ok(x_storage);
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
    [[nodiscard]] inline dp::mat::vector<T, dp::mat::Dynamic>
    lstsq_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a,
                  const simd::Vector<T, simd::Dynamic> &b) noexcept {
        auto r = try_lstsq_dynamic(a, b);
        if (r.is_ok()) {
            return r.value();
        }
        dp::mat::vector<T, dp::mat::Dynamic> result(a.cols());
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
    [[nodiscard]] inline dp::Result<dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>, dp::Error>
    try_lstsq_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a,
                      const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &b) noexcept {
        static_assert(std::is_floating_point_v<T>, "lstsq_dynamic() requires floating-point type");

        const std::size_t m = a.rows();
        const std::size_t n = a.cols();
        const std::size_t k = b.cols();

        if (m < n) {
            return dp::Result<dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>, dp::Error>::err(
                dp::Error::invalid_argument("lstsq requires m >= n (overdetermined system)"));
        }

        const auto f = qr_dynamic(a);

        dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> x_storage(n, k);
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> x(x_storage);

        // Solve for each column of B
        for (std::size_t col = 0; col < k; ++col) {
            // Extract column from B (column-major: B[:,col] is contiguous)
            const T *b_col = b.data() + col * m;

            // Compute y = Q^T * b_col using SIMD
            dp::mat::vector<T, dp::mat::Dynamic> y_storage(m);
            simd::Vector<T, simd::Dynamic> y(y_storage);
            lstsq_dynamic_detail::mat_transpose_vec_simd(y.data(), f.q, b_col, m);

            // Back substitution
            dp::mat::vector<T, dp::mat::Dynamic> x_col_storage(n);
            simd::Vector<T, simd::Dynamic> x_col(x_col_storage);
            for (std::size_t ii = 0; ii < n; ++ii) {
                const std::size_t i = n - 1 - ii;
                T sum = y[i];
                for (std::size_t j = i + 1; j < n; ++j) {
                    sum -= f.r(i, j) * x_col[j];
                }
                const T diag = f.r(i, i);
                if (std::abs(diag) < std::numeric_limits<T>::epsilon() * T{100}) {
                    return dp::Result<dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>, dp::Error>::err(
                        dp::Error::invalid_argument("rank deficient R"));
                }
                x_col[i] = sum / diag;
            }

            // Store in X (column-major: X[:,col] is contiguous)
            T *x_col_ptr = x.data() + col * n;
            for (std::size_t i = 0; i < n; ++i) {
                x_col_ptr[i] = x_col[i];
            }
        }

        return dp::Result<dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>, dp::Error>::ok(x_storage);
    }

    /**
     * @brief Solve multiple least squares problems
     *
     * Returns zero matrix on error.
     */
    template <typename T>
    [[nodiscard]] inline dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>
    lstsq_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a,
                  const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &b) noexcept {
        auto r = try_lstsq_dynamic(a, b);
        if (r.is_ok()) {
            return r.value();
        }
        dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> result(a.cols(), b.cols());
        result.fill(T{});
        return result;
    }

    /**
     * @brief Compute residual ||Ax - b||_2 for least squares solution
     *
     * Uses SIMD for matrix-vector product and norm computation.
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

        // Compute r = Ax - b
        dp::mat::vector<T, dp::mat::Dynamic> r_storage(m);
        simd::Vector<T, simd::Dynamic> r(r_storage);

        // Ax: for column-major A, A[:,j] is contiguous
        // (Ax)[i] = sum_j A[i,j] * x[j]
        // = sum_j (column j of A)[i] * x[j]
        r.fill(T{});
        constexpr std::size_t W = simd::backend::default_pack_width<T>();
        for (std::size_t j = 0; j < n; ++j) {
            const T xj = x[j];
            const T *a_col = a.data() + j * m;

            // SIMD: r += xj * A[:,j]
            const std::size_t main = (m / W) * W;
            const simd::pack<T, W> vxj(xj);

            for (std::size_t i = 0; i < main; i += W) {
                auto vr = simd::pack<T, W>::loadu(r.data() + i);
                auto va = simd::pack<T, W>::loadu(a_col + i);
                (simd::pack<T, W>::fma(vxj, va, vr)).storeu(r.data() + i);
            }
            for (std::size_t i = main; i < m; ++i) {
                r[i] += xj * a_col[i];
            }
        }

        // r = r - b using SIMD
        simd::backend::sub_runtime<T>(r.data(), r.data(), b.data(), m);

        // ||r||_2 using SIMD dot
        T norm_sq = simd::backend::dot_runtime<T>(r.data(), r.data(), m);
        return std::sqrt(norm_sq);
    }

    // =========================================================================
    // Overloads accepting owning types (dp::mat::matrix/vector)
    // =========================================================================

    /**
     * @brief Solve least squares problem for owning matrix/vector types
     */
    template <typename T>
    [[nodiscard]] inline dp::Result<dp::mat::vector<T, dp::mat::Dynamic>, dp::Error>
    try_lstsq_dynamic(const dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> &a,
                      const dp::mat::vector<T, dp::mat::Dynamic> &b) noexcept {
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> a_view(a);
        simd::Vector<T, simd::Dynamic> b_view(b);
        auto result = try_lstsq_dynamic(a_view, b_view);
        if (result.is_err()) {
            return dp::Result<dp::mat::vector<T, dp::mat::Dynamic>, dp::Error>::err(result.error());
        }
        // Copy view result to owning type
        dp::mat::vector<T, dp::mat::Dynamic> x(result.value().size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] = result.value()[i];
        }
        return dp::Result<dp::mat::vector<T, dp::mat::Dynamic>, dp::Error>::ok(x);
    }

    /**
     * @brief Solve least squares problem for owning types (returns zero on error)
     */
    template <typename T>
    [[nodiscard]] inline dp::mat::vector<T, dp::mat::Dynamic>
    lstsq_dynamic(const dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> &a,
                  const dp::mat::vector<T, dp::mat::Dynamic> &b) noexcept {
        auto r = try_lstsq_dynamic(a, b);
        if (r.is_ok()) {
            return r.value();
        }
        dp::mat::vector<T, dp::mat::Dynamic> result(a.cols());
        result.fill(T{});
        return result;
    }

    /**
     * @brief Solve multiple least squares problems for owning types
     */
    template <typename T>
    [[nodiscard]] inline dp::Result<dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>, dp::Error>
    try_lstsq_dynamic(const dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> &a,
                      const dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> &b) noexcept {
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> a_view(a);
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> b_view(b);
        auto result = try_lstsq_dynamic(a_view, b_view);
        if (result.is_err()) {
            return dp::Result<dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>, dp::Error>::err(result.error());
        }
        // Copy view result to owning type
        dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> x(result.value().rows(), result.value().cols());
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] = result.value()[i];
        }
        return dp::Result<dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>, dp::Error>::ok(x);
    }

    /**
     * @brief Solve multiple least squares problems for owning types (returns zero on error)
     */
    template <typename T>
    [[nodiscard]] inline dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>
    lstsq_dynamic(const dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> &a,
                  const dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> &b) noexcept {
        auto r = try_lstsq_dynamic(a, b);
        if (r.is_ok()) {
            return r.value();
        }
        dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> result(a.cols(), b.cols());
        result.fill(T{});
        return result;
    }

    /**
     * @brief Compute residual for owning types
     */
    template <typename T>
    [[nodiscard]] inline T lstsq_residual_dynamic(const dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> &a,
                                                  const dp::mat::vector<T, dp::mat::Dynamic> &x,
                                                  const dp::mat::vector<T, dp::mat::Dynamic> &b) noexcept {
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> a_view(a);
        simd::Vector<T, simd::Dynamic> x_view(x);
        simd::Vector<T, simd::Dynamic> b_view(b);
        return lstsq_residual_dynamic(a_view, x_view, b_view);
    }

} // namespace optinum::lina
