#pragma once

// =============================================================================
// optinum/lina/solve/solve_dynamic.hpp
// Linear system solvers for dynamic/runtime-sized matrices
// =============================================================================

#include <datapod/adapters/error.hpp>
#include <datapod/adapters/result.hpp>
#include <optinum/lina/decompose/lu_dynamic.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cstddef>

namespace optinum::lina {

    namespace dp = ::datapod;

    using DynVecD = simd::Vector<double, simd::Dynamic>;
    using DynVecF = simd::Vector<float, simd::Dynamic>;
    using DynMatD = simd::Matrix<double, simd::Dynamic, simd::Dynamic>;
    using DynMatF = simd::Matrix<float, simd::Dynamic, simd::Dynamic>;

    /**
     * @brief Solve linear system Ax = b for dynamic-size matrices (with error handling)
     *
     * Uses LU decomposition with partial pivoting.
     *
     * @tparam T Scalar type (float, double)
     * @param a Coefficient matrix (n x n)
     * @param b Right-hand side vector (n elements)
     * @return Result containing solution vector x, or error if singular
     */
    template <typename T>
    [[nodiscard]] inline dp::Result<dp::mat::vector<T, dp::mat::Dynamic>, dp::Error>
    try_solve_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a,
                      const simd::Vector<T, simd::Dynamic> &b) noexcept {
        const auto f = lu_dynamic(a);
        if (f.singular) {
            return dp::Result<dp::mat::vector<T, dp::mat::Dynamic>, dp::Error>::err(
                dp::Error::invalid_argument("matrix is singular"));
        }
        return dp::Result<dp::mat::vector<T, dp::mat::Dynamic>, dp::Error>::ok(lu_solve_dynamic(f, b));
    }

    /**
     * @brief Solve linear system Ax = b for dynamic-size matrices
     *
     * Returns zero vector if matrix is singular.
     *
     * @tparam T Scalar type
     * @param a Coefficient matrix (n x n)
     * @param b Right-hand side vector (n elements)
     * @return Solution vector x (or zeros if singular)
     */
    template <typename T>
    [[nodiscard]] inline dp::mat::vector<T, dp::mat::Dynamic>
    solve_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a,
                  const simd::Vector<T, simd::Dynamic> &b) noexcept {
        auto r = try_solve_dynamic(a, b);
        if (r.is_ok()) {
            return r.value();
        }
        dp::mat::vector<T, dp::mat::Dynamic> result(b.size());
        simd::Vector<T, simd::Dynamic> result_view(result);
        result_view.fill(T{});
        return result;
    }

    /**
     * @brief Solve multiple right-hand sides: AX = B for dynamic-size matrices
     *
     * @tparam T Scalar type
     * @param a Coefficient matrix (n x n)
     * @param b Right-hand side matrix (n x m)
     * @return Result containing solution matrix X (n x m), or error if singular
     */
    template <typename T>
    [[nodiscard]] inline dp::Result<dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>, dp::Error>
    try_solve_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a,
                      const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &b) noexcept {
        const std::size_t n = a.rows();
        const std::size_t m = b.cols();

        const auto f = lu_dynamic(a);
        if (f.singular) {
            return dp::Result<dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>, dp::Error>::err(
                dp::Error::invalid_argument("matrix is singular"));
        }

        dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> x_storage(n, m);
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> x(x_storage);

        // Solve for each column of B
        for (std::size_t col = 0; col < m; ++col) {
            // Extract column from B (owning storage + view)
            dp::mat::vector<T, dp::mat::Dynamic> rhs_storage(n);
            simd::Vector<T, simd::Dynamic> rhs(rhs_storage);
            for (std::size_t i = 0; i < n; ++i) {
                rhs[i] = b(i, col);
            }

            // Solve for this column
            auto sol = lu_solve_dynamic(f, rhs);

            // Store in X
            for (std::size_t i = 0; i < n; ++i) {
                x(i, col) = sol[i];
            }
        }

        return dp::Result<dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>, dp::Error>::ok(x_storage);
    }

    /**
     * @brief Solve multiple right-hand sides: AX = B for dynamic-size matrices
     *
     * Returns zero matrix if A is singular.
     */
    template <typename T>
    [[nodiscard]] inline dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>
    solve_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a,
                  const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &b) noexcept {
        auto r = try_solve_dynamic(a, b);
        if (r.is_ok()) {
            return r.value();
        }
        dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> result(a.rows(), b.cols());
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> result_view(result);
        result_view.fill(T{});
        return result;
    }

    /**
     * @brief Compute matrix inverse for dynamic-size matrix
     *
     * @tparam T Scalar type
     * @param a Input matrix (n x n)
     * @return Result containing inverse matrix, or error if singular
     */
    template <typename T>
    [[nodiscard]] inline dp::Result<dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>, dp::Error>
    try_inverse_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a) noexcept {
        const std::size_t n = a.rows();

        // Create identity matrix as RHS (owning storage + view)
        dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> identity_storage(n, n);
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> identity(identity_storage);
        identity.fill(T{});
        for (std::size_t i = 0; i < n; ++i) {
            identity(i, i) = T{1};
        }

        return try_solve_dynamic(a, identity);
    }

    /**
     * @brief Compute matrix inverse for dynamic-size matrix
     *
     * Returns zero matrix if singular.
     */
    template <typename T>
    [[nodiscard]] inline dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>
    inverse_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a) noexcept {
        auto r = try_inverse_dynamic(a);
        if (r.is_ok()) {
            return r.value();
        }
        dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> result(a.rows(), a.cols());
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> result_view(result);
        result_view.fill(T{});
        return result;
    }

    /**
     * @brief Compute determinant for dynamic-size matrix
     *
     * @tparam T Scalar type
     * @param a Input matrix (n x n)
     * @return Determinant value
     */
    template <typename T>
    [[nodiscard]] inline T determinant_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a) noexcept {
        const std::size_t n = a.rows();
        const auto f = lu_dynamic(a);
        if (f.singular) {
            return T{};
        }

        // det(A) = sign * product of U diagonal
        T det = static_cast<T>(f.sign);
        for (std::size_t i = 0; i < n; ++i) {
            det *= f.u(i, i);
        }
        return det;
    }

} // namespace optinum::lina
