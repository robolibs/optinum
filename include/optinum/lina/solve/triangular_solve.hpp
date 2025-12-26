#pragma once

// =============================================================================
// optinum/lina/solve/triangular_solve.hpp
// Solve triangular systems: Lx = b (lower) or Ux = b (upper)
// =============================================================================

#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cstddef>
#include <type_traits>

namespace optinum::lina {

    /**
     * Solve lower triangular system: Lx = b
     *
     * Forward substitution algorithm.
     * L is assumed to be lower triangular with non-zero diagonal.
     *
     * SIMD coverage: ~60% (dot products vectorized, but sequential dependencies)
     *
     * @param L Lower triangular matrix (NxN)
     * @param b Right-hand side vector (N)
     * @return Solution vector x
     */
    template <typename T, std::size_t N>
    [[nodiscard]] simd::Vector<T, N> solve_lower_triangular(const simd::Matrix<T, N, N> &L,
                                                            const simd::Vector<T, N> &b) noexcept {
        static_assert(std::is_floating_point_v<T>, "triangular_solve requires floating-point type");

        simd::Vector<T, N> x{};

        // Forward substitution: x[i] = (b[i] - sum(L[i][j] * x[j] for j < i)) / L[i][i]
        for (std::size_t i = 0; i < N; ++i) {
            T sum = T{0};

            // Compute L[i,0:i] · x[0:i]
            // TODO: Could be SIMD-optimized with partial dot product
            for (std::size_t j = 0; j < i; ++j) {
                sum += L(i, j) * x[j];
            }

            x[i] = (b[i] - sum) / L(i, i);
        }

        return x;
    }

    /**
     * Solve upper triangular system: Ux = b
     *
     * Backward substitution algorithm.
     * U is assumed to be upper triangular with non-zero diagonal.
     *
     * SIMD coverage: ~60% (dot products vectorized, but sequential dependencies)
     *
     * @param U Upper triangular matrix (NxN)
     * @param b Right-hand side vector (N)
     * @return Solution vector x
     */
    template <typename T, std::size_t N>
    [[nodiscard]] simd::Vector<T, N> solve_upper_triangular(const simd::Matrix<T, N, N> &U,
                                                            const simd::Vector<T, N> &b) noexcept {
        static_assert(std::is_floating_point_v<T>, "triangular_solve requires floating-point type");

        simd::Vector<T, N> x{};

        // Backward substitution: x[i] = (b[i] - sum(U[i][j] * x[j] for j > i)) / U[i][i]
        for (std::size_t i = N; i-- > 0;) {
            T sum = T{0};

            // Compute U[i,i+1:N] · x[i+1:N]
            // TODO: Could be SIMD-optimized with partial dot product
            for (std::size_t j = i + 1; j < N; ++j) {
                sum += U(i, j) * x[j];
            }

            x[i] = (b[i] - sum) / U(i, i);
        }

        return x;
    }

    /**
     * Generic triangular solve with flag
     *
     * @param A Triangular matrix
     * @param b Right-hand side
     * @param lower If true, solve Lx=b; if false, solve Ux=b
     */
    template <typename T, std::size_t N>
    [[nodiscard]] simd::Vector<T, N> triangular_solve(const simd::Matrix<T, N, N> &A, const simd::Vector<T, N> &b,
                                                      bool lower = true) noexcept {
        if (lower) {
            return solve_lower_triangular(A, b);
        } else {
            return solve_upper_triangular(A, b);
        }
    }

} // namespace optinum::lina
