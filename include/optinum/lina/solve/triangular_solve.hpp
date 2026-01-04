#pragma once

// =============================================================================
// optinum/lina/solve/triangular_solve.hpp
// Solve triangular systems: Lx = b (lower) or Ux = b (upper)
// =============================================================================

#include <datapod/datapod.hpp>
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
     * SIMD optimization: For large i, extract row L[i,0:i] to contiguous buffer
     * and use SIMD dot product. For small i, use scalar loop.
     *
     * @param L Lower triangular matrix (NxN)
     * @param b Right-hand side vector (N)
     * @return Solution vector x
     */
    template <typename T, std::size_t N>
    [[nodiscard]] datapod::mat::Vector<T, N> solve_lower_triangular(const simd::Matrix<T, N, N> &L,
                                                                    const simd::Vector<T, N> &b) noexcept {
        static_assert(std::is_floating_point_v<T>, "triangular_solve requires floating-point type");

        datapod::mat::Vector<T, N> x{};

        // Threshold for using SIMD (need enough elements for vectorization benefit)
        constexpr std::size_t simd_threshold = 8;

        // Forward substitution: x[i] = (b[i] - sum(L[i][j] * x[j] for j < i)) / L[i][i]
        for (std::size_t i = 0; i < N; ++i) {
            T sum = T{0};

            if (i >= simd_threshold) {
                // Extract row L[i, 0:i] to contiguous buffer for SIMD dot product
                // Column-major: L(i,j) = L.data()[j*N + i]
                alignas(32) T L_row[N];
                for (std::size_t j = 0; j < i; ++j) {
                    L_row[j] = L(i, j);
                }
                sum = simd::backend::dot_runtime<T>(L_row, x.data(), i);
            } else {
                // Scalar loop for small i
                for (std::size_t j = 0; j < i; ++j) {
                    sum += L(i, j) * x[j];
                }
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
     * SIMD optimization: For large remaining elements, extract row segment
     * U[i,i+1:N] to contiguous buffer and use SIMD dot product.
     *
     * @param U Upper triangular matrix (NxN)
     * @param b Right-hand side vector (N)
     * @return Solution vector x
     */
    template <typename T, std::size_t N>
    [[nodiscard]] datapod::mat::Vector<T, N> solve_upper_triangular(const simd::Matrix<T, N, N> &U,
                                                                    const simd::Vector<T, N> &b) noexcept {
        static_assert(std::is_floating_point_v<T>, "triangular_solve requires floating-point type");

        datapod::mat::Vector<T, N> x{};

        // Threshold for using SIMD
        constexpr std::size_t simd_threshold = 8;

        // Backward substitution: x[i] = (b[i] - sum(U[i][j] * x[j] for j > i)) / U[i][i]
        for (std::size_t i = N; i-- > 0;) {
            T sum = T{0};
            const std::size_t remaining = N - i - 1;

            if (remaining >= simd_threshold) {
                // Extract row segment U[i, i+1:N] to contiguous buffer
                // Column-major: U(i,j) = U.data()[j*N + i]
                alignas(32) T U_row[N];
                for (std::size_t j = i + 1; j < N; ++j) {
                    U_row[j - i - 1] = U(i, j);
                }
                sum = simd::backend::dot_runtime<T>(U_row, x.data() + i + 1, remaining);
            } else {
                // Scalar loop for small remaining
                for (std::size_t j = i + 1; j < N; ++j) {
                    sum += U(i, j) * x[j];
                }
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
    [[nodiscard]] datapod::mat::Vector<T, N> triangular_solve(const simd::Matrix<T, N, N> &A,
                                                              const simd::Vector<T, N> &b, bool lower = true) noexcept {
        if (lower) {
            return solve_lower_triangular(A, b);
        } else {
            return solve_upper_triangular(A, b);
        }
    }

} // namespace optinum::lina
