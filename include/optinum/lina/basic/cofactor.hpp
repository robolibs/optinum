#pragma once

// =============================================================================
// optinum/lina/basic/cofactor.hpp
// Cofactor matrix computation
// =============================================================================
//
// The cofactor matrix C of a square matrix A is defined as:
//   C_ij = (-1)^(i+j) * M_ij
// where M_ij is the minor (determinant of the submatrix with row i and col j removed)
//
// For small matrices (2x2, 3x3, 4x4), we use direct formulas.
// For larger matrices, we use submatrix extraction and determinant computation.
//
// =============================================================================

#include <optinum/lina/basic/determinant.hpp>
#include <optinum/simd/matrix.hpp>

namespace optinum::lina {

    namespace detail {
        // Extract submatrix by removing row i and column j
        template <typename T, std::size_t N>
        [[nodiscard]] constexpr simd::Matrix<T, N - 1, N - 1> submatrix(const simd::Matrix<T, N, N> &a, std::size_t row,
                                                                        std::size_t col) noexcept {
            simd::Matrix<T, N - 1, N - 1> result;

            std::size_t dst_row = 0;
            for (std::size_t i = 0; i < N; ++i) {
                if (i == row)
                    continue;

                std::size_t dst_col = 0;
                for (std::size_t j = 0; j < N; ++j) {
                    if (j == col)
                        continue;

                    result(dst_row, dst_col) = a(i, j);
                    ++dst_col;
                }
                ++dst_row;
            }

            return result;
        }

        // Cofactor element: C_ij = (-1)^(i+j) * M_ij
        template <typename T, std::size_t N>
        [[nodiscard]] constexpr T cofactor_element(const simd::Matrix<T, N, N> &a, std::size_t i,
                                                   std::size_t j) noexcept {
            if constexpr (N == 1) {
                // Cofactor of 1x1 matrix is just 1
                return T{1};
            } else {
                const auto sub = submatrix(a, i, j);
                const T minor = determinant(sub);
                const T sign = ((i + j) % 2 == 0) ? T{1} : T{-1};
                return sign * minor;
            }
        }
    } // namespace detail

    // =========================================================================
    // Cofactor Matrix - Specialized for 2x2
    // =========================================================================
    template <typename T>
    [[nodiscard]] constexpr simd::Matrix<T, 2, 2> cofactor(const simd::Matrix<T, 2, 2> &a) noexcept {
        // For 2x2: cofactor matrix is just [[a11, -a10], [-a01, a00]]
        simd::Matrix<T, 2, 2> result;

        result(0, 0) = a(1, 1);  // C_00 = a11
        result(0, 1) = -a(1, 0); // C_01 = -a10
        result(1, 0) = -a(0, 1); // C_10 = -a01
        result(1, 1) = a(0, 0);  // C_11 = a00

        return result;
    }

    // =========================================================================
    // Cofactor Matrix - Specialized for 3x3
    // =========================================================================
    template <typename T>
    [[nodiscard]] constexpr simd::Matrix<T, 3, 3> cofactor(const simd::Matrix<T, 3, 3> &a) noexcept {
        simd::Matrix<T, 3, 3> result;

        // Compute each cofactor using 2x2 determinants
        // C_ij = (-1)^(i+j) * det(minor_ij)

        // Row 0
        result(0, 0) = +(a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)); // C_00
        result(0, 1) = -(a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)); // C_01
        result(0, 2) = +(a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)); // C_02

        // Row 1
        result(1, 0) = -(a(0, 1) * a(2, 2) - a(0, 2) * a(2, 1)); // C_10
        result(1, 1) = +(a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)); // C_11
        result(1, 2) = -(a(0, 0) * a(2, 1) - a(0, 1) * a(2, 0)); // C_12

        // Row 2
        result(2, 0) = +(a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)); // C_20
        result(2, 1) = -(a(0, 0) * a(1, 2) - a(0, 2) * a(1, 0)); // C_21
        result(2, 2) = +(a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)); // C_22

        return result;
    }

    // =========================================================================
    // Cofactor Matrix - Specialized for 4x4
    // =========================================================================
    template <typename T>
    [[nodiscard]] constexpr simd::Matrix<T, 4, 4> cofactor(const simd::Matrix<T, 4, 4> &a) noexcept {
        simd::Matrix<T, 4, 4> result;

        // For 4x4, each cofactor is a 3x3 determinant
        // We compute them using the specialized 3x3 det formula

        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                result(i, j) = detail::cofactor_element(a, i, j);
            }
        }

        return result;
    }

    // =========================================================================
    // Cofactor Matrix - General case (N x N, N > 4)
    // =========================================================================
    template <typename T, std::size_t N>
    [[nodiscard]] constexpr simd::Matrix<T, N, N> cofactor(const simd::Matrix<T, N, N> &a) noexcept
    requires(N > 4)
    {
        simd::Matrix<T, N, N> result;

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                result(i, j) = detail::cofactor_element(a, i, j);
            }
        }

        return result;
    }

} // namespace optinum::lina
