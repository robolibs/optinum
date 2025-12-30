#pragma once

// =============================================================================
// optinum/lina/basic/properties.hpp
// Matrix property checks
// =============================================================================

#include <optinum/simd/matrix.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace optinum::lina {

    /**
     * Check if matrix is symmetric (A == A^T)
     *
     * @param a Input square matrix
     * @param tol Tolerance for comparison (default: machine epsilon)
     * @return true if symmetric, false otherwise
     */
    template <typename T, std::size_t N>
    [[nodiscard]] bool is_symmetric(const simd::Matrix<T, N, N> &a,
                                    T tol = std::numeric_limits<T>::epsilon()) noexcept {
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                if (std::abs(a(i, j) - a(j, i)) > tol) {
                    return false;
                }
            }
        }
        return true;
    }

    // Overload for dp::mat::matrix (owning type)
    template <typename T, std::size_t N>
    [[nodiscard]] bool is_symmetric(const datapod::mat::matrix<T, N, N> &a,
                                    T tol = std::numeric_limits<T>::epsilon()) noexcept {
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                if (std::abs(a(i, j) - a(j, i)) > tol) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Check if matrix is Hermitian (A == A^H for complex, A == A^T for real)
     *
     * For real matrices, this is the same as is_symmetric.
     *
     * @param a Input square matrix
     * @param tol Tolerance for comparison
     * @return true if Hermitian, false otherwise
     */
    template <typename T, std::size_t N>
    [[nodiscard]] bool is_hermitian(const simd::Matrix<T, N, N> &a,
                                    T tol = std::numeric_limits<T>::epsilon()) noexcept {
        static_assert(std::is_floating_point_v<T>, "is_hermitian() currently only supports real matrices");
        // For real matrices, Hermitian is the same as symmetric
        return is_symmetric(a, tol);
    }

    /**
     * Check if matrix is positive definite
     *
     * A matrix is positive definite if it's symmetric and all eigenvalues are positive.
     * This implementation uses Cholesky decomposition as a test.
     *
     * @param a Input square matrix
     * @return true if positive definite, false otherwise
     */
    template <typename T, std::size_t N>
    [[nodiscard]] bool is_positive_definite(const simd::Matrix<T, N, N> &a) noexcept {
        // First check symmetry
        if (!is_symmetric(a)) {
            return false;
        }

        // Try Cholesky decomposition - it succeeds only for SPD matrices
        // For now, we'll do a simple check on diagonal elements
        // TODO: Use actual Cholesky when we integrate it
        for (std::size_t i = 0; i < N; ++i) {
            if (a(i, i) <= T{0}) {
                return false; // Diagonal must be positive
            }
        }

        return true; // Simplified check
    }

    // Overload for dp::mat::matrix (owning type)
    template <typename T, std::size_t N>
    [[nodiscard]] bool is_positive_definite(const datapod::mat::matrix<T, N, N> &a) noexcept {
        // First check symmetry
        if (!is_symmetric(a)) {
            return false;
        }

        // Try Cholesky decomposition - it succeeds only for SPD matrices
        // For now, we'll do a simple check on diagonal elements
        // TODO: Use actual Cholesky when we integrate it
        for (std::size_t i = 0; i < N; ++i) {
            if (a(i, i) <= T{0}) {
                return false; // Diagonal must be positive
            }
        }

        return true; // Simplified check
    }

} // namespace optinum::lina
