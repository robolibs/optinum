#pragma once

// =============================================================================
// optinum/lina/basic/rank.hpp
// Matrix rank via SVD
// =============================================================================

#include <datapod/matrix/matrix.hpp>
#include <optinum/lina/decompose/svd.hpp>
#include <optinum/simd/matrix.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace optinum::lina {

    /**
     * Compute matrix rank via SVD
     *
     * Counts the number of singular values greater than tolerance.
     *
     * @param a Input matrix (M x N)
     * @param tol Tolerance for singular value (default: ε * max(M,N) * σ_max)
     * @return Matrix rank (number of non-zero singular values)
     */
    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] std::size_t rank(const simd::Matrix<T, M, N> &a, T tol = T{-1}) noexcept {
        static_assert(std::is_floating_point_v<T>, "rank() requires floating-point type");

        // Compute SVD
        auto result = svd(a);

        // Determine tolerance if not provided
        if (tol < T{0}) {
            // Default: ε * max(M, N) * σ_max
            constexpr std::size_t max_dim = (M > N) ? M : N;
            T sigma_max = result.s[0]; // Singular values are sorted in descending order
            tol = std::numeric_limits<T>::epsilon() * static_cast<T>(max_dim) * sigma_max;
        }

        // Count singular values > tolerance
        std::size_t r = 0;
        constexpr std::size_t min_dim = (M < N) ? M : N;
        for (std::size_t i = 0; i < min_dim; ++i) {
            if (result.s[i] > tol) {
                ++r;
            }
        }

        return r;
    }

    // Overload for dp::mat::matrix (owning type)
    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] std::size_t rank(const datapod::mat::Matrix<T, M, N> &a, T tol = T{-1}) noexcept {
        simd::Matrix<T, M, N> view(const_cast<datapod::mat::Matrix<T, M, N> &>(a));
        return rank(view, tol);
    }

} // namespace optinum::lina
