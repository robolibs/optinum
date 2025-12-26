#pragma once

// =============================================================================
// optinum/lina/basic/cond.hpp
// Condition number via SVD
// =============================================================================

#include <optinum/lina/decompose/svd.hpp>
#include <optinum/simd/matrix.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace optinum::lina {

    /**
     * Compute condition number (2-norm) via SVD
     *
     * cond(A) = σ_max / σ_min
     *
     * A large condition number indicates an ill-conditioned matrix.
     *
     * @param a Input square matrix (N x N)
     * @return Condition number (infinity if singular)
     */
    template <typename T, std::size_t N> [[nodiscard]] T cond(const simd::Matrix<T, N, N> &a) noexcept {
        static_assert(std::is_floating_point_v<T>, "cond() requires floating-point type");

        // Compute SVD
        auto result = svd(a);

        // cond = σ_max / σ_min
        T sigma_max = result.s[0];     // Largest singular value
        T sigma_min = result.s[N - 1]; // Smallest singular value

        // Check for singularity
        if (sigma_min <= std::numeric_limits<T>::epsilon() * sigma_max) {
            return std::numeric_limits<T>::infinity();
        }

        return sigma_max / sigma_min;
    }

    /**
     * Compute reciprocal condition number via SVD
     *
     * rcond(A) = σ_min / σ_max = 1 / cond(A)
     *
     * Values close to 0 indicate ill-conditioning.
     *
     * @param a Input square matrix (N x N)
     * @return Reciprocal condition number (0 if singular)
     */
    template <typename T, std::size_t N> [[nodiscard]] T rcond(const simd::Matrix<T, N, N> &a) noexcept {
        static_assert(std::is_floating_point_v<T>, "rcond() requires floating-point type");

        // Compute SVD
        auto result = svd(a);

        // rcond = σ_min / σ_max
        T sigma_max = result.s[0];
        T sigma_min = result.s[N - 1];

        // Check for singularity
        if (sigma_min <= std::numeric_limits<T>::epsilon() * sigma_max) {
            return T{0};
        }

        return sigma_min / sigma_max;
    }

} // namespace optinum::lina
