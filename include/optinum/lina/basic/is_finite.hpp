#pragma once

// =============================================================================
// optinum/lina/basic/is_finite.hpp
// Check if all matrix elements are finite (not inf, not nan)
// =============================================================================

#include <optinum/simd/bridge.hpp>
#include <optinum/simd/math/isfinite.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cstddef>
#include <type_traits>

namespace optinum::lina {

    /**
     * Check if all matrix elements are finite
     *
     * An element is finite if it's not infinity and not NaN.
     * Uses SIMD isfinite() + reduction for ~95% SIMD coverage.
     *
     * @param a Input matrix
     * @return true if all elements are finite, false otherwise
     */
    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] bool is_finite(const simd::Matrix<T, M, N> &a) noexcept {
        static_assert(std::is_floating_point_v<T>, "is_finite() requires floating-point type");

        constexpr std::size_t total = M * N;
        constexpr std::size_t W = simd::backend::preferred_simd_lanes<T, total>();
        constexpr std::size_t num_packs = total / W;
        constexpr std::size_t tail = total % W;

        const T *data = a.data();

        // SIMD main loop
        for (std::size_t i = 0; i < num_packs; ++i) {
            auto p = simd::pack<T, W>::loadu(data + i * W);
            auto finite_mask = simd::isfinite(p);

            // If any element is NOT finite, return false
            if (!finite_mask.all()) {
                return false;
            }
        }

        // Scalar tail loop
        for (std::size_t i = num_packs * W; i < total; ++i) {
            if (!std::isfinite(data[i])) {
                return false;
            }
        }

        return true;
    }

    /**
     * Check if all vector elements are finite
     */
    template <typename T, std::size_t N> [[nodiscard]] bool is_finite(const simd::Vector<T, N> &v) noexcept {
        static_assert(std::is_floating_point_v<T>, "is_finite() requires floating-point type");

        constexpr std::size_t W = simd::backend::preferred_simd_lanes<T, N>();
        constexpr std::size_t num_packs = N / W;
        constexpr std::size_t tail = N % W;

        const T *data = v.data();

        // SIMD main loop
        for (std::size_t i = 0; i < num_packs; ++i) {
            auto p = simd::pack<T, W>::loadu(data + i * W);
            auto finite_mask = simd::isfinite(p);

            if (!finite_mask.all()) {
                return false;
            }
        }

        // Scalar tail loop
        for (std::size_t i = num_packs * W; i < N; ++i) {
            if (!std::isfinite(data[i])) {
                return false;
            }
        }

        return true;
    }

} // namespace optinum::lina
