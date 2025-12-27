#pragma once

// =============================================================================
// optinum/simd/backend/norm.hpp
// Norms and normalization with SIMD when available
// =============================================================================

#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>

#include <cmath>

namespace optinum::simd::backend {

    // Runtime version for Dynamic sizes
    template <typename T>
    [[nodiscard]] OPTINUM_INLINE T norm_l2_runtime(const T *OPTINUM_RESTRICT src, std::size_t n) noexcept {
        return std::sqrt(dot_runtime<T>(src, src, n));
    }

    // Compile-time version for fixed sizes
    template <typename T, std::size_t N>
    [[nodiscard]] OPTINUM_INLINE T norm_l2(const T *OPTINUM_RESTRICT src) noexcept {
        return std::sqrt(dot<T, N>(src, src));
    }

    // Runtime version for Dynamic sizes
    template <typename T>
    OPTINUM_INLINE void normalize_runtime(T *OPTINUM_RESTRICT dst, const T *OPTINUM_RESTRICT src,
                                          std::size_t n) noexcept {
        const T norm = norm_l2_runtime<T>(src, n);
        if (norm > T{}) {
            div_scalar_runtime<T>(dst, src, norm, n);
        } else {
            // If norm is 0, return input unchanged.
            for (std::size_t i = 0; i < n; ++i) {
                dst[i] = src[i];
            }
        }
    }

    // Compile-time version for fixed sizes
    template <typename T, std::size_t N>
    OPTINUM_INLINE void normalize(T *OPTINUM_RESTRICT dst, const T *OPTINUM_RESTRICT src) noexcept {
        const T n = norm_l2<T, N>(src);
        if (n > T{}) {
            div_scalar<T, N>(dst, src, n);
        } else {
            // If norm is 0, return input unchanged.
            for (std::size_t i = 0; i < N; ++i) {
                dst[i] = src[i];
            }
        }
    }

} // namespace optinum::simd::backend
