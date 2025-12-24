#pragma once

// =============================================================================
// optinum/simd/backend/elementwise.hpp
// Element-wise operations (add/sub/mul/div) with SIMD when available
// =============================================================================

#include <optinum/simd/backend/backend.hpp>

namespace optinum::simd::backend {

    template <typename T, std::size_t N>
    OPTINUM_INLINE void add(T *OPTINUM_RESTRICT dst, const T *OPTINUM_RESTRICT a,
                            const T *OPTINUM_RESTRICT b) noexcept {
        constexpr std::size_t W = preferred_simd_lanes<T, N>();
        constexpr std::size_t main = main_loop_count<N, W>();

        for (std::size_t i = 0; i < main; i += W) {
            auto va = pack<T, W>::loadu(a + i);
            auto vb = pack<T, W>::loadu(b + i);
            (va + vb).storeu(dst + i);
        }

        for (std::size_t i = main; i < N; ++i) {
            dst[i] = a[i] + b[i];
        }
    }

    template <typename T, std::size_t N>
    OPTINUM_INLINE void sub(T *OPTINUM_RESTRICT dst, const T *OPTINUM_RESTRICT a,
                            const T *OPTINUM_RESTRICT b) noexcept {
        constexpr std::size_t W = preferred_simd_lanes<T, N>();
        constexpr std::size_t main = main_loop_count<N, W>();

        for (std::size_t i = 0; i < main; i += W) {
            auto va = pack<T, W>::loadu(a + i);
            auto vb = pack<T, W>::loadu(b + i);
            (va - vb).storeu(dst + i);
        }

        for (std::size_t i = main; i < N; ++i) {
            dst[i] = a[i] - b[i];
        }
    }

    template <typename T, std::size_t N>
    OPTINUM_INLINE void mul(T *OPTINUM_RESTRICT dst, const T *OPTINUM_RESTRICT a,
                            const T *OPTINUM_RESTRICT b) noexcept {
        constexpr std::size_t W = preferred_simd_lanes<T, N>();
        constexpr std::size_t main = main_loop_count<N, W>();

        for (std::size_t i = 0; i < main; i += W) {
            auto va = pack<T, W>::loadu(a + i);
            auto vb = pack<T, W>::loadu(b + i);
            (va * vb).storeu(dst + i);
        }

        for (std::size_t i = main; i < N; ++i) {
            dst[i] = a[i] * b[i];
        }
    }

    template <typename T, std::size_t N>
    OPTINUM_INLINE void div(T *OPTINUM_RESTRICT dst, const T *OPTINUM_RESTRICT a,
                            const T *OPTINUM_RESTRICT b) noexcept {
        constexpr std::size_t W = preferred_simd_lanes<T, N>();
        constexpr std::size_t main = main_loop_count<N, W>();

        for (std::size_t i = 0; i < main; i += W) {
            auto va = pack<T, W>::loadu(a + i);
            auto vb = pack<T, W>::loadu(b + i);
            (va / vb).storeu(dst + i);
        }

        for (std::size_t i = main; i < N; ++i) {
            dst[i] = a[i] / b[i];
        }
    }

    template <typename T, std::size_t N>
    OPTINUM_INLINE void mul_scalar(T *OPTINUM_RESTRICT dst, const T *OPTINUM_RESTRICT src, T scalar) noexcept {
        constexpr std::size_t W = preferred_simd_lanes<T, N>();
        constexpr std::size_t main = main_loop_count<N, W>();

        const pack<T, W> s(scalar);
        for (std::size_t i = 0; i < main; i += W) {
            auto v = pack<T, W>::loadu(src + i);
            (v * s).storeu(dst + i);
        }

        for (std::size_t i = main; i < N; ++i) {
            dst[i] = src[i] * scalar;
        }
    }

    template <typename T, std::size_t N>
    OPTINUM_INLINE void div_scalar(T *OPTINUM_RESTRICT dst, const T *OPTINUM_RESTRICT src, T scalar) noexcept {
        constexpr std::size_t W = preferred_simd_lanes<T, N>();
        constexpr std::size_t main = main_loop_count<N, W>();

        const pack<T, W> s(scalar);
        for (std::size_t i = 0; i < main; i += W) {
            auto v = pack<T, W>::loadu(src + i);
            (v / s).storeu(dst + i);
        }

        for (std::size_t i = main; i < N; ++i) {
            dst[i] = src[i] / scalar;
        }
    }

} // namespace optinum::simd::backend
