#pragma once

// =============================================================================
// optinum/simd/backend/reduce.hpp
// Reductions (sum/min/max) with SIMD when available
// =============================================================================

#include <optinum/simd/backend/backend.hpp>

namespace optinum::simd::backend {

    template <typename T, std::size_t N>
    [[nodiscard]] OPTINUM_INLINE T reduce_sum(const T *OPTINUM_RESTRICT src) noexcept {
        constexpr std::size_t W = preferred_simd_lanes<T, N>();
        constexpr std::size_t main = main_loop_count<N, W>();

        SIMDVec<T, W> acc(T{});
        for (std::size_t i = 0; i < main; i += W) {
            acc += SIMDVec<T, W>::loadu(src + i);
        }

        T result = acc.hsum();
        for (std::size_t i = main; i < N; ++i) {
            result += src[i];
        }
        return result;
    }

    template <typename T, std::size_t N>
    [[nodiscard]] OPTINUM_INLINE T reduce_min(const T *OPTINUM_RESTRICT src) noexcept {
        T result = src[0];
        constexpr std::size_t W = preferred_simd_lanes<T, N>();
        constexpr std::size_t main = main_loop_count<N, W>();

        if constexpr (W > 1) {
            if constexpr (N >= W) {
                SIMDVec<T, W> acc = SIMDVec<T, W>::loadu(src);
                for (std::size_t i = W; i < main; i += W) {
                    acc = SIMDVec<T, W>::min(acc, SIMDVec<T, W>::loadu(src + i));
                }
                result = acc.hmin();
                for (std::size_t i = main; i < N; ++i) {
                    result = (src[i] < result) ? src[i] : result;
                }
                return result;
            }
        }

        for (std::size_t i = 1; i < N; ++i) {
            result = (src[i] < result) ? src[i] : result;
        }
        return result;
    }

    template <typename T, std::size_t N>
    [[nodiscard]] OPTINUM_INLINE T reduce_max(const T *OPTINUM_RESTRICT src) noexcept {
        T result = src[0];
        constexpr std::size_t W = preferred_simd_lanes<T, N>();
        constexpr std::size_t main = main_loop_count<N, W>();

        if constexpr (W > 1) {
            if constexpr (N >= W) {
                SIMDVec<T, W> acc = SIMDVec<T, W>::loadu(src);
                for (std::size_t i = W; i < main; i += W) {
                    acc = SIMDVec<T, W>::max(acc, SIMDVec<T, W>::loadu(src + i));
                }
                result = acc.hmax();
                for (std::size_t i = main; i < N; ++i) {
                    result = (src[i] > result) ? src[i] : result;
                }
                return result;
            }
        }

        for (std::size_t i = 1; i < N; ++i) {
            result = (src[i] > result) ? src[i] : result;
        }
        return result;
    }

} // namespace optinum::simd::backend
