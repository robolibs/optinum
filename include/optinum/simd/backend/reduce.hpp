#pragma once

// =============================================================================
// optinum/simd/backend/reduce.hpp
// Reductions (sum/min/max) with SIMD when available
// =============================================================================

#include <optinum/simd/backend/backend.hpp>

namespace optinum::simd::backend {

    // Runtime version for Dynamic sizes
    template <typename T>
    [[nodiscard]] OPTINUM_INLINE T reduce_sum_runtime(const T *OPTINUM_RESTRICT src, std::size_t n) noexcept {
        const std::size_t W = preferred_simd_lanes_runtime<T>();
        const std::size_t main = main_loop_count_runtime(n, W);

        constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = pack<T, pack_width>;

        pack_t acc(T{});
        for (std::size_t i = 0; i < main; i += W) {
            acc += pack_t::loadu(src + i);
        }

        T result = acc.hsum();
        for (std::size_t i = main; i < n; ++i) {
            result += src[i];
        }
        return result;
    }

    // Compile-time version for fixed sizes
    template <typename T, std::size_t N>
    [[nodiscard]] OPTINUM_INLINE T reduce_sum(const T *OPTINUM_RESTRICT src) noexcept {
        constexpr std::size_t W = preferred_simd_lanes<T, N>();
        constexpr std::size_t main = main_loop_count<N, W>();

        pack<T, W> acc(T{});
        for (std::size_t i = 0; i < main; i += W) {
            acc += pack<T, W>::loadu(src + i);
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
                pack<T, W> acc = pack<T, W>::loadu(src);
                for (std::size_t i = W; i < main; i += W) {
                    acc = pack<T, W>::min(acc, pack<T, W>::loadu(src + i));
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
                pack<T, W> acc = pack<T, W>::loadu(src);
                for (std::size_t i = W; i < main; i += W) {
                    acc = pack<T, W>::max(acc, pack<T, W>::loadu(src + i));
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
