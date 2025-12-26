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

    // Fill array with constant value using SIMD
    template <typename T, std::size_t N> OPTINUM_INLINE void fill(T *OPTINUM_RESTRICT dst, T value) noexcept {
        constexpr std::size_t W = preferred_simd_lanes<T, N>();
        constexpr std::size_t main = main_loop_count<N, W>();

        const pack<T, W> v(value); // Broadcast value to all lanes
        for (std::size_t i = 0; i < main; i += W) {
            v.storeu(dst + i);
        }

        for (std::size_t i = main; i < N; ++i) {
            dst[i] = value;
        }
    }

    // Fill array with sequential values (iota) using SIMD
    template <typename T, std::size_t N> OPTINUM_INLINE void iota(T *OPTINUM_RESTRICT dst, T start, T step) noexcept {
        constexpr std::size_t W = preferred_simd_lanes<T, N>();
        constexpr std::size_t main = main_loop_count<N, W>();

        // Create pack with sequential offsets: [0, 1, 2, 3, ...] * step
        alignas(32) T offsets[W];
        for (std::size_t j = 0; j < W; ++j) {
            offsets[j] = static_cast<T>(j) * step;
        }
        const pack<T, W> offset_pack = pack<T, W>::loadu(offsets);
        const pack<T, W> step_pack(static_cast<T>(W) * step); // Increment per iteration

        pack<T, W> current(start);
        current = current + offset_pack; // [start, start+step, start+2*step, ...]

        for (std::size_t i = 0; i < main; i += W) {
            current.storeu(dst + i);
            current = current + step_pack;
        }

        // Tail elements
        for (std::size_t i = main; i < N; ++i) {
            dst[i] = start + static_cast<T>(i) * step;
        }
    }

    // Reverse array in-place using SIMD
    template <typename T, std::size_t N> OPTINUM_INLINE void reverse(T *OPTINUM_RESTRICT data) noexcept {
        constexpr std::size_t W = preferred_simd_lanes<T, N>();

        // For small arrays or when SIMD doesn't help, use scalar swap
        if constexpr (N < W * 2) {
            for (std::size_t i = 0; i < N / 2; ++i) {
                std::swap(data[i], data[N - 1 - i]);
            }
        } else {
            // SIMD reverse: load from both ends, reverse within packs, swap and store
            std::size_t left = 0;
            std::size_t right = (N / W) * W; // Align to pack boundary

            while (left < right - W) {
                auto left_pack = pack<T, W>::loadu(data + left);
                auto right_pack = pack<T, W>::loadu(data + right - W);

                // Reverse the packs and swap
                left_pack.reverse().storeu(data + right - W);
                right_pack.reverse().storeu(data + left);

                left += W;
                right -= W;
            }

            // Handle remaining elements with scalar swaps
            for (std::size_t i = left; i < N / 2; ++i) {
                std::swap(data[i], data[N - 1 - i]);
            }
        }
    }

} // namespace optinum::simd::backend
