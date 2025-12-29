#pragma once

// =============================================================================
// optinum/simd/backend/elementwise.hpp
// Element-wise operations (add/sub/mul/div) with SIMD when available
// =============================================================================

#include <optinum/simd/backend/backend.hpp>

namespace optinum::simd::backend {

    // Runtime versions for Dynamic sizes
    template <typename T>
    OPTINUM_INLINE void add_runtime(T *OPTINUM_RESTRICT dst, const T *OPTINUM_RESTRICT a, const T *OPTINUM_RESTRICT b,
                                    std::size_t n) noexcept {
        const std::size_t W = preferred_simd_lanes_runtime<T>();
        const std::size_t main = main_loop_count_runtime(n, W);

        constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = pack<T, pack_width>;

        for (std::size_t i = 0; i < main; i += W) {
            auto va = pack_t::loadu(a + i);
            auto vb = pack_t::loadu(b + i);
            (va + vb).storeu(dst + i);
        }

        for (std::size_t i = main; i < n; ++i) {
            dst[i] = a[i] + b[i];
        }
    }

    template <typename T>
    OPTINUM_INLINE void sub_runtime(T *OPTINUM_RESTRICT dst, const T *OPTINUM_RESTRICT a, const T *OPTINUM_RESTRICT b,
                                    std::size_t n) noexcept {
        const std::size_t W = preferred_simd_lanes_runtime<T>();
        const std::size_t main = main_loop_count_runtime(n, W);

        constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = pack<T, pack_width>;

        for (std::size_t i = 0; i < main; i += W) {
            auto va = pack_t::loadu(a + i);
            auto vb = pack_t::loadu(b + i);
            (va - vb).storeu(dst + i);
        }

        for (std::size_t i = main; i < n; ++i) {
            dst[i] = a[i] - b[i];
        }
    }

    template <typename T>
    OPTINUM_INLINE void mul_runtime(T *OPTINUM_RESTRICT dst, const T *OPTINUM_RESTRICT a, const T *OPTINUM_RESTRICT b,
                                    std::size_t n) noexcept {
        const std::size_t W = preferred_simd_lanes_runtime<T>();
        const std::size_t main = main_loop_count_runtime(n, W);

        constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = pack<T, pack_width>;

        for (std::size_t i = 0; i < main; i += W) {
            auto va = pack_t::loadu(a + i);
            auto vb = pack_t::loadu(b + i);
            (va * vb).storeu(dst + i);
        }

        for (std::size_t i = main; i < n; ++i) {
            dst[i] = a[i] * b[i];
        }
    }

    template <typename T>
    OPTINUM_INLINE void div_runtime(T *OPTINUM_RESTRICT dst, const T *OPTINUM_RESTRICT a, const T *OPTINUM_RESTRICT b,
                                    std::size_t n) noexcept {
        const std::size_t W = preferred_simd_lanes_runtime<T>();
        const std::size_t main = main_loop_count_runtime(n, W);

        constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = pack<T, pack_width>;

        for (std::size_t i = 0; i < main; i += W) {
            auto va = pack_t::loadu(a + i);
            auto vb = pack_t::loadu(b + i);
            (va / vb).storeu(dst + i);
        }

        for (std::size_t i = main; i < n; ++i) {
            dst[i] = a[i] / b[i];
        }
    }

    // Compile-time versions for fixed sizes
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

    // Runtime versions for Dynamic sizes
    template <typename T>
    OPTINUM_INLINE void mul_scalar_runtime(T *OPTINUM_RESTRICT dst, const T *OPTINUM_RESTRICT src, T scalar,
                                           std::size_t n) noexcept {
        const std::size_t W = preferred_simd_lanes_runtime<T>();
        const std::size_t main = main_loop_count_runtime(n, W);

        constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = pack<T, pack_width>;

        const pack_t s(scalar);
        for (std::size_t i = 0; i < main; i += W) {
            auto v = pack_t::loadu(src + i);
            (v * s).storeu(dst + i);
        }

        for (std::size_t i = main; i < n; ++i) {
            dst[i] = src[i] * scalar;
        }
    }

    template <typename T>
    OPTINUM_INLINE void div_scalar_runtime(T *OPTINUM_RESTRICT dst, const T *OPTINUM_RESTRICT src, T scalar,
                                           std::size_t n) noexcept {
        const std::size_t W = preferred_simd_lanes_runtime<T>();
        const std::size_t main = main_loop_count_runtime(n, W);

        constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = pack<T, pack_width>;

        const pack_t s(scalar);
        for (std::size_t i = 0; i < main; i += W) {
            auto v = pack_t::loadu(src + i);
            (v / s).storeu(dst + i);
        }

        for (std::size_t i = main; i < n; ++i) {
            dst[i] = src[i] / scalar;
        }
    }

    // Compile-time versions for fixed sizes
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

    // Runtime version for Dynamic sizes
    template <typename T> OPTINUM_INLINE void fill_runtime(T *OPTINUM_RESTRICT dst, std::size_t n, T value) noexcept {
        const std::size_t W = preferred_simd_lanes_runtime<T>();
        const std::size_t main = main_loop_count_runtime(n, W);

        constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = pack<T, pack_width>;

        const pack_t v(value); // Broadcast value to all lanes
        for (std::size_t i = 0; i < main; i += W) {
            v.storeu(dst + i);
        }

        for (std::size_t i = main; i < n; ++i) {
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

    // =============================================================================
    // Optimizer-specific SIMD utilities
    // These patterns are used extensively in gradient-based optimizers
    // =============================================================================

    // axpy_runtime: y = x + alpha * d (BLAS-style axpy)
    // Computes: dst[i] = x[i] + alpha * d[i] for all i
    template <typename T>
    OPTINUM_INLINE void axpy_runtime(T *OPTINUM_RESTRICT dst, const T *OPTINUM_RESTRICT x, T alpha,
                                     const T *OPTINUM_RESTRICT d, std::size_t n) noexcept {
        const std::size_t W = preferred_simd_lanes_runtime<T>();
        const std::size_t main = main_loop_count_runtime(n, W);

        constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = pack<T, pack_width>;

        const pack_t alpha_pack(alpha);
        for (std::size_t i = 0; i < main; i += W) {
            auto vx = pack_t::loadu(x + i);
            auto vd = pack_t::loadu(d + i);
            // y = x + alpha * d using FMA: fma(d, alpha, x)
            pack_t::fma(vd, alpha_pack, vx).storeu(dst + i);
        }

        for (std::size_t i = main; i < n; ++i) {
            dst[i] = x[i] + alpha * d[i];
        }
    }

    // axpy_inplace_runtime: x += alpha * d (in-place axpy)
    // Computes: x[i] += alpha * d[i] for all i
    template <typename T>
    OPTINUM_INLINE void axpy_inplace_runtime(T *OPTINUM_RESTRICT x, T alpha, const T *OPTINUM_RESTRICT d,
                                             std::size_t n) noexcept {
        const std::size_t W = preferred_simd_lanes_runtime<T>();
        const std::size_t main = main_loop_count_runtime(n, W);

        constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = pack<T, pack_width>;

        const pack_t alpha_pack(alpha);
        for (std::size_t i = 0; i < main; i += W) {
            auto vx = pack_t::loadu(x + i);
            auto vd = pack_t::loadu(d + i);
            // x = x + alpha * d using FMA: fma(d, alpha, x)
            pack_t::fma(vd, alpha_pack, vx).storeu(x + i);
        }

        for (std::size_t i = main; i < n; ++i) {
            x[i] += alpha * d[i];
        }
    }

    // scale_sub_runtime: x -= alpha * g (gradient descent step)
    // Computes: x[i] -= alpha * g[i] for all i
    // This is the most common optimizer update pattern
    template <typename T>
    OPTINUM_INLINE void scale_sub_runtime(T *OPTINUM_RESTRICT x, T alpha, const T *OPTINUM_RESTRICT g,
                                          std::size_t n) noexcept {
        const std::size_t W = preferred_simd_lanes_runtime<T>();
        const std::size_t main = main_loop_count_runtime(n, W);

        constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = pack<T, pack_width>;

        const pack_t alpha_pack(alpha);
        for (std::size_t i = 0; i < main; i += W) {
            auto vx = pack_t::loadu(x + i);
            auto vg = pack_t::loadu(g + i);
            // x = x - alpha * g using FMS: fms(g, alpha, x) = g*alpha - x, then negate
            // Or: x - alpha*g = x + (-alpha)*g = fma(g, -alpha, x)
            pack_t::fma(vg, pack_t(-alpha), vx).storeu(x + i);
        }

        for (std::size_t i = main; i < n; ++i) {
            x[i] -= alpha * g[i];
        }
    }

    // fms_runtime: out = a - s * b (fused multiply-subtract)
    // Computes: out[i] = a[i] - s * b[i] for all i
    template <typename T>
    OPTINUM_INLINE void fms_runtime(T *OPTINUM_RESTRICT out, const T *OPTINUM_RESTRICT a, T s,
                                    const T *OPTINUM_RESTRICT b, std::size_t n) noexcept {
        const std::size_t W = preferred_simd_lanes_runtime<T>();
        const std::size_t main = main_loop_count_runtime(n, W);

        constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = pack<T, pack_width>;

        const pack_t neg_s(-s);
        for (std::size_t i = 0; i < main; i += W) {
            auto va = pack_t::loadu(a + i);
            auto vb = pack_t::loadu(b + i);
            // out = a - s*b = a + (-s)*b = fma(b, -s, a)
            pack_t::fma(vb, neg_s, va).storeu(out + i);
        }

        for (std::size_t i = main; i < n; ++i) {
            out[i] = a[i] - s * b[i];
        }
    }

    // negate_runtime: out = -a
    // Computes: out[i] = -a[i] for all i
    template <typename T>
    OPTINUM_INLINE void negate_runtime(T *OPTINUM_RESTRICT out, const T *OPTINUM_RESTRICT a, std::size_t n) noexcept {
        const std::size_t W = preferred_simd_lanes_runtime<T>();
        const std::size_t main = main_loop_count_runtime(n, W);

        constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = pack<T, pack_width>;

        for (std::size_t i = 0; i < main; i += W) {
            auto va = pack_t::loadu(a + i);
            (-va).storeu(out + i);
        }

        for (std::size_t i = main; i < n; ++i) {
            out[i] = -a[i];
        }
    }

    // negate_inplace_runtime: x = -x
    // Computes: x[i] = -x[i] for all i
    template <typename T> OPTINUM_INLINE void negate_inplace_runtime(T *OPTINUM_RESTRICT x, std::size_t n) noexcept {
        const std::size_t W = preferred_simd_lanes_runtime<T>();
        const std::size_t main = main_loop_count_runtime(n, W);

        constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = pack<T, pack_width>;

        for (std::size_t i = 0; i < main; i += W) {
            auto vx = pack_t::loadu(x + i);
            (-vx).storeu(x + i);
        }

        for (std::size_t i = main; i < n; ++i) {
            x[i] = -x[i];
        }
    }

    // copy_runtime: dst = src
    // Computes: dst[i] = src[i] for all i
    template <typename T>
    OPTINUM_INLINE void copy_runtime(T *OPTINUM_RESTRICT dst, const T *OPTINUM_RESTRICT src, std::size_t n) noexcept {
        const std::size_t W = preferred_simd_lanes_runtime<T>();
        const std::size_t main = main_loop_count_runtime(n, W);

        constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = pack<T, pack_width>;

        for (std::size_t i = 0; i < main; i += W) {
            pack_t::loadu(src + i).storeu(dst + i);
        }

        for (std::size_t i = main; i < n; ++i) {
            dst[i] = src[i];
        }
    }

} // namespace optinum::simd::backend
