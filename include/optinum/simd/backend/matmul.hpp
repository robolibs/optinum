#pragma once

// =============================================================================
// optinum/simd/backend/matmul.hpp
// Column-major matrix multiplication and matvec with SIMD when available
// =============================================================================

#include <optinum/simd/backend/backend.hpp>

namespace optinum::simd::backend {

    // Column-major GEMM: C = A * B
    // A: (R x K), B: (K x C), C: (R x C)
    // Layout: column-major (datapod compatible)
    template <typename T, std::size_t R, std::size_t K, std::size_t C>
    OPTINUM_INLINE void matmul(T *OPTINUM_RESTRICT out, const T *OPTINUM_RESTRICT a,
                               const T *OPTINUM_RESTRICT b) noexcept {
        constexpr std::size_t W = preferred_simd_lanes<T, R>();
        constexpr std::size_t main_r = main_loop_count<R, W>();

        for (std::size_t j = 0; j < C; ++j) {
            const T *bcol = b + j * K; // B column j
            T *outcol = out + j * R;   // C column j

            for (std::size_t i = 0; i < main_r; i += W) {
                pack<T, W> acc(T{});
                for (std::size_t k = 0; k < K; ++k) {
                    const auto av = pack<T, W>::loadu(a + k * R + i);
                    const pack<T, W> bv(bcol[k]);
                    acc = pack<T, W>::fma(av, bv, acc);
                }
                acc.storeu(outcol + i);
            }

            for (std::size_t i = main_r; i < R; ++i) {
                T acc{};
                for (std::size_t k = 0; k < K; ++k) {
                    acc += a[k * R + i] * bcol[k];
                }
                outcol[i] = acc;
            }
        }
    }

    // Column-major matrix-vector multiply: y = A * x
    // A: (R x C), x: (C), y: (R)
    template <typename T, std::size_t R, std::size_t C>
    OPTINUM_INLINE void matvec(T *OPTINUM_RESTRICT out, const T *OPTINUM_RESTRICT a,
                               const T *OPTINUM_RESTRICT x) noexcept {
        constexpr std::size_t W = preferred_simd_lanes<T, R>();
        constexpr std::size_t main_r = main_loop_count<R, W>();

        for (std::size_t i = 0; i < main_r; i += W) {
            pack<T, W> acc(T{});
            for (std::size_t j = 0; j < C; ++j) {
                const auto av = pack<T, W>::loadu(a + j * R + i); // column j
                const pack<T, W> xv(x[j]);
                acc = pack<T, W>::fma(av, xv, acc);
            }
            acc.storeu(out + i);
        }

        for (std::size_t i = main_r; i < R; ++i) {
            T acc{};
            for (std::size_t j = 0; j < C; ++j) {
                acc += a[j * R + i] * x[j];
            }
            out[i] = acc;
        }
    }

} // namespace optinum::simd::backend
