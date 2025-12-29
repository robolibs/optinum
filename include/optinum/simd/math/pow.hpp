#pragma once

// =============================================================================
// optinum/simd/math/pow.hpp
// Vectorized pow() using pack<T,W> with SIMD intrinsics
// pow(x, y) = exp(y * log(x))
// Uses existing exp/log implementations
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/math/exp.hpp>
#include <optinum/simd/math/log.hpp>
#include <optinum/simd/pack/pack.hpp>
#if defined(OPTINUM_HAS_AVX)
#include <optinum/simd/pack/avx.hpp>
#endif
#if defined(OPTINUM_HAS_SSE2)
#include <optinum/simd/pack/sse.hpp>
#endif

#if defined(OPTINUM_HAS_NEON)
#include <optinum/simd/pack/neon.hpp>
#endif

namespace optinum::simd {

    // Forward declaration
    template <typename T, std::size_t W> pack<T, W> pow(const pack<T, W> &x, const pack<T, W> &y) noexcept;

    // =========================================================================
    // pack<float, 4> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE41)

#if defined(OPTINUM_HAS_SSE2)
    template <> inline pack<float, 4> pow(const pack<float, 4> &x, const pack<float, 4> &y) noexcept {
        // pow(x, y) = exp(y * log(x))
        pack<float, 4> log_x = log(x);
        __m128 y_log_x = _mm_mul_ps(y.data_, log_x.data_);
        return exp(pack<float, 4>(y_log_x));
    }

#endif // OPTINUM_HAS_SSE41

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

#endif // OPTINUM_HAS_SSE2

#if defined(OPTINUM_HAS_AVX)
    template <> inline pack<float, 8> pow(const pack<float, 8> &x, const pack<float, 8> &y) noexcept {
        pack<float, 8> log_x = log(x);
        __m256 y_log_x = _mm256_mul_ps(y.data_, log_x.data_);
        return exp(pack<float, 8>(y_log_x));
    }

#endif // OPTINUM_HAS_AVX

    // =========================================================================
#endif // OPTINUM_HAS_AVX

    // pack<double, 2> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE41)

    template <> inline pack<double, 2> pow(const pack<double, 2> &x, const pack<double, 2> &y) noexcept {
        pack<double, 2> log_x = log(x);
        __m128d y_log_x = _mm_mul_pd(y.data_, log_x.data_);
        return exp(pack<double, 2>(y_log_x));
    }

#endif // OPTINUM_HAS_SSE41

    // =========================================================================
    // pack<double, 4> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<double, 4> pow(const pack<double, 4> &x, const pack<double, 4> &y) noexcept {
        pack<double, 4> log_x = log(x);
        __m256d y_log_x = _mm256_mul_pd(y.data_, log_x.data_);
        return exp(pack<double, 4>(y_log_x));
    }

#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
