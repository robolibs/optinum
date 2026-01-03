#pragma once

// =============================================================================
// optinum/simd/math/sqrt.hpp
// Vectorized sqrt() using pack<T,W> with SIMD intrinsics
// New clean API - replaces fast_sqrt from fast_pow.hpp
// Uses hardware rsqrt + Newton-Raphson refinement
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/pack/pack.hpp>

#if defined(OPTINUM_HAS_SSE2)
#include <optinum/simd/pack/sse.hpp>
#endif

#if defined(OPTINUM_HAS_AVX)
#include <optinum/simd/pack/avx.hpp>
#endif

#if defined(OPTINUM_HAS_NEON)
#include <optinum/simd/pack/neon.hpp>
#endif

namespace optinum::simd {

    // Note: Generic scalar fallback for sqrt is provided in pack/pack.hpp
    // The specializations below provide SIMD-optimized versions

    // =========================================================================
    // pack<float, 4> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE41)

#if defined(OPTINUM_HAS_SSE2)
    template <> inline pack<float, 4> sqrt(const pack<float, 4> &x) noexcept {
        __m128 vx = x.data_;

        __m128 vrsqrt = _mm_rsqrt_ps(vx);

        __m128 vhalf = _mm_set1_ps(0.5f);
        __m128 vthree = _mm_set1_ps(3.0f);
        __m128 vy2 = _mm_mul_ps(vrsqrt, vrsqrt);
        __m128 vxy2 = _mm_mul_ps(vx, vy2);
        vrsqrt = _mm_mul_ps(vrsqrt, _mm_mul_ps(_mm_sub_ps(vthree, vxy2), vhalf));

        __m128 vresult = _mm_mul_ps(vx, vrsqrt);

        __m128 vzero = _mm_setzero_ps();
        __m128 vx_zero = _mm_cmpeq_ps(vx, vzero);
        vresult = _mm_blendv_ps(vresult, vzero, vx_zero);

        __m128 vnan = _mm_set1_ps(__builtin_nanf(""));
        __m128 vx_neg = _mm_cmplt_ps(vx, vzero);
        vresult = _mm_blendv_ps(vresult, vnan, vx_neg);

        return pack<float, 4>(vresult);
    }

#endif // OPTINUM_HAS_SSE41

    // =========================================================================
    // pack<float, 4> - NEON implementation
    // =========================================================================
#if defined(OPTINUM_HAS_NEON) && !defined(OPTINUM_HAS_SSE41)

    template <> inline pack<float, 4> sqrt(const pack<float, 4> &x) noexcept {
#ifdef __aarch64__
        return pack<float, 4>(vsqrtq_f32(x.data_));
#else
        // ARMv7: Use rsqrt estimate + Newton-Raphson
        float32x4_t vx = x.data_;
        float32x4_t estimate = vrsqrteq_f32(vx);
        estimate = vmulq_f32(estimate, vrsqrtsq_f32(vmulq_f32(vx, estimate), estimate));
        return pack<float, 4>(vmulq_f32(vx, estimate));
#endif
    }

#endif // OPTINUM_HAS_NEON && !OPTINUM_HAS_SSE41

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

#endif // OPTINUM_HAS_SSE2

#if defined(OPTINUM_HAS_AVX)
    template <> inline pack<float, 8> sqrt(const pack<float, 8> &x) noexcept {
        __m256 vx = x.data_;

        // Initial approximation using rsqrt (1/sqrt(x))
        __m256 vrsqrt = _mm256_rsqrt_ps(vx);

        // One Newton-Raphson iteration for rsqrt: y = y * (3 - x*y*y) / 2
        __m256 vhalf = _mm256_set1_ps(0.5f);
        __m256 vthree = _mm256_set1_ps(3.0f);
        __m256 vy2 = _mm256_mul_ps(vrsqrt, vrsqrt);
        __m256 vxy2 = _mm256_mul_ps(vx, vy2);
#ifdef OPTINUM_HAS_FMA
        vrsqrt = _mm256_mul_ps(vrsqrt, _mm256_fmadd_ps(_mm256_sub_ps(vthree, vxy2), vhalf, _mm256_setzero_ps()));
#else
        vrsqrt = _mm256_mul_ps(vrsqrt, _mm256_mul_ps(_mm256_sub_ps(vthree, vxy2), vhalf));
#endif

        // sqrt(x) = x * rsqrt(x)
        __m256 vresult = _mm256_mul_ps(vx, vrsqrt);

        // Handle x == 0 (avoid 0 * inf = NaN)
        __m256 vzero = _mm256_setzero_ps();
        __m256 vx_zero = _mm256_cmp_ps(vx, vzero, _CMP_EQ_OQ);
        vresult = _mm256_blendv_ps(vresult, vzero, vx_zero);

        // Handle x < 0 -> NaN
        __m256 vnan = _mm256_set1_ps(__builtin_nanf(""));
        __m256 vx_neg = _mm256_cmp_ps(vx, vzero, _CMP_LT_OQ);
        vresult = _mm256_blendv_ps(vresult, vnan, vx_neg);

        return pack<float, 8>(vresult);
    }

#endif // OPTINUM_HAS_AVX

    // =========================================================================
#endif // OPTINUM_HAS_AVX

    // pack<double, 2> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE2)

    template <> inline pack<double, 2> sqrt(const pack<double, 2> &x) noexcept {
        __m128d vx = x.data_;

        // Use hardware sqrt for double precision (more accurate than rsqrt + refinement)
        __m128d vresult = _mm_sqrt_pd(vx);

        // Handle special cases
        __m128d vzero = _mm_setzero_pd();
        __m128d vnan = _mm_set1_pd(__builtin_nan(""));

        // x < 0 -> NaN
        __m128d vx_neg = _mm_cmplt_pd(vx, vzero);
        vresult = _mm_blendv_pd(vresult, vnan, vx_neg);

        return pack<double, 2>(vresult);
    }

#endif // OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<double, 2> - NEON implementation (ARM64 only)
    // =========================================================================
#if defined(OPTINUM_HAS_NEON) && defined(__aarch64__) && !defined(OPTINUM_HAS_SSE2)

    template <> inline pack<double, 2> sqrt(const pack<double, 2> &x) noexcept {
        return pack<double, 2>(vsqrtq_f64(x.data_));
    }

#endif // OPTINUM_HAS_NEON && __aarch64__ && !OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<double, 4> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<double, 4> sqrt(const pack<double, 4> &x) noexcept {
        __m256d vx = x.data_;

        // Use hardware sqrt for double precision
        __m256d vresult = _mm256_sqrt_pd(vx);

        // Handle special cases
        __m256d vzero = _mm256_setzero_pd();
        __m256d vnan = _mm256_set1_pd(__builtin_nan(""));

        // x < 0 -> NaN
        __m256d vx_neg = _mm256_cmp_pd(vx, vzero, _CMP_LT_OQ);
        vresult = _mm256_blendv_pd(vresult, vnan, vx_neg);

        return pack<double, 4>(vresult);
    }

#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
