#pragma once

// =============================================================================
// optinum/simd/math/sqrt.hpp
// Vectorized sqrt() using pack<T,W> with SIMD intrinsics
// New clean API - replaces fast_sqrt from fast_pow.hpp
// Uses hardware rsqrt + Newton-Raphson refinement
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/pack/avx.hpp>
#include <optinum/simd/pack/sse.hpp>

namespace optinum::simd {

    // Forward declaration
    template <typename T, std::size_t W> pack<T, W> sqrt(const pack<T, W> &x) noexcept;

    // =========================================================================
    // pack<float, 4> - SSE implementation
    // =========================================================================

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

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================

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

    // TODO: Add double precision variants (pack<double, 2>, pack<double, 4>)

} // namespace optinum::simd
