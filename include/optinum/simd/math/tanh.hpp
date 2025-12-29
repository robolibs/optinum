#pragma once

// =============================================================================
// optinum/simd/math/tanh.hpp
// Vectorized tanh() using pack<T,W> with SIMD intrinsics
// New clean API - replaces fast_tanh from fast_hyp.hpp
// Critical for neural networks (activation function)
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/math/exp.hpp>
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
    template <typename T, std::size_t W> pack<T, W> tanh(const pack<T, W> &x) noexcept;

    // =========================================================================
    // pack<float, 4> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE2)

#if defined(OPTINUM_HAS_SSE2)
    template <> inline pack<float, 4> tanh(const pack<float, 4> &x) noexcept {
        __m128 vx = x.data_;

        __m128 vone = _mm_set1_ps(1.0f);
        __m128 vtwo = _mm_set1_ps(2.0f);
        __m128 vthreshold = _mm_set1_ps(0.625f);
        __m128 vsaturate = _mm_set1_ps(9.0f);

        __m128 vabs_x = _mm_andnot_ps(_mm_set1_ps(-0.0f), vx);
        __m128 vsign = _mm_and_ps(vx, _mm_set1_ps(-0.0f));

        __m128 vc1 = _mm_set1_ps(1.0f);
        __m128 vc3 = _mm_set1_ps(-0.33333333333333333f);
        __m128 vc5 = _mm_set1_ps(0.13333333333333333f);
        __m128 vc7 = _mm_set1_ps(-0.05396825396825397f);

        __m128 vx2 = _mm_mul_ps(vx, vx);

#ifdef OPTINUM_HAS_FMA
        __m128 vpoly = _mm_fmadd_ps(vc7, vx2, vc5);
        vpoly = _mm_fmadd_ps(vpoly, vx2, vc3);
        vpoly = _mm_fmadd_ps(vpoly, vx2, vc1);
#else
        __m128 vpoly = _mm_add_ps(_mm_mul_ps(vc7, vx2), vc5);
        vpoly = _mm_add_ps(_mm_mul_ps(vpoly, vx2), vc3);
        vpoly = _mm_add_ps(_mm_mul_ps(vpoly, vx2), vc1);
#endif
        __m128 vtanh_small = _mm_mul_ps(vx, vpoly);

        __m128 v2x = _mm_mul_ps(vabs_x, vtwo);
        pack<float, 4> exp_2x = exp(pack<float, 4>(v2x));
        __m128 vexp_2x = exp_2x.data_;

        __m128 vtanh_large = _mm_sub_ps(vone, _mm_div_ps(vtwo, _mm_add_ps(vexp_2x, vone)));
        vtanh_large = _mm_or_ps(vtanh_large, vsign);

        __m128 vmask_small = _mm_cmplt_ps(vabs_x, vthreshold);
        __m128 vresult = _mm_blendv_ps(vtanh_large, vtanh_small, vmask_small);

        __m128 vmask_saturate = _mm_cmpgt_ps(vabs_x, vsaturate);
        __m128 vsaturated = _mm_or_ps(vone, vsign);
        vresult = _mm_blendv_ps(vresult, vsaturated, vmask_saturate);

        return pack<float, 4>(vresult);
    }

#endif // OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

#endif // OPTINUM_HAS_SSE2

#if defined(OPTINUM_HAS_AVX)
    template <> inline pack<float, 8> tanh(const pack<float, 8> &x) noexcept {
        __m256 vx = x.data_;

        // Constants
        __m256 vone = _mm256_set1_ps(1.0f);
        __m256 vtwo = _mm256_set1_ps(2.0f);
        __m256 vthreshold = _mm256_set1_ps(0.625f);
        __m256 vsaturate = _mm256_set1_ps(9.0f);

        // Absolute value
        __m256 vabs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vx);
        __m256 vsign = _mm256_and_ps(vx, _mm256_set1_ps(-0.0f));

        // Polynomial coefficients for small |x|
        // tanh(x) ≈ x * (1 - x^2/3 + 2x^4/15 - 17x^6/315)
        __m256 vc1 = _mm256_set1_ps(1.0f);
        __m256 vc3 = _mm256_set1_ps(-0.33333333333333333f); // -1/3
        __m256 vc5 = _mm256_set1_ps(0.13333333333333333f);  // 2/15
        __m256 vc7 = _mm256_set1_ps(-0.05396825396825397f); // -17/315

        __m256 vx2 = _mm256_mul_ps(vx, vx);

        // Polynomial evaluation using Horner's method
#ifdef OPTINUM_HAS_FMA
        __m256 vpoly = _mm256_fmadd_ps(vc7, vx2, vc5);
        vpoly = _mm256_fmadd_ps(vpoly, vx2, vc3);
        vpoly = _mm256_fmadd_ps(vpoly, vx2, vc1);
#else
        __m256 vpoly = _mm256_add_ps(_mm256_mul_ps(vc7, vx2), vc5);
        vpoly = _mm256_add_ps(_mm256_mul_ps(vpoly, vx2), vc3);
        vpoly = _mm256_add_ps(_mm256_mul_ps(vpoly, vx2), vc1);
#endif
        __m256 vtanh_small = _mm256_mul_ps(vx, vpoly);

        // For larger |x|: tanh(x) = 1 - 2/(exp(2x) + 1)
        // Use absolute value and restore sign at the end
        __m256 v2x = _mm256_mul_ps(vabs_x, vtwo);

        // Compute exp(2|x|) using our new exp
        pack<float, 8> exp_2x = exp(pack<float, 8>(v2x));
        __m256 vexp_2x = exp_2x.data_;

        // tanh = 1 - 2/(exp(2x) + 1)
        __m256 vtanh_large = _mm256_sub_ps(vone, _mm256_div_ps(vtwo, _mm256_add_ps(vexp_2x, vone)));

        // Restore sign
        vtanh_large = _mm256_or_ps(vtanh_large, vsign);

        // Select based on |x| < threshold
        __m256 vmask_small = _mm256_cmp_ps(vabs_x, vthreshold, _CMP_LT_OQ);
        __m256 vresult = _mm256_blendv_ps(vtanh_large, vtanh_small, vmask_small);

        // For very large |x|, saturate to ±1
        __m256 vmask_saturate = _mm256_cmp_ps(vabs_x, vsaturate, _CMP_GT_OQ);
        __m256 vsaturated = _mm256_or_ps(vone, vsign);
        vresult = _mm256_blendv_ps(vresult, vsaturated, vmask_saturate);

        return pack<float, 8>(vresult);
    }

#endif // OPTINUM_HAS_AVX

    // =========================================================================
#endif // OPTINUM_HAS_AVX

    // pack<double, 2> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE41)

    template <> inline pack<double, 2> tanh(const pack<double, 2> &x) noexcept {
        __m128d vx = x.data_;

        __m128d vone = _mm_set1_pd(1.0);
        __m128d vtwo = _mm_set1_pd(2.0);
        __m128d vthreshold = _mm_set1_pd(0.625);
        __m128d vsaturate = _mm_set1_pd(9.0);

        __m128d vabs_x = _mm_andnot_pd(_mm_set1_pd(-0.0), vx);
        __m128d vsign = _mm_and_pd(vx, _mm_set1_pd(-0.0));

        // Higher-order polynomial for double precision
        __m128d vc1 = _mm_set1_pd(1.0);
        __m128d vc3 = _mm_set1_pd(-0.33333333333333333); // -1/3
        __m128d vc5 = _mm_set1_pd(0.13333333333333333);  // 2/15
        __m128d vc7 = _mm_set1_pd(-0.05396825396825397); // -17/315
        __m128d vc9 = _mm_set1_pd(0.021869488536155203); // 62/2835

        __m128d vx2 = _mm_mul_pd(vx, vx);

#ifdef OPTINUM_HAS_FMA
        __m128d vpoly = _mm_fmadd_pd(vc9, vx2, vc7);
        vpoly = _mm_fmadd_pd(vpoly, vx2, vc5);
        vpoly = _mm_fmadd_pd(vpoly, vx2, vc3);
        vpoly = _mm_fmadd_pd(vpoly, vx2, vc1);
#else
        __m128d vpoly = _mm_add_pd(_mm_mul_pd(vc9, vx2), vc7);
        vpoly = _mm_add_pd(_mm_mul_pd(vpoly, vx2), vc5);
        vpoly = _mm_add_pd(_mm_mul_pd(vpoly, vx2), vc3);
        vpoly = _mm_add_pd(_mm_mul_pd(vpoly, vx2), vc1);
#endif
        __m128d vtanh_small = _mm_mul_pd(vx, vpoly);

        __m128d v2x = _mm_mul_pd(vabs_x, vtwo);
        pack<double, 2> exp_2x = exp(pack<double, 2>(v2x));
        __m128d vexp_2x = exp_2x.data_;

        __m128d vtanh_large = _mm_sub_pd(vone, _mm_div_pd(vtwo, _mm_add_pd(vexp_2x, vone)));
        vtanh_large = _mm_or_pd(vtanh_large, vsign);

        __m128d vmask_small = _mm_cmplt_pd(vabs_x, vthreshold);
        __m128d vresult = _mm_blendv_pd(vtanh_large, vtanh_small, vmask_small);

        __m128d vmask_saturate = _mm_cmpgt_pd(vabs_x, vsaturate);
        __m128d vsaturated = _mm_or_pd(vone, vsign);
        vresult = _mm_blendv_pd(vresult, vsaturated, vmask_saturate);

        return pack<double, 2>(vresult);
    }

#endif // OPTINUM_HAS_SSE41

    // =========================================================================
    // pack<double, 4> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<double, 4> tanh(const pack<double, 4> &x) noexcept {
        __m256d vx = x.data_;

        __m256d vone = _mm256_set1_pd(1.0);
        __m256d vtwo = _mm256_set1_pd(2.0);
        __m256d vthreshold = _mm256_set1_pd(0.625);
        __m256d vsaturate = _mm256_set1_pd(9.0);

        __m256d vabs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), vx);
        __m256d vsign = _mm256_and_pd(vx, _mm256_set1_pd(-0.0));

        // Higher-order polynomial for double precision
        __m256d vc1 = _mm256_set1_pd(1.0);
        __m256d vc3 = _mm256_set1_pd(-0.33333333333333333);
        __m256d vc5 = _mm256_set1_pd(0.13333333333333333);
        __m256d vc7 = _mm256_set1_pd(-0.05396825396825397);
        __m256d vc9 = _mm256_set1_pd(0.021869488536155203);

        __m256d vx2 = _mm256_mul_pd(vx, vx);

#ifdef OPTINUM_HAS_FMA
        __m256d vpoly = _mm256_fmadd_pd(vc9, vx2, vc7);
        vpoly = _mm256_fmadd_pd(vpoly, vx2, vc5);
        vpoly = _mm256_fmadd_pd(vpoly, vx2, vc3);
        vpoly = _mm256_fmadd_pd(vpoly, vx2, vc1);
#else
        __m256d vpoly = _mm256_add_pd(_mm256_mul_pd(vc9, vx2), vc7);
        vpoly = _mm256_add_pd(_mm256_mul_pd(vpoly, vx2), vc5);
        vpoly = _mm256_add_pd(_mm256_mul_pd(vpoly, vx2), vc3);
        vpoly = _mm256_add_pd(_mm256_mul_pd(vpoly, vx2), vc1);
#endif
        __m256d vtanh_small = _mm256_mul_pd(vx, vpoly);

        __m256d v2x = _mm256_mul_pd(vabs_x, vtwo);
        pack<double, 4> exp_2x = exp(pack<double, 4>(v2x));
        __m256d vexp_2x = exp_2x.data_;

        __m256d vtanh_large = _mm256_sub_pd(vone, _mm256_div_pd(vtwo, _mm256_add_pd(vexp_2x, vone)));
        vtanh_large = _mm256_or_pd(vtanh_large, vsign);

        __m256d vmask_small = _mm256_cmp_pd(vabs_x, vthreshold, _CMP_LT_OQ);
        __m256d vresult = _mm256_blendv_pd(vtanh_large, vtanh_small, vmask_small);

        __m256d vmask_saturate = _mm256_cmp_pd(vabs_x, vsaturate, _CMP_GT_OQ);
        __m256d vsaturated = _mm256_or_pd(vone, vsign);
        vresult = _mm256_blendv_pd(vresult, vsaturated, vmask_saturate);

        return pack<double, 4>(vresult);
    }

#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
