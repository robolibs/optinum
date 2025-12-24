#pragma once

// Fast vectorized hyperbolic functions: tanh, sinh, cosh
// tanh is critical for neural networks (activation function)

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/intrinsic/simd_vec.hpp>
#include <optinum/simd/math/detail/constants.hpp>
#include <optinum/simd/math/fast_exp.hpp>

#if defined(OPTINUM_HAS_AVX)
#include <optinum/simd/intrinsic/avx.hpp>
#endif
#if defined(OPTINUM_HAS_SSE2)
#include <optinum/simd/intrinsic/sse.hpp>
#endif

namespace optinum::simd {

    // ============================================================================
    // Algorithm for tanh(x):
    //
    // For |x| < 0.625:
    //   tanh(x) ≈ x * (1 - x^2/3 + 2x^4/15 - 17x^6/315 + ...)
    //   (polynomial approximation)
    //
    // For |x| >= 0.625:
    //   tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    //   Or equivalently: tanh(x) = 1 - 2/(exp(2x) + 1)  (more stable for large x)
    //   Or: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    //
    // For |x| > 9: tanh(x) ≈ sign(x) (saturates to ±1)
    // ============================================================================

#if defined(OPTINUM_HAS_AVX)

    OPTINUM_INLINE SIMDVec<float, 8> fast_tanh(const SIMDVec<float, 8> &x) {
        __m256 vx = x.value;

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

        // Compute exp(2|x|) using fast_exp
        SIMDVec<float, 8> exp_2x = fast_exp(SIMDVec<float, 8>(v2x));
        __m256 vexp_2x = exp_2x.value;

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

        return SIMDVec<float, 8>(vresult);
    }

    OPTINUM_INLINE SIMDVec<float, 8> fast_sinh(const SIMDVec<float, 8> &x) {
        // sinh(x) = (exp(x) - exp(-x)) / 2
        __m256 vx = x.value;
        __m256 vhalf = _mm256_set1_ps(0.5f);

        SIMDVec<float, 8> exp_pos = fast_exp(x);
        SIMDVec<float, 8> exp_neg = fast_exp(SIMDVec<float, 8>(_mm256_sub_ps(_mm256_setzero_ps(), vx)));

        __m256 vresult = _mm256_mul_ps(_mm256_sub_ps(exp_pos.value, exp_neg.value), vhalf);

        return SIMDVec<float, 8>(vresult);
    }

    OPTINUM_INLINE SIMDVec<float, 8> fast_cosh(const SIMDVec<float, 8> &x) {
        // cosh(x) = (exp(x) + exp(-x)) / 2
        __m256 vx = x.value;
        __m256 vhalf = _mm256_set1_ps(0.5f);

        SIMDVec<float, 8> exp_pos = fast_exp(x);
        SIMDVec<float, 8> exp_neg = fast_exp(SIMDVec<float, 8>(_mm256_sub_ps(_mm256_setzero_ps(), vx)));

        __m256 vresult = _mm256_mul_ps(_mm256_add_ps(exp_pos.value, exp_neg.value), vhalf);

        return SIMDVec<float, 8>(vresult);
    }

#endif // OPTINUM_HAS_AVX

#if defined(OPTINUM_HAS_SSE41)

    OPTINUM_INLINE SIMDVec<float, 4> fast_tanh(const SIMDVec<float, 4> &x) {
        __m128 vx = x.value;

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
        SIMDVec<float, 4> exp_2x = fast_exp(SIMDVec<float, 4>(v2x));
        __m128 vexp_2x = exp_2x.value;

        __m128 vtanh_large = _mm_sub_ps(vone, _mm_div_ps(vtwo, _mm_add_ps(vexp_2x, vone)));
        vtanh_large = _mm_or_ps(vtanh_large, vsign);

        __m128 vmask_small = _mm_cmplt_ps(vabs_x, vthreshold);
        __m128 vresult = _mm_blendv_ps(vtanh_large, vtanh_small, vmask_small);

        __m128 vmask_saturate = _mm_cmpgt_ps(vabs_x, vsaturate);
        __m128 vsaturated = _mm_or_ps(vone, vsign);
        vresult = _mm_blendv_ps(vresult, vsaturated, vmask_saturate);

        return SIMDVec<float, 4>(vresult);
    }

    OPTINUM_INLINE SIMDVec<float, 4> fast_sinh(const SIMDVec<float, 4> &x) {
        __m128 vx = x.value;
        __m128 vhalf = _mm_set1_ps(0.5f);

        SIMDVec<float, 4> exp_pos = fast_exp(x);
        SIMDVec<float, 4> exp_neg = fast_exp(SIMDVec<float, 4>(_mm_sub_ps(_mm_setzero_ps(), vx)));

        __m128 vresult = _mm_mul_ps(_mm_sub_ps(exp_pos.value, exp_neg.value), vhalf);

        return SIMDVec<float, 4>(vresult);
    }

    OPTINUM_INLINE SIMDVec<float, 4> fast_cosh(const SIMDVec<float, 4> &x) {
        __m128 vx = x.value;
        __m128 vhalf = _mm_set1_ps(0.5f);

        SIMDVec<float, 4> exp_pos = fast_exp(x);
        SIMDVec<float, 4> exp_neg = fast_exp(SIMDVec<float, 4>(_mm_sub_ps(_mm_setzero_ps(), vx)));

        __m128 vresult = _mm_mul_ps(_mm_add_ps(exp_pos.value, exp_neg.value), vhalf);

        return SIMDVec<float, 4>(vresult);
    }

#endif // OPTINUM_HAS_SSE41

} // namespace optinum::simd
