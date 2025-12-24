#pragma once

// Fast vectorized log() implementation
// Uses range reduction + polynomial approximation
// Accuracy: ~3-5 ULP (good for ML/graphics, not scientific computing)

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/intrinsic/simd_vec.hpp>
#include <optinum/simd/math/detail/constants.hpp>

#if defined(OPTINUM_HAS_AVX)
#include <optinum/simd/intrinsic/avx.hpp>
#endif
#if defined(OPTINUM_HAS_SSE2)
#include <optinum/simd/intrinsic/sse.hpp>
#endif

namespace optinum::simd {

    // ============================================================================
    // Algorithm for log(x):
    //
    // 1. Extract exponent and mantissa: x = 2^n * m, where m in [1, 2)
    // 2. If m > sqrt(2), adjust: m' = m/2, n' = n+1
    // 3. Compute f = m - 1, so f in [sqrt(2)/2 - 1, sqrt(2) - 1] ≈ [-0.29, 0.41]
    // 4. Use s = f / (2 + f) to map to smaller range
    // 5. log(m) = log(1+f) = 2*s + 2*s^3/3 + 2*s^5/5 + ...
    // 6. result = n * ln(2) + log(m)
    // ============================================================================

#if defined(OPTINUM_HAS_AVX)

    OPTINUM_INLINE SIMDVec<float, 8> fast_log(const SIMDVec<float, 8> &x) {
        using namespace math_constants;

        __m256 vx = x.value;

        // Constants
        __m256 vone = _mm256_set1_ps(1.0f);
        __m256 vhalf = _mm256_set1_ps(0.5f);
        __m256 vsqrt2 = _mm256_set1_ps(SQRT2_F);
        __m256 vln2 = _mm256_set1_ps(LN2_F);

        // Extract exponent: floor(log2(x))
        // IEEE 754: bits 23-30 are exponent (biased by 127)
        __m256i vi = _mm256_castps_si256(vx);
        __m256i vexp = _mm256_srli_epi32(vi, 23);
        vexp = _mm256_sub_epi32(vexp, _mm256_set1_epi32(127));
        __m256 vn = _mm256_cvtepi32_ps(vexp);

        // Extract mantissa and set exponent to 0 (so value is in [1, 2))
        __m256i vmant_mask = _mm256_set1_epi32(0x007FFFFF);
        __m256i vexp_zero = _mm256_set1_epi32(0x3F800000); // exponent = 0 -> 1.0
        __m256i vmant = _mm256_or_si256(_mm256_and_si256(vi, vmant_mask), vexp_zero);
        __m256 vm = _mm256_castsi256_ps(vmant);

        // If m > sqrt(2), use m/2 and n+1
        __m256 vmask = _mm256_cmp_ps(vm, vsqrt2, _CMP_GT_OQ);
        vm = _mm256_blendv_ps(vm, _mm256_mul_ps(vm, vhalf), vmask);
        vn = _mm256_blendv_ps(vn, _mm256_add_ps(vn, vone), vmask);

        // f = m - 1
        __m256 vf = _mm256_sub_ps(vm, vone);

        // s = f / (2 + f)
        __m256 vs = _mm256_div_ps(vf, _mm256_add_ps(_mm256_set1_ps(2.0f), vf));
        __m256 vs2 = _mm256_mul_ps(vs, vs);

        // Polynomial: log(1+f) ≈ 2*s * (1 + s^2/3 + s^4/5 + s^6/7 + s^8/9)
        // = 2*s * P(s^2) where P(z) = 1 + z/3 + z^2/5 + z^3/7 + z^4/9
        __m256 vc9 = _mm256_set1_ps(0.11111111111111111f); // 1/9
        __m256 vc7 = _mm256_set1_ps(0.14285714285714285f); // 1/7
        __m256 vc5 = _mm256_set1_ps(0.2f);                 // 1/5
        __m256 vc3 = _mm256_set1_ps(0.33333333333333333f); // 1/3

#ifdef OPTINUM_HAS_FMA
        __m256 vp = _mm256_fmadd_ps(vc9, vs2, vc7);
        vp = _mm256_fmadd_ps(vp, vs2, vc5);
        vp = _mm256_fmadd_ps(vp, vs2, vc3);
        vp = _mm256_fmadd_ps(vp, vs2, vone);
#else
        __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc9, vs2), vc7);
        vp = _mm256_add_ps(_mm256_mul_ps(vp, vs2), vc5);
        vp = _mm256_add_ps(_mm256_mul_ps(vp, vs2), vc3);
        vp = _mm256_add_ps(_mm256_mul_ps(vp, vs2), vone);
#endif

        // log(m) = 2 * s * P(s^2)
        __m256 vlog_m = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), vs), vp);

        // result = n * ln(2) + log(m)
#ifdef OPTINUM_HAS_FMA
        __m256 vresult = _mm256_fmadd_ps(vn, vln2, vlog_m);
#else
        __m256 vresult = _mm256_add_ps(_mm256_mul_ps(vn, vln2), vlog_m);
#endif

        // Handle special cases: x <= 0 -> NaN, x == 0 -> -inf
        __m256 vzero = _mm256_setzero_ps();
        __m256 vneg_inf = _mm256_set1_ps(-__builtin_inff());
        __m256 vnan = _mm256_set1_ps(__builtin_nanf(""));

        vresult = _mm256_blendv_ps(vresult, vneg_inf, _mm256_cmp_ps(vx, vzero, _CMP_EQ_OQ));
        vresult = _mm256_blendv_ps(vresult, vnan, _mm256_cmp_ps(vx, vzero, _CMP_LT_OQ));

        return SIMDVec<float, 8>(vresult);
    }

#endif // OPTINUM_HAS_AVX

#if defined(OPTINUM_HAS_SSE41)

    OPTINUM_INLINE SIMDVec<float, 4> fast_log(const SIMDVec<float, 4> &x) {
        using namespace math_constants;

        __m128 vx = x.value;

        __m128 vone = _mm_set1_ps(1.0f);
        __m128 vhalf = _mm_set1_ps(0.5f);
        __m128 vsqrt2 = _mm_set1_ps(SQRT2_F);
        __m128 vln2 = _mm_set1_ps(LN2_F);

        // Extract exponent
        __m128i vi = _mm_castps_si128(vx);
        __m128i vexp = _mm_srli_epi32(vi, 23);
        vexp = _mm_sub_epi32(vexp, _mm_set1_epi32(127));
        __m128 vn = _mm_cvtepi32_ps(vexp);

        // Extract mantissa
        __m128i vmant_mask = _mm_set1_epi32(0x007FFFFF);
        __m128i vexp_zero = _mm_set1_epi32(0x3F800000);
        __m128i vmant = _mm_or_si128(_mm_and_si128(vi, vmant_mask), vexp_zero);
        __m128 vm = _mm_castsi128_ps(vmant);

        // If m > sqrt(2), adjust
        __m128 vmask = _mm_cmpgt_ps(vm, vsqrt2);
        vm = _mm_blendv_ps(vm, _mm_mul_ps(vm, vhalf), vmask);
        vn = _mm_blendv_ps(vn, _mm_add_ps(vn, vone), vmask);

        // f = m - 1
        __m128 vf = _mm_sub_ps(vm, vone);

        // s = f / (2 + f)
        __m128 vs = _mm_div_ps(vf, _mm_add_ps(_mm_set1_ps(2.0f), vf));
        __m128 vs2 = _mm_mul_ps(vs, vs);

        // Polynomial
        __m128 vc9 = _mm_set1_ps(0.11111111111111111f);
        __m128 vc7 = _mm_set1_ps(0.14285714285714285f);
        __m128 vc5 = _mm_set1_ps(0.2f);
        __m128 vc3 = _mm_set1_ps(0.33333333333333333f);

#ifdef OPTINUM_HAS_FMA
        __m128 vp = _mm_fmadd_ps(vc9, vs2, vc7);
        vp = _mm_fmadd_ps(vp, vs2, vc5);
        vp = _mm_fmadd_ps(vp, vs2, vc3);
        vp = _mm_fmadd_ps(vp, vs2, vone);
#else
        __m128 vp = _mm_add_ps(_mm_mul_ps(vc9, vs2), vc7);
        vp = _mm_add_ps(_mm_mul_ps(vp, vs2), vc5);
        vp = _mm_add_ps(_mm_mul_ps(vp, vs2), vc3);
        vp = _mm_add_ps(_mm_mul_ps(vp, vs2), vone);
#endif

        __m128 vlog_m = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(2.0f), vs), vp);

#ifdef OPTINUM_HAS_FMA
        __m128 vresult = _mm_fmadd_ps(vn, vln2, vlog_m);
#else
        __m128 vresult = _mm_add_ps(_mm_mul_ps(vn, vln2), vlog_m);
#endif

        // Handle special cases
        __m128 vzero = _mm_setzero_ps();
        __m128 vneg_inf = _mm_set1_ps(-__builtin_inff());
        __m128 vnan = _mm_set1_ps(__builtin_nanf(""));

        vresult = _mm_blendv_ps(vresult, vneg_inf, _mm_cmpeq_ps(vx, vzero));
        vresult = _mm_blendv_ps(vresult, vnan, _mm_cmplt_ps(vx, vzero));

        return SIMDVec<float, 4>(vresult);
    }

#endif // OPTINUM_HAS_SSE41

} // namespace optinum::simd
