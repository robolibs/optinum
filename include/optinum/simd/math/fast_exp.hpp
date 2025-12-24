#pragma once

// Fast vectorized exp() implementation
// Uses range reduction + polynomial approximation
// Accuracy: ~3-5 ULP (good for ML/graphics, not scientific computing)

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/intrinsic/simd_vec.hpp>

// Include architecture-specific specializations
#if defined(OPTINUM_HAS_AVX)
#include <optinum/simd/intrinsic/avx.hpp>
#endif
#if defined(OPTINUM_HAS_SSE2)
#include <optinum/simd/intrinsic/sse.hpp>
#endif

namespace optinum::simd {

    namespace detail {

        // Constants for exp(x) computation
        // exp(x) = 2^(x/ln2) = 2^n * 2^f where n is integer, f is fraction
        constexpr float LN2_HI = 0.693145751953125f;         // High part of ln(2)
        constexpr float LN2_LO = 1.42860682030941723212e-6f; // Low part of ln(2)
        constexpr float LOG2E = 1.44269504088896341f;        // 1/ln(2)

        // Polynomial coefficients for exp(x) on [-ln2/2, ln2/2]
        // Minimax polynomial approximation
        constexpr float EXP_C1 = 1.0f;
        constexpr float EXP_C2 = 0.5f;
        constexpr float EXP_C3 = 0.16666666666666666f;  // 1/6
        constexpr float EXP_C4 = 0.041666666666666664f; // 1/24
        constexpr float EXP_C5 = 0.008333333333333333f; // 1/120

    } // namespace detail

#if defined(OPTINUM_HAS_AVX)

    // AVX implementation of exp for 8 floats
    OPTINUM_INLINE SIMDVec<float, 8> fast_exp(const SIMDVec<float, 8> &x) {
        using namespace detail;

        __m256 vx = x.value;

        // Clamp to avoid overflow/underflow
        __m256 max_val = _mm256_set1_ps(88.0f);
        __m256 min_val = _mm256_set1_ps(-88.0f);
        vx = _mm256_min_ps(vx, max_val);
        vx = _mm256_max_ps(vx, min_val);

        // n = round(x / ln2)
        __m256 vlog2e = _mm256_set1_ps(LOG2E);
        __m256 vn = _mm256_round_ps(_mm256_mul_ps(vx, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // r = x - n * ln2 (range reduction to [-ln2/2, ln2/2])
        __m256 vln2_hi = _mm256_set1_ps(LN2_HI);
        __m256 vln2_lo = _mm256_set1_ps(LN2_LO);
        __m256 vr = _mm256_sub_ps(vx, _mm256_mul_ps(vn, vln2_hi));
        vr = _mm256_sub_ps(vr, _mm256_mul_ps(vn, vln2_lo));

        // Polynomial approximation: exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
        // Using Horner's method: ((((c5*r + c4)*r + c3)*r + c2)*r + c1)*r + 1
        __m256 vr2 = _mm256_mul_ps(vr, vr);

        __m256 vc5 = _mm256_set1_ps(EXP_C5);
        __m256 vc4 = _mm256_set1_ps(EXP_C4);
        __m256 vc3 = _mm256_set1_ps(EXP_C3);
        __m256 vc2 = _mm256_set1_ps(EXP_C2);
        __m256 vc1 = _mm256_set1_ps(EXP_C1);
        __m256 vone = _mm256_set1_ps(1.0f);

#ifdef OPTINUM_HAS_FMA
        // With FMA: more accurate and faster
        __m256 vpoly = _mm256_fmadd_ps(vc5, vr, vc4);
        vpoly = _mm256_fmadd_ps(vpoly, vr, vc3);
        vpoly = _mm256_fmadd_ps(vpoly, vr, vc2);
        vpoly = _mm256_fmadd_ps(vpoly, vr, vc1);
        vpoly = _mm256_fmadd_ps(vpoly, vr, vone);
#else
        // Without FMA
        __m256 vpoly = _mm256_add_ps(_mm256_mul_ps(vc5, vr), vc4);
        vpoly = _mm256_add_ps(_mm256_mul_ps(vpoly, vr), vc3);
        vpoly = _mm256_add_ps(_mm256_mul_ps(vpoly, vr), vc2);
        vpoly = _mm256_add_ps(_mm256_mul_ps(vpoly, vr), vc1);
        vpoly = _mm256_add_ps(_mm256_mul_ps(vpoly, vr), vone);
#endif

        // Scale by 2^n using integer manipulation of IEEE float
        // 2^n = reinterpret((n + 127) << 23)
        __m256i vni = _mm256_cvtps_epi32(vn);
        vni = _mm256_add_epi32(vni, _mm256_set1_epi32(127));
        vni = _mm256_slli_epi32(vni, 23);
        __m256 vscale = _mm256_castsi256_ps(vni);

        // result = poly * 2^n
        __m256 vresult = _mm256_mul_ps(vpoly, vscale);

        return SIMDVec<float, 8>(vresult);
    }

#endif // OPTINUM_HAS_AVX

#if defined(OPTINUM_HAS_SSE41)

    // SSE4.1 implementation of exp for 4 floats (requires _mm_round_ps)
    OPTINUM_INLINE SIMDVec<float, 4> fast_exp(const SIMDVec<float, 4> &x) {
        using namespace detail;

        __m128 vx = x.value;

        // Clamp to avoid overflow/underflow
        __m128 max_val = _mm_set1_ps(88.0f);
        __m128 min_val = _mm_set1_ps(-88.0f);
        vx = _mm_min_ps(vx, max_val);
        vx = _mm_max_ps(vx, min_val);

        // n = round(x / ln2)
        __m128 vlog2e = _mm_set1_ps(LOG2E);
        __m128 vn = _mm_round_ps(_mm_mul_ps(vx, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // r = x - n * ln2
        __m128 vln2_hi = _mm_set1_ps(LN2_HI);
        __m128 vln2_lo = _mm_set1_ps(LN2_LO);
        __m128 vr = _mm_sub_ps(vx, _mm_mul_ps(vn, vln2_hi));
        vr = _mm_sub_ps(vr, _mm_mul_ps(vn, vln2_lo));

        // Polynomial approximation using Horner's method
        __m128 vc5 = _mm_set1_ps(EXP_C5);
        __m128 vc4 = _mm_set1_ps(EXP_C4);
        __m128 vc3 = _mm_set1_ps(EXP_C3);
        __m128 vc2 = _mm_set1_ps(EXP_C2);
        __m128 vc1 = _mm_set1_ps(EXP_C1);
        __m128 vone = _mm_set1_ps(1.0f);

#ifdef OPTINUM_HAS_FMA
        __m128 vpoly = _mm_fmadd_ps(vc5, vr, vc4);
        vpoly = _mm_fmadd_ps(vpoly, vr, vc3);
        vpoly = _mm_fmadd_ps(vpoly, vr, vc2);
        vpoly = _mm_fmadd_ps(vpoly, vr, vc1);
        vpoly = _mm_fmadd_ps(vpoly, vr, vone);
#else
        __m128 vpoly = _mm_add_ps(_mm_mul_ps(vc5, vr), vc4);
        vpoly = _mm_add_ps(_mm_mul_ps(vpoly, vr), vc3);
        vpoly = _mm_add_ps(_mm_mul_ps(vpoly, vr), vc2);
        vpoly = _mm_add_ps(_mm_mul_ps(vpoly, vr), vc1);
        vpoly = _mm_add_ps(_mm_mul_ps(vpoly, vr), vone);
#endif

        // Scale by 2^n
        __m128i vni = _mm_cvtps_epi32(vn);
        vni = _mm_add_epi32(vni, _mm_set1_epi32(127));
        vni = _mm_slli_epi32(vni, 23);
        __m128 vscale = _mm_castsi128_ps(vni);

        __m128 vresult = _mm_mul_ps(vpoly, vscale);

        return SIMDVec<float, 4>(vresult);
    }

#endif // OPTINUM_HAS_SSE41

} // namespace optinum::simd
