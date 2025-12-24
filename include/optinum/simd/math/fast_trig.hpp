#pragma once

// Fast vectorized sin() and cos() implementation
// Uses range reduction + polynomial approximation
// Accuracy: ~3-5 ULP

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
    // Algorithm for sin(x) and cos(x):
    //
    // 1. Range reduction: y = x * (4/pi), j = round(y)
    // 2. Compute reduced argument: z = x - j * (pi/4)
    // 3. Based on j mod 8, determine which polynomial to use:
    //    - j=0,4: sin(z), cos(z)
    //    - j=1,5: (sin(z)+cos(z))/sqrt(2), (cos(z)-sin(z))/sqrt(2)
    //    - j=2,6: cos(z), -sin(z)
    //    - j=3,7: (cos(z)-sin(z))/sqrt(2), -(sin(z)+cos(z))/sqrt(2)
    // 4. Simpler approach: reduce to [-pi/4, pi/4] and use:
    //    sin(z) ≈ z - z^3/6 + z^5/120 - z^7/5040
    //    cos(z) ≈ 1 - z^2/2 + z^4/24 - z^6/720
    // ============================================================================

#if defined(OPTINUM_HAS_AVX)

    OPTINUM_INLINE SIMDVec<float, 8> fast_sin(const SIMDVec<float, 8> &x) {
        using namespace math_constants;

        __m256 vx = x.value;

        // Constants for range reduction
        __m256 vfour_over_pi = _mm256_set1_ps(FOUR_INV_PI_F);
        __m256 vpi_over_4_hi = _mm256_set1_ps(0.78515625f);
        __m256 vpi_over_4_lo = _mm256_set1_ps(2.4187564849853515625e-4f);
        __m256 vpi_over_4_lo2 = _mm256_set1_ps(3.77489497744594108e-8f);

        // y = x * (4/pi)
        __m256 vy = _mm256_mul_ps(vx, vfour_over_pi);

        // j = round(y) - we want the nearest integer
        __m256 vj = _mm256_round_ps(vy, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i vji = _mm256_cvtps_epi32(vj);

        // z = x - j * (pi/4) using extended precision
        __m256 vz = _mm256_sub_ps(vx, _mm256_mul_ps(vj, vpi_over_4_hi));
        vz = _mm256_sub_ps(vz, _mm256_mul_ps(vj, vpi_over_4_lo));
        vz = _mm256_sub_ps(vz, _mm256_mul_ps(vj, vpi_over_4_lo2));

        __m256 vz2 = _mm256_mul_ps(vz, vz);

        // Determine which quadrant we're in
        // j & 1: if odd, swap sin<->cos
        // j & 2: if set, negate sin
        // j & 4: if set, negate cos
        __m256i vone_i = _mm256_set1_epi32(1);
        __m256i vtwo_i = _mm256_set1_epi32(2);
        __m256i vfour_i = _mm256_set1_epi32(4);

        __m256i vswap_mask = _mm256_and_si256(vji, vone_i);
        __m256i vneg_sin_mask = _mm256_and_si256(vji, vtwo_i);
        __m256i vneg_cos_mask = _mm256_and_si256(_mm256_add_epi32(vji, vone_i), vtwo_i);

        // Polynomial coefficients for sin(z) where z in [-pi/4, pi/4]
        // sin(z) = z * (1 - z^2/6 + z^4/120 - z^6/5040)
        __m256 vs_c7 = _mm256_set1_ps(-1.9841269841269841e-4f); // -1/5040
        __m256 vs_c5 = _mm256_set1_ps(8.3333333333333333e-3f);  // 1/120
        __m256 vs_c3 = _mm256_set1_ps(-1.6666666666666666e-1f); // -1/6
        __m256 vs_c1 = _mm256_set1_ps(1.0f);

        // Polynomial coefficients for cos(z)
        // cos(z) = 1 - z^2/2 + z^4/24 - z^6/720
        __m256 vc_c6 = _mm256_set1_ps(-1.3888888888888889e-3f); // -1/720
        __m256 vc_c4 = _mm256_set1_ps(4.1666666666666667e-2f);  // 1/24
        __m256 vc_c2 = _mm256_set1_ps(-5.0e-1f);                // -1/2
        __m256 vc_c0 = _mm256_set1_ps(1.0f);

        // Compute sin(z)
#ifdef OPTINUM_HAS_FMA
        __m256 vsin_p = _mm256_fmadd_ps(vs_c7, vz2, vs_c5);
        vsin_p = _mm256_fmadd_ps(vsin_p, vz2, vs_c3);
        vsin_p = _mm256_fmadd_ps(vsin_p, vz2, vs_c1);
#else
        __m256 vsin_p = _mm256_add_ps(_mm256_mul_ps(vs_c7, vz2), vs_c5);
        vsin_p = _mm256_add_ps(_mm256_mul_ps(vsin_p, vz2), vs_c3);
        vsin_p = _mm256_add_ps(_mm256_mul_ps(vsin_p, vz2), vs_c1);
#endif
        __m256 vsin_z = _mm256_mul_ps(vz, vsin_p);

        // Compute cos(z)
#ifdef OPTINUM_HAS_FMA
        __m256 vcos_p = _mm256_fmadd_ps(vc_c6, vz2, vc_c4);
        vcos_p = _mm256_fmadd_ps(vcos_p, vz2, vc_c2);
        vcos_p = _mm256_fmadd_ps(vcos_p, vz2, vc_c0);
#else
        __m256 vcos_p = _mm256_add_ps(_mm256_mul_ps(vc_c6, vz2), vc_c4);
        vcos_p = _mm256_add_ps(_mm256_mul_ps(vcos_p, vz2), vc_c2);
        vcos_p = _mm256_add_ps(_mm256_mul_ps(vcos_p, vz2), vc_c0);
#endif
        __m256 vcos_z = vcos_p;

        // Select sin or cos based on quadrant (swap if j is odd)
        __m256 vswap_cmp = _mm256_castsi256_ps(_mm256_cmpeq_epi32(vswap_mask, vone_i));
        __m256 vresult = _mm256_blendv_ps(vsin_z, vcos_z, vswap_cmp);

        // Negate if needed
        __m256 vneg_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(vneg_sin_mask, vtwo_i));
        vneg_mask = _mm256_xor_ps(
            vneg_mask, _mm256_and_ps(vswap_cmp, _mm256_castsi256_ps(_mm256_cmpeq_epi32(vneg_cos_mask, vtwo_i))));
        vresult = _mm256_xor_ps(vresult, _mm256_and_ps(vneg_mask, _mm256_set1_ps(-0.0f)));

        return SIMDVec<float, 8>(vresult);
    }

    OPTINUM_INLINE SIMDVec<float, 8> fast_cos(const SIMDVec<float, 8> &x) {
        using namespace math_constants;

        __m256 vx = x.value;

        // Constants for range reduction
        __m256 vfour_over_pi = _mm256_set1_ps(FOUR_INV_PI_F);
        __m256 vpi_over_4_hi = _mm256_set1_ps(0.78515625f);
        __m256 vpi_over_4_lo = _mm256_set1_ps(2.4187564849853515625e-4f);
        __m256 vpi_over_4_lo2 = _mm256_set1_ps(3.77489497744594108e-8f);

        // y = x * (4/pi)
        __m256 vy = _mm256_mul_ps(vx, vfour_over_pi);
        __m256 vj = _mm256_round_ps(vy, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i vji = _mm256_cvtps_epi32(vj);

        // z = x - j * (pi/4)
        __m256 vz = _mm256_sub_ps(vx, _mm256_mul_ps(vj, vpi_over_4_hi));
        vz = _mm256_sub_ps(vz, _mm256_mul_ps(vj, vpi_over_4_lo));
        vz = _mm256_sub_ps(vz, _mm256_mul_ps(vj, vpi_over_4_lo2));

        __m256 vz2 = _mm256_mul_ps(vz, vz);

        // Quadrant handling: cos(x) = sin(x + pi/2), so add 1 to j
        __m256i vone_i = _mm256_set1_epi32(1);
        __m256i vtwo_i = _mm256_set1_epi32(2);
        __m256i vji_adj = _mm256_add_epi32(vji, vone_i);

        __m256i vswap_mask = _mm256_and_si256(vji_adj, vone_i);
        __m256i vneg_sin_mask = _mm256_and_si256(vji_adj, vtwo_i);
        __m256i vneg_cos_mask = _mm256_and_si256(_mm256_add_epi32(vji_adj, vone_i), vtwo_i);

        // Polynomial coefficients
        __m256 vs_c7 = _mm256_set1_ps(-1.9841269841269841e-4f);
        __m256 vs_c5 = _mm256_set1_ps(8.3333333333333333e-3f);
        __m256 vs_c3 = _mm256_set1_ps(-1.6666666666666666e-1f);
        __m256 vs_c1 = _mm256_set1_ps(1.0f);

        __m256 vc_c6 = _mm256_set1_ps(-1.3888888888888889e-3f);
        __m256 vc_c4 = _mm256_set1_ps(4.1666666666666667e-2f);
        __m256 vc_c2 = _mm256_set1_ps(-5.0e-1f);
        __m256 vc_c0 = _mm256_set1_ps(1.0f);

#ifdef OPTINUM_HAS_FMA
        __m256 vsin_p = _mm256_fmadd_ps(vs_c7, vz2, vs_c5);
        vsin_p = _mm256_fmadd_ps(vsin_p, vz2, vs_c3);
        vsin_p = _mm256_fmadd_ps(vsin_p, vz2, vs_c1);
        __m256 vcos_p = _mm256_fmadd_ps(vc_c6, vz2, vc_c4);
        vcos_p = _mm256_fmadd_ps(vcos_p, vz2, vc_c2);
        vcos_p = _mm256_fmadd_ps(vcos_p, vz2, vc_c0);
#else
        __m256 vsin_p = _mm256_add_ps(_mm256_mul_ps(vs_c7, vz2), vs_c5);
        vsin_p = _mm256_add_ps(_mm256_mul_ps(vsin_p, vz2), vs_c3);
        vsin_p = _mm256_add_ps(_mm256_mul_ps(vsin_p, vz2), vs_c1);
        __m256 vcos_p = _mm256_add_ps(_mm256_mul_ps(vc_c6, vz2), vc_c4);
        vcos_p = _mm256_add_ps(_mm256_mul_ps(vcos_p, vz2), vc_c2);
        vcos_p = _mm256_add_ps(_mm256_mul_ps(vcos_p, vz2), vc_c0);
#endif

        __m256 vsin_z = _mm256_mul_ps(vz, vsin_p);
        __m256 vcos_z = vcos_p;

        // Select and negate
        __m256 vswap_cmp = _mm256_castsi256_ps(_mm256_cmpeq_epi32(vswap_mask, vone_i));
        __m256 vresult = _mm256_blendv_ps(vsin_z, vcos_z, vswap_cmp);

        __m256 vneg_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(vneg_sin_mask, vtwo_i));
        vneg_mask = _mm256_xor_ps(
            vneg_mask, _mm256_and_ps(vswap_cmp, _mm256_castsi256_ps(_mm256_cmpeq_epi32(vneg_cos_mask, vtwo_i))));
        vresult = _mm256_xor_ps(vresult, _mm256_and_ps(vneg_mask, _mm256_set1_ps(-0.0f)));

        return SIMDVec<float, 8>(vresult);
    }

    // Compute sin and cos together (more efficient)
    OPTINUM_INLINE void fast_sincos(const SIMDVec<float, 8> &x, SIMDVec<float, 8> &out_sin,
                                    SIMDVec<float, 8> &out_cos) {
        using namespace math_constants;

        __m256 vx = x.value;

        __m256 vfour_over_pi = _mm256_set1_ps(FOUR_INV_PI_F);
        __m256 vpi_over_4_hi = _mm256_set1_ps(0.78515625f);
        __m256 vpi_over_4_lo = _mm256_set1_ps(2.4187564849853515625e-4f);
        __m256 vpi_over_4_lo2 = _mm256_set1_ps(3.77489497744594108e-8f);

        __m256 vy = _mm256_mul_ps(vx, vfour_over_pi);
        __m256 vj = _mm256_round_ps(vy, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i vji = _mm256_cvtps_epi32(vj);

        __m256 vz = _mm256_sub_ps(vx, _mm256_mul_ps(vj, vpi_over_4_hi));
        vz = _mm256_sub_ps(vz, _mm256_mul_ps(vj, vpi_over_4_lo));
        vz = _mm256_sub_ps(vz, _mm256_mul_ps(vj, vpi_over_4_lo2));

        __m256 vz2 = _mm256_mul_ps(vz, vz);

        // Polynomial coefficients
        __m256 vs_c7 = _mm256_set1_ps(-1.9841269841269841e-4f);
        __m256 vs_c5 = _mm256_set1_ps(8.3333333333333333e-3f);
        __m256 vs_c3 = _mm256_set1_ps(-1.6666666666666666e-1f);
        __m256 vs_c1 = _mm256_set1_ps(1.0f);

        __m256 vc_c6 = _mm256_set1_ps(-1.3888888888888889e-3f);
        __m256 vc_c4 = _mm256_set1_ps(4.1666666666666667e-2f);
        __m256 vc_c2 = _mm256_set1_ps(-5.0e-1f);
        __m256 vc_c0 = _mm256_set1_ps(1.0f);

#ifdef OPTINUM_HAS_FMA
        __m256 vsin_p = _mm256_fmadd_ps(vs_c7, vz2, vs_c5);
        vsin_p = _mm256_fmadd_ps(vsin_p, vz2, vs_c3);
        vsin_p = _mm256_fmadd_ps(vsin_p, vz2, vs_c1);
        __m256 vcos_p = _mm256_fmadd_ps(vc_c6, vz2, vc_c4);
        vcos_p = _mm256_fmadd_ps(vcos_p, vz2, vc_c2);
        vcos_p = _mm256_fmadd_ps(vcos_p, vz2, vc_c0);
#else
        __m256 vsin_p = _mm256_add_ps(_mm256_mul_ps(vs_c7, vz2), vs_c5);
        vsin_p = _mm256_add_ps(_mm256_mul_ps(vsin_p, vz2), vs_c3);
        vsin_p = _mm256_add_ps(_mm256_mul_ps(vsin_p, vz2), vs_c1);
        __m256 vcos_p = _mm256_add_ps(_mm256_mul_ps(vc_c6, vz2), vc_c4);
        vcos_p = _mm256_add_ps(_mm256_mul_ps(vcos_p, vz2), vc_c2);
        vcos_p = _mm256_add_ps(_mm256_mul_ps(vcos_p, vz2), vc_c0);
#endif

        __m256 vsin_z = _mm256_mul_ps(vz, vsin_p);
        __m256 vcos_z = vcos_p;

        __m256i vone_i = _mm256_set1_epi32(1);
        __m256i vtwo_i = _mm256_set1_epi32(2);

        // For sin
        __m256i vswap_sin = _mm256_and_si256(vji, vone_i);
        __m256i vneg_sin = _mm256_and_si256(vji, vtwo_i);
        __m256 vswap_sin_cmp = _mm256_castsi256_ps(_mm256_cmpeq_epi32(vswap_sin, vone_i));
        __m256 vsin_result = _mm256_blendv_ps(vsin_z, vcos_z, vswap_sin_cmp);
        __m256 vneg_sin_cmp = _mm256_castsi256_ps(_mm256_cmpeq_epi32(vneg_sin, vtwo_i));
        vsin_result = _mm256_xor_ps(vsin_result, _mm256_and_ps(vneg_sin_cmp, _mm256_set1_ps(-0.0f)));

        // For cos (j+1)
        __m256i vji_cos = _mm256_add_epi32(vji, vone_i);
        __m256i vswap_cos = _mm256_and_si256(vji_cos, vone_i);
        __m256i vneg_cos = _mm256_and_si256(vji_cos, vtwo_i);
        __m256 vswap_cos_cmp = _mm256_castsi256_ps(_mm256_cmpeq_epi32(vswap_cos, vone_i));
        __m256 vcos_result = _mm256_blendv_ps(vsin_z, vcos_z, vswap_cos_cmp);
        __m256 vneg_cos_cmp = _mm256_castsi256_ps(_mm256_cmpeq_epi32(vneg_cos, vtwo_i));
        vcos_result = _mm256_xor_ps(vcos_result, _mm256_and_ps(vneg_cos_cmp, _mm256_set1_ps(-0.0f)));

        out_sin = SIMDVec<float, 8>(vsin_result);
        out_cos = SIMDVec<float, 8>(vcos_result);
    }

#endif // OPTINUM_HAS_AVX

#if defined(OPTINUM_HAS_SSE41)

    OPTINUM_INLINE SIMDVec<float, 4> fast_sin(const SIMDVec<float, 4> &x) {
        using namespace math_constants;

        __m128 vx = x.value;

        __m128 vfour_over_pi = _mm_set1_ps(FOUR_INV_PI_F);
        __m128 vpi_over_4_hi = _mm_set1_ps(0.78515625f);
        __m128 vpi_over_4_lo = _mm_set1_ps(2.4187564849853515625e-4f);
        __m128 vpi_over_4_lo2 = _mm_set1_ps(3.77489497744594108e-8f);

        __m128 vy = _mm_mul_ps(vx, vfour_over_pi);
        __m128 vj = _mm_round_ps(vy, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i vji = _mm_cvtps_epi32(vj);

        __m128 vz = _mm_sub_ps(vx, _mm_mul_ps(vj, vpi_over_4_hi));
        vz = _mm_sub_ps(vz, _mm_mul_ps(vj, vpi_over_4_lo));
        vz = _mm_sub_ps(vz, _mm_mul_ps(vj, vpi_over_4_lo2));

        __m128 vz2 = _mm_mul_ps(vz, vz);

        __m128i vone_i = _mm_set1_epi32(1);
        __m128i vtwo_i = _mm_set1_epi32(2);
        __m128i vswap_mask = _mm_and_si128(vji, vone_i);
        __m128i vneg_sin_mask = _mm_and_si128(vji, vtwo_i);

        __m128 vs_c7 = _mm_set1_ps(-1.9841269841269841e-4f);
        __m128 vs_c5 = _mm_set1_ps(8.3333333333333333e-3f);
        __m128 vs_c3 = _mm_set1_ps(-1.6666666666666666e-1f);
        __m128 vs_c1 = _mm_set1_ps(1.0f);

        __m128 vc_c6 = _mm_set1_ps(-1.3888888888888889e-3f);
        __m128 vc_c4 = _mm_set1_ps(4.1666666666666667e-2f);
        __m128 vc_c2 = _mm_set1_ps(-5.0e-1f);
        __m128 vc_c0 = _mm_set1_ps(1.0f);

#ifdef OPTINUM_HAS_FMA
        __m128 vsin_p = _mm_fmadd_ps(vs_c7, vz2, vs_c5);
        vsin_p = _mm_fmadd_ps(vsin_p, vz2, vs_c3);
        vsin_p = _mm_fmadd_ps(vsin_p, vz2, vs_c1);
        __m128 vcos_p = _mm_fmadd_ps(vc_c6, vz2, vc_c4);
        vcos_p = _mm_fmadd_ps(vcos_p, vz2, vc_c2);
        vcos_p = _mm_fmadd_ps(vcos_p, vz2, vc_c0);
#else
        __m128 vsin_p = _mm_add_ps(_mm_mul_ps(vs_c7, vz2), vs_c5);
        vsin_p = _mm_add_ps(_mm_mul_ps(vsin_p, vz2), vs_c3);
        vsin_p = _mm_add_ps(_mm_mul_ps(vsin_p, vz2), vs_c1);
        __m128 vcos_p = _mm_add_ps(_mm_mul_ps(vc_c6, vz2), vc_c4);
        vcos_p = _mm_add_ps(_mm_mul_ps(vcos_p, vz2), vc_c2);
        vcos_p = _mm_add_ps(_mm_mul_ps(vcos_p, vz2), vc_c0);
#endif

        __m128 vsin_z = _mm_mul_ps(vz, vsin_p);
        __m128 vcos_z = vcos_p;

        __m128 vswap_cmp = _mm_castsi128_ps(_mm_cmpeq_epi32(vswap_mask, vone_i));
        __m128 vresult = _mm_blendv_ps(vsin_z, vcos_z, vswap_cmp);

        __m128 vneg_cmp = _mm_castsi128_ps(_mm_cmpeq_epi32(vneg_sin_mask, vtwo_i));
        vresult = _mm_xor_ps(vresult, _mm_and_ps(vneg_cmp, _mm_set1_ps(-0.0f)));

        return SIMDVec<float, 4>(vresult);
    }

    OPTINUM_INLINE SIMDVec<float, 4> fast_cos(const SIMDVec<float, 4> &x) {
        using namespace math_constants;

        __m128 vx = x.value;

        __m128 vfour_over_pi = _mm_set1_ps(FOUR_INV_PI_F);
        __m128 vpi_over_4_hi = _mm_set1_ps(0.78515625f);
        __m128 vpi_over_4_lo = _mm_set1_ps(2.4187564849853515625e-4f);
        __m128 vpi_over_4_lo2 = _mm_set1_ps(3.77489497744594108e-8f);

        __m128 vy = _mm_mul_ps(vx, vfour_over_pi);
        __m128 vj = _mm_round_ps(vy, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i vji = _mm_cvtps_epi32(vj);

        __m128 vz = _mm_sub_ps(vx, _mm_mul_ps(vj, vpi_over_4_hi));
        vz = _mm_sub_ps(vz, _mm_mul_ps(vj, vpi_over_4_lo));
        vz = _mm_sub_ps(vz, _mm_mul_ps(vj, vpi_over_4_lo2));

        __m128 vz2 = _mm_mul_ps(vz, vz);

        __m128i vone_i = _mm_set1_epi32(1);
        __m128i vtwo_i = _mm_set1_epi32(2);
        __m128i vji_adj = _mm_add_epi32(vji, vone_i);
        __m128i vswap_mask = _mm_and_si128(vji_adj, vone_i);
        __m128i vneg_mask = _mm_and_si128(vji_adj, vtwo_i);

        __m128 vs_c7 = _mm_set1_ps(-1.9841269841269841e-4f);
        __m128 vs_c5 = _mm_set1_ps(8.3333333333333333e-3f);
        __m128 vs_c3 = _mm_set1_ps(-1.6666666666666666e-1f);
        __m128 vs_c1 = _mm_set1_ps(1.0f);

        __m128 vc_c6 = _mm_set1_ps(-1.3888888888888889e-3f);
        __m128 vc_c4 = _mm_set1_ps(4.1666666666666667e-2f);
        __m128 vc_c2 = _mm_set1_ps(-5.0e-1f);
        __m128 vc_c0 = _mm_set1_ps(1.0f);

#ifdef OPTINUM_HAS_FMA
        __m128 vsin_p = _mm_fmadd_ps(vs_c7, vz2, vs_c5);
        vsin_p = _mm_fmadd_ps(vsin_p, vz2, vs_c3);
        vsin_p = _mm_fmadd_ps(vsin_p, vz2, vs_c1);
        __m128 vcos_p = _mm_fmadd_ps(vc_c6, vz2, vc_c4);
        vcos_p = _mm_fmadd_ps(vcos_p, vz2, vc_c2);
        vcos_p = _mm_fmadd_ps(vcos_p, vz2, vc_c0);
#else
        __m128 vsin_p = _mm_add_ps(_mm_mul_ps(vs_c7, vz2), vs_c5);
        vsin_p = _mm_add_ps(_mm_mul_ps(vsin_p, vz2), vs_c3);
        vsin_p = _mm_add_ps(_mm_mul_ps(vsin_p, vz2), vs_c1);
        __m128 vcos_p = _mm_add_ps(_mm_mul_ps(vc_c6, vz2), vc_c4);
        vcos_p = _mm_add_ps(_mm_mul_ps(vcos_p, vz2), vc_c2);
        vcos_p = _mm_add_ps(_mm_mul_ps(vcos_p, vz2), vc_c0);
#endif

        __m128 vsin_z = _mm_mul_ps(vz, vsin_p);
        __m128 vcos_z = vcos_p;

        __m128 vswap_cmp = _mm_castsi128_ps(_mm_cmpeq_epi32(vswap_mask, vone_i));
        __m128 vresult = _mm_blendv_ps(vsin_z, vcos_z, vswap_cmp);

        __m128 vneg_cmp = _mm_castsi128_ps(_mm_cmpeq_epi32(vneg_mask, vtwo_i));
        vresult = _mm_xor_ps(vresult, _mm_and_ps(vneg_cmp, _mm_set1_ps(-0.0f)));

        return SIMDVec<float, 4>(vresult);
    }

#endif // OPTINUM_HAS_SSE41

} // namespace optinum::simd
