#pragma once

// =============================================================================
// optinum/simd/math/cos.hpp
// Vectorized cos() using pack<T,W> with SIMD intrinsics
// Uses quadrant-based range reduction (π/2 steps) for accuracy
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/math/detail/constants.hpp>
#include <optinum/simd/pack/avx.hpp>
#include <optinum/simd/pack/sse.hpp>

namespace optinum::simd {

    // Forward declaration
    template <typename T, std::size_t W> pack<T, W> cos(const pack<T, W> &x) noexcept;

    // =========================================================================
    // pack<float, 4> - SSE implementation
    // =========================================================================

    template <> inline pack<float, 4> cos(const pack<float, 4> &x) noexcept {
        __m128 vx = x.data_;
        __m128 sign_mask = _mm_set1_ps(-0.0f);

        // Constants for range reduction to [-π/4, π/4]
        __m128 two_over_pi = _mm_set1_ps(0.6366197723675814f);  // 2/π
        __m128 pi_over_2_hi = _mm_set1_ps(1.5707963267948966f); // π/2 high part

        // Work with |x| (cos is even function)
        __m128 abs_x = _mm_andnot_ps(sign_mask, vx);

        // Range reduction: y = |x| * (2/π), q = round(y)
        __m128 y = _mm_mul_ps(abs_x, two_over_pi);
        __m128 q = _mm_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i qi = _mm_cvtps_epi32(q);

        // For cos(x), we use the identity cos(x) = sin(x + π/2)
        // So we add 1 to the quadrant index
        qi = _mm_add_epi32(qi, _mm_set1_epi32(1));

        // z = |x| - q * (π/2)
        __m128 z = _mm_sub_ps(abs_x, _mm_mul_ps(q, pi_over_2_hi));
        __m128 z2 = _mm_mul_ps(z, z);

        // Polynomial for sin(z) on [-π/4, π/4]
        __m128 s_c5 = _mm_set1_ps(0.00833333333f);  // 1/120
        __m128 s_c3 = _mm_set1_ps(-0.16666666667f); // -1/6

#ifdef OPTINUM_HAS_FMA
        __m128 sin_p = _mm_fmadd_ps(s_c5, z2, s_c3);
        sin_p = _mm_fmadd_ps(sin_p, z2, _mm_set1_ps(1.0f));
#else
        __m128 sin_p = _mm_add_ps(_mm_mul_ps(s_c5, z2), s_c3);
        sin_p = _mm_add_ps(_mm_mul_ps(sin_p, z2), _mm_set1_ps(1.0f));
#endif
        __m128 sin_z = _mm_mul_ps(z, sin_p);

        // Polynomial for cos(z) on [-π/4, π/4]
        __m128 c_c4 = _mm_set1_ps(0.04166666667f); // 1/24
        __m128 c_c2 = _mm_set1_ps(-0.5f);          // -1/2

#ifdef OPTINUM_HAS_FMA
        __m128 cos_p = _mm_fmadd_ps(c_c4, z2, c_c2);
        cos_p = _mm_fmadd_ps(cos_p, z2, _mm_set1_ps(1.0f));
#else
        __m128 cos_p = _mm_add_ps(_mm_mul_ps(c_c4, z2), c_c2);
        cos_p = _mm_add_ps(_mm_mul_ps(cos_p, z2), _mm_set1_ps(1.0f));
#endif
        __m128 cos_z = cos_p;

        // Select sin or cos based on shifted quadrant ((q+1) mod 4)
        // (q+1)&1 == 1: use cos, else use sin
        __m128i q_and_1 = _mm_and_si128(qi, _mm_set1_epi32(1));
        __m128 use_cos = _mm_castsi128_ps(_mm_cmpeq_epi32(q_and_1, _mm_set1_epi32(1)));
        __m128 result = _mm_blendv_ps(sin_z, cos_z, use_cos);

        // Negate based on shifted quadrant ((q+1) mod 4)
        // (q+1)&2 == 2: negate result
        __m128i q_and_2 = _mm_and_si128(qi, _mm_set1_epi32(2));
        __m128 negate = _mm_castsi128_ps(_mm_cmpeq_epi32(q_and_2, _mm_set1_epi32(2)));
        result = _mm_xor_ps(result, _mm_and_ps(negate, sign_mask));

        return pack<float, 4>(result);
    }

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================

    template <> inline pack<float, 8> cos(const pack<float, 8> &x) noexcept {
        __m256 vx = x.data_;
        __m256 sign_mask = _mm256_set1_ps(-0.0f);

        // Constants for range reduction to [-π/4, π/4]
        __m256 two_over_pi = _mm256_set1_ps(0.6366197723675814f);  // 2/π
        __m256 pi_over_2_hi = _mm256_set1_ps(1.5707963267948966f); // π/2 high part

        // Work with |x| (cos is even function)
        __m256 abs_x = _mm256_andnot_ps(sign_mask, vx);

        // Range reduction: y = |x| * (2/π), q = round(y)
        __m256 y = _mm256_mul_ps(abs_x, two_over_pi);
        __m256 q = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i qi = _mm256_cvtps_epi32(q);

        // For cos(x), we use the identity cos(x) = sin(x + π/2)
        // So we add 1 to the quadrant index
        qi = _mm256_add_epi32(qi, _mm256_set1_epi32(1));

        // z = |x| - q * (π/2)
        __m256 z = _mm256_sub_ps(abs_x, _mm256_mul_ps(q, pi_over_2_hi));
        __m256 z2 = _mm256_mul_ps(z, z);

        // Polynomial for sin(z) on [-π/4, π/4]
        __m256 s_c5 = _mm256_set1_ps(0.00833333333f);  // 1/120
        __m256 s_c3 = _mm256_set1_ps(-0.16666666667f); // -1/6

#ifdef OPTINUM_HAS_FMA
        __m256 sin_p = _mm256_fmadd_ps(s_c5, z2, s_c3);
        sin_p = _mm256_fmadd_ps(sin_p, z2, _mm256_set1_ps(1.0f));
#else
        __m256 sin_p = _mm256_add_ps(_mm256_mul_ps(s_c5, z2), s_c3);
        sin_p = _mm256_add_ps(_mm256_mul_ps(sin_p, z2), _mm256_set1_ps(1.0f));
#endif
        __m256 sin_z = _mm256_mul_ps(z, sin_p);

        // Polynomial for cos(z) on [-π/4, π/4]
        __m256 c_c4 = _mm256_set1_ps(0.04166666667f); // 1/24
        __m256 c_c2 = _mm256_set1_ps(-0.5f);          // -1/2

#ifdef OPTINUM_HAS_FMA
        __m256 cos_p = _mm256_fmadd_ps(c_c4, z2, c_c2);
        cos_p = _mm256_fmadd_ps(cos_p, z2, _mm256_set1_ps(1.0f));
#else
        __m256 cos_p = _mm256_add_ps(_mm256_mul_ps(c_c4, z2), c_c2);
        cos_p = _mm256_add_ps(_mm256_mul_ps(cos_p, z2), _mm256_set1_ps(1.0f));
#endif
        __m256 cos_z = cos_p;

        // Select sin or cos based on shifted quadrant ((q+1) mod 4)
        // (q+1)&1 == 1: use cos, else use sin
        __m256i q_and_1 = _mm256_and_si256(qi, _mm256_set1_epi32(1));
        __m256 use_cos = _mm256_castsi256_ps(_mm256_cmpeq_epi32(q_and_1, _mm256_set1_epi32(1)));
        __m256 result = _mm256_blendv_ps(sin_z, cos_z, use_cos);

        // Negate based on shifted quadrant ((q+1) mod 4)
        // (q+1)&2 == 2: negate result
        __m256i q_and_2 = _mm256_and_si256(qi, _mm256_set1_epi32(2));
        __m256 negate = _mm256_castsi256_ps(_mm256_cmpeq_epi32(q_and_2, _mm256_set1_epi32(2)));
        result = _mm256_xor_ps(result, _mm256_and_ps(negate, sign_mask));

        return pack<float, 8>(result);
    }

    // =========================================================================
    // pack<double, 2> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE41)

    template <> inline pack<double, 2> cos(const pack<double, 2> &x) noexcept {
        __m128d vx = x.data_;
        __m128d sign_mask = _mm_set1_pd(-0.0);

        // Constants for range reduction to [-π/4, π/4]
        __m128d two_over_pi = _mm_set1_pd(0.6366197723675814);     // 2/π
        __m128d pi_over_2_hi = _mm_set1_pd(1.5707963267948966);    // π/2 high part
        __m128d pi_over_2_lo = _mm_set1_pd(6.123233995736766e-17); // π/2 low

        // Work with |x| (cos is even function)
        __m128d abs_x = _mm_andnot_pd(sign_mask, vx);

        // Range reduction: y = |x| * (2/π), q = round(y)
        __m128d y = _mm_mul_pd(abs_x, two_over_pi);
        __m128d q = _mm_round_pd(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i qi32 = _mm_cvtpd_epi32(q); // Converts 2 doubles to 2 int32s

        // For cos(x), we use the identity cos(x) = sin(x + π/2)
        // So we add 1 to the quadrant index
        qi32 = _mm_add_epi32(qi32, _mm_set1_epi32(1));

        // z = |x| - q * (π/2)
        __m128d z = _mm_sub_pd(abs_x, _mm_mul_pd(q, pi_over_2_hi));
        z = _mm_sub_pd(z, _mm_mul_pd(q, pi_over_2_lo));
        __m128d z2 = _mm_mul_pd(z, z);

        // Higher-order polynomial for sin(z) on [-π/4, π/4]
        __m128d s_c7 = _mm_set1_pd(-1.98412698412698e-4);
        __m128d s_c5 = _mm_set1_pd(8.33333333333333e-3);
        __m128d s_c3 = _mm_set1_pd(-0.166666666666667);

#ifdef OPTINUM_HAS_FMA
        __m128d sin_p = _mm_fmadd_pd(s_c7, z2, s_c5);
        sin_p = _mm_fmadd_pd(sin_p, z2, s_c3);
        sin_p = _mm_fmadd_pd(sin_p, z2, _mm_set1_pd(1.0));
#else
        __m128d sin_p = _mm_add_pd(_mm_mul_pd(s_c7, z2), s_c5);
        sin_p = _mm_add_pd(_mm_mul_pd(sin_p, z2), s_c3);
        sin_p = _mm_add_pd(_mm_mul_pd(sin_p, z2), _mm_set1_pd(1.0));
#endif
        __m128d sin_z = _mm_mul_pd(z, sin_p);

        // Higher-order polynomial for cos(z) on [-π/4, π/4]
        __m128d c_c6 = _mm_set1_pd(-1.38888888888889e-3);
        __m128d c_c4 = _mm_set1_pd(4.16666666666667e-2);
        __m128d c_c2 = _mm_set1_pd(-0.5);

#ifdef OPTINUM_HAS_FMA
        __m128d cos_p = _mm_fmadd_pd(c_c6, z2, c_c4);
        cos_p = _mm_fmadd_pd(cos_p, z2, c_c2);
        cos_p = _mm_fmadd_pd(cos_p, z2, _mm_set1_pd(1.0));
#else
        __m128d cos_p = _mm_add_pd(_mm_mul_pd(c_c6, z2), c_c4);
        cos_p = _mm_add_pd(_mm_mul_pd(cos_p, z2), c_c2);
        cos_p = _mm_add_pd(_mm_mul_pd(cos_p, z2), _mm_set1_pd(1.0));
#endif
        __m128d cos_z = cos_p;

        // Select sin or cos based on shifted quadrant ((q+1) mod 4)
        // (q+1)&1 == 1: use cos, else use sin
        __m128i q_and_1 = _mm_and_si128(qi32, _mm_set1_epi32(1));
        __m128i use_cos_mask = _mm_cmpeq_epi32(q_and_1, _mm_set1_epi32(1));
        __m128i use_cos_mask64 = _mm_shuffle_epi32(use_cos_mask, _MM_SHUFFLE(1, 1, 0, 0));
        __m128d use_cos = _mm_castsi128_pd(use_cos_mask64);
        __m128d result = _mm_blendv_pd(sin_z, cos_z, use_cos);

        // Negate based on shifted quadrant ((q+1) mod 4)
        // (q+1)&2 == 2: negate result
        __m128i q_and_2 = _mm_and_si128(qi32, _mm_set1_epi32(2));
        __m128i negate_mask = _mm_cmpeq_epi32(q_and_2, _mm_set1_epi32(2));
        __m128i negate_mask64 = _mm_shuffle_epi32(negate_mask, _MM_SHUFFLE(1, 1, 0, 0));
        __m128d negate = _mm_castsi128_pd(negate_mask64);
        result = _mm_xor_pd(result, _mm_and_pd(negate, sign_mask));

        return pack<double, 2>(result);
    }

#endif // OPTINUM_HAS_SSE41

    // =========================================================================
    // pack<double, 4> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<double, 4> cos(const pack<double, 4> &x) noexcept {
        __m256d vx = x.data_;
        __m256d sign_mask = _mm256_set1_pd(-0.0);

        // Constants for range reduction to [-π/4, π/4]
        __m256d two_over_pi = _mm256_set1_pd(0.6366197723675814);
        __m256d pi_over_2_hi = _mm256_set1_pd(1.5707963267948966);
        __m256d pi_over_2_lo = _mm256_set1_pd(6.123233995736766e-17);

        // Work with |x| (cos is even function)
        __m256d abs_x = _mm256_andnot_pd(sign_mask, vx);

        // Range reduction: y = |x| * (2/π), q = round(y)
        __m256d y = _mm256_mul_pd(abs_x, two_over_pi);
        __m256d q = _mm256_round_pd(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i qi32 = _mm256_cvtpd_epi32(q); // 4 doubles -> 4 int32s

        // For cos(x), use identity cos(x) = sin(x + π/2)
        qi32 = _mm_add_epi32(qi32, _mm_set1_epi32(1));

        // z = |x| - q * (π/2)
        __m256d z = _mm256_sub_pd(abs_x, _mm256_mul_pd(q, pi_over_2_hi));
        z = _mm256_sub_pd(z, _mm256_mul_pd(q, pi_over_2_lo));
        __m256d z2 = _mm256_mul_pd(z, z);

        // Higher-order polynomial for sin(z)
        __m256d s_c7 = _mm256_set1_pd(-1.98412698412698e-4);
        __m256d s_c5 = _mm256_set1_pd(8.33333333333333e-3);
        __m256d s_c3 = _mm256_set1_pd(-0.166666666666667);

#ifdef OPTINUM_HAS_FMA
        __m256d sin_p = _mm256_fmadd_pd(s_c7, z2, s_c5);
        sin_p = _mm256_fmadd_pd(sin_p, z2, s_c3);
        sin_p = _mm256_fmadd_pd(sin_p, z2, _mm256_set1_pd(1.0));
#else
        __m256d sin_p = _mm256_add_pd(_mm256_mul_pd(s_c7, z2), s_c5);
        sin_p = _mm256_add_pd(_mm256_mul_pd(sin_p, z2), s_c3);
        sin_p = _mm256_add_pd(_mm256_mul_pd(sin_p, z2), _mm256_set1_pd(1.0));
#endif
        __m256d sin_z = _mm256_mul_pd(z, sin_p);

        // Higher-order polynomial for cos(z)
        __m256d c_c6 = _mm256_set1_pd(-1.38888888888889e-3);
        __m256d c_c4 = _mm256_set1_pd(4.16666666666667e-2);
        __m256d c_c2 = _mm256_set1_pd(-0.5);

#ifdef OPTINUM_HAS_FMA
        __m256d cos_p = _mm256_fmadd_pd(c_c6, z2, c_c4);
        cos_p = _mm256_fmadd_pd(cos_p, z2, c_c2);
        cos_p = _mm256_fmadd_pd(cos_p, z2, _mm256_set1_pd(1.0));
#else
        __m256d cos_p = _mm256_add_pd(_mm256_mul_pd(c_c6, z2), c_c4);
        cos_p = _mm256_add_pd(_mm256_mul_pd(cos_p, z2), c_c2);
        cos_p = _mm256_add_pd(_mm256_mul_pd(cos_p, z2), _mm256_set1_pd(1.0));
#endif
        __m256d cos_z = cos_p;

        // Select sin or cos based on shifted quadrant
        __m128i q_and_1 = _mm_and_si128(qi32, _mm_set1_epi32(1));
        __m128i use_cos_mask32 = _mm_cmpeq_epi32(q_and_1, _mm_set1_epi32(1));
        __m256i use_cos_mask64 = _mm256_cvtepi32_epi64(use_cos_mask32);
        __m256d use_cos = _mm256_castsi256_pd(use_cos_mask64);
        __m256d result = _mm256_blendv_pd(sin_z, cos_z, use_cos);

        // Negate based on shifted quadrant
        __m128i q_and_2 = _mm_and_si128(qi32, _mm_set1_epi32(2));
        __m128i negate_mask32 = _mm_cmpeq_epi32(q_and_2, _mm_set1_epi32(2));
        __m256i negate_mask64 = _mm256_cvtepi32_epi64(negate_mask32);
        __m256d negate = _mm256_castsi256_pd(negate_mask64);
        result = _mm256_xor_pd(result, _mm256_and_pd(negate, sign_mask));

        return pack<double, 4>(result);
    }

#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
