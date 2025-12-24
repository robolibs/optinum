#pragma once

// =============================================================================
// optinum/simd/math/sin.hpp
// Vectorized sin() using pack<T,W> with SIMD intrinsics
// Uses quadrant-based range reduction (π/2 steps) for accuracy
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/math/detail/constants.hpp>
#include <optinum/simd/pack/avx.hpp>
#include <optinum/simd/pack/sse.hpp>

namespace optinum::simd {

    // Forward declaration
    template <typename T, std::size_t W> pack<T, W> sin(const pack<T, W> &x) noexcept;

    // =========================================================================
    // pack<float, 4> - SSE implementation
    // =========================================================================

    template <> inline pack<float, 4> sin(const pack<float, 4> &x) noexcept {
        __m128 vx = x.data_;
        __m128 sign_mask = _mm_set1_ps(-0.0f);

        // Constants for range reduction to [-π/4, π/4]
        __m128 two_over_pi = _mm_set1_ps(0.6366197723675814f);     // 2/π
        __m128 pi_over_2_hi = _mm_set1_ps(1.5707963267948966f);    // π/2 high part
        __m128 pi_over_2_lo = _mm_set1_ps(6.123233995736766e-17f); // π/2 low (precision)

        // Extract sign and work with |x|
        __m128 abs_x = _mm_andnot_ps(sign_mask, vx);
        __m128 sign_x = _mm_and_ps(vx, sign_mask);

        // Range reduction: y = |x| * (2/π), q = round(y)
        __m128 y = _mm_mul_ps(abs_x, two_over_pi);
        __m128 q = _mm_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i qi = _mm_cvtps_epi32(q);

        // z = |x| - q * (π/2)
        __m128 z = _mm_sub_ps(abs_x, _mm_mul_ps(q, pi_over_2_hi));
        __m128 z2 = _mm_mul_ps(z, z);

        // Polynomial for sin(z) on [-π/4, π/4]
        // sin(z) ≈ z * (1 - z²/6 + z⁴/120 - z⁶/5040)
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
        // cos(z) ≈ 1 - z²/2 + z⁴/24 - z⁶/720
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

        // Select sin or cos based on quadrant (q mod 4)
        // q&1 == 1: use cos, else use sin
        __m128i q_and_1 = _mm_and_si128(qi, _mm_set1_epi32(1));
        __m128 use_cos = _mm_castsi128_ps(_mm_cmpeq_epi32(q_and_1, _mm_set1_epi32(1)));
        __m128 result = _mm_blendv_ps(sin_z, cos_z, use_cos);

        // Negate based on quadrant (q mod 4)
        // q&2 == 2: negate result
        __m128i q_and_2 = _mm_and_si128(qi, _mm_set1_epi32(2));
        __m128 negate = _mm_castsi128_ps(_mm_cmpeq_epi32(q_and_2, _mm_set1_epi32(2)));
        result = _mm_xor_ps(result, _mm_and_ps(negate, sign_mask));

        // Apply original sign of x
        result = _mm_xor_ps(result, sign_x);

        return pack<float, 4>(result);
    }

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================

    template <> inline pack<float, 8> sin(const pack<float, 8> &x) noexcept {
        __m256 vx = x.data_;
        __m256 sign_mask = _mm256_set1_ps(-0.0f);

        // Constants for range reduction to [-π/4, π/4]
        __m256 two_over_pi = _mm256_set1_ps(0.6366197723675814f);  // 2/π
        __m256 pi_over_2_hi = _mm256_set1_ps(1.5707963267948966f); // π/2 high part

        // Extract sign and work with |x|
        __m256 abs_x = _mm256_andnot_ps(sign_mask, vx);
        __m256 sign_x = _mm256_and_ps(vx, sign_mask);

        // Range reduction: y = |x| * (2/π), q = round(y)
        __m256 y = _mm256_mul_ps(abs_x, two_over_pi);
        __m256 q = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i qi = _mm256_cvtps_epi32(q);

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

        // Select sin or cos based on quadrant (q mod 4)
        // q&1 == 1: use cos, else use sin
        __m256i q_and_1 = _mm256_and_si256(qi, _mm256_set1_epi32(1));
        __m256 use_cos = _mm256_castsi256_ps(_mm256_cmpeq_epi32(q_and_1, _mm256_set1_epi32(1)));
        __m256 result = _mm256_blendv_ps(sin_z, cos_z, use_cos);

        // Negate based on quadrant (q mod 4)
        // q&2 == 2: negate result
        __m256i q_and_2 = _mm256_and_si256(qi, _mm256_set1_epi32(2));
        __m256 negate = _mm256_castsi256_ps(_mm256_cmpeq_epi32(q_and_2, _mm256_set1_epi32(2)));
        result = _mm256_xor_ps(result, _mm256_and_ps(negate, sign_mask));

        // Apply original sign of x
        result = _mm256_xor_ps(result, sign_x);

        return pack<float, 8>(result);
    }

    // TODO: Add double precision variants (pack<double, 2>, pack<double, 4>)

} // namespace optinum::simd
