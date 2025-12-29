#pragma once

// =============================================================================
// optinum/simd/math/cbrt.hpp
// Vectorized cbrt() using pack<T,W> with SIMD intrinsics
// Cube root: cbrt(x) = x^(1/3)
// Uses Newton-Raphson iteration: y_{n+1} = (2*y_n + x/(y_n²)) / 3
// =============================================================================

#include <cmath>
#include <cstdint>
#include <cstring>
#include <optinum/simd/arch/arch.hpp>
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
    template <typename T, std::size_t W> pack<T, W> cbrt(const pack<T, W> &x) noexcept;

    // =========================================================================
    // pack<float, 4> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE2)

#if defined(OPTINUM_HAS_SSE2)
    template <> inline pack<float, 4> cbrt(const pack<float, 4> &x) noexcept {
        __m128 vx = x.data_;

        // Extract sign and work with absolute value
        __m128i sign_mask = _mm_set1_epi32(0x80000000);
        __m128 vsign = _mm_and_ps(vx, _mm_castsi128_ps(sign_mask));
        __m128 vabs = _mm_andnot_ps(_mm_castsi128_ps(sign_mask), vx);

        // Initial approximation using bit manipulation
        // cbrt(x) ≈ 2^(exponent/3) * (mantissa^(1/3))
        __m128i vi = _mm_castps_si128(vabs);
        __m128i vexp = _mm_srli_epi32(vi, 23);
        vexp = _mm_sub_epi32(vexp, _mm_set1_epi32(127));
        __m128i vexp_div3 = _mm_cvttps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(vexp), _mm_set1_ps(1.0f / 3.0f)));
        vexp_div3 = _mm_add_epi32(vexp_div3, _mm_set1_epi32(127));
        vexp_div3 = _mm_slli_epi32(vexp_div3, 23);
        __m128 vy = _mm_castsi128_ps(vexp_div3);

        // Newton-Raphson iterations: y = (2*y + x/y²) / 3
        __m128 vone_third = _mm_set1_ps(1.0f / 3.0f);
        __m128 vtwo_thirds = _mm_set1_ps(2.0f / 3.0f);

        for (int i = 0; i < 3; ++i) {
            __m128 vy2 = _mm_mul_ps(vy, vy);
            __m128 vterm = _mm_div_ps(vabs, vy2);
            vy = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(vy, vtwo_thirds), _mm_mul_ps(vterm, vone_third)), _mm_set1_ps(1.0f));
        }

        // Restore sign
        __m128 vresult = _mm_or_ps(vy, vsign);

        // Handle special cases
        __m128 vzero = _mm_setzero_ps();
        __m128 vmask_zero = _mm_cmpeq_ps(vx, vzero);
        vresult = _mm_blendv_ps(vresult, vzero, vmask_zero);

        return pack<float, 4>(vresult);
    }

#endif // OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

#endif // OPTINUM_HAS_SSE2

#if defined(OPTINUM_HAS_AVX)
    template <> inline pack<float, 8> cbrt(const pack<float, 8> &x) noexcept {
        __m256 vx = x.data_;

        // Extract sign and work with absolute value
        __m256i sign_mask = _mm256_set1_epi32(0x80000000);
        __m256 vsign = _mm256_and_ps(vx, _mm256_castsi256_ps(sign_mask));
        __m256 vabs = _mm256_andnot_ps(_mm256_castsi256_ps(sign_mask), vx);

        // Initial approximation using bit manipulation
        __m256i vi = _mm256_castps_si256(vabs);
        __m256i vexp = _mm256_srli_epi32(vi, 23);
        vexp = _mm256_sub_epi32(vexp, _mm256_set1_epi32(127));
        __m256i vexp_div3 = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(vexp), _mm256_set1_ps(1.0f / 3.0f)));
        vexp_div3 = _mm256_add_epi32(vexp_div3, _mm256_set1_epi32(127));
        vexp_div3 = _mm256_slli_epi32(vexp_div3, 23);
        __m256 vy = _mm256_castsi256_ps(vexp_div3);

        // Newton-Raphson iterations: y = (2*y + x/y²) / 3
        __m256 vone_third = _mm256_set1_ps(1.0f / 3.0f);
        __m256 vtwo_thirds = _mm256_set1_ps(2.0f / 3.0f);

        for (int i = 0; i < 3; ++i) {
            __m256 vy2 = _mm256_mul_ps(vy, vy);
            __m256 vterm = _mm256_div_ps(vabs, vy2);
            vy = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(vy, vtwo_thirds), _mm256_mul_ps(vterm, vone_third)),
                               _mm256_set1_ps(1.0f));
        }

        // Restore sign
        __m256 vresult = _mm256_or_ps(vy, vsign);

        // Handle special cases
        __m256 vzero = _mm256_setzero_ps();
        __m256 vmask_zero = _mm256_cmp_ps(vx, vzero, _CMP_EQ_OQ);
        vresult = _mm256_blendv_ps(vresult, vzero, vmask_zero);

        return pack<float, 8>(vresult);
    }

#endif // OPTINUM_HAS_AVX

    // =========================================================================
#endif // OPTINUM_HAS_AVX

    // pack<double, 2> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE2)

    template <> inline pack<double, 2> cbrt(const pack<double, 2> &x) noexcept {
        __m128d vx = x.data_;

        // Extract sign and work with absolute value
        __m128i sign_mask = _mm_set1_epi64x(0x8000000000000000LL);
        __m128d vsign = _mm_and_pd(vx, _mm_castsi128_pd(sign_mask));
        __m128d vabs = _mm_andnot_pd(_mm_castsi128_pd(sign_mask), vx);

        // Initial approximation using bit manipulation
        // Extract exponent, divide by 3, reconstruct as 2^(exp/3)
        __m128i vi = _mm_castpd_si128(vabs);

        // Mask to extract exponent bits [62:52]
        __m128i exp_mask = _mm_set1_epi64x(0x7FF0000000000000LL);
        __m128i vexp_bits = _mm_and_si128(vi, exp_mask);
        __m128i vexp = _mm_srli_epi64(vexp_bits, 52);

        // Subtract bias (1023) - treat as unsigned since we know exp >= 0 for positive normals
        // For proper handling, convert to double, do arithmetic, convert back
        __m128d vexp_f = _mm_cvtepi32_pd(_mm_shuffle_epi32(vexp, _MM_SHUFFLE(2, 0, 2, 0)));
        __m128d vexp_unbiased = _mm_sub_pd(vexp_f, _mm_set1_pd(1023.0));

        // Divide by 3
        __m128d vexp_div3_f = _mm_mul_pd(vexp_unbiased, _mm_set1_pd(1.0 / 3.0));

        // Round and convert back to integer
        __m128i vexp_div3_32 = _mm_cvtpd_epi32(vexp_div3_f);

        // Convert back to double and add bias
        __m128d vexp_div3_d = _mm_cvtepi32_pd(vexp_div3_32);
        __m128d vexp_biased = _mm_add_pd(vexp_div3_d, _mm_set1_pd(1023.0));

        // Convert to integer and shift to exponent position
        __m128i vexp_new_32 = _mm_cvtpd_epi32(vexp_biased);

        // Expand 32-bit to 64-bit and shift left 52
        // vexp_new_32 has two 32-bit ints in low 64 bits: [int1, int0, 0, 0]
        // We need: [int1 << 52 in 64-bit, int0 << 52 in 64-bit]
        __m128i vexp_64_lo = _mm_cvtepu32_epi64(vexp_new_32);
        __m128i vexp_shifted = _mm_slli_epi64(vexp_64_lo, 52);

        __m128d vy = _mm_castsi128_pd(vexp_shifted);

        // Newton-Raphson iterations: y = (2*y + x/y²) / 3
        __m128d vone_third = _mm_set1_pd(1.0 / 3.0);
        __m128d vtwo_thirds = _mm_set1_pd(2.0 / 3.0);

        for (int i = 0; i < 4; ++i) {
            __m128d vy2 = _mm_mul_pd(vy, vy);
            __m128d vterm = _mm_div_pd(vabs, vy2);
            vy = _mm_add_pd(_mm_mul_pd(vy, vtwo_thirds), _mm_mul_pd(vterm, vone_third));
        }

        // Restore sign
        __m128d vresult = _mm_or_pd(vy, vsign);

        // Handle special cases
        __m128d vzero = _mm_setzero_pd();
        __m128d vmask_zero = _mm_cmpeq_pd(vx, vzero);
        vresult = _mm_blendv_pd(vresult, vzero, vmask_zero);

        return pack<double, 2>(vresult);
    }

#endif // OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<double, 4> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<double, 4> cbrt(const pack<double, 4> &x) noexcept {
        __m256d vx = x.data_;

        // Extract sign and work with absolute value
        __m256i sign_mask = _mm256_set1_epi64x(0x8000000000000000LL);
        __m256d vsign = _mm256_and_pd(vx, _mm256_castsi256_pd(sign_mask));
        __m256d vabs = _mm256_andnot_pd(_mm256_castsi256_pd(sign_mask), vx);

        // Initial approximation using bit manipulation
        __m256i vi = _mm256_castpd_si256(vabs);

        // Extract exponent bits [62:52]
        __m256i exp_mask = _mm256_set1_epi64x(0x7FF0000000000000LL);
        __m256i vexp_bits = _mm256_and_si256(vi, exp_mask);
        __m256i vexp = _mm256_srli_epi64(vexp_bits, 52);

        // Convert to double for arithmetic (4 x 64-bit int -> 4 x double)
        // Extract low 32 bits from each 64-bit lane
        __m128i vexp_lo_lane = _mm256_castsi256_si128(vexp);
        __m128i vexp_hi_lane = _mm256_extracti128_si256(vexp, 1);

        // Shuffle to get low 32-bit parts
        __m128i vexp_lo_32 = _mm_shuffle_epi32(vexp_lo_lane, _MM_SHUFFLE(2, 0, 2, 0));
        __m128i vexp_hi_32 = _mm_shuffle_epi32(vexp_hi_lane, _MM_SHUFFLE(2, 0, 2, 0));

        // Convert to double
        __m128d vexp_lo_d = _mm_cvtepi32_pd(vexp_lo_32);
        __m128d vexp_hi_d = _mm_cvtepi32_pd(vexp_hi_32);
        __m256d vexp_d = _mm256_set_m128d(vexp_hi_d, vexp_lo_d);

        // Subtract bias and divide by 3
        __m256d vexp_unbiased = _mm256_sub_pd(vexp_d, _mm256_set1_pd(1023.0));
        __m256d vexp_div3_d = _mm256_mul_pd(vexp_unbiased, _mm256_set1_pd(1.0 / 3.0));

        // Add bias back
        __m256d vexp_biased = _mm256_add_pd(vexp_div3_d, _mm256_set1_pd(1023.0));

        // Convert back to 32-bit integers
        __m128i vexp_new_lo_32 = _mm_cvtpd_epi32(_mm256_castpd256_pd128(vexp_biased));
        __m128i vexp_new_hi_32 = _mm_cvtpd_epi32(_mm256_extractf128_pd(vexp_biased, 1));

        // Expand to 64-bit and shift to exponent position
        __m128i vexp_new_lo_64 = _mm_cvtepu32_epi64(vexp_new_lo_32);
        __m128i vexp_new_hi_64 = _mm_cvtepu32_epi64(vexp_new_hi_32);
        __m128i vexp_shifted_lo = _mm_slli_epi64(vexp_new_lo_64, 52);
        __m128i vexp_shifted_hi = _mm_slli_epi64(vexp_new_hi_64, 52);

        __m256i vexp_shifted = _mm256_set_m128i(vexp_shifted_hi, vexp_shifted_lo);
        __m256d vy = _mm256_castsi256_pd(vexp_shifted);

        // Newton-Raphson iterations: y = (2*y + x/y²) / 3
        __m256d vone_third = _mm256_set1_pd(1.0 / 3.0);
        __m256d vtwo_thirds = _mm256_set1_pd(2.0 / 3.0);

        for (int i = 0; i < 4; ++i) {
            __m256d vy2 = _mm256_mul_pd(vy, vy);
            __m256d vterm = _mm256_div_pd(vabs, vy2);
            vy = _mm256_mul_pd(_mm256_add_pd(_mm256_mul_pd(vy, vtwo_thirds), _mm256_mul_pd(vterm, vone_third)),
                               _mm256_set1_pd(1.0));
        }

        // Restore sign
        __m256d vresult = _mm256_or_pd(vy, vsign);

        // Handle special cases
        __m256d vzero = _mm256_setzero_pd();
        __m256d vmask_zero = _mm256_cmp_pd(vx, vzero, _CMP_EQ_OQ);
        vresult = _mm256_blendv_pd(vresult, vzero, vmask_zero);

        return pack<double, 4>(vresult);
    }

#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
