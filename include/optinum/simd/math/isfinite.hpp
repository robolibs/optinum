#pragma once

// =============================================================================
// optinum/simd/math/isfinite.hpp
// Vectorized isfinite() using pack<T,W> with SIMD intrinsics
// Tests if values are finite (not infinity and not NaN)
// Returns mask<T,W> where true indicates finite value
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/mask.hpp>
#include <optinum/simd/pack/avx.hpp>
#include <optinum/simd/pack/sse.hpp>

namespace optinum::simd {

    // Forward declaration
    template <typename T, std::size_t W> mask<T, W> isfinite(const pack<T, W> &x) noexcept;

    // =========================================================================
    // pack<float, 4> - SSE implementation
    // =========================================================================

    template <> inline mask<float, 4> isfinite(const pack<float, 4> &x) noexcept {
        __m128 vx = x.data_;

        // Finite: exponent is not all 1s
        // For float: exponent mask = 0x7F800000
        // If (x & 0x7F800000) != 0x7F800000, then finite
        __m128i exp_mask = _mm_set1_epi32(0x7F800000);
        __m128i vx_int = _mm_castps_si128(vx);
        __m128i exp_bits = _mm_and_si128(vx_int, exp_mask);

        // Compare: exp_bits != exp_mask means finite
        __m128i cmp_result = _mm_cmpeq_epi32(exp_bits, exp_mask);

        // Invert the result (we want true when NOT equal)
        __m128 vresult = _mm_xor_ps(_mm_castsi128_ps(cmp_result), _mm_castsi128_ps(_mm_set1_epi32(-1)));

        return mask<float, 4>(vresult);
    }

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================

    template <> inline mask<float, 8> isfinite(const pack<float, 8> &x) noexcept {
        __m256 vx = x.data_;

        // Finite: exponent is not all 1s
        // For float: exponent mask = 0x7F800000
        // If (x & 0x7F800000) != 0x7F800000, then finite
        __m256i exp_mask = _mm256_set1_epi32(0x7F800000);
        __m256i vx_int = _mm256_castps_si256(vx);
        __m256i exp_bits = _mm256_and_si256(vx_int, exp_mask);

        // Compare: exp_bits != exp_mask means finite
        __m256i cmp_result = _mm256_cmpeq_epi32(exp_bits, exp_mask);

        // Invert the result (we want true when NOT equal)
        __m256 vresult = _mm256_xor_ps(_mm256_castsi256_ps(cmp_result), _mm256_castsi256_ps(_mm256_set1_epi32(-1)));

        return mask<float, 8>(vresult);
    }

    // =========================================================================
    // pack<double, 2> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE2)

    template <> inline mask<double, 2> isfinite(const pack<double, 2> &x) noexcept {
        __m128d vx = x.data_;

        // Finite: exponent is not all 1s
        // For double: exponent mask = 0x7FF0000000000000
        // If (x & 0x7FF0000000000000) != 0x7FF0000000000000, then finite
        __m128i exp_mask = _mm_set1_epi64x(0x7FF0000000000000LL);
        __m128i vx_int = _mm_castpd_si128(vx);
        __m128i exp_bits = _mm_and_si128(vx_int, exp_mask);

        // Compare: exp_bits != exp_mask means finite
        __m128i cmp_result = _mm_cmpeq_epi64(exp_bits, exp_mask);

        // Invert the result (we want true when NOT equal)
        __m128d vresult = _mm_xor_pd(_mm_castsi128_pd(cmp_result), _mm_castsi128_pd(_mm_set1_epi64x(-1LL)));

        return mask<double, 2>(vresult);
    }

#endif // OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<double, 4> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline mask<double, 4> isfinite(const pack<double, 4> &x) noexcept {
        __m256d vx = x.data_;

        // Finite: exponent is not all 1s
        // For double: exponent mask = 0x7FF0000000000000
        // If (x & 0x7FF0000000000000) != 0x7FF0000000000000, then finite
        __m256i exp_mask = _mm256_set1_epi64x(0x7FF0000000000000LL);
        __m256i vx_int = _mm256_castpd_si256(vx);
        __m256i exp_bits = _mm256_and_si256(vx_int, exp_mask);

        // Compare: exp_bits != exp_mask means finite
        __m256i cmp_result = _mm256_cmpeq_epi64(exp_bits, exp_mask);

        // Invert the result (we want true when NOT equal)
        __m256d vresult = _mm256_xor_pd(_mm256_castsi256_pd(cmp_result), _mm256_castsi256_pd(_mm256_set1_epi64x(-1LL)));

        return mask<double, 4>(vresult);
    }

#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
