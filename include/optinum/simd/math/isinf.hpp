#pragma once

// =============================================================================
// optinum/simd/math/isinf.hpp
// Vectorized isinf() using pack<T,W> with SIMD intrinsics
// Tests if values are positive or negative infinity
// Returns mask<T,W> where true indicates infinity
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/mask.hpp>
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
    template <typename T, std::size_t W> mask<T, W> isinf(const pack<T, W> &x) noexcept;

    // =========================================================================
    // pack<float, 4> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE2)

#if defined(OPTINUM_HAS_SSE2)
    template <> inline mask<float, 4> isinf(const pack<float, 4> &x) noexcept {
        __m128 vx = x.data_;

        // Infinity: exponent = all 1s (0x7F800000), mantissa = all 0s
        // Clear sign bit and compare to 0x7F800000
        __m128i mask_abs = _mm_set1_epi32(0x7FFFFFFF);
        __m128i inf_pattern = _mm_set1_epi32(0x7F800000);

        __m128 vx_abs = _mm_and_ps(vx, _mm_castsi128_ps(mask_abs));
        __m128 vresult = _mm_cmpeq_ps(vx_abs, _mm_castsi128_ps(inf_pattern));

        return mask<float, 4>(vresult);
    }

#endif // OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

#endif // OPTINUM_HAS_SSE2

#if defined(OPTINUM_HAS_AVX)
    template <> inline mask<float, 8> isinf(const pack<float, 8> &x) noexcept {
        __m256 vx = x.data_;

        // Infinity: exponent = all 1s (0x7F800000), mantissa = all 0s
        // Clear sign bit and compare to 0x7F800000
        __m256i mask_abs = _mm256_set1_epi32(0x7FFFFFFF);
        __m256i inf_pattern = _mm256_set1_epi32(0x7F800000);

        __m256 vx_abs = _mm256_and_ps(vx, _mm256_castsi256_ps(mask_abs));
        __m256 vresult = _mm256_cmp_ps(vx_abs, _mm256_castsi256_ps(inf_pattern), _CMP_EQ_OQ);

        return mask<float, 8>(vresult);
    }

#endif // OPTINUM_HAS_AVX

    // =========================================================================
#endif // OPTINUM_HAS_AVX

    // pack<double, 2> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE2)

    template <> inline mask<double, 2> isinf(const pack<double, 2> &x) noexcept {
        __m128d vx = x.data_;

        // Infinity: exponent = all 1s (0x7FF0000000000000), mantissa = all 0s
        // Clear sign bit and compare to 0x7FF0000000000000
        __m128i mask_abs = _mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL);
        __m128i inf_pattern = _mm_set1_epi64x(0x7FF0000000000000LL);

        __m128d vx_abs = _mm_and_pd(vx, _mm_castsi128_pd(mask_abs));
        __m128d vresult = _mm_cmpeq_pd(vx_abs, _mm_castsi128_pd(inf_pattern));

        return mask<double, 2>(vresult);
    }

#endif // OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<double, 4> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline mask<double, 4> isinf(const pack<double, 4> &x) noexcept {
        __m256d vx = x.data_;

        // Infinity: exponent = all 1s (0x7FF0000000000000), mantissa = all 0s
        // Clear sign bit and compare to 0x7FF0000000000000
        __m256i mask_abs = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL);
        __m256i inf_pattern = _mm256_set1_epi64x(0x7FF0000000000000LL);

        __m256d vx_abs = _mm256_and_pd(vx, _mm256_castsi256_pd(mask_abs));
        __m256d vresult = _mm256_cmp_pd(vx_abs, _mm256_castsi256_pd(inf_pattern), _CMP_EQ_OQ);

        return mask<double, 4>(vresult);
    }

#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
