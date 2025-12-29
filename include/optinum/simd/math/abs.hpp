#pragma once

// =============================================================================
// optinum/simd/math/abs.hpp
// Vectorized abs() using pack<T,W> with SIMD intrinsics
// Absolute value: abs(x) = |x|
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/pack/pack.hpp>

#if defined(OPTINUM_HAS_SSE2)
#include <optinum/simd/pack/sse.hpp>
#endif

#if defined(OPTINUM_HAS_AVX)
#include <optinum/simd/pack/avx.hpp>
#endif

#if defined(OPTINUM_HAS_NEON)
#include <optinum/simd/pack/neon.hpp>
#endif

namespace optinum::simd {

    // Forward declaration
    template <typename T, std::size_t W> pack<T, W> abs(const pack<T, W> &x) noexcept;

    // =========================================================================
    // pack<float, 4> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE2)

    template <> inline pack<float, 4> abs(const pack<float, 4> &x) noexcept {
        __m128 vx = x.data_;

        // Clear sign bit: abs(x) = x & 0x7FFFFFFF
        __m128i mask = _mm_set1_epi32(0x7FFFFFFF);
        __m128 vresult = _mm_and_ps(vx, _mm_castsi128_ps(mask));

        return pack<float, 4>(vresult);
    }

#endif // OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<float, 4> - NEON implementation
    // =========================================================================
#if defined(OPTINUM_HAS_NEON) && !defined(OPTINUM_HAS_SSE2)

    template <> inline pack<float, 4> abs(const pack<float, 4> &x) noexcept {
        return pack<float, 4>(vabsq_f32(x.data_));
    }

#endif // OPTINUM_HAS_NEON && !OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<float, 8> abs(const pack<float, 8> &x) noexcept {
        __m256 vx = x.data_;

        // Clear sign bit: abs(x) = x & 0x7FFFFFFF
        __m256i mask = _mm256_set1_epi32(0x7FFFFFFF);
        __m256 vresult = _mm256_and_ps(vx, _mm256_castsi256_ps(mask));

        return pack<float, 8>(vresult);
    }

#endif // OPTINUM_HAS_AVX

    // =========================================================================
    // pack<double, 2> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE2)

    template <> inline pack<double, 2> abs(const pack<double, 2> &x) noexcept {
        __m128d vx = x.data_;

        // Clear sign bit: abs(x) = x & 0x7FFFFFFFFFFFFFFF
        __m128i mask = _mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL);
        __m128d vresult = _mm_and_pd(vx, _mm_castsi128_pd(mask));

        return pack<double, 2>(vresult);
    }

#endif // OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<double, 2> - NEON implementation (ARM64 only)
    // =========================================================================
#if defined(OPTINUM_HAS_NEON) && defined(__aarch64__) && !defined(OPTINUM_HAS_SSE2)

    template <> inline pack<double, 2> abs(const pack<double, 2> &x) noexcept {
        return pack<double, 2>(vabsq_f64(x.data_));
    }

#endif // OPTINUM_HAS_NEON && __aarch64__ && !OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<double, 4> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<double, 4> abs(const pack<double, 4> &x) noexcept {
        __m256d vx = x.data_;

        // Clear sign bit: abs(x) = x & 0x7FFFFFFFFFFFFFFF
        __m256i mask = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL);
        __m256d vresult = _mm256_and_pd(vx, _mm256_castsi256_pd(mask));

        return pack<double, 4>(vresult);
    }

#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
