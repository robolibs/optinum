#pragma once

// =============================================================================
// optinum/simd/math/clamp.hpp
// Vectorized clamp() using pack<T,W> with SIMD intrinsics
// Clamp to range: clamp(x, lo, hi) = max(lo, min(x, hi))
// =============================================================================

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
    template <typename T, std::size_t W>
    pack<T, W> clamp(const pack<T, W> &x, const pack<T, W> &lo, const pack<T, W> &hi) noexcept;

    // =========================================================================
    // pack<float, 4> - SSE implementation
    // =========================================================================

    template <>
    inline pack<float, 4> clamp(const pack<float, 4> &x, const pack<float, 4> &lo, const pack<float, 4> &hi) noexcept {
        __m128 vx = x.data_;
        __m128 vlo = lo.data_;
        __m128 vhi = hi.data_;

        // clamp(x, lo, hi) = max(lo, min(x, hi))
        __m128 vresult = _mm_min_ps(vx, vhi);
        vresult = _mm_max_ps(vresult, vlo);

        return pack<float, 4>(vresult);
    }

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================

    template <>
    inline pack<float, 8> clamp(const pack<float, 8> &x, const pack<float, 8> &lo, const pack<float, 8> &hi) noexcept {
        __m256 vx = x.data_;
        __m256 vlo = lo.data_;
        __m256 vhi = hi.data_;

        // clamp(x, lo, hi) = max(lo, min(x, hi))
        __m256 vresult = _mm256_min_ps(vx, vhi);
        vresult = _mm256_max_ps(vresult, vlo);

        return pack<float, 8>(vresult);
    }

    // =========================================================================
    // pack<double, 2> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE2)

    template <>
    inline pack<double, 2> clamp(const pack<double, 2> &x, const pack<double, 2> &lo,
                                 const pack<double, 2> &hi) noexcept {
        __m128d vx = x.data_;
        __m128d vlo = lo.data_;
        __m128d vhi = hi.data_;

        // clamp(x, lo, hi) = max(lo, min(x, hi))
        __m128d vresult = _mm_min_pd(vx, vhi);
        vresult = _mm_max_pd(vresult, vlo);

        return pack<double, 2>(vresult);
    }

#endif // OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<double, 4> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <>
    inline pack<double, 4> clamp(const pack<double, 4> &x, const pack<double, 4> &lo,
                                 const pack<double, 4> &hi) noexcept {
        __m256d vx = x.data_;
        __m256d vlo = lo.data_;
        __m256d vhi = hi.data_;

        // clamp(x, lo, hi) = max(lo, min(x, hi))
        __m256d vresult = _mm256_min_pd(vx, vhi);
        vresult = _mm256_max_pd(vresult, vlo);

        return pack<double, 4>(vresult);
    }

#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
