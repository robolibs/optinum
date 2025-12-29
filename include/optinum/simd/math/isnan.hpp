#pragma once

// =============================================================================
// optinum/simd/math/isnan.hpp
// Vectorized isnan() using pack<T,W> with SIMD intrinsics
// Tests if values are NaN (Not a Number)
// Returns mask<T,W> where true indicates NaN
// Property: NaN != NaN, so we compare x with itself
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
    template <typename T, std::size_t W> mask<T, W> isnan(const pack<T, W> &x) noexcept;

    // =========================================================================
    // pack<float, 4> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE2)

    template <> inline mask<float, 4> isnan(const pack<float, 4> &x) noexcept {
        __m128 vx = x.data_;

        // NaN property: NaN != NaN (unordered comparison)
        // Use cmpneq to check if x != x
        __m128 vresult = _mm_cmpneq_ps(vx, vx);

        return mask<float, 4>(vresult);
    }

#endif // OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline mask<float, 8> isnan(const pack<float, 8> &x) noexcept {
        __m256 vx = x.data_;

        // NaN property: NaN != NaN (unordered comparison)
        // Use CMP_NEQ_UQ (unordered, non-signaling)
        __m256 vresult = _mm256_cmp_ps(vx, vx, _CMP_UNORD_Q);

        return mask<float, 8>(vresult);
    }

#endif // OPTINUM_HAS_AVX

    // =========================================================================
    // pack<double, 2> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE2)

    template <> inline mask<double, 2> isnan(const pack<double, 2> &x) noexcept {
        __m128d vx = x.data_;

        // NaN property: NaN != NaN (unordered comparison)
        // Use cmpneq to check if x != x
        __m128d vresult = _mm_cmpneq_pd(vx, vx);

        return mask<double, 2>(vresult);
    }

#endif // OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<double, 4> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline mask<double, 4> isnan(const pack<double, 4> &x) noexcept {
        __m256d vx = x.data_;

        // NaN property: NaN != NaN (unordered comparison)
        // Use CMP_UNORD_Q (unordered, non-signaling)
        __m256d vresult = _mm256_cmp_pd(vx, vx, _CMP_UNORD_Q);

        return mask<double, 4>(vresult);
    }

#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
