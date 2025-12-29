#pragma once

// =============================================================================
// optinum/simd/math/floor.hpp
// Vectorized floor() using pack<T,W> with SIMD intrinsics
// Rounds each element toward negative infinity
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
    template <typename T, std::size_t W> pack<T, W> floor(const pack<T, W> &x) noexcept;

    // =========================================================================
    // pack<float, 4> - SSE4.1 implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE41)

    template <> inline pack<float, 4> floor(const pack<float, 4> &x) noexcept {
        return pack<float, 4>(_mm_floor_ps(x.data_));
    }

#endif // OPTINUM_HAS_SSE41

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<float, 8> floor(const pack<float, 8> &x) noexcept {
        return pack<float, 8>(_mm256_floor_ps(x.data_));
    }

#endif // OPTINUM_HAS_AVX

    // =========================================================================
    // pack<double, 2> - SSE4.1 implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE41)

    template <> inline pack<double, 2> floor(const pack<double, 2> &x) noexcept {
        return pack<double, 2>(_mm_floor_pd(x.data_));
    }

#endif // OPTINUM_HAS_SSE41

    // =========================================================================
    // pack<double, 4> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<double, 4> floor(const pack<double, 4> &x) noexcept {
        return pack<double, 4>(_mm256_floor_pd(x.data_));
    }

#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
