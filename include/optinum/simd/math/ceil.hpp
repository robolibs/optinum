#pragma once

// =============================================================================
// optinum/simd/math/ceil.hpp
// Vectorized ceil() using pack<T,W> with SIMD intrinsics
// Rounds each element toward positive infinity
// =============================================================================

#include <cmath>
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

    // =========================================================================
    // Generic scalar fallback - works for any pack<T, W>
    // =========================================================================
    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> ceil(const pack<T, W> &x) noexcept {
        pack<T, W> result;
        for (std::size_t i = 0; i < W; ++i) {
            result.data_[i] = std::ceil(x.data_[i]);
        }
        return result;
    }

    // =========================================================================
    // pack<float, 4> - SSE4.1 implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE41)
    template <> inline pack<float, 4> ceil(const pack<float, 4> &x) noexcept {
        return pack<float, 4>(_mm_ceil_ps(x.data_));
    }
#endif // OPTINUM_HAS_SSE41

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)
    template <> inline pack<float, 8> ceil(const pack<float, 8> &x) noexcept {
        return pack<float, 8>(_mm256_ceil_ps(x.data_));
    }
#endif // OPTINUM_HAS_AVX

    // =========================================================================
    // pack<double, 2> - SSE4.1 implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE41)
    template <> inline pack<double, 2> ceil(const pack<double, 2> &x) noexcept {
        return pack<double, 2>(_mm_ceil_pd(x.data_));
    }
#endif // OPTINUM_HAS_SSE41

    // =========================================================================
    // pack<double, 4> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)
    template <> inline pack<double, 4> ceil(const pack<double, 4> &x) noexcept {
        return pack<double, 4>(_mm256_ceil_pd(x.data_));
    }
#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
