#pragma once

// =============================================================================
// optinum/simd/math/tan.hpp
// Vectorized tan() using pack<T,W> with SIMD intrinsics
// tan(x) = sin(x) / cos(x)
// Uses existing sin/cos implementations for accuracy
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/math/cos.hpp>
#include <optinum/simd/math/sin.hpp>
#include <optinum/simd/pack/avx.hpp>
#include <optinum/simd/pack/sse.hpp>

namespace optinum::simd {

    // Forward declaration
    template <typename T, std::size_t W> pack<T, W> tan(const pack<T, W> &x) noexcept;

    // =========================================================================
    // pack<float, 4> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE41)

    template <> inline pack<float, 4> tan(const pack<float, 4> &x) noexcept {
        // Compute sin(x) and cos(x)
        pack<float, 4> sin_x = sin(x);
        pack<float, 4> cos_x = cos(x);

        // tan(x) = sin(x) / cos(x)
        __m128 result = _mm_div_ps(sin_x.data_, cos_x.data_);

        return pack<float, 4>(result);
    }

#endif // OPTINUM_HAS_SSE41

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<float, 8> tan(const pack<float, 8> &x) noexcept {
        // Compute sin(x) and cos(x)
        pack<float, 8> sin_x = sin(x);
        pack<float, 8> cos_x = cos(x);

        // tan(x) = sin(x) / cos(x)
        __m256 result = _mm256_div_ps(sin_x.data_, cos_x.data_);

        return pack<float, 8>(result);
    }

#endif // OPTINUM_HAS_AVX

    // =========================================================================
    // pack<double, 2> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE41)

    template <> inline pack<double, 2> tan(const pack<double, 2> &x) noexcept {
        // Compute sin(x) and cos(x)
        pack<double, 2> sin_x = sin(x);
        pack<double, 2> cos_x = cos(x);

        // tan(x) = sin(x) / cos(x)
        __m128d result = _mm_div_pd(sin_x.data_, cos_x.data_);

        return pack<double, 2>(result);
    }

#endif // OPTINUM_HAS_SSE41

    // =========================================================================
    // pack<double, 4> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<double, 4> tan(const pack<double, 4> &x) noexcept {
        // Compute sin(x) and cos(x)
        pack<double, 4> sin_x = sin(x);
        pack<double, 4> cos_x = cos(x);

        // tan(x) = sin(x) / cos(x)
        __m256d result = _mm256_div_pd(sin_x.data_, cos_x.data_);

        return pack<double, 4>(result);
    }

#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
