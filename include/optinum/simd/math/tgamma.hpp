#pragma once

// =============================================================================
// optinum/simd/math/tgamma.hpp
// Vectorized tgamma() (gamma function) using pack<T,W> with SIMD
//
// Γ(x) = ∫₀^∞ t^(x-1) e^(-t) dt
//
// Uses Lanczos approximation then exponentiation:
//   Γ(x) = exp(lgamma(x))
//
// Note: For efficiency, we compute via lgamma since it's more stable
// =============================================================================

#include <cmath>
#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/math/exp.hpp>
#include <optinum/simd/math/lgamma.hpp>
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
    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> tgamma(const pack<T, W> &x) noexcept {
        pack<T, W> result;
        for (std::size_t i = 0; i < W; ++i) {
            result.data_[i] = std::tgamma(x.data_[i]);
        }
        return result;
    }

// =============================================================================
// SSE Implementation for float (W=4)
// =============================================================================
#if defined(OPTINUM_HAS_SSE41)
    template <> inline pack<float, 4> tgamma(const pack<float, 4> &x) noexcept {
        // tgamma(x) = exp(lgamma(x))
        return exp(lgamma(x));
    }
#endif // OPTINUM_HAS_SSE41

// =============================================================================
// AVX Implementation for float (W=8)
// =============================================================================
#if defined(OPTINUM_HAS_AVX)
    template <> inline pack<float, 8> tgamma(const pack<float, 8> &x) noexcept {
        // tgamma(x) = exp(lgamma(x))
        return exp(lgamma(x));
    }
#endif // OPTINUM_HAS_AVX

// =============================================================================
// SSE Implementation for double (W=2)
// =============================================================================
#if defined(OPTINUM_HAS_SSE41)
    template <> inline pack<double, 2> tgamma(const pack<double, 2> &x) noexcept {
        // tgamma(x) = exp(lgamma(x))
        return exp(lgamma(x));
    }
#endif // OPTINUM_HAS_SSE41

// =============================================================================
// AVX Implementation for double (W=4)
// =============================================================================
#if defined(OPTINUM_HAS_AVX)
    template <> inline pack<double, 4> tgamma(const pack<double, 4> &x) noexcept {
        // tgamma(x) = exp(lgamma(x))
        return exp(lgamma(x));
    }
#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
