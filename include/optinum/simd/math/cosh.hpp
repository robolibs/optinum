#pragma once

// =============================================================================
// optinum/simd/math/cosh.hpp
// Hyperbolic cosine: cosh(x) = (exp(x) + exp(-x)) / 2
// =============================================================================

#include <optinum/simd/math/exp.hpp>
#include <optinum/simd/pack/pack.hpp>

namespace optinum::simd {

    // =============================================================================
    // cosh() - Hyperbolic cosine
    // Uses the definition: cosh(x) = (exp(x) + exp(-x)) / 2
    // =============================================================================

    // -------------------------------------------------------------------------
    // pack<float, 4> - SSE
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 4> cosh(const pack<float, 4> &x) noexcept {
        auto exp_pos = exp(x);  // exp(x)
        auto exp_neg = exp(-x); // exp(-x)
        return (exp_pos + exp_neg) * pack<float, 4>(0.5f);
    }

    // -------------------------------------------------------------------------
    // pack<float, 8> - AVX
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 8> cosh(const pack<float, 8> &x) noexcept {
        auto exp_pos = exp(x);  // exp(x)
        auto exp_neg = exp(-x); // exp(-x)
        return (exp_pos + exp_neg) * pack<float, 8>(0.5f);
    }

    // -------------------------------------------------------------------------
    // pack<double, 2> - SSE (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 2> cosh(const pack<double, 2> &x) noexcept {
        auto exp_pos = exp(x);  // exp(x)
        auto exp_neg = exp(-x); // exp(-x)
        return (exp_pos + exp_neg) * pack<double, 2>(0.5);
    }

    // -------------------------------------------------------------------------
    // pack<double, 4> - AVX (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 4> cosh(const pack<double, 4> &x) noexcept {
        auto exp_pos = exp(x);  // exp(x)
        auto exp_neg = exp(-x); // exp(-x)
        return (exp_pos + exp_neg) * pack<double, 4>(0.5);
    }

} // namespace optinum::simd
