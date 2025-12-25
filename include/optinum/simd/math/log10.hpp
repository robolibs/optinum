#pragma once

// =============================================================================
// optinum/simd/math/log10.hpp
// Base-10 logarithm: log10(x) = log(x) / ln(10)
// =============================================================================

#include <optinum/simd/math/detail/constants.hpp>
#include <optinum/simd/math/log.hpp>
#include <optinum/simd/pack/pack.hpp>

namespace optinum::simd {

    // =============================================================================
    // log10() - Base-10 logarithm
    // Uses the identity: log10(x) = log(x) / ln(10) = log(x) * log10(e)
    // where log10(e) = 1/ln(10)
    // =============================================================================

    // -------------------------------------------------------------------------
    // pack<float, 4> - SSE
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 4> log10(const pack<float, 4> &x) noexcept {
        return log(x) * pack<float, 4>(math_constants::LOG10E_F);
    }

    // -------------------------------------------------------------------------
    // pack<float, 8> - AVX
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 8> log10(const pack<float, 8> &x) noexcept {
        return log(x) * pack<float, 8>(math_constants::LOG10E_F);
    }

    // -------------------------------------------------------------------------
    // pack<double, 2> - SSE (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 2> log10(const pack<double, 2> &x) noexcept {
        return log(x) * pack<double, 2>(math_constants::LOG10E_D);
    }

    // -------------------------------------------------------------------------
    // pack<double, 4> - AVX (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 4> log10(const pack<double, 4> &x) noexcept {
        return log(x) * pack<double, 4>(math_constants::LOG10E_D);
    }

} // namespace optinum::simd
