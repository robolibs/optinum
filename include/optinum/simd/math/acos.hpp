#pragma once

// =============================================================================
// optinum/simd/math/acos.hpp
// Arc cosine: acos(x) = π/2 - asin(x)
// Valid for x in [-1, 1]
// =============================================================================

#include <optinum/simd/math/asin.hpp>
#include <optinum/simd/math/detail/constants.hpp>
#include <optinum/simd/pack/pack.hpp>

namespace optinum::simd {

    // =============================================================================
    // acos(x) - Arc cosine
    //
    // Uses the identity: acos(x) = π/2 - asin(x)
    //
    // Valid range: [-1, 1]
    // Returns: [0, π]
    // =============================================================================

    // -------------------------------------------------------------------------
    // pack<float, 4> - SSE
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 4> acos(const pack<float, 4> &x) noexcept {
        return pack<float, 4>(math_constants::PI_2_F) - asin(x);
    }

    // -------------------------------------------------------------------------
    // pack<float, 8> - AVX
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 8> acos(const pack<float, 8> &x) noexcept {
        return pack<float, 8>(math_constants::PI_2_F) - asin(x);
    }

    // -------------------------------------------------------------------------
    // pack<double, 2> - SSE (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 2> acos(const pack<double, 2> &x) noexcept {
        return pack<double, 2>(math_constants::PI_2_D) - asin(x);
    }

    // -------------------------------------------------------------------------
    // pack<double, 4> - AVX (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 4> acos(const pack<double, 4> &x) noexcept {
        return pack<double, 4>(math_constants::PI_2_D) - asin(x);
    }

} // namespace optinum::simd
