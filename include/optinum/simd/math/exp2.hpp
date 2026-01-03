#pragma once

// =============================================================================
// optinum/simd/math/exp2.hpp
// Base-2 exponential: 2^x = exp(x * ln(2))
// =============================================================================

#include <cmath>
#include <optinum/simd/math/detail/constants.hpp>
#include <optinum/simd/math/exp.hpp>
#include <optinum/simd/pack/pack.hpp>

namespace optinum::simd {

    // =========================================================================
    // Generic scalar fallback - works for any pack<T, W>
    // =========================================================================
    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> exp2(const pack<T, W> &x) noexcept {
        pack<T, W> result;
        for (std::size_t i = 0; i < W; ++i) {
            result.data_[i] = std::exp2(x.data_[i]);
        }
        return result;
    }

    // =============================================================================
    // exp2() - Base-2 exponential (2^x)
    // Uses the identity: 2^x = exp(x * ln(2))
    // =============================================================================

    // -------------------------------------------------------------------------
    // pack<float, 4> - SSE
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 4> exp2(const pack<float, 4> &x) noexcept {
        return exp(x * pack<float, 4>(math_constants::LN2_F));
    }

    // -------------------------------------------------------------------------
    // pack<float, 8> - AVX
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 8> exp2(const pack<float, 8> &x) noexcept {
        return exp(x * pack<float, 8>(math_constants::LN2_F));
    }

    // -------------------------------------------------------------------------
    // pack<double, 2> - SSE (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 2> exp2(const pack<double, 2> &x) noexcept {
        return exp(x * pack<double, 2>(math_constants::LN2_D));
    }

    // -------------------------------------------------------------------------
    // pack<double, 4> - AVX (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 4> exp2(const pack<double, 4> &x) noexcept {
        return exp(x * pack<double, 4>(math_constants::LN2_D));
    }

} // namespace optinum::simd
