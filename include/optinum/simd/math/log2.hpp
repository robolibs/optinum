#pragma once

// =============================================================================
// optinum/simd/math/log2.hpp
// Base-2 logarithm: log2(x) = log(x) / ln(2)
// =============================================================================

#include <cmath>
#include <optinum/simd/math/detail/constants.hpp>
#include <optinum/simd/math/log.hpp>
#include <optinum/simd/pack/pack.hpp>

namespace optinum::simd {

    // =========================================================================
    // Generic scalar fallback - works for any pack<T, W>
    // =========================================================================
    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> log2(const pack<T, W> &x) noexcept {
        pack<T, W> result;
        for (std::size_t i = 0; i < W; ++i) {
            result.data_[i] = std::log2(x.data_[i]);
        }
        return result;
    }

    // =============================================================================
    // log2() - Base-2 logarithm
    // Uses the identity: log2(x) = log(x) / ln(2) = log(x) * log2(e)
    // where log2(e) = 1/ln(2)
    // =============================================================================

    // -------------------------------------------------------------------------
    // pack<float, 4> - SSE
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 4> log2(const pack<float, 4> &x) noexcept {
        return log(x) * pack<float, 4>(math_constants::LOG2E_F);
    }

    // -------------------------------------------------------------------------
    // pack<float, 8> - AVX
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 8> log2(const pack<float, 8> &x) noexcept {
        return log(x) * pack<float, 8>(math_constants::LOG2E_F);
    }

    // -------------------------------------------------------------------------
    // pack<double, 2> - SSE (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 2> log2(const pack<double, 2> &x) noexcept {
        return log(x) * pack<double, 2>(math_constants::LOG2E_D);
    }

    // -------------------------------------------------------------------------
    // pack<double, 4> - AVX (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 4> log2(const pack<double, 4> &x) noexcept {
        return log(x) * pack<double, 4>(math_constants::LOG2E_D);
    }

} // namespace optinum::simd
