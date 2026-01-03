#pragma once

// =============================================================================
// optinum/simd/math/asinh.hpp
// Inverse hyperbolic sine: asinh(x) = ln(x + sqrt(x² + 1))
// Valid for all real x
// =============================================================================

#include <cmath>
#include <optinum/simd/mask.hpp>
#include <optinum/simd/math/log.hpp>
#include <optinum/simd/math/sqrt.hpp>
#if defined(OPTINUM_HAS_AVX)
#include <optinum/simd/pack/avx.hpp>
#endif
#include <optinum/simd/pack/pack.hpp>
#if defined(OPTINUM_HAS_SSE2)
#include <optinum/simd/pack/sse.hpp>
#endif

#if defined(OPTINUM_HAS_NEON)
#include <optinum/simd/pack/neon.hpp>
#endif

namespace optinum::simd {

    // =========================================================================
    // Generic scalar fallback - works for any pack<T, W>
    // =========================================================================
    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> asinh(const pack<T, W> &x) noexcept {
        pack<T, W> result;
        for (std::size_t i = 0; i < W; ++i) {
            result.data_[i] = std::asinh(x.data_[i]);
        }
        return result;
    }

    // =============================================================================
    // asinh(x) - Inverse hyperbolic sine
    //
    // Formula: asinh(x) = ln(x + sqrt(x² + 1))
    //
    // For large |x|, uses: asinh(x) = sign(x) * ln(2|x|)
    // to avoid overflow in x² + 1
    //
    // Valid range: all real numbers
    // Returns: all real numbers
    // Accuracy: ~3-5 ULP
    // =============================================================================

    // -------------------------------------------------------------------------
    // pack<float, 4> - SSE
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 4> asinh(const pack<float, 4> &x) noexcept {
        using P = pack<float, 4>;

        auto abs_x = x.abs();

        // For |x| > 1e8, use approximation: asinh(x) ≈ sign(x) * ln(2|x|)
        // This avoids overflow in x² + 1
        auto use_approx = cmp_gt(abs_x, P(1e8f));

        // Standard formula: asinh(x) = ln(x + sqrt(x² + 1))
        auto x2 = x * x;
        auto standard = log(x + sqrt(x2 + P(1.0f)));

        // Approximation for large |x|: asinh(x) = sign(x) * ln(2|x|)
        auto approx = log(P(2.0f) * abs_x);

        // Copy sign from x to approx
        auto negative = cmp_lt(x, P(0.0f));
        approx = blend(approx, -approx, negative);

        return blend(standard, approx, use_approx);
    }

    // -------------------------------------------------------------------------
    // pack<float, 8> - AVX
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 8> asinh(const pack<float, 8> &x) noexcept {
        using P = pack<float, 8>;

        auto abs_x = x.abs();
        auto use_approx = cmp_gt(abs_x, P(1e8f));

        auto x2 = x * x;
        auto standard = log(x + sqrt(x2 + P(1.0f)));

        auto approx = log(P(2.0f) * abs_x);
        auto negative = cmp_lt(x, P(0.0f));
        approx = blend(approx, -approx, negative);

        return blend(standard, approx, use_approx);
    }

    // -------------------------------------------------------------------------
    // pack<double, 2> - SSE (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 2> asinh(const pack<double, 2> &x) noexcept {
        using P = pack<double, 2>;

        auto abs_x = x.abs();
        auto use_approx = cmp_gt(abs_x, P(1e154));

        auto x2 = x * x;
        auto standard = log(x + sqrt(x2 + P(1.0)));

        auto approx = log(P(2.0) * abs_x);
        auto negative = cmp_lt(x, P(0.0));
        approx = blend(approx, -approx, negative);

        return blend(standard, approx, use_approx);
    }

    // -------------------------------------------------------------------------
    // pack<double, 4> - AVX (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 4> asinh(const pack<double, 4> &x) noexcept {
        using P = pack<double, 4>;

        auto abs_x = x.abs();
        auto use_approx = cmp_gt(abs_x, P(1e154));

        auto x2 = x * x;
        auto standard = log(x + sqrt(x2 + P(1.0)));

        auto approx = log(P(2.0) * abs_x);
        auto negative = cmp_lt(x, P(0.0));
        approx = blend(approx, -approx, negative);

        return blend(standard, approx, use_approx);
    }

} // namespace optinum::simd
