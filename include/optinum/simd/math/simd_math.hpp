#pragma once

// =============================================================================
// optinum/simd/math/simd_math.hpp
// High-performance SIMD math API using pack<T,W>
//
// Functions available:
//   simd::exp(pack<T,W>)   - exponential (e^x)
//   simd::exp2(pack<T,W>)  - base-2 exponential (2^x)
//   simd::log(pack<T,W>)   - natural logarithm (ln)
//   simd::log2(pack<T,W>)  - base-2 logarithm
//   simd::log10(pack<T,W>) - base-10 logarithm
//   simd::sin(pack<T,W>)   - sine
//   simd::cos(pack<T,W>)   - cosine
//   simd::tan(pack<T,W>)   - tangent
//   simd::asin(pack<T,W>)  - arc sine
//   simd::acos(pack<T,W>)  - arc cosine
//   simd::atan(pack<T,W>)  - arc tangent
//   simd::atan2(pack<T,W>, pack<T,W>) - two-argument arc tangent
//   simd::sinh(pack<T,W>)  - hyperbolic sine
//   simd::cosh(pack<T,W>)  - hyperbolic cosine
//   simd::tanh(pack<T,W>)  - hyperbolic tangent
//   simd::asinh(pack<T,W>) - inverse hyperbolic sine
//   simd::acosh(pack<T,W>) - inverse hyperbolic cosine
//   simd::atanh(pack<T,W>) - inverse hyperbolic tangent
//   simd::expm1(pack<T,W>) - exp(x) - 1 (accurate for small x)
//   simd::log1p(pack<T,W>) - log(1 + x) (accurate for small x)
//   simd::sqrt(pack<T,W>)  - square root
//   simd::pow(pack<T,W>, pack<T,W>) - power
//   simd::abs(pack<T,W>)   - absolute value
//   simd::cbrt(pack<T,W>)  - cube root
//   simd::clamp(pack<T,W>, pack<T,W>, pack<T,W>) - clamp to range [lo, hi]
//   simd::hypot(pack<T,W>, pack<T,W>) - hypotenuse sqrt(x² + y²)
//   simd::ceil(pack<T,W>)  - ceiling (round up)
//   simd::floor(pack<T,W>) - floor (round down)
//   simd::round(pack<T,W>) - round to nearest
//   simd::trunc(pack<T,W>) - truncate (round toward zero)
//   simd::isinf(pack<T,W>)  - test for infinity (returns mask<T,W>)
//   simd::isnan(pack<T,W>)  - test for NaN (returns mask<T,W>)
//   simd::isfinite(pack<T,W>) - test for finite values (returns mask<T,W>)
//   simd::erf(pack<T,W>)   - error function
//   simd::tgamma(pack<T,W>) - gamma function (Γ)
//   simd::lgamma(pack<T,W>) - log gamma function (log Γ)
//
// All functions are optimized for speed (~3-5 ULP accuracy).
//
// Benchmark Results (1M elements, 100 iterations):
//   | Function | SIMD (ms) | Scalar (ms) | Speedup |
//   |----------|-----------|-------------|---------|
//   | exp      | 20.16     | 159.97      | 7.94x   |
//   | log      | 36.13     | 173.57      | 4.80x   |
//   | sin      | 24.63     | 564.93      | 22.94x  |
//   | cos      | 24.54     | 540.34      | 22.02x  |
//   | tanh     | 40.80     | 1123.93     | 27.55x  |
//   | sqrt     | 14.96     | 60.30       | 4.03x   |
// =============================================================================

// Native SIMD implementations using pack<T,W>
#include <optinum/simd/math/abs.hpp>
#include <optinum/simd/math/acos.hpp>
#include <optinum/simd/math/acosh.hpp>
#include <optinum/simd/math/asin.hpp>
#include <optinum/simd/math/asinh.hpp>
#include <optinum/simd/math/atan.hpp>
#include <optinum/simd/math/atan2.hpp>
#include <optinum/simd/math/atanh.hpp>
#include <optinum/simd/math/cbrt.hpp>
#include <optinum/simd/math/ceil.hpp>
#include <optinum/simd/math/clamp.hpp>
#include <optinum/simd/math/cos.hpp>
#include <optinum/simd/math/cosh.hpp>
#include <optinum/simd/math/exp.hpp>
#include <optinum/simd/math/exp2.hpp>
#include <optinum/simd/math/expm1.hpp>
#include <optinum/simd/math/floor.hpp>
#include <optinum/simd/math/hypot.hpp>
#include <optinum/simd/math/isfinite.hpp>
#include <optinum/simd/math/isinf.hpp>
#include <optinum/simd/math/isnan.hpp>
#include <optinum/simd/math/log.hpp>
#include <optinum/simd/math/log10.hpp>
#include <optinum/simd/math/log1p.hpp>
#include <optinum/simd/math/log2.hpp>
#include <optinum/simd/math/pow.hpp>
#include <optinum/simd/math/round.hpp>
#include <optinum/simd/math/sin.hpp>
#include <optinum/simd/math/sinh.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/math/tan.hpp>
#include <optinum/simd/math/tanh.hpp>
#include <optinum/simd/math/tgamma.hpp>
#include <optinum/simd/math/trunc.hpp>

// Special functions
#include <optinum/simd/math/erf.hpp>
#include <optinum/simd/math/lgamma.hpp>

// Complex math functions
#include <optinum/simd/math/complex_math.hpp>
