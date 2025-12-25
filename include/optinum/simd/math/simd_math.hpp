#pragma once

// =============================================================================
// optinum/simd/math/simd_math.hpp
// High-performance SIMD math API using pack<T,W>
//
// Functions available:
//   simd::exp(pack<T,W>)   - exponential
//   simd::log(pack<T,W>)   - natural logarithm
//   simd::sin(pack<T,W>)   - sine
//   simd::cos(pack<T,W>)   - cosine
//   simd::tan(pack<T,W>)   - tangent
//   simd::tanh(pack<T,W>)  - hyperbolic tangent
//   simd::sqrt(pack<T,W>)  - square root
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
#include <optinum/simd/math/cos.hpp>
#include <optinum/simd/math/exp.hpp>
#include <optinum/simd/math/log.hpp>
#include <optinum/simd/math/sin.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/math/tan.hpp>
#include <optinum/simd/math/tanh.hpp>
