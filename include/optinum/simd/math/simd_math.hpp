#pragma once

// =============================================================================
// optinum/simd/math/simd_math.hpp
// High-performance SIMD math API for SIMDVec<T,Width>
//
// Provides vectorized implementations of common math functions:
// - Exponential: exp, log
// - Trigonometric: sin, cos, sincos
// - Hyperbolic: tanh, sinh, cosh
// - Power: pow, powi, sqrt, rsqrt, cbrt
// - Elementary: abs, min, max, clamp, etc.
//
// All functions are optimized for speed (~3-5 ULP accuracy).
// =============================================================================

// Bring in SIMDVec specializations
#include <optinum/simd/intrinsic/avx.hpp>
#include <optinum/simd/intrinsic/avx512.hpp>
#include <optinum/simd/intrinsic/neon.hpp>
#include <optinum/simd/intrinsic/simd_vec.hpp>
#include <optinum/simd/intrinsic/sse.hpp>

// Template fallback implementations (lane-by-lane using std::)
#include <optinum/simd/math/elementary.hpp>
#include <optinum/simd/math/exponential.hpp>
#include <optinum/simd/math/hyperbolic.hpp>
#include <optinum/simd/math/pow.hpp>
#include <optinum/simd/math/special.hpp>
#include <optinum/simd/math/trig.hpp>

// Native SIMD implementations (fast, no external dependencies)
// These provide non-template overloads that take precedence over the template fallbacks
#include <optinum/simd/math/fast_exp.hpp>
#include <optinum/simd/math/fast_hyp.hpp>
#include <optinum/simd/math/fast_log.hpp>
#include <optinum/simd/math/fast_pow.hpp>
#include <optinum/simd/math/fast_trig.hpp>
