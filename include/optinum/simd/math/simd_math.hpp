#pragma once

// =============================================================================
// optinum/simd/math/simd_math.hpp
// Fastor-like SIMD math API for SIMDVec<T,Width>
// =============================================================================

// Bring in SIMDVec specializations if available.
#include <optinum/simd/intrinsic/avx.hpp>
#include <optinum/simd/intrinsic/avx512.hpp>
#include <optinum/simd/intrinsic/neon.hpp>
#include <optinum/simd/intrinsic/simd_vec.hpp>
#include <optinum/simd/intrinsic/sse.hpp>

// Base math functions (template fallback using std::)
#include <optinum/simd/math/elementary.hpp>
#include <optinum/simd/math/exponential.hpp>
#include <optinum/simd/math/hyperbolic.hpp>
#include <optinum/simd/math/pow.hpp>
#include <optinum/simd/math/special.hpp>
#include <optinum/simd/math/trig.hpp>

// Our own fast native SIMD implementations (no external deps)
#include <optinum/simd/math/fast_exp.hpp>
#include <optinum/simd/math/fast_hyp.hpp>
#include <optinum/simd/math/fast_log.hpp>
#include <optinum/simd/math/fast_pow.hpp>
#include <optinum/simd/math/fast_trig.hpp>

// Optional: SLEEF bindings (when OPTINUM_USE_SLEEF is defined)
// SLEEF provides higher precision (~1 ULP) but our fast_* are faster (~3-5 ULP)
#include <optinum/simd/math/sleef.hpp>

// Optional: Intel SVML hooks
#include <optinum/simd/math/svml.hpp>
