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

#include <optinum/simd/math/elementary.hpp>
#include <optinum/simd/math/exponential.hpp>
#include <optinum/simd/math/hyperbolic.hpp>
#include <optinum/simd/math/pow.hpp>
#include <optinum/simd/math/sleef.hpp>
#include <optinum/simd/math/special.hpp>
#include <optinum/simd/math/svml.hpp>
#include <optinum/simd/math/trig.hpp>
