#pragma once

// Architecture detection and macros
#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/arch/macros.hpp>

// SIMD intrinsics
#include <optinum/simd/intrinsic/avx.hpp>
#include <optinum/simd/intrinsic/avx512.hpp>
#include <optinum/simd/intrinsic/neon.hpp>
#include <optinum/simd/intrinsic/simd_vec.hpp>
#include <optinum/simd/intrinsic/sse.hpp>

// SIMD math
#include <optinum/simd/math/simd_math.hpp>

// User-facing types
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/scalar.hpp>
#include <optinum/simd/tensor.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::simd {} // namespace optinum::simd
