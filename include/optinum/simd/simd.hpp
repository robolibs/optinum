#pragma once

// Architecture detection and macros
#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/arch/macros.hpp>

// pack<T,W> - SIMD register abstraction
#include <optinum/simd/pack/avx.hpp>
#include <optinum/simd/pack/avx512.hpp>
#include <optinum/simd/pack/complex.hpp>
#include <optinum/simd/pack/neon.hpp>
#include <optinum/simd/pack/pack.hpp>
#include <optinum/simd/pack/sse.hpp>

// mask<T,W> - comparison results
#include <optinum/simd/mask.hpp>

// SIMD math functions (exp, log, sin, cos, tanh, sqrt)
#include <optinum/simd/math/simd_math.hpp>

// Views - non-owning views over datapod types
#include <optinum/simd/view/view.hpp>

// Bridge - view<W>(dp_obj) factory
#include <optinum/simd/bridge.hpp>

// Algorithms on views
#include <optinum/simd/algo/elementwise.hpp>
#include <optinum/simd/algo/reduce.hpp>
#include <optinum/simd/algo/transform.hpp>

// Backend - low-level SIMD operations on raw arrays
#include <optinum/simd/backend/backend.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/backend/matmul.hpp>
#include <optinum/simd/backend/norm.hpp>
#include <optinum/simd/backend/reduce.hpp>
#include <optinum/simd/backend/transpose.hpp>

// User-facing types (legacy, wrapping datapod)
#include <optinum/simd/complex.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/scalar.hpp>
#include <optinum/simd/tensor.hpp>
#include <optinum/simd/vector.hpp>

// I/O utilities
#include <optinum/simd/io.hpp>

namespace optinum::simd {} // namespace optinum::simd
