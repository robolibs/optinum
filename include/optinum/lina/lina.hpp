#pragma once

// =============================================================================
// optinum/lina/lina.hpp
// Linear algebra (fixed-size) on optinum::simd types
// =============================================================================

#include <optinum/lina/algebra/contraction.hpp>
#include <optinum/lina/algebra/einsum.hpp>
#include <optinum/lina/algebra/kron.hpp>

#include <optinum/lina/basic/adjoint.hpp>
#include <optinum/lina/basic/cofactor.hpp>
#include <optinum/lina/basic/cond.hpp>
#include <optinum/lina/basic/determinant.hpp>
#include <optinum/lina/basic/expmat.hpp>
#include <optinum/lina/basic/inverse.hpp>
#include <optinum/lina/basic/is_finite.hpp>
#include <optinum/lina/basic/log_det.hpp>
#include <optinum/lina/basic/matmul.hpp>
#include <optinum/lina/basic/norm.hpp>
#include <optinum/lina/basic/null.hpp>
#include <optinum/lina/basic/orth.hpp>
#include <optinum/lina/basic/pinv.hpp>
#include <optinum/lina/basic/properties.hpp>
#include <optinum/lina/basic/rank.hpp>
#include <optinum/lina/basic/transpose.hpp>

#include <optinum/lina/decompose/cholesky.hpp>
#include <optinum/lina/decompose/eigen.hpp>
#include <optinum/lina/decompose/lu.hpp>
#include <optinum/lina/decompose/qr.hpp>
#include <optinum/lina/decompose/svd.hpp>

#include <optinum/lina/expr/expr.hpp>

#include <optinum/lina/solve/lstsq.hpp>
#include <optinum/lina/solve/solve.hpp>
#include <optinum/lina/solve/triangular_solve.hpp>

namespace optinum::lina {}
