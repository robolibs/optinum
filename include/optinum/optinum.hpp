#pragma once

// =============================================================================
// optinum/optinum.hpp
// Unified public API - everything exposed through optinum:: namespace
// =============================================================================

#include <optinum/lina/lina.hpp>
#include <optinum/simd/simd.hpp>
// #include <optinum/opti/opti.hpp>  // TODO: implement

namespace optinum {

    // =========================================================================
    // Core Types (from simd::)
    // =========================================================================

    /// Matrix type - column-major, fixed-size
    template <typename T, std::size_t R, std::size_t C> using Matrix = simd::Matrix<T, R, C>;

    /// Vector type - fixed-size 1D array
    template <typename T, std::size_t N> using Vector = simd::Vector<T, N>;

    /// Tensor type - fixed-size N-dimensional array
    template <typename T, std::size_t... Dims> using Tensor = simd::Tensor<T, Dims...>;

    /// Scalar wrapper type
    template <typename T> using Scalar = simd::Scalar<T>;

    /// Complex number type (array of complex numbers)
    template <typename T, std::size_t N> using Complex = simd::Complex<T, N>;

    // =========================================================================
    // Basic Linear Algebra Operations (from lina::)
    // =========================================================================

    // Basic operations
    using lina::adjoint;
    using lina::adjugate; // Alias for adjoint
    using lina::cofactor;
    using lina::determinant;
    using lina::inverse;
    using lina::matmul;
    using lina::transpose;

    // Norms and dot products
    using lina::cross;
    using lina::dot;
    using lina::norm;
    using lina::norm_fro;

    // BLAS-like operations
    using lina::axpy;
    using lina::scale;

    // =========================================================================
    // Matrix Decompositions (from lina::)
    // =========================================================================

    using lina::cholesky;
    using lina::eigen_sym;
    using lina::lu;
    using lina::qr;
    using lina::svd;

    // =========================================================================
    // Linear System Solvers (from lina::)
    // =========================================================================

    using lina::lstsq;
    using lina::solve;

    // =========================================================================
    // Tensor Algebra (from lina::)
    // =========================================================================

    using lina::einsum;
    using lina::hadamard;
    using lina::inner;
    using lina::outer;

    // =========================================================================
    // SIMD Math & Transform Functions (from simd::)
    // =========================================================================

    // Basic math
    using simd::abs;
    using simd::exp;
    using simd::log;
    using simd::pow;
    using simd::sqrt;

    // Trigonometric
    using simd::acos;
    using simd::asin;
    using simd::atan;
    using simd::atan2;
    using simd::cos;
    using simd::sin;
    using simd::tan;

    // Hyperbolic
    using simd::acosh;
    using simd::asinh;
    using simd::atanh;
    using simd::cosh;
    using simd::sinh;
    using simd::tanh;

    // Rounding
    using simd::ceil;
    using simd::floor;
    using simd::round;
    using simd::trunc;

    // Additional math
    using simd::cbrt;
    using simd::clamp;
    using simd::exp2;
    using simd::expm1;
    using simd::hypot;
    using simd::log10;
    using simd::log1p;
    using simd::log2;

    // Boolean/status
    using simd::isfinite;
    using simd::isinf;
    using simd::isnan;

    // Special functions
    using simd::erf;
    using simd::lgamma;
    using simd::tgamma;

    // =========================================================================
    // SIMD Algorithm Functions (from simd::)
    // =========================================================================

    // Reduce operations on views
    using simd::max;
    using simd::min;
    using simd::sum;

    // Elementwise operations on views (add, sub, mul, div, fill already in simd/algo)
    // Note: These work on views, exposed for advanced users
    using simd::add;
    using simd::copy;
    using simd::div;
    using simd::fill;
    using simd::mul;
    using simd::sub;

    // =========================================================================
    // Utility Functions (from simd::)
    // =========================================================================

    // View factory
    using simd::view;

    // Layout conversion
    using simd::tocolumnmajor;
    using simd::torowmajor;

    // Voigt notation (mechanics)
    using simd::from_voigt;
    using simd::to_voigt;

    // Optimization hints
    using simd::noalias;

    // =========================================================================
    // Expression Templates (from lina::)
    // =========================================================================

    // Expression templates are implementation details, not directly exposed
    // Users get them automatically through operator overloading

    // =========================================================================
    // Future: Optimization Module (from opti::)
    // =========================================================================

    // When opti:: is implemented, add here:
    // using GradientDescent = opti::GradientDescent;
    // using Adam = opti::Adam;
    // using opti::minimize;
    // etc.

} // namespace optinum

// =============================================================================
// Short namespace alias (optional)
// =============================================================================

#if defined(SHORT_NAMESPACE)
namespace on = optinum;
#endif
