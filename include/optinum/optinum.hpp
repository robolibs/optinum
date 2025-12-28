#pragma once

// =============================================================================
// optinum/optinum.hpp
// Unified public API - everything exposed through optinum:: namespace
// =============================================================================

#include <optinum/lie/lie.hpp>
#include <optinum/lina/lina.hpp>
#include <optinum/opti/opti.hpp>
#include <optinum/simd/simd.hpp>

namespace optinum {

    // =========================================================================
    // Core Types (from simd::)
    // =========================================================================

    /// Dynamic size constant (for runtime-sized containers)
    /// Use as template parameter: Vector<double, Dynamic>
    using simd::Dynamic;

    /// Matrix type - column-major
    /// Fixed-size: Matrix<double, 3, 4>
    /// Dynamic-size: Matrix<double, Dynamic, Dynamic>
    template <typename T, std::size_t R, std::size_t C> using Matrix = simd::Matrix<T, R, C>;

    /// Vector type - 1D array
    /// Fixed-size: Vector<double, 10>
    /// Dynamic-size: Vector<double, Dynamic>
    template <typename T, std::size_t N> using Vector = simd::Vector<T, N>;

    /// Tensor type - N-dimensional array
    /// Fixed-size: Tensor<double, 2, 3, 4>
    /// Dynamic-size: Tensor<double, Dynamic, Dynamic, Dynamic>
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

    // Numerical differentiation
    using lina::gradient;
    using lina::jacobian;
    using lina::jacobian_error;

    // Matrix properties
    using lina::cond;
    using lina::is_finite;
    using lina::is_hermitian;
    using lina::is_positive_definite;
    using lina::is_symmetric;
    using lina::log_det;
    using lina::rank;
    using lina::rcond;
    using lina::slogdet;

    // Advanced operations
    using lina::expmat;
    using lina::null;
    using lina::orth;
    using lina::pinv;

    // Also expose from simd:: (same functions, different namespace)
    using simd::frobenius_norm;
    using simd::normalized;
    using simd::trace;

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

    using lina::dare;
    using lina::lqr_gain;
    using lina::lstsq;
    using lina::solve;
    using lina::solve_lower_triangular;
    using lina::solve_upper_triangular;
    using lina::triangular_solve;

    // =========================================================================
    // Tensor Algebra (from lina::)
    // =========================================================================

    using lina::einsum;
    using lina::hadamard;
    using lina::inner;
    using lina::kron;
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

    // Tensor permutations
    using simd::permute_012_to_201;
    using simd::transpose_3d;

    // Optimization hints
    using simd::noalias;

    // =========================================================================
    // Slicing & Indexing (from simd::)
    // =========================================================================

    // Slice types
    using simd::all;
    using simd::fix;
    using simd::fseq;
    using simd::seq;

    // View types (non-owning)
    using simd::matrix_view;
    using simd::scalar_view;
    using simd::tensor_view;
    using simd::vector_view;

    // =========================================================================
    // Expression Templates (from lina::)
    // =========================================================================

    // Expression templates are implementation details, not directly exposed
    // Users get them automatically through operator overloading

    // =========================================================================
    // Optimization Module (from opti::)
    // =========================================================================

    // First-order optimizers (gradient-based)
    template <typename UpdatePolicy = opti::VanillaUpdate, typename DecayPolicy = opti::NoDecay>
    using GradientDescent = opti::GradientDescent<UpdatePolicy, DecayPolicy>;

    // Update policies
    using opti::AdamUpdate;
    using opti::MomentumUpdate;
    using opti::RMSPropUpdate;
    using opti::VanillaUpdate;

    // Convenient optimizer aliases
    using Momentum = GradientDescent<opti::MomentumUpdate>;
    using RMSprop = GradientDescent<opti::RMSPropUpdate>;
    using Adam = GradientDescent<opti::AdamUpdate>;

    // Second-order optimizers (quasi-Newton methods)
    template <typename T = double> using GaussNewton = opti::GaussNewton<T>;
    template <typename T = double> using LevenbergMarquardt = opti::LevenbergMarquardt<T>;

    // Decay policies
    using opti::NoDecay;

    // Callbacks
    using opti::EarlyStoppingCallback;
    using opti::LogCallback;
    using opti::NoCallback;

    // Result types
    using opti::IterationInfo;
    using opti::OptimizationResult;

    // Test problems
    using opti::Sphere;

    // =========================================================================
    // Lie Groups (from lie::)
    // =========================================================================

    // 2D rotation group (unit complex number)
    template <typename T = double> using SO2 = lie::SO2<T>;
    using SO2f = lie::SO2f;
    using SO2d = lie::SO2d;

    // 2D rigid transform group (rotation + translation)
    template <typename T = double> using SE2 = lie::SE2<T>;
    using SE2f = lie::SE2f;
    using SE2d = lie::SE2d;

    // 3D rotation group (unit quaternion)
    template <typename T = double> using SO3 = lie::SO3<T>;
    using SO3f = lie::SO3f;
    using SO3d = lie::SO3d;

    // Lie group utilities
    using lie::interpolate;
    // using lie::average;  // Has template deduction issues, use lie::average directly

    // Constants (from lie::)
    // These are already namespaced as lie::pi<T>, lie::epsilon<T>, etc.

} // namespace optinum

// =============================================================================
// Short namespace alias (optional)
// =============================================================================

#if defined(SHORT_NAMESPACE)
namespace on = optinum;
#endif
