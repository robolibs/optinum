#pragma once

// =============================================================================
// optinum/optinum.hpp
// Public API - Core types in optinum::, modules in their own namespaces
// =============================================================================

#include <optinum/lie/lie.hpp>
#include <optinum/lina/lina.hpp>
#include <optinum/meta/meta.hpp>
#include <optinum/opti/opti.hpp>
#include <optinum/simd/simd.hpp>

namespace optinum {

    // =========================================================================
    // Core Types (always exposed)
    // =========================================================================

    /// Dynamic size constant (for runtime-sized containers)
    using simd::Dynamic;

    /// Matrix type - column-major
    template <typename T, std::size_t R, std::size_t C> using Matrix = simd::Matrix<T, R, C>;

    /// Vector type - 1D array
    template <typename T, std::size_t N> using Vector = simd::Vector<T, N>;

    /// Tensor type - N-dimensional array
    template <typename T, std::size_t... Dims> using Tensor = simd::Tensor<T, Dims...>;

    /// Scalar wrapper type
    template <typename T> using Scalar = simd::Scalar<T>;

    /// Complex number type
    template <typename T, std::size_t N> using Complex = simd::Complex<T, N>;

    // =========================================================================
    // Slicing (always exposed - needed for indexing)
    // =========================================================================

    using simd::all;
    using simd::fix;
    using simd::fseq;
    using simd::seq;

    // View types (non-owning)
    using simd::matrix_view;
    using simd::scalar_view;
    using simd::tensor_view;
    using simd::vector_view;

} // namespace optinum

// =============================================================================
// OPTINUM_EXPOSE_ALL: Expose all functions from submodules into optinum::
// Enable via -DOPTINUM_EXPOSE_ALL or cmake/xmake option
// =============================================================================

#if defined(OPTINUM_EXPOSE_ALL)

namespace optinum {

    // =========================================================================
    // Basic Linear Algebra Operations (from lina::)
    // =========================================================================

    using lina::adjoint;
    using lina::adjugate;
    using lina::cofactor;
    using lina::determinant;
    using lina::inverse;
    using lina::matmul;
    using lina::transpose;

    using lina::cross;
    using lina::dot;
    using lina::norm;
    using lina::norm_fro;

    using lina::gradient;
    using lina::jacobian;
    using lina::jacobian_error;

    using lina::cond;
    using lina::is_finite;
    using lina::is_hermitian;
    using lina::is_positive_definite;
    using lina::is_symmetric;
    using lina::log_det;
    using lina::rank;
    using lina::rcond;
    using lina::slogdet;

    using lina::expmat;
    using lina::null;
    using lina::orth;
    using lina::pinv;

    using simd::frobenius_norm;
    using simd::normalized;
    using simd::trace;

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
    // SIMD Math Functions (from simd::)
    // =========================================================================

    using simd::abs;
    using simd::exp;
    using simd::log;
    using simd::pow;
    using simd::sqrt;

    using simd::acos;
    using simd::asin;
    using simd::atan;
    using simd::atan2;
    using simd::cos;
    using simd::sin;
    using simd::tan;

    using simd::acosh;
    using simd::asinh;
    using simd::atanh;
    using simd::cosh;
    using simd::sinh;
    using simd::tanh;

    using simd::ceil;
    using simd::floor;
    using simd::round;
    using simd::trunc;

    using simd::cbrt;
    using simd::clamp;
    using simd::exp2;
    using simd::expm1;
    using simd::hypot;
    using simd::log10;
    using simd::log1p;
    using simd::log2;

    using simd::isfinite;
    using simd::isinf;
    using simd::isnan;

    using simd::erf;
    using simd::lgamma;
    using simd::tgamma;

    // =========================================================================
    // SIMD Algorithm Functions (from simd::)
    // =========================================================================

    using simd::max;
    using simd::min;
    using simd::sum;

    using simd::add;
    using simd::copy;
    using simd::div;
    using simd::fill;
    using simd::mul;
    using simd::sub;

    // =========================================================================
    // Utility Functions (from simd::)
    // =========================================================================

    using simd::from_voigt;
    using simd::noalias;
    using simd::permute_012_to_201;
    using simd::to_voigt;
    using simd::tocolumnmajor;
    using simd::torowmajor;
    using simd::transpose_3d;
    using simd::view;

    // =========================================================================
    // Optimization Module (from opti::)
    // =========================================================================

    template <typename UpdatePolicy = opti::VanillaUpdate, typename DecayPolicy = opti::NoDecay>
    using GradientDescent = opti::GradientDescent<UpdatePolicy, DecayPolicy>;

    using opti::AdamUpdate;
    using opti::AMSGradUpdate;
    using opti::MomentumUpdate;
    using opti::RMSPropUpdate;
    using opti::VanillaUpdate;

    using Momentum = GradientDescent<opti::MomentumUpdate>;
    using RMSprop = GradientDescent<opti::RMSPropUpdate>;
    using Adam = GradientDescent<opti::AdamUpdate>;
    using AMSGrad = GradientDescent<opti::AMSGradUpdate>;

    template <typename T = double> using GaussNewton = opti::GaussNewton<T>;
    template <typename T = double> using LevenbergMarquardt = opti::LevenbergMarquardt<T>;

    using opti::EarlyStoppingCallback;
    using opti::IterationInfo;
    using opti::LogCallback;
    using opti::NoCallback;
    using opti::NoDecay;
    using opti::OptimizationResult;
    using opti::Sphere;

    // =========================================================================
    // Metaheuristic Optimizers (from meta::)
    // =========================================================================

    template <typename T = double> using PSO = meta::PSO<T>;
    template <typename T> using PSOResult = meta::PSOResult<T>;

    template <typename T = double> using CMAES = meta::CMAES<T>;
    template <typename T> using CMAESResult = meta::CMAESResult<T>;

    template <typename T = double> using DifferentialEvolution = meta::DifferentialEvolution<T>;
    template <typename T> using DEResult = meta::DEResult<T>;

    template <typename T = double> using GeneticAlgorithm = meta::GeneticAlgorithm<T>;
    template <typename T> using GAResult = meta::GAResult<T>;

    // =========================================================================
    // Lie Groups (from lie::)
    // =========================================================================

    template <typename T = double> using SO2 = lie::SO2<T>;
    using SO2f = lie::SO2f;
    using SO2d = lie::SO2d;

    template <typename T = double> using SE2 = lie::SE2<T>;
    using SE2f = lie::SE2f;
    using SE2d = lie::SE2d;

    template <typename T = double> using SO3 = lie::SO3<T>;
    using SO3f = lie::SO3f;
    using SO3d = lie::SO3d;

    template <typename T = double> using SE3 = lie::SE3<T>;
    using SE3f = lie::SE3f;
    using SE3d = lie::SE3d;

    using lie::interpolate;

} // namespace optinum

#endif // OPTINUM_EXPOSE_ALL

// =============================================================================
// Short namespace alias (optional)
// =============================================================================

#if defined(SHORT_NAMESPACE)
namespace on = optinum;
#endif
