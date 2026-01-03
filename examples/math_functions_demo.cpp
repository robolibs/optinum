// =============================================================================
// Math Functions Demo - optinum library namespace organization
// =============================================================================

#include <iostream>
#include <optinum/optinum.hpp>

namespace dp = datapod;

int main() {
    std::cout << "=== Optinum Library - Namespace Organization Demo ===\n\n";

    // =============================================================================
    // Part 1: Core Types (directly in optinum::)
    // =============================================================================

    std::cout << "1. Core Types (from optinum::):\n";
    std::cout << "   optinum::Matrix, Vector, Tensor, Scalar, Complex\n\n";

    // =============================================================================
    // Part 2: Linear Algebra (optinum::lina::)
    // =============================================================================

    std::cout << "2. Linear Algebra Functions (optinum::lina::):\n";

    // Use dp::mat types for owning storage
    dp::mat::Matrix<double, 3, 3> A_storage;
    optinum::Matrix<double, 3, 3> A(A_storage); // Non-owning view
    A.fill(1.0);
    A(0, 0) = 4.0;
    A(1, 1) = 3.0;
    A(2, 2) = 5.0;

    dp::mat::Vector<double, 3> b_storage;
    optinum::Vector<double, 3> b(b_storage); // Non-owning view
    b.fill(1.0);

    auto x = optinum::lina::solve(A, b);
    auto d = optinum::lina::determinant(A);
    auto Ainv = optinum::lina::inverse(A);
    auto lu_result = optinum::lina::lu(A);

    std::cout << "   solve, determinant, inverse, lu, qr, svd\n";
    std::cout << "   cholesky, eigen_sym, lstsq\n";
    std::cout << "   matmul, transpose, adjoint, cofactor\n";
    std::cout << "   dot, norm, cross, einsum\n\n";

    // =============================================================================
    // Part 3: SIMD Math Functions (optinum::simd::)
    // =============================================================================

    std::cout << "3. SIMD Math Functions (optinum::simd:: - 40+ available!):\n";
    std::cout << "   Exponential/Log:\n";
    std::cout << "     exp, log, exp2, expm1, log2, log10, log1p\n";
    std::cout << "   Trigonometric:\n";
    std::cout << "     sin, cos, tan, asin, acos, atan, atan2\n";
    std::cout << "   Hyperbolic:\n";
    std::cout << "     sinh, cosh, tanh, asinh, acosh, atanh\n";
    std::cout << "   Rounding:\n";
    std::cout << "     ceil, floor, round, trunc\n";
    std::cout << "   Other:\n";
    std::cout << "     sqrt, pow, abs, cbrt, clamp, hypot\n";
    std::cout << "     isnan, isinf, isfinite\n";
    std::cout << "     erf, tgamma, lgamma\n\n";

    // =============================================================================
    // Part 4: Optimization (optinum::opti::)
    // =============================================================================

    std::cout << "4. Optimization (optinum::opti::):\n";
    std::cout << "   Gradient-based: GradientDescent, Adam, RMSprop, Momentum\n";
    std::cout << "   Quasi-Newton: GaussNewton, LevenbergMarquardt, LBFGS\n";
    std::cout << "   Problems: Sphere, Rosenbrock, Rastrigin, Ackley\n\n";

    // =============================================================================
    // Part 5: Metaheuristics (optinum::meta::)
    // =============================================================================

    std::cout << "5. Metaheuristics (optinum::meta::):\n";
    std::cout << "   PSO, DE, CEM, CMAES, GA, SA, MPPI\n\n";

    // =============================================================================
    // Part 6: Lie Groups (optinum::lie::)
    // =============================================================================

    std::cout << "6. Lie Groups (optinum::lie::):\n";
    std::cout << "   SO2, SO3, SE2, SE3, Sim3\n\n";

    // =============================================================================
    // Summary
    // =============================================================================

    std::cout << "=== Summary ===\n";
    std::cout << "Namespace organization:\n";
    std::cout << "  optinum::        - Core types (Matrix, Vector, etc.)\n";
    std::cout << "  optinum::lina::  - Linear algebra\n";
    std::cout << "  optinum::simd::  - SIMD operations\n";
    std::cout << "  optinum::opti::  - Optimization algorithms\n";
    std::cout << "  optinum::meta::  - Metaheuristic optimizers\n";
    std::cout << "  optinum::lie::   - Lie groups\n";
    std::cout << "\nDefine OPTINUM_EXPOSE_ALL to expose everything in optinum::\n";

    return 0;
}
