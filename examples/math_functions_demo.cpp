// =============================================================================
// Math Functions Demo - optinum:: namespace exposes everything
// =============================================================================

#include <iostream>
#include <optinum/optinum.hpp>

int main() {
    std::cout << "=== Optinum Unified API - Complete Demo ===\n\n";

    // =============================================================================
    // Part 1: Types (all from optinum::)
    // =============================================================================

    std::cout << "1. Types (from optinum::):\n";
    std::cout << "   optinum::Matrix, Vector, Tensor, Scalar, Complex\n\n";

    // =============================================================================
    // Part 2: Linear Algebra (all from optinum::)
    // =============================================================================

    std::cout << "2. Linear Algebra Functions:\n";
    optinum::Matrix<double, 3, 3> A{};
    A.fill(1.0);
    A(0, 0) = 4.0;
    A(1, 1) = 3.0;
    A(2, 2) = 5.0;

    optinum::Vector<double, 3> b{};
    b.fill(1.0);

    auto x = optinum::solve(A, b);
    auto d = optinum::determinant(A);
    auto Ainv = optinum::inverse(A);
    auto lu_result = optinum::lu(A);

    std::cout << "   ✓ solve, determinant, inverse, lu, qr, svd\n";
    std::cout << "   ✓ cholesky, eigen_sym, lstsq\n";
    std::cout << "   ✓ matmul, transpose, adjoint, cofactor\n";
    std::cout << "   ✓ dot, norm, cross, einsum\n\n";

    // =============================================================================
    // Part 3: SIMD Math Functions (40+ functions!)
    // =============================================================================

    std::cout << "3. SIMD Math Functions (40+ available!):\n";
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
    // Part 4: SIMD Algorithms
    // =============================================================================

    std::cout << "4. SIMD Algorithm Functions:\n";
    std::cout << "   Reduce: sum, min, max\n";
    std::cout << "   Element-wise: add, sub, mul, div, fill, copy\n";
    std::cout << "   Utilities: view, noalias, torowmajor, tocolumnmajor\n";
    std::cout << "   Mechanics: to_voigt, from_voigt\n\n";

    // =============================================================================
    // Summary
    // =============================================================================

    std::cout << "=== Summary ===\n";
    std::cout << "Everything exposed through optinum:: namespace!\n";
    std::cout << "- 5 core types\n";
    std::cout << "- 20+ linear algebra functions\n";
    std::cout << "- 40+ SIMD math functions\n";
    std::cout << "- 10+ algorithm functions\n";
    std::cout << "- 5+ utility functions\n";
    std::cout << "\nClean, unified, Armadillo-style API! ✓\n";
    std::cout << "\nOr use SHORT_NAMESPACE: on::Matrix, on::solve, etc.\n";

    return 0;
}
