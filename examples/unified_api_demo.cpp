// =============================================================================
// Unified API Demo - Using optinum:: namespace
// This demonstrates the namespace organization
// =============================================================================

#include <iostream>
#include <optinum/optinum.hpp>

int main() {
    std::cout << "=== Optinum Unified API Demo ===\n\n";

    // =========================================================================
    // Create matrices and vectors using optinum:: types
    // =========================================================================

    std::cout << "1. Creating matrices and vectors:\n";
    optinum::Matrix<double, 3, 3> A{};
    A.fill(2.0);
    A(0, 0) = 4.0;
    A(1, 1) = 3.0;
    A(2, 2) = 5.0;

    optinum::Vector<double, 3> b{};
    b.fill(1.0);

    std::cout << "Matrix A (3x3):\n" << A << "\n";
    std::cout << "Vector b (3):\n" << b << "\n";

    // =========================================================================
    // Basic linear algebra operations (optinum::lina::)
    // =========================================================================

    std::cout << "\n2. Basic operations (from optinum::lina::):\n";

    double d = optinum::lina::determinant(A);
    std::cout << "det(A) = " << d << "\n";

    auto Ainv = optinum::lina::inverse(A);
    std::cout << "A^(-1) = \n" << Ainv << "\n";

    // =========================================================================
    // Solving linear systems
    // =========================================================================

    std::cout << "\n3. Solving Ax = b (using optinum::lina::solve):\n";

    auto x = optinum::lina::solve(A, b);
    std::cout << "Solution x = \n" << x << "\n";

    // =========================================================================
    // Matrix decompositions
    // =========================================================================

    std::cout << "\n4. Decompositions (from optinum::lina::):\n";

    auto lu_result = optinum::lina::lu(A);
    std::cout << "LU decomposition computed\n";
    std::cout << "Permutation sign: " << lu_result.sign << "\n";

    auto qr_result = optinum::lina::qr(A);
    std::cout << "QR decomposition computed\n";

    // =========================================================================
    // Summary
    // =========================================================================

    std::cout << "\n=== Summary ===\n";
    std::cout << "Namespace organization:\n";
    std::cout << "  optinum::        - Core types (Matrix, Vector, etc.)\n";
    std::cout << "  optinum::lina::  - Linear algebra (solve, inverse, lu, qr, etc.)\n";
    std::cout << "  optinum::simd::  - SIMD operations (trace, frobenius_norm, etc.)\n";
    std::cout << "  optinum::opti::  - Optimization algorithms\n";
    std::cout << "  optinum::meta::  - Metaheuristic optimizers\n";
    std::cout << "  optinum::lie::   - Lie groups\n";

    return 0;
}
