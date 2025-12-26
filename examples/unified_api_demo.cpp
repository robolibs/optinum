// =============================================================================
// Unified API Demo - Using optinum:: namespace
// This demonstrates the clean, unified API
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
    // Basic linear algebra operations
    // =========================================================================

    std::cout << "\n2. Basic operations (all from optinum::):\n";

    double d = optinum::determinant(A);
    std::cout << "det(A) = " << d << "\n";

    auto Ainv = optinum::inverse(A);
    std::cout << "A^(-1) = \n" << Ainv << "\n";

    // =========================================================================
    // Solving linear systems
    // =========================================================================

    std::cout << "\n3. Solving Ax = b (using optinum::solve):\n";

    auto x = optinum::solve(A, b);
    std::cout << "Solution x = \n" << x << "\n";

    // =========================================================================
    // Matrix decompositions
    // =========================================================================

    std::cout << "\n4. Decompositions (all from optinum::):\n";

    auto lu_result = optinum::lu(A);
    std::cout << "LU decomposition computed\n";
    std::cout << "Permutation sign: " << lu_result.sign << "\n";

    auto qr_result = optinum::qr(A);
    std::cout << "QR decomposition computed\n";

    // =========================================================================
    // Summary
    // =========================================================================

    std::cout << "\n=== Summary ===\n";
    std::cout << "All operations used the optinum:: namespace!\n";
    std::cout << "Types: Matrix, Vector, Tensor, Scalar, Complex\n";
    std::cout << "Functions: solve, determinant, inverse, lu, qr, svd, etc.\n";
    std::cout << "\nClean, unified API! ✓\n";

    return 0;
}
