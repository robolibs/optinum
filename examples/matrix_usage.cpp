#include <iostream>
#include <optinum/optinum.hpp>

int main() {
    // Create matrices
    optinum::Matrix<float, 3, 3> A;
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;
    A(2, 0) = 7;
    A(2, 1) = 8;
    A(2, 2) = 9;

    optinum::Matrix<float, 3, 3> B;
    B.set_identity();

    std::cout << "A =\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [" << A(i, 0) << ", " << A(i, 1) << ", " << A(i, 2) << "]\n";
    }

    std::cout << "B (identity) =\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [" << B(i, 0) << ", " << B(i, 1) << ", " << B(i, 2) << "]\n";
    }

    // Matrix multiplication
    auto C = A * B;
    std::cout << "A * B =\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [" << C(i, 0) << ", " << C(i, 1) << ", " << C(i, 2) << "]\n";
    }

    // Element-wise addition
    auto D = A + B;
    std::cout << "A + B =\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [" << D(i, 0) << ", " << D(i, 1) << ", " << D(i, 2) << "]\n";
    }

    // Scalar multiplication
    auto E = A * 2.0f;
    std::cout << "A * 2 =\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [" << E(i, 0) << ", " << E(i, 1) << ", " << E(i, 2) << "]\n";
    }

    // Transpose
    auto At = optinum::lina::transpose(A);
    std::cout << "transpose(A) =\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [" << At(i, 0) << ", " << At(i, 1) << ", " << At(i, 2) << "]\n";
    }

    // Trace
    std::cout << "trace(A) = " << optinum::simd::trace(A) << "\n";

    // Frobenius norm
    std::cout << "frobenius_norm(A) = " << optinum::simd::frobenius_norm(A) << "\n";

    // Matrix-vector multiplication
    optinum::Vector<float, 3> v;
    v[0] = 1;
    v[1] = 2;
    v[2] = 3;
    auto Av = A * v;
    std::cout << "A * v = [" << Av[0] << ", " << Av[1] << ", " << Av[2] << "]\n";

    // Non-square matrix
    optinum::Matrix<float, 2, 3> M;
    M(0, 0) = 1;
    M(0, 1) = 2;
    M(0, 2) = 3;
    M(1, 0) = 4;
    M(1, 1) = 5;
    M(1, 2) = 6;

    optinum::Matrix<float, 3, 2> N;
    N(0, 0) = 1;
    N(0, 1) = 2;
    N(1, 0) = 3;
    N(1, 1) = 4;
    N(2, 0) = 5;
    N(2, 1) = 6;

    auto MN = M * N; // 2x3 * 3x2 = 2x2
    std::cout << "M (2x3) * N (3x2) = (2x2)\n";
    std::cout << "  [" << MN(0, 0) << ", " << MN(0, 1) << "]\n";
    std::cout << "  [" << MN(1, 0) << ", " << MN(1, 1) << "]\n";

    // Access underlying datapod
    datapod::mat::matrix<float, 3, 3> &pod = A.pod();
    std::cout << "pod(0,0) = " << pod(0, 0) << "\n";

    // Identity factory
    optinum::Matrix<float, 4, 4> I;
    I.set_identity();
    std::cout << "identity<4>() diagonal = [" << I(0, 0) << ", " << I(1, 1) << ", " << I(2, 2) << ", " << I(3, 3)
              << "]\n";

#if defined(SHORT_NAMESPACE)
    // Short namespace
    on::Matrix<float, 3, 3> R;
    R.set_identity();
    std::cout << "on::Matrix<float, 3, 3> trace = " << optinum::simd::trace(R) << "\n";
#endif

    return 0;
}
