#include <iostream>
#include <optinum/lina/lina.hpp>
#include <optinum/simd/simd.hpp>

namespace dp = datapod;
namespace lina = optinum::lina;
namespace simd = optinum::simd;

int main() {
    // Create matrices using dp::mat::matrix
    dp::mat::Matrix<float, 3, 3> A;
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;
    A(2, 0) = 7;
    A(2, 1) = 8;
    A(2, 2) = 9;

    dp::mat::Matrix<float, 3, 3> B;
    B.set_identity();

    std::cout << "A =\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [" << A(i, 0) << ", " << A(i, 1) << ", " << A(i, 2) << "]\n";
    }

    std::cout << "B (identity) =\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [" << B(i, 0) << ", " << B(i, 1) << ", " << B(i, 2) << "]\n";
    }

    // Matrix multiplication using SIMD backend
    dp::mat::Matrix<float, 3, 3> C;
    simd::backend::matmul<float, 3, 3, 3>(C.data(), A.data(), B.data());
    std::cout << "A * B =\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [" << C(i, 0) << ", " << C(i, 1) << ", " << C(i, 2) << "]\n";
    }

    // Element-wise addition using SIMD backend
    dp::mat::Matrix<float, 3, 3> D;
    simd::backend::add<float, 9>(D.data(), A.data(), B.data());
    std::cout << "A + B =\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [" << D(i, 0) << ", " << D(i, 1) << ", " << D(i, 2) << "]\n";
    }

    // Scalar multiplication using SIMD backend
    dp::mat::Matrix<float, 3, 3> E;
    simd::backend::mul_scalar<float, 9>(E.data(), A.data(), 2.0f);
    std::cout << "A * 2 =\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [" << E(i, 0) << ", " << E(i, 1) << ", " << E(i, 2) << "]\n";
    }

    // Transpose using SIMD backend
    dp::mat::Matrix<float, 3, 3> At;
    simd::backend::transpose<float, 3, 3>(At.data(), A.data());
    std::cout << "transpose(A) =\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [" << At(i, 0) << ", " << At(i, 1) << ", " << At(i, 2) << "]\n";
    }

    // Trace (sum of diagonal)
    float trace = A(0, 0) + A(1, 1) + A(2, 2);
    std::cout << "trace(A) = " << trace << "\n";

    // Frobenius norm using SIMD backend
    std::cout << "frobenius_norm(A) = " << simd::backend::norm_l2<float, 9>(A.data()) << "\n";

    // Matrix-vector multiplication using SIMD backend
    dp::mat::Vector<float, 3> v{1, 2, 3};
    dp::mat::Vector<float, 3> Av;
    simd::backend::matvec<float, 3, 3>(Av.data(), A.data(), v.data());
    std::cout << "A * v = [" << Av[0] << ", " << Av[1] << ", " << Av[2] << "]\n";

    // Non-square matrix
    dp::mat::Matrix<float, 2, 3> M;
    M(0, 0) = 1;
    M(0, 1) = 2;
    M(0, 2) = 3;
    M(1, 0) = 4;
    M(1, 1) = 5;
    M(1, 2) = 6;

    dp::mat::Matrix<float, 3, 2> N;
    N(0, 0) = 1;
    N(0, 1) = 2;
    N(1, 0) = 3;
    N(1, 1) = 4;
    N(2, 0) = 5;
    N(2, 1) = 6;

    // 2x3 * 3x2 = 2x2
    dp::mat::Matrix<float, 2, 2> MN;
    simd::backend::matmul<float, 2, 3, 2>(MN.data(), M.data(), N.data());
    std::cout << "M (2x3) * N (3x2) = (2x2)\n";
    std::cout << "  [" << MN(0, 0) << ", " << MN(0, 1) << "]\n";
    std::cout << "  [" << MN(1, 0) << ", " << MN(1, 1) << "]\n";

    // Direct access to dp::mat::matrix
    std::cout << "A(0,0) = " << A(0, 0) << "\n";

    // Identity matrix
    dp::mat::Matrix<float, 4, 4> I;
    I.set_identity();
    std::cout << "identity<4>() diagonal = [" << I(0, 0) << ", " << I(1, 1) << ", " << I(2, 2) << ", " << I(3, 3)
              << "]\n";

    return 0;
}
