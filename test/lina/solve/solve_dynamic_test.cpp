#include <doctest/doctest.h>
#include <optinum/lina/solve/solve_dynamic.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>

using optinum::lina::determinant_dynamic;
using optinum::lina::inverse_dynamic;
using optinum::lina::solve_dynamic;
using optinum::lina::try_inverse_dynamic;
using optinum::lina::try_solve_dynamic;
using optinum::simd::Dynamic;
using optinum::simd::Matrix;
using optinum::simd::Vector;

using DynMat = Matrix<double, Dynamic, Dynamic>;
using DynVec = Vector<double, Dynamic>;

TEST_CASE("lina::solve_dynamic solves Ax=b for 3x3 system") {
    // A = [3, 2, -1; 2, -2, 4; -1, 0.5, -1]
    DynMat a(3, 3);
    a(0, 0) = 3.0;
    a(0, 1) = 2.0;
    a(0, 2) = -1.0;
    a(1, 0) = 2.0;
    a(1, 1) = -2.0;
    a(1, 2) = 4.0;
    a(2, 0) = -1.0;
    a(2, 1) = 0.5;
    a(2, 2) = -1.0;

    DynVec b(3);
    b[0] = 1.0;
    b[1] = -2.0;
    b[2] = 0.0;

    const auto x = solve_dynamic(a, b);

    // Verify Ax = b
    for (std::size_t i = 0; i < 3; ++i) {
        double ax_i = 0.0;
        for (std::size_t j = 0; j < 3; ++j) {
            ax_i += a(i, j) * x[j];
        }
        CHECK(ax_i == doctest::Approx(b[i]).epsilon(1e-9));
    }
}

TEST_CASE("lina::solve_dynamic solves 2x2 system") {
    DynMat a(2, 2);
    a(0, 0) = 4.0;
    a(0, 1) = 3.0;
    a(1, 0) = 6.0;
    a(1, 1) = 3.0;

    DynVec b(2);
    b[0] = 10.0;
    b[1] = 12.0;

    const auto x = solve_dynamic(a, b);

    // Expected: x = [1, 2]
    CHECK(x[0] == doctest::Approx(1.0).epsilon(1e-9));
    CHECK(x[1] == doctest::Approx(2.0).epsilon(1e-9));
}

TEST_CASE("lina::solve_dynamic solves 4x4 system") {
    DynMat a(4, 4);
    a(0, 0) = 2.0;
    a(0, 1) = 1.0;
    a(0, 2) = 0.0;
    a(0, 3) = 0.0;
    a(1, 0) = 1.0;
    a(1, 1) = 3.0;
    a(1, 2) = 1.0;
    a(1, 3) = 0.0;
    a(2, 0) = 0.0;
    a(2, 1) = 1.0;
    a(2, 2) = 4.0;
    a(2, 3) = 1.0;
    a(3, 0) = 0.0;
    a(3, 1) = 0.0;
    a(3, 2) = 1.0;
    a(3, 3) = 5.0;

    DynVec b(4);
    b[0] = 1.0;
    b[1] = 2.0;
    b[2] = 3.0;
    b[3] = 4.0;

    const auto x = solve_dynamic(a, b);

    // Verify Ax = b
    for (std::size_t i = 0; i < 4; ++i) {
        double ax_i = 0.0;
        for (std::size_t j = 0; j < 4; ++j) {
            ax_i += a(i, j) * x[j];
        }
        CHECK(ax_i == doctest::Approx(b[i]).epsilon(1e-9));
    }
}

TEST_CASE("lina::try_solve_dynamic detects singular matrix") {
    // Singular: second row is 2x first row
    DynMat a(2, 2);
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(1, 0) = 2.0;
    a(1, 1) = 4.0;

    DynVec b(2);
    b[0] = 1.0;
    b[1] = 2.0;

    const auto r = try_solve_dynamic(a, b);
    CHECK(r.is_err());
}

TEST_CASE("lina::solve_dynamic handles multiple RHS (AX=B)") {
    DynMat a(2, 2);
    a(0, 0) = 2.0;
    a(0, 1) = 1.0;
    a(1, 0) = 1.0;
    a(1, 1) = 3.0;

    DynMat b(2, 2);
    b(0, 0) = 5.0;
    b(0, 1) = 6.0;
    b(1, 0) = 7.0;
    b(1, 1) = 8.0;

    const auto x = solve_dynamic(a, b);

    // Verify AX = B
    for (std::size_t col = 0; col < 2; ++col) {
        for (std::size_t i = 0; i < 2; ++i) {
            double ax_i = 0.0;
            for (std::size_t j = 0; j < 2; ++j) {
                ax_i += a(i, j) * x(j, col);
            }
            CHECK(ax_i == doctest::Approx(b(i, col)).epsilon(1e-9));
        }
    }
}

TEST_CASE("lina::inverse_dynamic computes matrix inverse") {
    DynMat a(2, 2);
    a(0, 0) = 4.0;
    a(0, 1) = 7.0;
    a(1, 0) = 2.0;
    a(1, 1) = 6.0;

    const auto inv = inverse_dynamic(a);

    // Verify A * inv = I
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < 2; ++k) {
                sum += a(i, k) * inv(k, j);
            }
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(sum == doctest::Approx(expected).epsilon(1e-9));
        }
    }
}

TEST_CASE("lina::try_inverse_dynamic detects singular matrix") {
    DynMat a(2, 2);
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(1, 0) = 2.0;
    a(1, 1) = 4.0;

    const auto r = try_inverse_dynamic(a);
    CHECK(r.is_err());
}

TEST_CASE("lina::determinant_dynamic computes determinant") {
    // A = [3, 8; 4, 6], det = 3*6 - 8*4 = -14
    DynMat a(2, 2);
    a(0, 0) = 3.0;
    a(0, 1) = 8.0;
    a(1, 0) = 4.0;
    a(1, 1) = 6.0;

    const auto det = determinant_dynamic(a);
    CHECK(det == doctest::Approx(-14.0).epsilon(1e-9));
}

TEST_CASE("lina::determinant_dynamic returns 0 for singular matrix") {
    DynMat a(2, 2);
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(1, 0) = 2.0;
    a(1, 1) = 4.0;

    const auto det = determinant_dynamic(a);
    CHECK(det == doctest::Approx(0.0).epsilon(1e-9));
}

TEST_CASE("lina::determinant_dynamic for 3x3 matrix") {
    // A = [6, 1, 1; 4, -2, 5; 2, 8, 7]
    // det = 6*(-2*7 - 5*8) - 1*(4*7 - 5*2) + 1*(4*8 - (-2)*2)
    //     = 6*(-14 - 40) - 1*(28 - 10) + 1*(32 + 4)
    //     = 6*(-54) - 18 + 36 = -324 - 18 + 36 = -306
    DynMat a(3, 3);
    a(0, 0) = 6.0;
    a(0, 1) = 1.0;
    a(0, 2) = 1.0;
    a(1, 0) = 4.0;
    a(1, 1) = -2.0;
    a(1, 2) = 5.0;
    a(2, 0) = 2.0;
    a(2, 1) = 8.0;
    a(2, 2) = 7.0;

    const auto det = determinant_dynamic(a);
    CHECK(det == doctest::Approx(-306.0).epsilon(1e-6));
}

TEST_CASE("lina::solve_dynamic for larger system (5x5)") {
    // Diagonally dominant matrix for numerical stability
    DynMat a(5, 5);
    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            if (i == j) {
                a(i, j) = 10.0;
            } else {
                a(i, j) = 1.0;
            }
        }
    }

    DynVec b(5);
    b[0] = 1.0;
    b[1] = 2.0;
    b[2] = 3.0;
    b[3] = 4.0;
    b[4] = 5.0;

    const auto x = solve_dynamic(a, b);

    // Verify Ax = b
    for (std::size_t i = 0; i < 5; ++i) {
        double ax_i = 0.0;
        for (std::size_t j = 0; j < 5; ++j) {
            ax_i += a(i, j) * x[j];
        }
        CHECK(ax_i == doctest::Approx(b[i]).epsilon(1e-9));
    }
}
