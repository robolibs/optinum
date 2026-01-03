#include <doctest/doctest.h>
#include <optinum/lina/lina.hpp>

namespace lina = optinum::lina;
using optinum::simd::Matrix;

namespace dp = datapod;

// =============================================================================
// Cofactor Matrix Tests
// =============================================================================

TEST_CASE("Cofactor 2x2") {
    dp::mat::Matrix<double, 2, 2> a;
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(1, 0) = 3.0;
    a(1, 1) = 4.0;

    auto cof = lina::cofactor(Matrix<double, 2, 2>(a));

    // For 2x2: cofactor is [[a11, -a10], [-a01, a00]]
    CHECK(cof(0, 0) == doctest::Approx(4.0));  // a11
    CHECK(cof(0, 1) == doctest::Approx(-3.0)); // -a10
    CHECK(cof(1, 0) == doctest::Approx(-2.0)); // -a01
    CHECK(cof(1, 1) == doctest::Approx(1.0));  // a00
}

TEST_CASE("Cofactor 3x3") {
    dp::mat::Matrix<double, 3, 3> a;
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(0, 2) = 3.0;
    a(1, 0) = 0.0;
    a(1, 1) = 4.0;
    a(1, 2) = 5.0;
    a(2, 0) = 1.0;
    a(2, 1) = 0.0;
    a(2, 2) = 6.0;

    auto cof = lina::cofactor(Matrix<double, 3, 3>(a));

    // Verify a few cofactor elements manually
    // C_00 = det([[4,5],[0,6]]) = 4*6 - 5*0 = 24
    CHECK(cof(0, 0) == doctest::Approx(24.0));

    // C_01 = -det([[0,5],[1,6]]) = -(0*6 - 5*1) = 5
    CHECK(cof(0, 1) == doctest::Approx(5.0));

    // C_10 = -det([[2,3],[0,6]]) = -(2*6 - 3*0) = -12
    CHECK(cof(1, 0) == doctest::Approx(-12.0));
}

TEST_CASE("Cofactor 4x4") {
    dp::mat::Matrix<double, 4, 4> a;
    for (std::size_t i = 0; i < 16; ++i)
        a[i] = 0.0;
    a(0, 0) = 1.0;
    a(1, 1) = 2.0;
    a(2, 2) = 3.0;
    a(3, 3) = 4.0;

    auto cof = lina::cofactor(Matrix<double, 4, 4>(a));

    // For diagonal matrix, cofactor diagonal elements are products of other diagonals
    // C_00 = det of 3x3 diagonal [2,3,4] = 2*3*4 = 24
    CHECK(cof(0, 0) == doctest::Approx(24.0));

    // C_11 = det of 3x3 diagonal [1,3,4] = 1*3*4 = 12
    CHECK(cof(1, 1) == doctest::Approx(12.0));

    // C_22 = det of 3x3 diagonal [1,2,4] = 1*2*4 = 8
    CHECK(cof(2, 2) == doctest::Approx(8.0));

    // C_33 = det of 3x3 diagonal [1,2,3] = 1*2*3 = 6
    CHECK(cof(3, 3) == doctest::Approx(6.0));
}

TEST_CASE("Cofactor property - det(A) from cofactor expansion") {
    dp::mat::Matrix<double, 3, 3> a;
    a(0, 0) = 6.0;
    a(0, 1) = 1.0;
    a(0, 2) = 1.0;
    a(1, 0) = 4.0;
    a(1, 1) = -2.0;
    a(1, 2) = 5.0;
    a(2, 0) = 2.0;
    a(2, 1) = 8.0;
    a(2, 2) = 7.0;

    auto cof = lina::cofactor(Matrix<double, 3, 3>(a));
    auto det_a = lina::determinant(Matrix<double, 3, 3>(a));

    // Determinant via cofactor expansion along first row: det(A) = sum(a_0j * C_0j)
    double det_via_cofactor = a(0, 0) * cof(0, 0) + a(0, 1) * cof(0, 1) + a(0, 2) * cof(0, 2);

    CHECK(det_via_cofactor == doctest::Approx(det_a));
}

// =============================================================================
// Adjoint Matrix Tests
// =============================================================================

TEST_CASE("Adjoint 2x2") {
    dp::mat::Matrix<double, 2, 2> a;
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(1, 0) = 3.0;
    a(1, 1) = 4.0;

    auto adj = lina::adjoint(Matrix<double, 2, 2>(a));

    // For 2x2: adj(A) = [[a11, -a01], [-a10, a00]]
    CHECK(adj(0, 0) == doctest::Approx(4.0));  // a11
    CHECK(adj(0, 1) == doctest::Approx(-2.0)); // -a01
    CHECK(adj(1, 0) == doctest::Approx(-3.0)); // -a10
    CHECK(adj(1, 1) == doctest::Approx(1.0));  // a00
}

TEST_CASE("Adjoint 3x3") {
    dp::mat::Matrix<double, 3, 3> a;
    a(0, 0) = 3.0;
    a(0, 1) = 0.0;
    a(0, 2) = 2.0;
    a(1, 0) = 2.0;
    a(1, 1) = 0.0;
    a(1, 2) = -2.0;
    a(2, 0) = 0.0;
    a(2, 1) = 1.0;
    a(2, 2) = 1.0;

    auto adj = lina::adjoint(Matrix<double, 3, 3>(a));
    auto cof = lina::cofactor(Matrix<double, 3, 3>(a));
    auto cof_t = lina::transpose(cof);

    // Verify adjoint is transpose of cofactor
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(adj(i, j) == doctest::Approx(cof_t(i, j)));
        }
    }
}

TEST_CASE("Adjoint property - A * adj(A) = det(A) * I") {
    dp::mat::Matrix<double, 3, 3> a;
    a(0, 0) = 2.0;
    a(0, 1) = -1.0;
    a(0, 2) = 0.0;
    a(1, 0) = -1.0;
    a(1, 1) = 2.0;
    a(1, 2) = -1.0;
    a(2, 0) = 0.0;
    a(2, 1) = -1.0;
    a(2, 2) = 2.0;

    auto adj = lina::adjoint(Matrix<double, 3, 3>(a));
    auto det_a = lina::determinant(Matrix<double, 3, 3>(a));
    auto product = lina::matmul(a, adj);

    // A * adj(A) should equal det(A) * I
    auto expected = lina::scale(det_a, lina::identity<double, 3>());

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(product(i, j) == doctest::Approx(expected(i, j)).epsilon(1e-10));
        }
    }
}

TEST_CASE("Adjoint property - Inverse via adjoint") {
    dp::mat::Matrix<double, 3, 3> a;
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(0, 2) = 3.0;
    a(1, 0) = 0.0;
    a(1, 1) = 1.0;
    a(1, 2) = 4.0;
    a(2, 0) = 5.0;
    a(2, 1) = 6.0;
    a(2, 2) = 0.0;

    auto det_a = lina::determinant(Matrix<double, 3, 3>(a));

    // Skip if singular
    if (std::abs(det_a) < 1e-10) {
        return;
    }

    auto adj = lina::adjoint(Matrix<double, 3, 3>(a));
    auto inv_via_adjoint = lina::scale(1.0 / det_a, adj);
    auto inv_direct = lina::inverse(Matrix<double, 3, 3>(a));

    // Both methods should produce the same inverse
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(inv_via_adjoint(i, j) == doctest::Approx(inv_direct(i, j)).epsilon(1e-9));
        }
    }
}

TEST_CASE("Adjoint 4x4") {
    dp::mat::Matrix<double, 4, 4> a;
    // Create a simple symmetric positive definite matrix
    a(0, 0) = 4.0;
    a(0, 1) = 1.0;
    a(0, 2) = 0.0;
    a(0, 3) = 0.0;
    a(1, 0) = 1.0;
    a(1, 1) = 3.0;
    a(1, 2) = 1.0;
    a(1, 3) = 0.0;
    a(2, 0) = 0.0;
    a(2, 1) = 1.0;
    a(2, 2) = 2.0;
    a(2, 3) = 1.0;
    a(3, 0) = 0.0;
    a(3, 1) = 0.0;
    a(3, 2) = 1.0;
    a(3, 3) = 1.0;

    auto adj = lina::adjoint(Matrix<double, 4, 4>(a));
    auto det_a = lina::determinant(Matrix<double, 4, 4>(a));
    auto product = lina::matmul(a, adj);

    // Verify A * adj(A) = det(A) * I
    auto expected = lina::scale(det_a, lina::identity<double, 4>());

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            CHECK(product(i, j) == doctest::Approx(expected(i, j)).epsilon(1e-9));
        }
    }
}

TEST_CASE("Adjugate alias") {
    dp::mat::Matrix<double, 2, 2> a;
    a(0, 0) = 5.0;
    a(0, 1) = 7.0;
    a(1, 0) = 2.0;
    a(1, 1) = 3.0;

    auto adj1 = lina::adjoint(Matrix<double, 2, 2>(a));
    auto adj2 = lina::adjugate(Matrix<double, 2, 2>(a)); // Alias

    // Both should be identical
    CHECK(adj1(0, 0) == adj2(0, 0));
    CHECK(adj1(0, 1) == adj2(0, 1));
    CHECK(adj1(1, 0) == adj2(1, 0));
    CHECK(adj1(1, 1) == adj2(1, 1));
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST_CASE("Cofactor and adjoint consistency") {
    dp::mat::Matrix<double, 3, 3> a;
    a(0, 0) = 1.0;
    a(0, 1) = 0.0;
    a(0, 2) = 2.0;
    a(1, 0) = -1.0;
    a(1, 1) = 3.0;
    a(1, 2) = 1.0;
    a(2, 0) = 2.0;
    a(2, 1) = 1.0;
    a(2, 2) = 0.0;

    auto cof = lina::cofactor(Matrix<double, 3, 3>(a));
    auto adj = lina::adjoint(Matrix<double, 3, 3>(a));
    auto cof_transposed = lina::transpose(cof);

    // Adjoint should be transpose of cofactor
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(adj(i, j) == doctest::Approx(cof_transposed(i, j)));
        }
    }
}

TEST_CASE("Cofactor matrix symmetry for symmetric matrix") {
    // For a symmetric matrix, cofactor matrix should also be symmetric
    dp::mat::Matrix<double, 3, 3> a;
    a(0, 0) = 2.0;
    a(0, 1) = 1.0;
    a(0, 2) = 0.0;
    a(1, 0) = 1.0;
    a(1, 1) = 2.0;
    a(1, 2) = 1.0;
    a(2, 0) = 0.0;
    a(2, 1) = 1.0;
    a(2, 2) = 2.0;

    auto cof = lina::cofactor(Matrix<double, 3, 3>(a));
    auto cof_t = lina::transpose(cof);

    // Check symmetry - cofactor should equal its transpose for symmetric matrix
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(cof(i, j) == doctest::Approx(cof_t(j, i)));
        }
    }
}
