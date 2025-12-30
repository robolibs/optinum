#include <doctest/doctest.h>
#include <optinum/lina/solve/lstsq_dynamic.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>

using optinum::lina::lstsq_dynamic;
using optinum::lina::lstsq_residual_dynamic;
using optinum::lina::try_lstsq_dynamic;
using optinum::simd::Dynamic;
using optinum::simd::Matrix;
using optinum::simd::Vector;

namespace dp = datapod;

// Owning storage types
using DynMatStorage = dp::mat::matrix<double, dp::mat::Dynamic, dp::mat::Dynamic>;
using DynVecStorage = dp::mat::vector<double, dp::mat::Dynamic>;

// View types
using DynMat = Matrix<double, Dynamic, Dynamic>;
using DynVec = Vector<double, Dynamic>;

TEST_CASE("lina::lstsq_dynamic recovers exact solution for consistent overdetermined system") {
    // A (3x2), x_true (2), b = A * x_true
    DynMatStorage a_storage(3, 2);
    DynMat a(a_storage);
    a(0, 0) = 1.0;
    a(0, 1) = 4.0;
    a(1, 0) = 2.0;
    a(1, 1) = 5.0;
    a(2, 0) = 3.0;
    a(2, 1) = 6.0;

    DynVecStorage x_true_storage(2);
    DynVec x_true(x_true_storage);
    x_true[0] = 1.0;
    x_true[1] = -1.0;

    // b = A * x_true
    DynVecStorage b_storage(3);
    DynVec b(b_storage);
    for (std::size_t i = 0; i < 3; ++i) {
        b[i] = a(i, 0) * x_true[0] + a(i, 1) * x_true[1];
    }

    const auto x = lstsq_dynamic(a, b);

    CHECK(x[0] == doctest::Approx(x_true[0]).epsilon(1e-6));
    CHECK(x[1] == doctest::Approx(x_true[1]).epsilon(1e-6));
}

TEST_CASE("lina::lstsq_dynamic solves overdetermined system (line fitting)") {
    // Fit y = a + b*x to points (0,1), (1,2), (2,3), (3,4)
    // This is exactly y = 1 + x
    DynMatStorage a_storage(4, 2);
    DynMat a(a_storage);
    a(0, 0) = 1.0;
    a(0, 1) = 0.0;
    a(1, 0) = 1.0;
    a(1, 1) = 1.0;
    a(2, 0) = 1.0;
    a(2, 1) = 2.0;
    a(3, 0) = 1.0;
    a(3, 1) = 3.0;

    DynVecStorage b_storage(4);
    DynVec b(b_storage);
    b[0] = 1.0;
    b[1] = 2.0;
    b[2] = 3.0;
    b[3] = 4.0;

    const auto x = lstsq_dynamic(a, b);

    // Expected: x = [1, 1] (intercept=1, slope=1)
    CHECK(x[0] == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(x[1] == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("lina::lstsq_dynamic handles noisy data") {
    // Fit y = a + b*x to noisy data
    // True line: y = 2 + 0.5*x
    DynMatStorage a_storage(5, 2);
    DynMat a(a_storage);
    a(0, 0) = 1.0;
    a(0, 1) = 0.0;
    a(1, 0) = 1.0;
    a(1, 1) = 1.0;
    a(2, 0) = 1.0;
    a(2, 1) = 2.0;
    a(3, 0) = 1.0;
    a(3, 1) = 3.0;
    a(4, 0) = 1.0;
    a(4, 1) = 4.0;

    DynVecStorage b_storage(5);
    DynVec b(b_storage);
    b[0] = 2.1;
    b[1] = 2.4;
    b[2] = 3.1;
    b[3] = 3.4;
    b[4] = 4.1; // y = 2 + 0.5*x + noise

    const auto x = lstsq_dynamic(a, b);

    // Should be close to [2, 0.5]
    CHECK(x[0] == doctest::Approx(2.0).epsilon(0.2));
    CHECK(x[1] == doctest::Approx(0.5).epsilon(0.1));
}

TEST_CASE("lina::try_lstsq_dynamic returns error for underdetermined system") {
    // m < n is not allowed
    DynMatStorage a_storage(2, 3);
    DynMat a(a_storage);
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(0, 2) = 3.0;
    a(1, 0) = 4.0;
    a(1, 1) = 5.0;
    a(1, 2) = 6.0;

    DynVecStorage b_storage(2);
    DynVec b(b_storage);
    b[0] = 1.0;
    b[1] = 2.0;

    const auto r = try_lstsq_dynamic(a, b);
    CHECK(r.is_err());
}

TEST_CASE("lina::lstsq_dynamic for square system") {
    // Square system should give exact solution
    DynMatStorage a_storage(2, 2);
    DynMat a(a_storage);
    a(0, 0) = 2.0;
    a(0, 1) = 1.0;
    a(1, 0) = 1.0;
    a(1, 1) = 3.0;

    DynVecStorage b_storage(2);
    DynVec b(b_storage);
    b[0] = 5.0;
    b[1] = 8.0;

    const auto x = lstsq_dynamic(a, b);

    // Verify Ax = b
    for (std::size_t i = 0; i < 2; ++i) {
        double ax_i = a(i, 0) * x[0] + a(i, 1) * x[1];
        CHECK(ax_i == doctest::Approx(b[i]).epsilon(1e-9));
    }
}

TEST_CASE("lina::lstsq_residual_dynamic computes residual norm") {
    // Overdetermined system with no exact solution
    DynMatStorage a_storage(3, 2);
    DynMat a(a_storage);
    a(0, 0) = 1.0;
    a(0, 1) = 0.0;
    a(1, 0) = 0.0;
    a(1, 1) = 1.0;
    a(2, 0) = 1.0;
    a(2, 1) = 1.0;

    DynVecStorage b_storage(3);
    DynVec b(b_storage);
    b[0] = 1.0;
    b[1] = 2.0;
    b[2] = 4.0; // No exact solution

    const auto x_result = lstsq_dynamic(a, b);
    DynVec x(x_result);
    const auto residual = lstsq_residual_dynamic(a, x, b);

    // Residual should be positive (no exact solution)
    CHECK(residual > 0.0);

    // Verify residual is computed correctly
    double manual_residual = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        double ax_i = a(i, 0) * x[0] + a(i, 1) * x[1];
        double diff = ax_i - b[i];
        manual_residual += diff * diff;
    }
    manual_residual = std::sqrt(manual_residual);

    CHECK(residual == doctest::Approx(manual_residual).epsilon(1e-12));
}

TEST_CASE("lina::lstsq_dynamic for consistent system has zero residual") {
    // Consistent overdetermined system
    DynMatStorage a_storage(3, 2);
    DynMat a(a_storage);
    a(0, 0) = 1.0;
    a(0, 1) = 0.0;
    a(1, 0) = 0.0;
    a(1, 1) = 1.0;
    a(2, 0) = 1.0;
    a(2, 1) = 1.0;

    DynVecStorage x_true_storage(2);
    DynVec x_true(x_true_storage);
    x_true[0] = 1.0;
    x_true[1] = 2.0;

    // b = A * x_true (consistent)
    DynVecStorage b_storage(3);
    DynVec b(b_storage);
    for (std::size_t i = 0; i < 3; ++i) {
        b[i] = a(i, 0) * x_true[0] + a(i, 1) * x_true[1];
    }

    const auto x_result = lstsq_dynamic(a, b);
    DynVec x(x_result);
    const auto residual = lstsq_residual_dynamic(a, x, b);

    CHECK(residual == doctest::Approx(0.0).epsilon(1e-9));
}

TEST_CASE("lina::lstsq_dynamic with multiple RHS") {
    // A (4x2), B (4x3)
    DynMatStorage a_storage(4, 2);
    DynMat a(a_storage);
    a(0, 0) = 1.0;
    a(0, 1) = 0.0;
    a(1, 0) = 0.0;
    a(1, 1) = 1.0;
    a(2, 0) = 1.0;
    a(2, 1) = 1.0;
    a(3, 0) = 2.0;
    a(3, 1) = 1.0;

    // Create consistent RHS
    DynMatStorage x_true_storage(2, 3);
    DynMat x_true(x_true_storage);
    x_true(0, 0) = 1.0;
    x_true(0, 1) = 2.0;
    x_true(0, 2) = 3.0;
    x_true(1, 0) = 4.0;
    x_true(1, 1) = 5.0;
    x_true(1, 2) = 6.0;

    DynMatStorage b_storage(4, 3);
    DynMat b(b_storage);
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t col = 0; col < 3; ++col) {
            b(i, col) = a(i, 0) * x_true(0, col) + a(i, 1) * x_true(1, col);
        }
    }

    const auto x = lstsq_dynamic(a, b);

    // Verify solution
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            CHECK(x(i, j) == doctest::Approx(x_true(i, j)).epsilon(1e-6));
        }
    }
}

TEST_CASE("lina::lstsq_dynamic for larger system (10x3)") {
    // Polynomial fitting: y = a + b*x + c*x^2
    DynMatStorage a_storage(10, 3);
    DynMat a(a_storage);
    DynVecStorage b_storage(10);
    DynVec b(b_storage);

    // True coefficients
    double a0 = 1.0, a1 = 2.0, a2 = -0.5;

    for (std::size_t i = 0; i < 10; ++i) {
        double xi = static_cast<double>(i);
        a(i, 0) = 1.0;
        a(i, 1) = xi;
        a(i, 2) = xi * xi;
        b[i] = a0 + a1 * xi + a2 * xi * xi;
    }

    const auto x = lstsq_dynamic(a, b);

    CHECK(x[0] == doctest::Approx(a0).epsilon(1e-6));
    CHECK(x[1] == doctest::Approx(a1).epsilon(1e-6));
    CHECK(x[2] == doctest::Approx(a2).epsilon(1e-6));
}
