#include <doctest/doctest.h>

#include <optinum/lina/basic/hessian.hpp>
#include <optinum/lina/basic/jacobian.hpp>
#include <optinum/lina/basic/properties.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>

namespace lina = optinum::lina;
using optinum::simd::Matrix;
using optinum::simd::Vector;

namespace dp = datapod;

// =============================================================================
// TEST SUITE: Hessian Computation
// =============================================================================

TEST_CASE("hessian - Quadratic function 2D") {
    // f(x,y) = x^2 + 2*x*y + 3*y^2
    // Hessian: [[2, 2], [2, 6]] (constant)
    auto f = [](const Vector<double, 2> &x) { return x[0] * x[0] + 2.0 * x[0] * x[1] + 3.0 * x[1] * x[1]; };

    dp::mat::Vector<double, 2> x;
    x[0] = 1.0;
    x[1] = 2.0;

    auto H = lina::hessian(f, Vector<double, 2>(x));

    CHECK(H.rows() == 2);
    CHECK(H.cols() == 2);

    // Analytical Hessian: [[2, 2], [2, 6]]
    CHECK(H(0, 0) == doctest::Approx(2.0).epsilon(1e-4));
    CHECK(H(0, 1) == doctest::Approx(2.0).epsilon(1e-4));
    CHECK(H(1, 0) == doctest::Approx(2.0).epsilon(1e-4));
    CHECK(H(1, 1) == doctest::Approx(6.0).epsilon(1e-4));
}

TEST_CASE("hessian - Sphere function") {
    // f(x) = x^2 + y^2 + z^2
    // Hessian: 2*I (identity scaled by 2)
    auto f = [](const Vector<double, 3> &x) { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; };

    dp::mat::Vector<double, 3> x;
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;

    auto H = lina::hessian(f, Vector<double, 3>(x));

    CHECK(H.rows() == 3);
    CHECK(H.cols() == 3);

    // Diagonal elements should be 2
    CHECK(H(0, 0) == doctest::Approx(2.0).epsilon(1e-4));
    CHECK(H(1, 1) == doctest::Approx(2.0).epsilon(1e-4));
    CHECK(H(2, 2) == doctest::Approx(2.0).epsilon(1e-4));

    // Off-diagonal elements should be 0
    CHECK(H(0, 1) == doctest::Approx(0.0).epsilon(1e-4));
    CHECK(H(0, 2) == doctest::Approx(0.0).epsilon(1e-4));
    CHECK(H(1, 0) == doctest::Approx(0.0).epsilon(1e-4));
    CHECK(H(1, 2) == doctest::Approx(0.0).epsilon(1e-4));
    CHECK(H(2, 0) == doctest::Approx(0.0).epsilon(1e-4));
    CHECK(H(2, 1) == doctest::Approx(0.0).epsilon(1e-4));
}

TEST_CASE("hessian - Rosenbrock function") {
    // f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    // Hessian at (x,y):
    //   H[0,0] = 2 - 400*(y - x^2) + 1200*x^2
    //   H[0,1] = H[1,0] = -400*x
    //   H[1,1] = 200
    auto f = [](const Vector<double, 2> &x) {
        double t1 = 1.0 - x[0];
        double t2 = x[1] - x[0] * x[0];
        return t1 * t1 + 100.0 * t2 * t2;
    };

    dp::mat::Vector<double, 2> x;
    x[0] = 1.0;
    x[1] = 1.0;

    auto H = lina::hessian(f, Vector<double, 2>(x));

    // At (1, 1): y - x^2 = 0
    // H[0,0] = 2 - 0 + 1200 = 1202
    // H[0,1] = H[1,0] = -400
    // H[1,1] = 200
    CHECK(H(0, 0) == doctest::Approx(802.0).epsilon(1e-2)); // 2 + 1200*1 - 400*0 = 802
    CHECK(H(0, 1) == doctest::Approx(-400.0).epsilon(1e-2));
    CHECK(H(1, 0) == doctest::Approx(-400.0).epsilon(1e-2));
    CHECK(H(1, 1) == doctest::Approx(200.0).epsilon(1e-2));
}

TEST_CASE("hessian - Symmetry") {
    // Hessian should be symmetric for smooth functions
    auto f = [](const Vector<double, 3> &x) {
        return x[0] * x[1] * x[2] + std::sin(x[0]) * std::cos(x[1]) + x[2] * x[2];
    };

    dp::mat::Vector<double, 3> x;
    x[0] = 0.5;
    x[1] = 1.0;
    x[2] = 1.5;

    auto H = lina::hessian(f, Vector<double, 3>(x));

    // Check symmetry
    CHECK(H(0, 1) == doctest::Approx(H(1, 0)).epsilon(1e-6));
    CHECK(H(0, 2) == doctest::Approx(H(2, 0)).epsilon(1e-6));
    CHECK(H(1, 2) == doctest::Approx(H(2, 1)).epsilon(1e-6));
}

TEST_CASE("hessian - Float precision") {
    auto f = [](const Vector<float, 2> &x) { return x[0] * x[0] + x[1] * x[1]; };

    dp::mat::Vector<float, 2> x;
    x[0] = 1.0f;
    x[1] = 2.0f;

    // Use larger step size for float precision (second derivatives are sensitive)
    auto H = lina::hessian(f, Vector<float, 2>(x), 1e-2f);

    // Float precision for second derivatives is limited
    CHECK(H(0, 0) == doctest::Approx(2.0f).epsilon(0.5f));
    CHECK(H(1, 1) == doctest::Approx(2.0f).epsilon(0.5f));
    CHECK(H(0, 1) == doctest::Approx(0.0f).epsilon(0.5f));
}

// =============================================================================
// TEST SUITE: Hessian-Vector Product
// =============================================================================

TEST_CASE("hessian_vector_product - Quadratic function") {
    // f(x,y) = x^2 + y^2
    // Hessian: [[2, 0], [0, 2]]
    // H*v = 2*v
    auto f = [](const Vector<double, 2> &x) { return x[0] * x[0] + x[1] * x[1]; };

    dp::mat::Vector<double, 2> x;
    x[0] = 1.0;
    x[1] = 2.0;

    dp::mat::Vector<double, 2> v;
    v[0] = 1.0;
    v[1] = 0.0;

    auto Hv = lina::hessian_vector_product(f, Vector<double, 2>(x), Vector<double, 2>(v));

    // H*v = [[2, 0], [0, 2]] * [1, 0] = [2, 0]
    CHECK(Hv[0] == doctest::Approx(2.0).epsilon(1e-3));
    CHECK(Hv[1] == doctest::Approx(0.0).epsilon(1e-3));
}

TEST_CASE("hessian_vector_product - Mixed quadratic") {
    // f(x,y) = x^2 + 2*x*y + 3*y^2
    // Hessian: [[2, 2], [2, 6]]
    auto f = [](const Vector<double, 2> &x) { return x[0] * x[0] + 2.0 * x[0] * x[1] + 3.0 * x[1] * x[1]; };

    dp::mat::Vector<double, 2> x;
    x[0] = 1.0;
    x[1] = 2.0;

    dp::mat::Vector<double, 2> v;
    v[0] = 1.0;
    v[1] = 1.0;

    auto Hv = lina::hessian_vector_product(f, Vector<double, 2>(x), Vector<double, 2>(v));

    // H*v = [[2, 2], [2, 6]] * [1, 1] = [4, 8]
    CHECK(Hv[0] == doctest::Approx(4.0).epsilon(1e-2));
    CHECK(Hv[1] == doctest::Approx(8.0).epsilon(1e-2));
}

TEST_CASE("hessian_vector_product - Consistency with full Hessian") {
    auto f = [](const Vector<double, 3> &x) { return x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[2] * x[2]; };

    dp::mat::Vector<double, 3> x;
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;

    dp::mat::Vector<double, 3> v;
    v[0] = 0.5;
    v[1] = 1.0;
    v[2] = 1.5;

    // Compute using full Hessian
    auto H = lina::hessian(f, Vector<double, 3>(x));
    dp::mat::Vector<double, 3> Hv_full;
    for (std::size_t i = 0; i < 3; ++i) {
        Hv_full[i] = 0.0;
        for (std::size_t j = 0; j < 3; ++j) {
            Hv_full[i] += H(i, j) * v[j];
        }
    }

    // Compute using Hessian-vector product
    auto Hv_direct = lina::hessian_vector_product(f, Vector<double, 3>(x), Vector<double, 3>(v));

    // Should match
    CHECK(Hv_direct[0] == doctest::Approx(Hv_full[0]).epsilon(1e-2));
    CHECK(Hv_direct[1] == doctest::Approx(Hv_full[1]).epsilon(1e-2));
    CHECK(Hv_direct[2] == doctest::Approx(Hv_full[2]).epsilon(1e-2));
}

// =============================================================================
// TEST SUITE: Positive Definiteness
// =============================================================================

TEST_CASE("is_positive_definite - Identity matrix") {
    dp::mat::Matrix<double, 3, 3> I;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            I(i, j) = (i == j) ? 1.0 : 0.0;
        }
    }

    CHECK(lina::is_positive_definite(Matrix<double, 3, 3>(I)) == true);
}

TEST_CASE("is_positive_definite - Scaled identity") {
    dp::mat::Matrix<double, 2, 2> H;
    H(0, 0) = 2.0;
    H(0, 1) = 0.0;
    H(1, 0) = 0.0;
    H(1, 1) = 2.0;

    CHECK(lina::is_positive_definite(Matrix<double, 2, 2>(H)) == true);
}

TEST_CASE("is_positive_definite - Positive definite matrix") {
    // [[4, 2], [2, 3]] has eigenvalues 5 and 2 (both positive)
    dp::mat::Matrix<double, 2, 2> H;
    H(0, 0) = 4.0;
    H(0, 1) = 2.0;
    H(1, 0) = 2.0;
    H(1, 1) = 3.0;

    CHECK(lina::is_positive_definite(Matrix<double, 2, 2>(H)) == true);
}

TEST_CASE("is_positive_definite - Not positive definite (negative diagonal)") {
    // Matrix with negative diagonal element
    dp::mat::Matrix<double, 2, 2> H;
    H(0, 0) = -1.0;
    H(0, 1) = 0.0;
    H(1, 0) = 0.0;
    H(1, 1) = 1.0;

    CHECK(lina::is_positive_definite(Matrix<double, 2, 2>(H)) == false);
}

TEST_CASE("is_positive_definite - Zero matrix") {
    dp::mat::Matrix<double, 2, 2> H;
    H(0, 0) = 0.0;
    H(0, 1) = 0.0;
    H(1, 0) = 0.0;
    H(1, 1) = 0.0;

    CHECK(lina::is_positive_definite(Matrix<double, 2, 2>(H)) == false);
}

TEST_CASE("is_positive_definite - Negative definite") {
    dp::mat::Matrix<double, 2, 2> H;
    H(0, 0) = -2.0;
    H(0, 1) = 0.0;
    H(1, 0) = 0.0;
    H(1, 1) = -3.0;

    CHECK(lina::is_positive_definite(Matrix<double, 2, 2>(H)) == false);
}

// =============================================================================
// TEST SUITE: Laplacian
// =============================================================================

TEST_CASE("laplacian - Sphere function") {
    // f(x) = x^2 + y^2 + z^2
    // Laplacian = 2 + 2 + 2 = 6
    auto f = [](const Vector<double, 3> &x) { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; };

    dp::mat::Vector<double, 3> x;
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;

    double lap = lina::laplacian(f, Vector<double, 3>(x));

    CHECK(lap == doctest::Approx(6.0).epsilon(1e-4));
}

TEST_CASE("laplacian - Weighted quadratic") {
    // f(x) = x^2 + 2*y^2 + 3*z^2
    // Laplacian = 2 + 4 + 6 = 12
    auto f = [](const Vector<double, 3> &x) { return x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[2] * x[2]; };

    dp::mat::Vector<double, 3> x;
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;

    double lap = lina::laplacian(f, Vector<double, 3>(x));

    CHECK(lap == doctest::Approx(12.0).epsilon(1e-4));
}

TEST_CASE("laplacian - Consistency with Hessian trace") {
    auto f = [](const Vector<double, 3> &x) {
        return x[0] * x[0] + 2.0 * x[0] * x[1] + 3.0 * x[1] * x[1] + x[2] * x[2];
    };

    dp::mat::Vector<double, 3> x;
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;

    // Compute Laplacian directly
    double lap = lina::laplacian(f, Vector<double, 3>(x));

    // Compute trace of Hessian
    auto H = lina::hessian(f, Vector<double, 3>(x));
    double trace = H(0, 0) + H(1, 1) + H(2, 2);

    CHECK(lap == doctest::Approx(trace).epsilon(1e-3));
}

// =============================================================================
// TEST SUITE: Hessian Error Helper
// =============================================================================

TEST_CASE("hessian_error - Identical matrices") {
    dp::mat::Matrix<double, 2, 2> H1;
    H1(0, 0) = 2.0;
    H1(0, 1) = 1.0;
    H1(1, 0) = 1.0;
    H1(1, 1) = 3.0;

    dp::mat::Matrix<double, 2, 2> H2 = H1;

    double error = lina::hessian_error(Matrix<double, 2, 2>(H1), Matrix<double, 2, 2>(H2));
    CHECK(error == doctest::Approx(0.0).epsilon(1e-12));
}

TEST_CASE("hessian_error - Small difference") {
    dp::mat::Matrix<double, 2, 2> H1;
    H1(0, 0) = 2.0;
    H1(0, 1) = 1.0;
    H1(1, 0) = 1.0;
    H1(1, 1) = 3.0;

    dp::mat::Matrix<double, 2, 2> H2;
    H2(0, 0) = 2.002;
    H2(0, 1) = 1.001;
    H2(1, 0) = 1.001;
    H2(1, 1) = 3.003;

    double error = lina::hessian_error(Matrix<double, 2, 2>(H1), Matrix<double, 2, 2>(H2));
    CHECK(error == doctest::Approx(0.001).epsilon(1e-6));
}

// =============================================================================
// TEST SUITE: Hessian at Critical Points
// =============================================================================

TEST_CASE("hessian - At minimum (positive definite)") {
    // f(x,y) = (x-1)^2 + (y-2)^2
    // Minimum at (1, 2), Hessian = [[2, 0], [0, 2]] (positive definite)
    auto f = [](const Vector<double, 2> &x) {
        double dx = x[0] - 1.0;
        double dy = x[1] - 2.0;
        return dx * dx + dy * dy;
    };

    dp::mat::Vector<double, 2> x;
    x[0] = 1.0;
    x[1] = 2.0;

    auto H = lina::hessian(f, Vector<double, 2>(x));

    CHECK(H(0, 0) == doctest::Approx(2.0).epsilon(1e-4));
    CHECK(H(1, 1) == doctest::Approx(2.0).epsilon(1e-4));
    CHECK(H(0, 1) == doctest::Approx(0.0).epsilon(1e-4));

    CHECK(lina::is_positive_definite(H) == true);
}

TEST_CASE("hessian - At saddle point (indefinite)") {
    // f(x,y) = x^2 - y^2
    // Saddle at (0, 0), Hessian = [[2, 0], [0, -2]] (indefinite)
    auto f = [](const Vector<double, 2> &x) { return x[0] * x[0] - x[1] * x[1]; };

    dp::mat::Vector<double, 2> x;
    x[0] = 0.0;
    x[1] = 0.0;

    auto H = lina::hessian(f, Vector<double, 2>(x));

    CHECK(H(0, 0) == doctest::Approx(2.0).epsilon(1e-4));
    CHECK(H(1, 1) == doctest::Approx(-2.0).epsilon(1e-4));
    CHECK(H(0, 1) == doctest::Approx(0.0).epsilon(1e-4));

    CHECK(lina::is_positive_definite(Matrix<double, 2, 2>(H)) == false);
}
