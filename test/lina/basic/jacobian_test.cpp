#include <doctest/doctest.h>

#include <optinum/lina/basic/jacobian.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>

using optinum::lina::gradient;
using optinum::lina::jacobian;
using optinum::simd::Dynamic;
using optinum::simd::Matrix;
using optinum::simd::Vector;

// =============================================================================
// TEST SUITE: Jacobian Computation
// =============================================================================

TEST_CASE("jacobian - Linear 2D function") {
    // f(x) = A*x where A = [[1, 2], [3, 4]]
    // Jacobian should be constant = A
    auto f = [](const Vector<double, 2> &x) {
        Vector<double, 2> result;
        result[0] = 1.0 * x[0] + 2.0 * x[1];
        result[1] = 3.0 * x[0] + 4.0 * x[1];
        return result;
    };

    Vector<double, 2> x;
    x[0] = 5.0;
    x[1] = 7.0;

    auto J = jacobian(f, x);

    CHECK(J.rows() == 2);
    CHECK(J.cols() == 2);

    // Analytical: [[1, 2], [3, 4]]
    CHECK(J(0, 0) == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(J(0, 1) == doctest::Approx(2.0).epsilon(1e-6));
    CHECK(J(1, 0) == doctest::Approx(3.0).epsilon(1e-6));
    CHECK(J(1, 1) == doctest::Approx(4.0).epsilon(1e-6));
}

TEST_CASE("jacobian - Nonlinear 2D->2D function") {
    // f(x,y) = [x^2 + y, x*y]
    // Analytical Jacobian: [[2x, 1], [y, x]]
    auto f = [](const Vector<double, 2> &x) {
        Vector<double, 2> result;
        result[0] = x[0] * x[0] + x[1];
        result[1] = x[0] * x[1];
        return result;
    };

    Vector<double, 2> x;
    x[0] = 3.0;
    x[1] = 4.0;

    auto J = jacobian(f, x);

    CHECK(J.rows() == 2);
    CHECK(J.cols() == 2);

    // At (3, 4): [[2*3, 1], [4, 3]] = [[6, 1], [4, 3]]
    CHECK(J(0, 0) == doctest::Approx(6.0).epsilon(1e-6));
    CHECK(J(0, 1) == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(J(1, 0) == doctest::Approx(4.0).epsilon(1e-6));
    CHECK(J(1, 1) == doctest::Approx(3.0).epsilon(1e-6));
}

TEST_CASE("jacobian - 3D->2D function") {
    // f(x,y,z) = [x*y + z^2, sin(x) + y*z]
    // Analytical Jacobian:
    //   [[y, x, 2z],
    //    [cos(x), z, y]]
    auto f = [](const Vector<double, 3> &x) {
        Vector<double, 2> result;
        result[0] = x[0] * x[1] + x[2] * x[2];
        result[1] = std::sin(x[0]) + x[1] * x[2];
        return result;
    };

    Vector<double, 3> x;
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;

    auto J = jacobian(f, x);

    CHECK(J.rows() == 2);
    CHECK(J.cols() == 3);

    // At (1, 2, 3):
    //   [[2, 1, 6],
    //    [cos(1), 3, 2]]
    CHECK(J(0, 0) == doctest::Approx(2.0).epsilon(1e-6));
    CHECK(J(0, 1) == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(J(0, 2) == doctest::Approx(6.0).epsilon(1e-6));
    CHECK(J(1, 0) == doctest::Approx(std::cos(1.0)).epsilon(1e-6));
    CHECK(J(1, 1) == doctest::Approx(3.0).epsilon(1e-6));
    CHECK(J(1, 2) == doctest::Approx(2.0).epsilon(1e-6));
}

TEST_CASE("jacobian - 2D->3D function") {
    // f(x,y) = [x^2, x*y, y^2]
    // Analytical Jacobian:
    //   [[2x, 0],
    //    [y, x],
    //    [0, 2y]]
    auto f = [](const Vector<double, 2> &x) {
        Vector<double, 3> result;
        result[0] = x[0] * x[0];
        result[1] = x[0] * x[1];
        result[2] = x[1] * x[1];
        return result;
    };

    Vector<double, 2> x;
    x[0] = 2.0;
    x[1] = 3.0;

    auto J = jacobian(f, x);

    CHECK(J.rows() == 3);
    CHECK(J.cols() == 2);

    // At (2, 3):
    //   [[4, 0],
    //    [3, 2],
    //    [0, 6]]
    CHECK(J(0, 0) == doctest::Approx(4.0).epsilon(1e-6));
    CHECK(J(0, 1) == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(J(1, 0) == doctest::Approx(3.0).epsilon(1e-6));
    CHECK(J(1, 1) == doctest::Approx(2.0).epsilon(1e-6));
    CHECK(J(2, 0) == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(J(2, 1) == doctest::Approx(6.0).epsilon(1e-6));
}

TEST_CASE("jacobian - Forward vs Central differences") {
    // Test that central differences are more accurate
    // f(x) = [sin(x), cos(x)]
    auto f = [](const Vector<double, 1> &x) {
        Vector<double, 2> result;
        result[0] = std::sin(x[0]);
        result[1] = std::cos(x[0]);
        return result;
    };

    Vector<double, 1> x;
    x[0] = 0.5;

    // Use larger step size to make difference visible
    double h = 1e-5;
    auto J_central = jacobian(f, x, h, true);
    auto J_forward = jacobian(f, x, h, false);

    // Analytical: [[cos(x)], [-sin(x)]] at x=0.5
    double expected_00 = std::cos(0.5);
    double expected_10 = -std::sin(0.5);

    // Central should be more accurate
    double error_central_00 = std::abs(J_central(0, 0) - expected_00);
    double error_forward_00 = std::abs(J_forward(0, 0) - expected_00);
    double error_central_10 = std::abs(J_central(1, 0) - expected_10);
    double error_forward_10 = std::abs(J_forward(1, 0) - expected_10);

    CHECK(error_central_00 < error_forward_00);
    CHECK(error_central_10 < error_forward_10);

    // Both should still be reasonably accurate
    CHECK(J_central(0, 0) == doctest::Approx(expected_00).epsilon(1e-8));
    CHECK(J_forward(0, 0) == doctest::Approx(expected_00).epsilon(1e-4));
}

TEST_CASE("jacobian - Dynamic-sized vectors") {
    // Test with Dynamic-sized inputs/outputs
    auto f = [](const Vector<double, Dynamic> &x) {
        Vector<double, Dynamic> result;
        result.resize(2);
        result[0] = x[0] * x[0] + x[1];
        result[1] = x[0] * x[1];
        return result;
    };

    Vector<double, Dynamic> x;
    x.resize(2);
    x[0] = 3.0;
    x[1] = 4.0;

    auto J = jacobian(f, x);

    CHECK(J.rows() == 2);
    CHECK(J.cols() == 2);

    // At (3, 4): [[6, 1], [4, 3]]
    CHECK(J(0, 0) == doctest::Approx(6.0).epsilon(1e-6));
    CHECK(J(0, 1) == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(J(1, 0) == doctest::Approx(4.0).epsilon(1e-6));
    CHECK(J(1, 1) == doctest::Approx(3.0).epsilon(1e-6));
}

// =============================================================================
// TEST SUITE: Gradient Computation
// =============================================================================

TEST_CASE("gradient - Sphere function") {
    // f(x) = x^2 + y^2
    // Gradient: [2x, 2y]
    auto f = [](const Vector<double, 2> &x) { return x[0] * x[0] + x[1] * x[1]; };

    Vector<double, 2> x;
    x[0] = 3.0;
    x[1] = 4.0;

    auto grad = gradient(f, x);

    CHECK(grad.size() == 2);
    CHECK(grad[0] == doctest::Approx(6.0).epsilon(1e-6));
    CHECK(grad[1] == doctest::Approx(8.0).epsilon(1e-6));
}

TEST_CASE("gradient - Rosenbrock function") {
    // f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    // Gradient: [-2(1-x) - 400x(y-x^2), 200(y-x^2)]
    auto f = [](const Vector<double, 2> &x) {
        double term1 = 1.0 - x[0];
        double term2 = x[1] - x[0] * x[0];
        return term1 * term1 + 100.0 * term2 * term2;
    };

    Vector<double, 2> x;
    x[0] = 0.5;
    x[1] = 0.25;

    auto grad = gradient(f, x);

    // Analytical gradient at (0.5, 0.25):
    // df/dx = -2(1-0.5) - 400*0.5*(0.25-0.25) = -1.0
    // df/dy = 200*(0.25-0.25) = 0.0
    CHECK(grad.size() == 2);
    CHECK(grad[0] == doctest::Approx(-1.0).epsilon(1e-6));
    CHECK(grad[1] == doctest::Approx(0.0).epsilon(1e-6));
}

TEST_CASE("gradient - 3D quadratic function") {
    // f(x,y,z) = x^2 + 2*y^2 + 3*z^2
    // Gradient: [2x, 4y, 6z]
    auto f = [](const Vector<double, 3> &x) { return x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[2] * x[2]; };

    Vector<double, 3> x;
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;

    auto grad = gradient(f, x);

    CHECK(grad.size() == 3);
    CHECK(grad[0] == doctest::Approx(2.0).epsilon(1e-6));
    CHECK(grad[1] == doctest::Approx(8.0).epsilon(1e-6));
    CHECK(grad[2] == doctest::Approx(18.0).epsilon(1e-6));
}

TEST_CASE("gradient - Forward vs Central differences") {
    // Test that central differences are more accurate
    // f(x) = sin(x)
    auto f = [](const Vector<double, 1> &x) { return std::sin(x[0]); };

    Vector<double, 1> x;
    x[0] = 0.5;

    // Use larger step size to make difference visible
    double h = 1e-5;
    auto grad_central = gradient(f, x, h, true);
    auto grad_forward = gradient(f, x, h, false);

    // Analytical: cos(0.5)
    double expected = std::cos(0.5);

    double error_central = std::abs(grad_central[0] - expected);
    double error_forward = std::abs(grad_forward[0] - expected);

    // Central should be more accurate
    CHECK(error_central < error_forward);

    // Both should still be reasonably accurate
    CHECK(grad_central[0] == doctest::Approx(expected).epsilon(1e-8));
    CHECK(grad_forward[0] == doctest::Approx(expected).epsilon(1e-4));
}

TEST_CASE("gradient - Dynamic-sized vector") {
    // Test with Dynamic-sized input
    auto f = [](const Vector<double, Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

    Vector<double, Dynamic> x;
    x.resize(2);
    x[0] = 3.0;
    x[1] = 4.0;

    auto grad = gradient(f, x);

    CHECK(grad.size() == 2);
    CHECK(grad[0] == doctest::Approx(6.0).epsilon(1e-6));
    CHECK(grad[1] == doctest::Approx(8.0).epsilon(1e-6));
}

TEST_CASE("gradient - Zero gradient at minimum") {
    // f(x,y) = (x-1)^2 + (y-2)^2
    // Minimum at (1, 2), gradient should be zero there
    auto f = [](const Vector<double, 2> &x) {
        double dx = x[0] - 1.0;
        double dy = x[1] - 2.0;
        return dx * dx + dy * dy;
    };

    Vector<double, 2> x;
    x[0] = 1.0;
    x[1] = 2.0;

    auto grad = gradient(f, x);

    CHECK(grad[0] == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(grad[1] == doctest::Approx(0.0).epsilon(1e-6));
}

// =============================================================================
// TEST SUITE: Jacobian Error Helper
// =============================================================================

TEST_CASE("jacobian_error - Identical matrices") {
    Matrix<double, 2, 2> J1;
    J1(0, 0) = 1.0;
    J1(0, 1) = 2.0;
    J1(1, 0) = 3.0;
    J1(1, 1) = 4.0;

    Matrix<double, 2, 2> J2 = J1;

    double error = optinum::lina::jacobian_error(J1, J2);
    CHECK(error == doctest::Approx(0.0).epsilon(1e-12));
}

TEST_CASE("jacobian_error - Small difference") {
    Matrix<double, 2, 2> J1;
    J1(0, 0) = 1.0;
    J1(0, 1) = 2.0;
    J1(1, 0) = 3.0;
    J1(1, 1) = 4.0;

    Matrix<double, 2, 2> J2;
    J2(0, 0) = 1.001;
    J2(0, 1) = 2.002;
    J2(1, 0) = 3.003;
    J2(1, 1) = 4.004;

    double error = optinum::lina::jacobian_error(J1, J2);
    CHECK(error == doctest::Approx(0.001).epsilon(1e-6));
}

TEST_CASE("jacobian_error - Dynamic matrices") {
    Matrix<double, Dynamic, Dynamic> J1;
    J1.resize(2, 2);
    J1(0, 0) = 1.0;
    J1(0, 1) = 2.0;
    J1(1, 0) = 3.0;
    J1(1, 1) = 4.0;

    Matrix<double, Dynamic, Dynamic> J2;
    J2.resize(2, 2);
    J2(0, 0) = 1.0001;
    J2(0, 1) = 2.0002;
    J2(1, 0) = 3.0003;
    J2(1, 1) = 4.0004;

    double error = optinum::lina::jacobian_error(J1, J2);
    CHECK(error == doctest::Approx(0.0001).epsilon(1e-6));
}
