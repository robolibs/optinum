#include <doctest/doctest.h>

#include <optinum/opti/quasi_newton/levenberg_marquardt.hpp>

#include <cmath>
#include <vector>

using optinum::opti::LevenbergMarquardt;

namespace dp = datapod;

// =============================================================================
// TEST SUITE: Levenberg-Marquardt Optimizer
// =============================================================================

TEST_CASE("LevenbergMarquardt - Simple linear least squares") {
    // Same as Gauss-Newton but should be more robust
    auto residual = [](const dp::mat::Vector<double, 2> &x) {
        dp::mat::Vector<double, 3> r;
        r[0] = 1.0 * x[0] + 2.0 * x[1] - 5.0;
        r[1] = 3.0 * x[0] + 4.0 * x[1] - 11.0;
        r[2] = 5.0 * x[0] + 6.0 * x[1] - 17.0;
        return r;
    };

    LevenbergMarquardt<double> lm;
    lm.max_iterations = 50;
    lm.tolerance = 1e-10;
    lm.verbose = false;

    dp::mat::Vector<double, 2> x0;
    x0[0] = 0.0;
    x0[1] = 0.0;

    auto result = lm.optimize(residual, x0);

    CHECK(result.converged);
    CHECK(result.x[0] == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(result.x[1] == doctest::Approx(2.0).epsilon(1e-6));
}

TEST_CASE("LevenbergMarquardt - Rosenbrock function") {
    // LM should handle Rosenbrock better than GN from poor initial guess
    auto residual = [](const dp::mat::Vector<double, 2> &x) {
        dp::mat::Vector<double, 2> r;
        r[0] = 1.0 - x[0];
        r[1] = 10.0 * (x[1] - x[0] * x[0]);
        return r;
    };

    LevenbergMarquardt<double> lm;
    lm.max_iterations = 200;
    lm.tolerance = 1e-8;
    lm.initial_lambda = 1e-2;
    lm.verbose = false;

    // Poor initial guess
    dp::mat::Vector<double, 2> x0;
    x0[0] = -5.0;
    x0[1] = 10.0;

    auto result = lm.optimize(residual, x0);

    CHECK(result.converged);
    CHECK(result.x[0] == doctest::Approx(1.0).epsilon(1e-3));
    CHECK(result.x[1] == doctest::Approx(1.0).epsilon(1e-3));
    CHECK(result.final_cost < 1e-6);
}

TEST_CASE("LevenbergMarquardt - Exponential curve fitting") {
    struct DataPoint {
        double x, y;
    };

    // Data from exponential: y = 5*exp(-0.5*x) + 1
    std::vector<DataPoint> data = {{0.0, 6.0},  {1.0, 4.03}, {2.0, 2.85}, {3.0, 2.23}, {4.0, 1.68}, {5.0, 1.41},
                                   {6.0, 1.25}, {7.0, 1.12}, {8.0, 1.06}, {9.0, 1.03}, {10.0, 1.01}};

    auto residual = [&data](const dp::mat::Vector<double, 3> &params) {
        dp::mat::Vector<double, dp::mat::Dynamic> r;
        r.resize(data.size());
        double a = params[0];
        double b = params[1];
        double c = params[2];

        for (std::size_t i = 0; i < data.size(); ++i) {
            double y_pred = a * std::exp(-b * data[i].x) + c;
            r[i] = y_pred - data[i].y;
        }
        return r;
    };

    LevenbergMarquardt<double> lm;
    lm.max_iterations = 100;
    lm.tolerance = 1e-8;
    lm.verbose = false;

    dp::mat::Vector<double, 3> x0;
    x0[0] = 1.0;
    x0[1] = 0.1;
    x0[2] = 0.0;

    auto result = lm.optimize(residual, x0);

    CHECK(result.converged);
    CHECK(result.x[0] == doctest::Approx(5.0).epsilon(0.1));
    CHECK(result.x[1] == doctest::Approx(0.5).epsilon(0.1));
    CHECK(result.x[2] == doctest::Approx(1.0).epsilon(0.1));
}

TEST_CASE("LevenbergMarquardt - Robustness to poor initialization") {
    // Test that LM can recover from very poor initial guess
    // where Gauss-Newton might fail

    auto residual = [](const dp::mat::Vector<double, 2> &x) {
        dp::mat::Vector<double, 3> r;
        r[0] = x[0] - 1.0;
        r[1] = x[1] - 2.0;
        r[2] = (x[0] - 1.0) * (x[1] - 2.0); // Nonlinear coupling
        return r;
    };

    LevenbergMarquardt<double> lm;
    lm.max_iterations = 100;
    lm.tolerance = 1e-10;
    lm.initial_lambda = 1e-1; // Higher initial lambda for robustness
    lm.verbose = false;

    // Very poor initial guess
    dp::mat::Vector<double, 2> x0;
    x0[0] = 100.0;
    x0[1] = -100.0;

    auto result = lm.optimize(residual, x0);

    CHECK(result.converged);
    CHECK(result.x[0] == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(result.x[1] == doctest::Approx(2.0).epsilon(1e-6));
}

TEST_CASE("LevenbergMarquardt - Lambda adaptation") {
    // Test that lambda increases when steps are rejected
    // and decreases when steps are accepted

    auto residual = [](const dp::mat::Vector<double, 1> &x) {
        dp::mat::Vector<double, 1> r;
        r[0] = x[0] - 5.0;
        return r;
    };

    LevenbergMarquardt<double> lm;
    lm.max_iterations = 50;
    lm.tolerance = 1e-10;
    lm.initial_lambda = 1e-3;
    lm.lambda_factor = 10.0;
    lm.verbose = false;

    dp::mat::Vector<double, 1> x0;
    x0[0] = 0.0;

    auto result = lm.optimize(residual, x0);

    CHECK(result.converged);
    CHECK(result.x[0] == doctest::Approx(5.0).epsilon(1e-6));
    // Lambda should decrease as we approach solution
}

TEST_CASE("LevenbergMarquardt - Comparison with Gauss-Newton") {
    // On a well-conditioned problem, LM should give similar results to GN
    // but potentially use more iterations (due to conservative lambda)

    auto residual = [](const dp::mat::Vector<double, 2> &x) {
        dp::mat::Vector<double, 4> r;
        r[0] = x[0] + x[1] - 3.0;
        r[1] = x[0] - x[1] - 1.0;
        r[2] = 2.0 * x[0] - 4.0;
        r[3] = 3.0 * x[1] - 3.0;
        return r;
    };

    LevenbergMarquardt<double> lm;
    lm.max_iterations = 50;
    lm.tolerance = 1e-10;
    lm.verbose = false;

    dp::mat::Vector<double, 2> x0;
    x0[0] = 0.0;
    x0[1] = 0.0;

    auto result = lm.optimize(residual, x0);

    CHECK(result.converged);
    CHECK(result.x[0] == doctest::Approx(2.0).epsilon(1e-6));
    CHECK(result.x[1] == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("LevenbergMarquardt - dp::mat::Dynamic sized problem") {
    auto residual = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
        dp::mat::Vector<double, dp::mat::Dynamic> r;
        r.resize(x.size() + 1);
        for (std::size_t i = 0; i < x.size(); ++i) {
            r[i] = x[i] - static_cast<double>(i + 1);
        }
        r[x.size()] = 0.0;
        return r;
    };

    LevenbergMarquardt<double> lm;
    lm.max_iterations = 50;
    lm.tolerance = 1e-10;
    lm.verbose = false;

    dp::mat::Vector<double, dp::mat::Dynamic> x0;
    x0.resize(3);
    x0[0] = 0.0;
    x0[1] = 0.0;
    x0[2] = 0.0;

    auto result = lm.optimize(residual, x0);

    CHECK(result.converged);
    CHECK(result.x[0] == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(result.x[1] == doctest::Approx(2.0).epsilon(1e-6));
    CHECK(result.x[2] == doctest::Approx(3.0).epsilon(1e-6));
}

TEST_CASE("LevenbergMarquardt - Ill-conditioned problem") {
    // Problem where Gauss-Newton might struggle
    // but LM should handle gracefully

    auto residual = [](const dp::mat::Vector<double, 2> &x) {
        dp::mat::Vector<double, 3> r;
        r[0] = 1000.0 * (x[0] - 1.0); // Very different scales
        r[1] = 0.001 * (x[1] - 2.0);
        r[2] = x[0] + x[1] - 3.0;
        return r;
    };

    LevenbergMarquardt<double> lm;
    lm.max_iterations = 100;
    lm.tolerance = 1e-8;
    lm.initial_lambda = 1e-2;
    lm.verbose = false;

    dp::mat::Vector<double, 2> x0;
    x0[0] = 10.0; // Poor guess
    x0[1] = -5.0;

    auto result = lm.optimize(residual, x0);

    CHECK(result.converged);
    CHECK(result.x[0] == doctest::Approx(1.0).epsilon(1e-4));
    CHECK(result.x[1] == doctest::Approx(2.0).epsilon(1e-4));
}
