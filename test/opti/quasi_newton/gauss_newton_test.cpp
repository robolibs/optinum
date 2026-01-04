#include <doctest/doctest.h>

#include <optinum/opti/quasi_newton/gauss_newton.hpp>

#include <cmath>
#include <vector>

using datapod::mat::Dynamic;
using optinum::opti::GaussNewton;
namespace dp = datapod;

// =============================================================================
// TEST SUITE: Gauss-Newton Optimizer
// =============================================================================

TEST_CASE("GaussNewton - Simple linear least squares") {
    // Solve: min ||Ax - b||^2 where A*x = b has exact solution
    // A = [[1, 2], [3, 4], [5, 6]], b = [5, 11, 17]
    // Solution: x = [1, 2]

    auto residual = [](const dp::mat::Vector<double, 2> &x) {
        dp::mat::Vector<double, 3> r;
        r[0] = 1.0 * x[0] + 2.0 * x[1] - 5.0;
        r[1] = 3.0 * x[0] + 4.0 * x[1] - 11.0;
        r[2] = 5.0 * x[0] + 6.0 * x[1] - 17.0;
        return r;
    };

    GaussNewton<double> gn;
    gn.max_iterations = 50;
    gn.tolerance = 1e-10;
    gn.verbose = false;

    dp::mat::Vector<double, 2> x0;
    x0[0] = 0.0;
    x0[1] = 0.0;

    auto result = gn.optimize(residual, x0);

    CHECK(result.converged);
    CHECK(result.x[0] == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(result.x[1] == doctest::Approx(2.0).epsilon(1e-6));
    CHECK(result.iterations < 10); // Should converge very fast for linear problem
}

TEST_CASE("GaussNewton - Nonlinear curve fitting (exponential decay)") {
    // Fit y = a * exp(-b*x) + c to noisy data
    // True parameters: a=5, b=0.5, c=1

    struct DataPoint {
        double x, y;
    };

    std::vector<DataPoint> data = {{0.0, 6.0},  {1.0, 4.03}, {2.0, 2.85}, {3.0, 2.23}, {4.0, 1.68},  {5.0, 1.41},
                                   {6.0, 1.25}, {7.0, 1.12}, {8.0, 1.06}, {9.0, 1.03}, {10.0, 1.01}, {11.0, 1.0},
                                   {12.0, 1.0}, {13.0, 1.0}, {14.0, 1.0}, {15.0, 1.0}, {16.0, 1.0},  {17.0, 1.0},
                                   {18.0, 1.0}, {19.0, 1.0}, {20.0, 1.01}};

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

    GaussNewton<double> gn;
    gn.max_iterations = 50;
    gn.tolerance = 1e-8;
    gn.use_line_search = true; // Use line search for robustness
    gn.verbose = false;

    dp::mat::Vector<double, 3> x0;
    x0[0] = 1.0; // Initial guess for a
    x0[1] = 0.1; // Initial guess for b
    x0[2] = 0.0; // Initial guess for c

    auto result = gn.optimize(residual, x0);

    CHECK(result.converged);
    CHECK(result.x[0] == doctest::Approx(5.0).epsilon(0.1)); // a ≈ 5
    CHECK(result.x[1] == doctest::Approx(0.5).epsilon(0.1)); // b ≈ 0.5
    CHECK(result.x[2] == doctest::Approx(1.0).epsilon(0.1)); // c ≈ 1
    CHECK(result.iterations < 30);                           // Reasonable convergence
    CHECK(result.final_cost < 0.1);                          // Low final error
}

TEST_CASE("GaussNewton - Circle fitting") {
    // Fit circle (x-cx)^2 + (y-cy)^2 = r^2 to noisy points
    // True circle: center (2, 3), radius 5

    struct Point2D {
        double x, y;
    };

    // Points generated on circle (2, 3) with radius ~5 and small noise
    std::vector<Point2D> points = {{7.03, 3.00},  {6.53, 4.88},  {5.50, 6.50},   {3.89, 7.57},
                                   {2.00, 8.05},  {0.07, 7.65},  {-1.59, 6.59},  {-2.54, 4.88},
                                   {-2.98, 3.00}, {-2.53, 1.12}, {-1.50, -0.50}, {0.09, -1.62},
                                   {2.00, -1.91}, {3.89, -1.56}, {5.56, -0.56},  {6.63, 1.08}};

    // Residual: algebraic distance (not geometric, but simpler)
    // r_i = (x_i - cx)^2 + (y_i - cy)^2 - r^2
    auto residual = [&points](const dp::mat::Vector<double, 3> &params) {
        dp::mat::Vector<double, dp::mat::Dynamic> r;
        r.resize(points.size());
        double cx = params[0];
        double cy = params[1];
        double radius = params[2];

        for (std::size_t i = 0; i < points.size(); ++i) {
            double dx = points[i].x - cx;
            double dy = points[i].y - cy;
            r[i] = dx * dx + dy * dy - radius * radius;
        }
        return r;
    };

    GaussNewton<double> gn;
    gn.max_iterations = 50;
    gn.tolerance = 1e-8;
    gn.use_line_search = true; // Circle fitting can benefit from line search
    gn.verbose = false;

    dp::mat::Vector<double, 3> x0;
    x0[0] = 0.0; // Initial guess for cx
    x0[1] = 0.0; // Initial guess for cy
    x0[2] = 3.0; // Initial guess for r

    auto result = gn.optimize(residual, x0);

    CHECK(result.converged);
    CHECK(result.x[0] == doctest::Approx(2.0).epsilon(0.05)); // cx ≈ 2
    CHECK(result.x[1] == doctest::Approx(3.0).epsilon(0.05)); // cy ≈ 3
    CHECK(result.x[2] == doctest::Approx(5.0).epsilon(0.05)); // r ≈ 5
    CHECK(result.iterations < 20);
}

TEST_CASE("GaussNewton - Rosenbrock as least squares") {
    // Rosenbrock function reformulated as least squares:
    // f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    //        = r1^2 + r2^2
    // where r1 = 1-x, r2 = 10*(y-x^2)
    // Minimum at (1, 1)

    auto residual = [](const dp::mat::Vector<double, 2> &x) {
        dp::mat::Vector<double, 2> r;
        r[0] = 1.0 - x[0];
        r[1] = 10.0 * (x[1] - x[0] * x[0]);
        return r;
    };

    GaussNewton<double> gn;
    gn.max_iterations = 100;
    gn.tolerance = 1e-8;
    gn.use_line_search = true; // Rosenbrock benefits from line search
    gn.verbose = false;

    dp::mat::Vector<double, 2> x0;
    x0[0] = -1.0;
    x0[1] = 2.0;

    auto result = gn.optimize(residual, x0);

    CHECK(result.converged);
    CHECK(result.x[0] == doctest::Approx(1.0).epsilon(1e-4));
    CHECK(result.x[1] == doctest::Approx(1.0).epsilon(1e-4));
    CHECK(result.final_cost < 1e-10);
}

TEST_CASE("GaussNewton - Bundle adjustment toy problem") {
    // Simplified bundle adjustment: estimate 3D point from 2D observations
    // 3D point: P = (X, Y, Z) in world frame
    // 2 cameras at different positions observe P
    // Projection: (x, y) = (X/Z, Y/Z) (simple pinhole)

    struct Observation {
        double x, y;   // Observed 2D point
        double tx, ty; // Camera translation
    };

    // True 3D point: (4, 3, 10)
    // Camera 1 at (0, 0), observes (0.4, 0.3)
    // Camera 2 at (1, 0), observes (0.3, 0.3)
    std::vector<Observation> obs = {
        {0.4, 0.3, 0.0, 0.0},
        {0.3, 0.3, 1.0, 0.0},
    };

    auto residual = [&obs](const dp::mat::Vector<double, 3> &P) {
        dp::mat::Vector<double, dp::mat::Dynamic> r;
        r.resize(obs.size() * 2); // 2 residuals per observation

        for (std::size_t i = 0; i < obs.size(); ++i) {
            // Transform point to camera frame
            double X_cam = P[0] - obs[i].tx;
            double Y_cam = P[1] - obs[i].ty;
            double Z_cam = P[2];

            // Project to image plane
            double x_pred = X_cam / Z_cam;
            double y_pred = Y_cam / Z_cam;

            // Residuals
            r[2 * i + 0] = x_pred - obs[i].x;
            r[2 * i + 1] = y_pred - obs[i].y;
        }
        return r;
    };

    GaussNewton<double> gn;
    gn.max_iterations = 50;
    gn.tolerance = 1e-10;
    gn.verbose = false;

    dp::mat::Vector<double, 3> P0;
    P0[0] = 5.0;  // Initial guess X
    P0[1] = 4.0;  // Initial guess Y
    P0[2] = 12.0; // Initial guess Z

    auto result = gn.optimize(residual, P0);

    CHECK(result.converged);
    CHECK(result.x[0] == doctest::Approx(4.0).epsilon(1e-4));  // X ≈ 4
    CHECK(result.x[1] == doctest::Approx(3.0).epsilon(1e-4));  // Y ≈ 3
    CHECK(result.x[2] == doctest::Approx(10.0).epsilon(1e-4)); // Z ≈ 10
    CHECK(result.final_cost < 1e-12);
}

TEST_CASE("GaussNewton - QR solver vs Normal equations solver") {
    // Test that QR solver produces same result as normal equations
    // QR is more stable for ill-conditioned problems

    auto residual = [](const dp::mat::Vector<double, 2> &x) {
        dp::mat::Vector<double, 3> r;
        r[0] = x[0] + 2.0 * x[1] - 3.0;
        r[1] = 2.0 * x[0] + x[1] - 2.0;
        r[2] = x[0] - x[1] - 1.0;
        return r;
    };

    dp::mat::Vector<double, 2> x0;
    x0[0] = 0.0;
    x0[1] = 0.0;

    // Test with normal equations (J^T*J)
    GaussNewton<double> gn_normal;
    gn_normal.max_iterations = 50;
    gn_normal.tolerance = 1e-10;
    gn_normal.linear_solver = "normal";
    gn_normal.verbose = false;

    auto result_normal = gn_normal.optimize(residual, x0);

    // Test with QR
    GaussNewton<double> gn_qr;
    gn_qr.max_iterations = 50;
    gn_qr.tolerance = 1e-10;
    gn_qr.linear_solver = "qr";
    gn_qr.verbose = false;

    auto result_qr = gn_qr.optimize(residual, x0);

    // Both should converge to same solution
    CHECK(result_normal.converged);
    CHECK(result_qr.converged);
    CHECK(result_normal.x[0] == doctest::Approx(result_qr.x[0]).epsilon(1e-6));
    CHECK(result_normal.x[1] == doctest::Approx(result_qr.x[1]).epsilon(1e-6));
}

TEST_CASE("GaussNewton - Line search effectiveness") {
    // Test that line search helps with poor initial guess
    // Rosenbrock is a good test case

    auto residual = [](const dp::mat::Vector<double, 2> &x) {
        dp::mat::Vector<double, 2> r;
        r[0] = 1.0 - x[0];
        r[1] = 10.0 * (x[1] - x[0] * x[0]);
        return r;
    };

    dp::mat::Vector<double, 2> x0;
    x0[0] = -5.0; // Very poor initial guess
    x0[1] = 10.0;

    // Without line search (may fail or diverge)
    GaussNewton<double> gn_no_ls;
    gn_no_ls.max_iterations = 100;
    gn_no_ls.tolerance = 1e-8;
    gn_no_ls.use_line_search = false;
    gn_no_ls.verbose = false;

    auto result_no_ls = gn_no_ls.optimize(residual, x0);

    // With line search (should be more robust)
    GaussNewton<double> gn_ls;
    gn_ls.max_iterations = 100;
    gn_ls.tolerance = 1e-8;
    gn_ls.use_line_search = true;
    gn_ls.verbose = false;

    auto result_ls = gn_ls.optimize(residual, x0);

    // Line search version should converge better
    CHECK(result_ls.converged);
    if (result_ls.converged) {
        CHECK(result_ls.final_cost < 1e-6);
    }
}

TEST_CASE("GaussNewton - Convergence criteria") {
    // Test different convergence criteria

    auto residual = [](const dp::mat::Vector<double, 2> &x) {
        dp::mat::Vector<double, 2> r;
        r[0] = x[0] - 1.0;
        r[1] = x[1] - 2.0;
        return r;
    };

    // Test 1: Tolerance on error decrease
    {
        GaussNewton<double> gn;
        gn.max_iterations = 100;
        gn.tolerance = 1e-6;
        gn.min_step_norm = 1e-20;     // Effectively disabled
        gn.min_gradient_norm = 1e-20; // Effectively disabled
        gn.verbose = false;

        dp::mat::Vector<double, 2> x0;
        x0[0] = 0.0;
        x0[1] = 0.0;
        auto result = gn.optimize(residual, x0);

        CHECK(result.converged);
        CHECK(result.termination_reason.find("error decrease") != std::string::npos);
    }

    // Test 2: Minimum step norm
    {
        GaussNewton<double> gn;
        gn.max_iterations = 100;
        gn.tolerance = 1e-20; // Effectively disabled
        gn.min_step_norm = 1e-6;
        gn.min_gradient_norm = 1e-20; // Effectively disabled
        gn.verbose = false;

        dp::mat::Vector<double, 2> x0;
        x0[0] = 0.9;
        x0[1] = 1.9;
        auto result = gn.optimize(residual, x0);

        CHECK(result.converged);
        CHECK(result.termination_reason.find("step norm") != std::string::npos);
    }

    // Test 3: Gradient norm
    {
        GaussNewton<double> gn;
        gn.max_iterations = 100;
        gn.tolerance = 1e-20;     // Effectively disabled
        gn.min_step_norm = 1e-20; // Effectively disabled
        gn.min_gradient_norm = 1e-3;
        gn.verbose = false;

        dp::mat::Vector<double, 2> x0;
        x0[0] = 0.5;
        x0[1] = 1.5;
        auto result = gn.optimize(residual, x0);

        CHECK(result.converged);
        CHECK(result.termination_reason.find("gradient norm") != std::string::npos);
    }
}

TEST_CASE("GaussNewton - dp::mat::Dynamic sized problems") {
    // Test with dp::mat::Dynamic-sized vectors

    auto residual = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
        std::cout << "residual called with x.size() = " << x.size() << std::endl;
        dp::mat::Vector<double, dp::mat::Dynamic> r;
        r.resize(x.size() + 1);
        std::cout << "r.size() = " << r.size() << std::endl;
        for (std::size_t i = 0; i < x.size(); ++i) {
            r[i] = x[i] - static_cast<double>(i + 1);
        }
        r[x.size()] = 0.0; // Overdetermined
        return r;
    };

    GaussNewton<double> gn;
    gn.max_iterations = 50;
    gn.tolerance = 1e-10;
    gn.verbose = true;

    dp::mat::Vector<double, dp::mat::Dynamic> x0;
    x0.resize(3);
    std::cout << "x0.size() = " << x0.size() << std::endl;
    x0[0] = 0.0;
    x0[1] = 0.0;
    x0[2] = 0.0;

    std::cout << "Starting optimization..." << std::endl;
    auto result = gn.optimize(residual, x0);
    std::cout << "Optimization complete" << std::endl;

    CHECK(result.converged);
    CHECK(result.x[0] == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(result.x[1] == doctest::Approx(2.0).epsilon(1e-6));
    CHECK(result.x[2] == doctest::Approx(3.0).epsilon(1e-6));
}
