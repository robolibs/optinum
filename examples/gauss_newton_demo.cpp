// =============================================================================
// gauss_newton_demo.cpp
// Demonstrates Gauss-Newton optimizer for nonlinear least squares problems
// =============================================================================

#include <iomanip>
#include <iostream>
#include <optinum/optinum.hpp>

using namespace optinum;

// =============================================================================
// Example 1: Exponential Curve Fitting
// =============================================================================
// Fit the model: y = a * exp(-b * x) + c
// to noisy data points

void example_exponential_curve_fitting() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Example 1: Exponential Curve Fitting\n";
    std::cout << "========================================\n\n";

    // Generate synthetic data: y = 2.0 * exp(-0.5 * x) + 1.0 + noise
    std::cout << "True parameters: a = 2.0, b = 0.5, c = 1.0\n\n";

    struct DataPoint {
        double x, y;
    };

    std::vector<DataPoint> data = {{0.0, 3.02}, {0.5, 2.56}, {1.0, 2.21}, {1.5, 1.92}, {2.0, 1.74},
                                   {2.5, 1.55}, {3.0, 1.45}, {3.5, 1.32}, {4.0, 1.27}, {4.5, 1.20},
                                   {5.0, 1.16}, {5.5, 1.10}, {6.0, 1.08}, {6.5, 1.05}, {7.0, 1.03},
                                   {7.5, 1.02}, {8.0, 1.01}, {8.5, 1.00}, {9.0, 0.99}, {9.5, 0.99}};

    // Define residual function: r_i = y_pred - y_actual
    auto residual = [&data](const Vector<double, 3> &params) {
        Vector<double, Dynamic> r;
        r.resize(data.size());

        double a = params[0];
        double b = params[1];
        double c = params[2];

        for (size_t i = 0; i < data.size(); ++i) {
            double y_pred = a * exp(-b * data[i].x) + c;
            r[i] = y_pred - data[i].y;
        }
        return r;
    };

    // Configure Gauss-Newton optimizer
    opti::GaussNewton<double> gn;
    gn.max_iterations = 50;
    gn.tolerance = 1e-8;
    gn.min_gradient_norm = 1e-10;
    gn.verbose = true;

    // Initial guess
    Vector<double, 3> x0;
    x0[0] = 1.0;
    x0[1] = 0.1;
    x0[2] = 0.0;
    std::cout << "Initial guess: [" << x0[0] << ", " << x0[1] << ", " << x0[2] << "]\n\n";

    // Optimize!
    auto result = gn.optimize(residual, x0);

    // Print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nResults:\n";
    std::cout << "  Converged: " << (result.converged ? "Yes" : "No") << "\n";
    std::cout << "  Iterations: " << result.iterations << "\n";
    std::cout << "  Final parameters: [" << result.x[0] << ", " << result.x[1] << ", " << result.x[2] << "]\n";
    std::cout << "  Final error: " << result.final_cost << "\n";
    std::cout << "  Gradient norm: " << result.gradient_norm << "\n";
    std::cout << "  Reason: " << result.termination_reason << "\n";
}

// =============================================================================
// Example 2: Circle Fitting
// =============================================================================
// Fit a circle (x - cx)^2 + (y - cy)^2 = r^2 to 2D points

void example_circle_fitting() {
    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "Example 2: Circle Fitting\n";
    std::cout << "========================================\n\n";

    // Points roughly on circle centered at (2, 3) with radius ~5
    std::vector<Vector<double, 2>> points(16);
    points[0] = {7.03, 3.00};
    points[1] = {6.53, 4.88};
    points[2] = {5.50, 6.50};
    points[3] = {3.89, 7.57};
    points[4] = {2.00, 8.05};
    points[5] = {0.07, 7.65};
    points[6] = {-1.59, 6.59};
    points[7] = {-2.54, 4.88};
    points[8] = {-2.98, 3.00};
    points[9] = {-2.53, 1.12};
    points[10] = {-1.50, -0.50};
    points[11] = {0.09, -1.62};
    points[12] = {2.00, -1.91};
    points[13] = {3.89, -1.56};
    points[14] = {5.56, -0.56};
    points[15] = {6.63, 1.08};

    std::cout << "Fitting circle to " << points.size() << " 2D points\n\n";

    // Algebraic distance residual: r_i = (x_i - cx)^2 + (y_i - cy)^2 - r^2
    auto residual = [&points](const Vector<double, 3> &params) {
        Vector<double, Dynamic> r;
        r.resize(points.size());

        double cx = params[0];
        double cy = params[1];
        double radius = params[2];

        for (size_t i = 0; i < points.size(); ++i) {
            double dx = points[i][0] - cx;
            double dy = points[i][1] - cy;
            r[i] = dx * dx + dy * dy - radius * radius;
        }
        return r;
    };

    // Configure optimizer
    opti::GaussNewton<double> gn;
    gn.max_iterations = 100;
    gn.tolerance = 1e-10;
    gn.use_line_search = true;
    gn.verbose = true;

    // Initial guess (centroid of points, approximate radius)
    Vector<double, 3> x0;
    x0[0] = 0.0;
    x0[1] = 0.0;
    x0[2] = 3.0;
    std::cout << "Initial guess: cx=" << x0[0] << ", cy=" << x0[1] << ", r=" << x0[2] << "\n\n";

    // Optimize!
    auto result = gn.optimize(residual, x0);

    // Print results
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nResults:\n";
    std::cout << "  Circle center: (" << result.x[0] << ", " << result.x[1] << ")\n";
    std::cout << "  Circle radius: " << result.x[2] << "\n";
    std::cout << "  Final error: " << result.final_cost << "\n";
    std::cout << "  Iterations: " << result.iterations << "\n";
    std::cout << "  Expected: center ≈ (2.0, 3.0), radius ≈ 5.0\n";
}

// =============================================================================
// Example 3: Rosenbrock Function (2D)
// =============================================================================
// Minimize f(x,y) = (1-x)^2 + 100*(y-x^2)^2
// This is actually a scalar function, but we can treat it as minimizing ||f||^2

void example_rosenbrock_2d() {
    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "Example 3: Rosenbrock 2D\n";
    std::cout << "========================================\n\n";

    std::cout << "Minimize: f(x,y) = (1-x)^2 + 100*(y-x^2)^2\n";
    std::cout << "Global minimum at (1, 1) with f = 0\n\n";

    // Residual form: split the function into two residuals
    auto residual = [](const Vector<double, 2> &x) {
        Vector<double, 2> r;
        r[0] = 1.0 - x[0];                  // First term
        r[1] = 10.0 * (x[1] - x[0] * x[0]); // Second term (scaled sqrt)
        return r;
    };

    // Configure optimizer
    opti::GaussNewton<double> gn;
    gn.max_iterations = 100;
    gn.tolerance = 1e-12;
    gn.use_line_search = true; // Important for Rosenbrock!
    gn.verbose = true;

    // Initial guess
    Vector<double, 2> x0;
    x0[0] = -1.0;
    x0[1] = 2.0;
    std::cout << "Initial guess: [" << x0[0] << ", " << x0[1] << "]\n\n";

    // Optimize!
    auto result = gn.optimize(residual, x0);

    // Print results
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "\nResults:\n";
    std::cout << "  Solution: [" << result.x[0] << ", " << result.x[1] << "]\n";
    std::cout << "  Expected: [1.0, 1.0]\n";
    std::cout << "  Final cost: " << result.final_cost << "\n";
    std::cout << "  Iterations: " << result.iterations << "\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════╗\n";
    std::cout << "║   Gauss-Newton Optimizer Demonstration        ║\n";
    std::cout << "║   Nonlinear Least Squares Optimization        ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n";

    example_exponential_curve_fitting();
    example_circle_fitting();
    example_rosenbrock_2d();

    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "Summary\n";
    std::cout << "========================================\n\n";
    std::cout << "Gauss-Newton is a powerful method for nonlinear least squares:\n";
    std::cout << "  ✓ Quadratic convergence near the solution\n";
    std::cout << "  ✓ No hyperparameters to tune (unlike gradient descent)\n";
    std::cout << "  ✓ Works best with good initial guesses\n";
    std::cout << "  ✓ Use line search for challenging problems\n\n";
    std::cout << "For poor initializations, try Levenberg-Marquardt!\n\n";

    return 0;
}
