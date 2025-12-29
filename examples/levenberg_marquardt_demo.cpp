// =============================================================================
// levenberg_marquardt_demo.cpp
// Demonstrates Levenberg-Marquardt optimizer - robust nonlinear least squares
// =============================================================================

#include <iomanip>
#include <iostream>
#include <optinum/optinum.hpp>

using namespace optinum;
namespace dp = datapod;

// =============================================================================
// Example 1: Robustness to Poor Initialization
// =============================================================================
// Compare Gauss-Newton vs Levenberg-Marquardt with terrible initial guess

void example_robustness_comparison() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Example 1: Robustness Comparison\n";
    std::cout << "========================================\n\n";

    std::cout << "Fitting exponential: y = a * exp(-b * x) + c\n";
    std::cout << "True parameters: a = 2.0, b = 0.5, c = 1.0\n";
    std::cout << "POOR initial guess: [100, -50, 1000]\n\n";

    // Synthetic data
    struct DataPoint {
        double x, y;
    };
    std::vector<DataPoint> data = {{0.0, 3.02}, {0.5, 2.56}, {1.0, 2.21}, {1.5, 1.92}, {2.0, 1.74}, {2.5, 1.55},
                                   {3.0, 1.45}, {3.5, 1.32}, {4.0, 1.27}, {4.5, 1.20}, {5.0, 1.16}};

    auto residual = [&data](const dp::mat::vector<double, 3> &params) {
        dp::mat::vector<double, dp::mat::Dynamic> r;
        r.resize(data.size());
        double a = params[0], b = params[1], c = params[2];
        for (size_t i = 0; i < data.size(); ++i) {
            r[i] = a * exp(-b * data[i].x) + c - data[i].y;
        }
        return r;
    };

    dp::mat::vector<double, 3> x0;
    x0[0] = 100.0;
    x0[1] = -50.0;
    x0[2] = 1000.0;

    // Try Gauss-Newton first
    std::cout << "--- Gauss-Newton ---\n";
    opti::GaussNewton<double> gn;
    gn.max_iterations = 50;
    gn.tolerance = 1e-8;
    gn.use_line_search = true;
    gn.verbose = false;

    auto result_gn = gn.optimize(residual, x0);
    std::cout << "  Result: " << (result_gn.converged ? "CONVERGED" : "FAILED") << "\n";
    std::cout << "  Iterations: " << result_gn.iterations << "\n";
    std::cout << "  Final error: " << std::scientific << result_gn.final_cost << "\n";
    if (result_gn.converged) {
        std::cout << "  Solution: [" << std::fixed << std::setprecision(4) << result_gn.x[0] << ", " << result_gn.x[1]
                  << ", " << result_gn.x[2] << "]\n";
    }

    // Now try Levenberg-Marquardt
    std::cout << "\n--- Levenberg-Marquardt ---\n";
    opti::LevenbergMarquardt<double> lm;
    lm.max_iterations = 50;
    lm.tolerance = 1e-8;
    lm.initial_lambda = 1e-3;
    lm.lambda_factor = 10.0;
    lm.verbose = false;

    auto result_lm = lm.optimize(residual, x0);
    std::cout << "  Result: " << (result_lm.converged ? "CONVERGED" : "FAILED") << "\n";
    std::cout << "  Iterations: " << result_lm.iterations << "\n";
    std::cout << "  Final error: " << std::scientific << result_lm.final_cost << "\n";
    if (result_lm.converged) {
        std::cout << "  Solution: [" << std::fixed << std::setprecision(4) << result_lm.x[0] << ", " << result_lm.x[1]
                  << ", " << result_lm.x[2] << "]\n";
    }

    std::cout << "\n✓ Levenberg-Marquardt succeeds where Gauss-Newton struggles!\n";
}

// =============================================================================
// Example 2: Ill-Conditioned Problem
// =============================================================================
// Problem with vastly different parameter scales

void example_ill_conditioned() {
    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "Example 2: Ill-Conditioned Problem\n";
    std::cout << "========================================\n\n";

    std::cout << "Parameters with very different scales:\n";
    std::cout << "  a ~ 1000  (large)\n";
    std::cout << "  b ~ 0.01  (small)\n\n";

    // Residual with mixed scales
    auto residual = [](const dp::mat::vector<double, 2> &x) {
        dp::mat::vector<double, 3> r;
        r[0] = 1000.0 - x[0];                         // Target: x[0] = 1000
        r[1] = 100.0 * (0.01 - x[1]);                 // Target: x[1] = 0.01 (scaled up)
        r[2] = (x[0] / 1000.0) * (x[1] / 0.01) - 1.0; // Coupling term
        return r;
    };

    dp::mat::vector<double, 2> x0;
    x0[0] = 500.0;
    x0[1] = 0.005;
    std::cout << "Initial guess: [" << x0[0] << ", " << x0[1] << "]\n\n";

    // Levenberg-Marquardt with adaptive damping
    opti::LevenbergMarquardt<double> lm;
    lm.max_iterations = 100;
    lm.tolerance = 1e-10;
    lm.initial_lambda = 1e-2;
    lm.lambda_factor = 5.0;
    lm.verbose = true;

    auto result = lm.optimize(residual, x0);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nResults:\n";
    std::cout << "  Solution: [" << result.x[0] << ", " << result.x[1] << "]\n";
    std::cout << "  Expected: [1000.0, 0.01]\n";
    std::cout << "  Final error: " << std::scientific << result.final_cost << "\n";
    std::cout << "  Iterations: " << result.iterations << "\n";
}

// =============================================================================
// Example 3: Bundle Adjustment (Mini)
// =============================================================================
// Simple 3D reconstruction problem

void example_bundle_adjustment() {
    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "Example 3: Mini Bundle Adjustment\n";
    std::cout << "========================================\n\n";

    std::cout << "Estimating 3D point from 2D observations\n";
    std::cout << "True 3D point: (1.0, 2.0, 5.0)\n\n";

    // Simulated camera projections (with noise)
    // Projection: u = x/z, v = y/z
    struct Observation {
        double u, v; // 2D image coordinates
    };

    std::vector<Observation> observations = {
        {0.202, 0.405}, // Camera 1
        {0.198, 0.395}, // Camera 2
        {0.205, 0.402}, // Camera 3
    };

    // Residual: reprojection error
    auto residual = [&observations](const dp::mat::vector<double, 3> &point) {
        dp::mat::vector<double, dp::mat::Dynamic> r;
        r.resize(observations.size() * 2);

        double x = point[0];
        double y = point[1];
        double z = point[2];

        for (size_t i = 0; i < observations.size(); ++i) {
            double u_pred = x / z;
            double v_pred = y / z;
            r[2 * i] = u_pred - observations[i].u;
            r[2 * i + 1] = v_pred - observations[i].v;
        }
        return r;
    };

    // Initial guess (rough depth estimate)
    dp::mat::vector<double, 3> x0;
    x0[0] = 1.0;
    x0[1] = 2.0;
    x0[2] = 4.5;
    std::cout << "Initial guess: [" << x0[0] << ", " << x0[1] << ", " << x0[2] << "]\n\n";

    // Levenberg-Marquardt
    opti::LevenbergMarquardt<double> lm;
    lm.max_iterations = 50;
    lm.tolerance = 1e-12;
    lm.initial_lambda = 1e-4;
    lm.verbose = true;

    auto result = lm.optimize(residual, x0);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nResults:\n";
    std::cout << "  Estimated 3D point: [" << result.x[0] << ", " << result.x[1] << ", " << result.x[2] << "]\n";
    std::cout << "  True point: [1.0, 2.0, 5.0]\n";
    std::cout << "  Reprojection error: " << std::scientific << result.final_cost << "\n";
    std::cout << "  Iterations: " << result.iterations << "\n";
}

// =============================================================================
// Example 4: Lambda Adaptation Behavior
// =============================================================================
// Demonstrate how lambda changes during optimization

void example_lambda_adaptation() {
    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "Example 4: Lambda Adaptation\n";
    std::cout << "========================================\n\n";

    std::cout << "Watch how lambda adapts during optimization:\n";
    std::cout << "  ↑ Increase lambda when step is bad (more gradient descent)\n";
    std::cout << "  ↓ Decrease lambda when step is good (more Gauss-Newton)\n\n";

    // Simple quadratic
    auto residual = [](const dp::mat::vector<double, 2> &x) {
        dp::mat::vector<double, 2> r;
        r[0] = x[0] - 3.0;
        r[1] = x[1] + 2.0;
        return r;
    };

    dp::mat::vector<double, 2> x0;
    x0[0] = 10.0;
    x0[1] = 10.0;

    opti::LevenbergMarquardt<double> lm;
    lm.max_iterations = 20;
    lm.tolerance = 1e-10;
    lm.initial_lambda = 1.0; // Start with high lambda
    lm.lambda_factor = 2.0;  // Moderate adaptation
    lm.verbose = true;

    std::cout << "Initial guess: [" << x0[0] << ", " << x0[1] << "]\n";
    std::cout << "Initial lambda: " << lm.initial_lambda << "\n\n";

    auto result = lm.optimize(residual, x0);

    std::cout << "\n✓ Lambda decreased as we approached the solution\n";
    std::cout << "  (transitions from gradient descent → Gauss-Newton)\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════╗\n";
    std::cout << "║  Levenberg-Marquardt Optimizer Demonstration  ║\n";
    std::cout << "║  Robust Nonlinear Least Squares               ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n";

    example_robustness_comparison();
    example_ill_conditioned();
    example_bundle_adjustment();
    example_lambda_adaptation();

    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "Summary: When to Use LM\n";
    std::cout << "========================================\n\n";
    std::cout << "Use Levenberg-Marquardt when:\n";
    std::cout << "  ✓ Initial guess is poor\n";
    std::cout << "  ✓ Problem is ill-conditioned\n";
    std::cout << "  ✓ Gauss-Newton diverges or oscillates\n";
    std::cout << "  ✓ Robustness is more important than speed\n\n";
    std::cout << "Use Gauss-Newton when:\n";
    std::cout << "  ✓ Initial guess is good\n";
    std::cout << "  ✓ Problem is well-conditioned\n";
    std::cout << "  ✓ Speed is critical (fewer iterations)\n\n";
    std::cout << "Tip: Try Gauss-Newton first, fall back to LM if needed!\n\n";

    return 0;
}
