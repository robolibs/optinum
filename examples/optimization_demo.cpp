// =============================================================================
// Optimization Demo - Using optinum:: unified API
// Demonstrates gradient descent optimizer
// =============================================================================

#include <iostream>
#include <optinum/optinum.hpp>

namespace dp = datapod;

int main() {
    std::cout << "=== Optinum Optimization Demo ===\n\n";

    // =========================================================================
    // Problem 1: Optimize Sphere Function - Simple Quadratic
    // =========================================================================

    std::cout << "1. Sphere Function Optimization\n";
    std::cout << "   f(x, y) = x² + y²\n";
    std::cout << "   Minimum at (0, 0) with f = 0\n\n";

    // Create optimizer using unified API
    optinum::opti::GradientDescent<> gd;
    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    // Create test problem
    optinum::opti::Sphere<double, 2> sphere;

    // Initial point - use dp::mat::vector for owning storage
    dp::mat::vector<double, 2> x0{{5.0, 3.0}};

    std::cout << "   Initial point: [" << x0[0] << ", " << x0[1] << "]\n";
    std::cout << "   Initial cost:  " << sphere.evaluate(x0) << "\n\n";

    // Optimize!
    auto result = gd.optimize(sphere, x0);

    std::cout << "   Results:\n";
    std::cout << "   ✓ Converged: " << (result.converged ? "Yes" : "No") << "\n";
    std::cout << "   ✓ Iterations: " << result.iterations << "\n";
    std::cout << "   ✓ Final cost: " << result.final_cost << "\n";
    std::cout << "   ✓ Solution: [" << x0[0] << ", " << x0[1] << "]\n";
    std::cout << "   ✓ Reason: " << result.termination_reason << "\n\n";

    // =========================================================================
    // Problem 2: Custom Quadratic with Offset Minimum
    // =========================================================================

    std::cout << "2. Custom Quadratic Function\n";
    std::cout << "   f(x, y) = (x - 2)² + (y + 3)²\n";
    std::cout << "   Minimum at (2, -3) with f = 0\n\n";

    // Define custom function inline
    struct CustomQuadratic {
        using tensor_type = dp::mat::vector<double, 2>;

        double evaluate(const tensor_type &x) const {
            double dx = x[0] - 2.0;
            double dy = x[1] + 3.0;
            return dx * dx + dy * dy;
        }

        void gradient(const tensor_type &x, tensor_type &g) const {
            g[0] = 2.0 * (x[0] - 2.0);
            g[1] = 2.0 * (x[1] + 3.0);
        }

        double evaluate_with_gradient(const tensor_type &x, tensor_type &g) const {
            gradient(x, g);
            return evaluate(x);
        }
    };

    CustomQuadratic custom_func;
    dp::mat::vector<double, 2> x1{{0.0, 0.0}};

    std::cout << "   Initial point: [" << x1[0] << ", " << x1[1] << "]\n";
    std::cout << "   Initial cost:  " << custom_func.evaluate(x1) << "\n\n";

    auto result2 = gd.optimize(custom_func, x1);

    std::cout << "   Results:\n";
    std::cout << "   ✓ Converged: " << (result2.converged ? "Yes" : "No") << "\n";
    std::cout << "   ✓ Iterations: " << result2.iterations << "\n";
    std::cout << "   ✓ Final cost: " << result2.final_cost << "\n";
    std::cout << "   ✓ Solution: [" << x1[0] << ", " << x1[1] << "]\n";
    std::cout << "   ✓ Expected: [2.0, -3.0]\n\n";

    // =========================================================================
    // Problem 3: Higher Dimensional - 10D Sphere
    // =========================================================================

    std::cout << "3. High-Dimensional Sphere (10D)\n";
    std::cout << "   f(x) = Σ xᵢ²\n";
    std::cout << "   Minimum at origin with f = 0\n\n";

    optinum::opti::Sphere<double, 10> sphere_10d;
    optinum::opti::GradientDescent<> gd_10d;
    gd_10d.step_size = 0.05; // Smaller step for higher dimensions
    gd_10d.max_iterations = 2000;
    gd_10d.tolerance = 1e-6;

    dp::mat::vector<double, 10> x10;
    for (std::size_t i = 0; i < 10; ++i) {
        x10[i] = static_cast<double>(i) - 5.0; // [-5, -4, ..., 4]
    }

    double initial_cost_10d = sphere_10d.evaluate(x10);
    std::cout << "   Initial cost: " << initial_cost_10d << "\n\n";

    auto result3 = gd_10d.optimize(sphere_10d, x10);

    std::cout << "   Results:\n";
    std::cout << "   ✓ Converged: " << (result3.converged ? "Yes" : "No") << "\n";
    std::cout << "   ✓ Iterations: " << result3.iterations << "\n";
    std::cout << "   ✓ Final cost: " << result3.final_cost << "\n";
    std::cout << "   ✓ Solution norm: " << optinum::simd::view(x10).norm() << "\n\n";

    // =========================================================================
    // Problem 4: With Early Stopping Callback
    // =========================================================================

    std::cout << "4. Optimization with Early Stopping\n";
    std::cout << "   Stop when objective < 1.0\n\n";

    dp::mat::vector<double, 2> x_callback{{10.0, 10.0}};
    optinum::opti::EarlyStoppingCallback<double> callback(1.0); // Stop at f < 1.0

    auto result4 = gd.optimize(sphere, x_callback, callback);

    std::cout << "   Results:\n";
    std::cout << "   ✓ Converged: " << (result4.converged ? "Yes" : "No") << "\n";
    std::cout << "   ✓ Iterations: " << result4.iterations << "\n";
    std::cout << "   ✓ Final cost: " << result4.final_cost << "\n";
    std::cout << "   ✓ Reason: " << result4.termination_reason << "\n\n";

    std::cout << "Clean, unified optimization API! ✓\n";
    std::cout << "All through optinum:: namespace\n";

    return 0;
}
