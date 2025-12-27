// =============================================================================
// Optimizer Comparison Demo - Comparing Vanilla, Momentum, RMSprop, and Adam
// Demonstrates the performance difference between different optimization algorithms
// =============================================================================

#include <iomanip>
#include <iostream>
#include <optinum/optinum.hpp>

int main() {
    std::cout << "=== Optimizer Comparison Demo ===\n\n";

    // =========================================================================
    // Problem Setup: Sphere Function in 2D
    // =========================================================================

    std::cout << "Problem: Minimize f(x, y) = x² + y²\n";
    std::cout << "Minimum at (0, 0) with f = 0\n";
    std::cout << "Initial point: (5, 3)\n\n";

    optinum::Sphere<double, 2> sphere;
    optinum::Vector<double, 2> x_init(datapod::mat::vector<double, 2>{5.0, 3.0});

    // Common settings
    const double step_size = 0.1;
    const std::size_t max_iterations = 1000;
    const double tolerance = 1e-6;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "-----------------------------------------------------------\n";
    std::cout << "Optimizer         Iterations    Final Cost      Solution\n";
    std::cout << "-----------------------------------------------------------\n";

    // =========================================================================
    // 1. Vanilla Gradient Descent
    // =========================================================================

    {
        optinum::GradientDescent<> vanilla_gd;
        vanilla_gd.step_size = step_size;
        vanilla_gd.max_iterations = max_iterations;
        vanilla_gd.tolerance = tolerance;

        optinum::Vector<double, 2> x = x_init;
        auto result = vanilla_gd.optimize(sphere, x);

        std::cout << "Vanilla GD        " << std::setw(10) << result.iterations << "    " << std::setw(12)
                  << result.final_cost << "    (" << std::setw(8) << x[0] << ", " << std::setw(8) << x[1] << ")\n";
    }

    // =========================================================================
    // 2. Momentum Gradient Descent
    // =========================================================================

    {
        optinum::Momentum momentum;
        momentum.step_size = step_size;
        momentum.max_iterations = max_iterations;
        momentum.tolerance = tolerance;
        momentum.get_update_policy().momentum = 0.9;

        optinum::Vector<double, 2> x = x_init;
        auto result = momentum.optimize(sphere, x);

        std::cout << "Momentum          " << std::setw(10) << result.iterations << "    " << std::setw(12)
                  << result.final_cost << "    (" << std::setw(8) << x[0] << ", " << std::setw(8) << x[1] << ")\n";
    }

    // =========================================================================
    // 3. RMSprop
    // =========================================================================

    {
        optinum::RMSprop rmsprop;
        rmsprop.step_size = step_size;
        rmsprop.max_iterations = max_iterations;
        rmsprop.tolerance = tolerance;
        rmsprop.get_update_policy().alpha = 0.99;
        rmsprop.get_update_policy().epsilon = 1e-8;

        optinum::Vector<double, 2> x = x_init;
        auto result = rmsprop.optimize(sphere, x);

        std::cout << "RMSprop           " << std::setw(10) << result.iterations << "    " << std::setw(12)
                  << result.final_cost << "    (" << std::setw(8) << x[0] << ", " << std::setw(8) << x[1] << ")\n";
    }

    // =========================================================================
    // 4. Adam
    // =========================================================================

    {
        optinum::Adam adam;
        adam.step_size = step_size;
        adam.max_iterations = max_iterations;
        adam.tolerance = tolerance;
        adam.get_update_policy().beta1 = 0.9;
        adam.get_update_policy().beta2 = 0.999;
        adam.get_update_policy().epsilon = 1e-8;

        optinum::Vector<double, 2> x = x_init;
        auto result = adam.optimize(sphere, x);

        std::cout << "Adam              " << std::setw(10) << result.iterations << "    " << std::setw(12)
                  << result.final_cost << "    (" << std::setw(8) << x[0] << ", " << std::setw(8) << x[1] << ")\n";
    }

    std::cout << "-----------------------------------------------------------\n\n";

    // =========================================================================
    // Higher Dimensional Problem: 10D Sphere
    // =========================================================================

    std::cout << "\n=== 10D Sphere Function ===\n\n";
    std::cout << "Problem: Minimize f(x) = Σ xᵢ²  (i=1 to 10)\n";
    std::cout << "Initial point: [-5, -4, -3, ..., 3, 4]\n\n";

    optinum::Sphere<double, 10> sphere_10d;
    optinum::Vector<double, 10> x10_init;
    for (std::size_t i = 0; i < 10; ++i) {
        x10_init[i] = static_cast<double>(i) - 5.0;
    }

    const double step_size_10d = 0.05;
    const std::size_t max_iterations_10d = 2000;

    std::cout << "-----------------------------------------------------------\n";
    std::cout << "Optimizer         Iterations    Final Cost      ||x||\n";
    std::cout << "-----------------------------------------------------------\n";

    // Vanilla GD
    {
        optinum::GradientDescent<> vanilla_gd;
        vanilla_gd.step_size = step_size_10d;
        vanilla_gd.max_iterations = max_iterations_10d;
        vanilla_gd.tolerance = tolerance;

        optinum::Vector<double, 10> x = x10_init;
        auto result = vanilla_gd.optimize(sphere_10d, x);

        double norm = optinum::norm(x);
        std::cout << "Vanilla GD        " << std::setw(10) << result.iterations << "    " << std::setw(12)
                  << result.final_cost << "    " << std::setw(10) << norm << "\n";
    }

    // Momentum
    {
        optinum::Momentum momentum;
        momentum.step_size = step_size_10d;
        momentum.max_iterations = max_iterations_10d;
        momentum.tolerance = tolerance;

        optinum::Vector<double, 10> x = x10_init;
        auto result = momentum.optimize(sphere_10d, x);

        double norm = optinum::norm(x);
        std::cout << "Momentum          " << std::setw(10) << result.iterations << "    " << std::setw(12)
                  << result.final_cost << "    " << std::setw(10) << norm << "\n";
    }

    // RMSprop
    {
        optinum::RMSprop rmsprop;
        rmsprop.step_size = step_size_10d;
        rmsprop.max_iterations = max_iterations_10d;
        rmsprop.tolerance = tolerance;

        optinum::Vector<double, 10> x = x10_init;
        auto result = rmsprop.optimize(sphere_10d, x);

        double norm = optinum::norm(x);
        std::cout << "RMSprop           " << std::setw(10) << result.iterations << "    " << std::setw(12)
                  << result.final_cost << "    " << std::setw(10) << norm << "\n";
    }

    // Adam
    {
        optinum::Adam adam;
        adam.step_size = step_size_10d;
        adam.max_iterations = max_iterations_10d;
        adam.tolerance = tolerance;

        optinum::Vector<double, 10> x = x10_init;
        auto result = adam.optimize(sphere_10d, x);

        double norm = optinum::norm(x);
        std::cout << "Adam              " << std::setw(10) << result.iterations << "    " << std::setw(12)
                  << result.final_cost << "    " << std::setw(10) << norm << "\n";
    }

    std::cout << "-----------------------------------------------------------\n\n";

    // =========================================================================
    // Demonstration of Custom Parameters
    // =========================================================================

    std::cout << "\n=== Custom Hyperparameter Tuning ===\n\n";
    std::cout << "Adam with different beta values:\n\n";

    std::cout << "-----------------------------------------------------------\n";
    std::cout << "β₁     β₂      Iterations    Final Cost\n";
    std::cout << "-----------------------------------------------------------\n";

    // Try different beta1 values
    for (double beta1 : {0.8, 0.9, 0.95}) {
        optinum::Adam adam;
        adam.step_size = 0.1;
        adam.max_iterations = 1000;
        adam.tolerance = 1e-6;
        adam.get_update_policy().beta1 = beta1;
        adam.get_update_policy().beta2 = 0.999;

        optinum::Vector<double, 2> x = x_init;
        auto result = adam.optimize(sphere, x);

        std::cout << std::setw(4) << beta1 << "   0.999   " << std::setw(10) << result.iterations << "    "
                  << std::setw(12) << result.final_cost << "\n";
    }

    std::cout << "-----------------------------------------------------------\n\n";

    std::cout << "Key Observations:\n";
    std::cout << "  • Vanilla GD: Simple but effective baseline\n";
    std::cout << "  • Momentum: Often faster convergence with acceleration\n";
    std::cout << "  • RMSprop: Adaptive learning rates for each parameter\n";
    std::cout << "  • Adam: Combines momentum + adaptive rates + bias correction\n\n";

    return 0;
}
