#include <iostream>
#include <optinum/opti/problem/sphere.hpp>
#include <optinum/optinum.hpp>

int main() {
    // Sphere function: f(x) = sum(x_i^2)
    // Global minimum at origin: f(0,0,0) = 0
    optinum::opti::Sphere<double, 3> sphere;

    // Starting point
    optinum::simd::Tensor<double, 3> x;
    x[0] = 5.0;
    x[1] = -3.0;
    x[2] = 7.0;

    // Gradient storage
    optinum::simd::Tensor<double, 3> g;

    // Simple gradient descent
    double lr = 0.1; // learning rate
    int max_iter = 100;

    std::cout << "Optimizing Sphere function with Gradient Descent\n";
    std::cout << "Starting point: [" << x[0] << ", " << x[1] << ", " << x[2] << "]\n";
    std::cout << "Initial f(x) = " << sphere.evaluate(x) << "\n\n";

    for (int i = 0; i < max_iter; ++i) {
        double f = sphere.evaluate_with_gradient(x, g);

        // Update: x = x - lr * gradient
        x[0] -= lr * g[0];
        x[1] -= lr * g[1];
        x[2] -= lr * g[2];

        if (i < 10 || i % 10 == 0) {
            std::cout << "iter " << i << ": f(x) = " << f << "  x = [" << x[0] << ", " << x[1] << ", " << x[2] << "]\n";
        }

        // Stop if converged
        if (f < 1e-10) {
            std::cout << "\nConverged at iteration " << i << "\n";
            break;
        }
    }

    std::cout << "\nFinal point: [" << x[0] << ", " << x[1] << ", " << x[2] << "]\n";
    std::cout << "Final f(x) = " << sphere.evaluate(x) << "\n";
    std::cout << "Expected minimum: f(0,0,0) = 0\n";

    return 0;
}
