#include <iostream>
#include <optinum/optinum.hpp>

using namespace optinum;

int main() {
    std::cout << "Testing Dynamic optimization\n";

    std::size_t n = 3;
    std::cout << "Creating dynamic vector of size " << n << "\n";

    Vector<double, Dynamic> x(n);
    std::cout << "Vector created, size = " << x.size() << "\n";

    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) + 1.0;
        std::cout << "x[" << i << "] = " << x[i] << "\n";
    }

    std::cout << "Creating Sphere function\n";
    Sphere<double, Dynamic> sphere;

    std::cout << "Evaluating sphere function\n";
    double f = sphere.evaluate(x);
    std::cout << "f(x) = " << f << "\n";

    std::cout << "Computing gradient\n";
    Vector<double, Dynamic> g(n);
    sphere.gradient(x, g);
    std::cout << "Gradient computed\n";
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << "g[" << i << "] = " << g[i] << "\n";
    }

    std::cout << "Creating optimizer\n";
    GradientDescent<> gd;
    gd.step_size = 0.1;
    gd.max_iterations = 10;
    gd.tolerance = 1e-6;

    std::cout << "Running optimization...\n";
    auto result = gd.optimize(sphere, x);

    std::cout << "Optimization complete!\n";
    std::cout << "Converged: " << result.converged << "\n";
    std::cout << "Iterations: " << result.iterations << "\n";
    std::cout << "Final cost: " << result.final_cost << "\n";

    return 0;
}
