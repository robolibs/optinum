#include <doctest/doctest.h>

#include <datapod/matrix/vector.hpp>
#include <optinum/opti/problem/sphere.hpp>
#include <optinum/opti/quasi_newton/lbfgs.hpp>

#include <cmath>

using namespace optinum;
using namespace optinum::opti;
namespace dp = datapod;

// =============================================================================
// Test Functions
// =============================================================================

/**
 * @brief Simple quadratic function: f(x) = 0.5 * x^T * x
 *
 * Minimum at origin, gradient = x
 */
template <typename T, std::size_t N> struct QuadraticFunction {
    using vector_type = dp::mat::Vector<T, N>;

    T evaluate(const vector_type &x) const {
        T sum = T(0);
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i];
        }
        return sum / T(2);
    }

    void gradient(const vector_type &x, vector_type &g) const {
        for (std::size_t i = 0; i < x.size(); ++i) {
            g[i] = x[i];
        }
    }

    T evaluate_with_gradient(const vector_type &x, vector_type &g) const {
        gradient(x, g);
        return evaluate(x);
    }
};

/**
 * @brief Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
 *
 * Minimum at (1, 1), challenging for optimization
 */
struct RosenbrockFunction {
    using vector_type = dp::mat::Vector<double, 2>;

    double evaluate(const vector_type &x) const {
        double a = 1.0 - x[0];
        double b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    }

    void gradient(const vector_type &x, vector_type &g) const {
        double x0 = x[0];
        double x1 = x[1];
        g[0] = -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0);
        g[1] = 200.0 * (x1 - x0 * x0);
    }

    double evaluate_with_gradient(const vector_type &x, vector_type &g) const {
        gradient(x, g);
        return evaluate(x);
    }
};

/**
 * @brief N-dimensional Rosenbrock function
 *
 * f(x) = sum_{i=0}^{n-2} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
 * Minimum at x = (1, 1, ..., 1)
 */
template <typename T, std::size_t N> struct RosenbrockND {
    using vector_type = dp::mat::Vector<T, N>;

    T evaluate(const vector_type &x) const {
        const std::size_t n = x.size();
        T sum = T(0);
        for (std::size_t i = 0; i < n - 1; ++i) {
            T a = T(1) - x[i];
            T b = x[i + 1] - x[i] * x[i];
            sum += a * a + T(100) * b * b;
        }
        return sum;
    }

    void gradient(const vector_type &x, vector_type &g) const {
        const std::size_t n = x.size();
        g.fill(T(0));

        for (std::size_t i = 0; i < n - 1; ++i) {
            T xi = x[i];
            T xi1 = x[i + 1];
            // df/dx_i from (1 - x_i)^2 term: -2*(1 - x_i)
            // df/dx_i from 100*(x_{i+1} - x_i^2)^2 term: -400*x_i*(x_{i+1} - x_i^2)
            g[i] += -T(2) * (T(1) - xi) - T(400) * xi * (xi1 - xi * xi);
            // df/dx_{i+1} from 100*(x_{i+1} - x_i^2)^2 term: 200*(x_{i+1} - x_i^2)
            g[i + 1] += T(200) * (xi1 - xi * xi);
        }
    }

    T evaluate_with_gradient(const vector_type &x, vector_type &g) const {
        gradient(x, g);
        return evaluate(x);
    }
};

/**
 * @brief Beale's function (2D)
 *
 * f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
 * Minimum at (3, 0.5) with f = 0
 */
struct BealeFunction {
    using vector_type = dp::mat::Vector<double, 2>;

    double evaluate(const vector_type &x) const {
        double x0 = x[0], x1 = x[1];
        double a = 1.5 - x0 + x0 * x1;
        double b = 2.25 - x0 + x0 * x1 * x1;
        double c = 2.625 - x0 + x0 * x1 * x1 * x1;
        return a * a + b * b + c * c;
    }

    void gradient(const vector_type &x, vector_type &g) const {
        double x0 = x[0], x1 = x[1];
        double a = 1.5 - x0 + x0 * x1;
        double b = 2.25 - x0 + x0 * x1 * x1;
        double c = 2.625 - x0 + x0 * x1 * x1 * x1;

        // da/dx0 = -1 + x1, da/dx1 = x0
        // db/dx0 = -1 + x1^2, db/dx1 = 2*x0*x1
        // dc/dx0 = -1 + x1^3, dc/dx1 = 3*x0*x1^2

        g[0] = 2.0 * a * (-1.0 + x1) + 2.0 * b * (-1.0 + x1 * x1) + 2.0 * c * (-1.0 + x1 * x1 * x1);
        g[1] = 2.0 * a * x0 + 2.0 * b * 2.0 * x0 * x1 + 2.0 * c * 3.0 * x0 * x1 * x1;
    }

    double evaluate_with_gradient(const vector_type &x, vector_type &g) const {
        gradient(x, g);
        return evaluate(x);
    }
};

/**
 * @brief Booth's function (2D)
 *
 * f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
 * Minimum at (1, 3) with f = 0
 */
struct BoothFunction {
    using vector_type = dp::mat::Vector<double, 2>;

    double evaluate(const vector_type &x) const {
        double a = x[0] + 2.0 * x[1] - 7.0;
        double b = 2.0 * x[0] + x[1] - 5.0;
        return a * a + b * b;
    }

    void gradient(const vector_type &x, vector_type &g) const {
        double a = x[0] + 2.0 * x[1] - 7.0;
        double b = 2.0 * x[0] + x[1] - 5.0;
        g[0] = 2.0 * a + 4.0 * b;
        g[1] = 4.0 * a + 2.0 * b;
    }

    double evaluate_with_gradient(const vector_type &x, vector_type &g) const {
        gradient(x, g);
        return evaluate(x);
    }
};

// =============================================================================
// Basic L-BFGS Tests
// =============================================================================

TEST_CASE("LBFGS - Quadratic function (2D)") {
    using Vec2 = dp::mat::Vector<double, 2>;
    QuadraticFunction<double, 2> func;
    LBFGS<double> optimizer;

    optimizer.max_iterations = 100;
    optimizer.gradient_tolerance = 1e-8;

    SUBCASE("Starting from (1, 1)") {
        Vec2 x0(dp::mat::Vector<double, 2>{1.0, 1.0});
        auto result = optimizer.optimize(func, x0);

        CHECK(result.converged);
        CHECK(result.iterations < 20); // Should converge quickly
        CHECK(std::abs(result.x[0]) < 1e-6);
        CHECK(std::abs(result.x[1]) < 1e-6);
        CHECK(result.final_cost < 1e-12);
    }

    SUBCASE("Starting from (5, -3)") {
        Vec2 x0(dp::mat::Vector<double, 2>{5.0, -3.0});
        auto result = optimizer.optimize(func, x0);

        CHECK(result.converged);
        CHECK(std::abs(result.x[0]) < 1e-6);
        CHECK(std::abs(result.x[1]) < 1e-6);
    }

    SUBCASE("Starting from origin (already optimal)") {
        Vec2 x0(dp::mat::Vector<double, 2>{0.0, 0.0});
        auto result = optimizer.optimize(func, x0);

        CHECK(result.converged);
        CHECK(result.iterations == 0); // Should converge immediately
    }
}

TEST_CASE("LBFGS - Quadratic function (higher dimensions)") {
    SUBCASE("5D quadratic") {
        using Vec5 = dp::mat::Vector<double, 5>;
        QuadraticFunction<double, 5> func;
        LBFGS<double> optimizer;
        optimizer.gradient_tolerance = 1e-8;

        Vec5 x0;
        for (std::size_t i = 0; i < 5; ++i) {
            x0[i] = static_cast<double>(i + 1);
        }

        auto result = optimizer.optimize(func, x0);

        CHECK(result.converged);
        for (std::size_t i = 0; i < 5; ++i) {
            CHECK(std::abs(result.x[i]) < 1e-6);
        }
    }

    SUBCASE("10D quadratic") {
        using Vec10 = dp::mat::Vector<double, 10>;
        QuadraticFunction<double, 10> func;
        LBFGS<double> optimizer;
        optimizer.gradient_tolerance = 1e-8;

        Vec10 x0;
        for (std::size_t i = 0; i < 10; ++i) {
            x0[i] = static_cast<double>(i) - 5.0;
        }

        auto result = optimizer.optimize(func, x0);

        CHECK(result.converged);
        for (std::size_t i = 0; i < 10; ++i) {
            CHECK(std::abs(result.x[i]) < 1e-6);
        }
    }
}

TEST_CASE("LBFGS - Rosenbrock function (2D)") {
    using Vec2 = dp::mat::Vector<double, 2>;
    RosenbrockFunction func;
    LBFGS<double> optimizer;

    optimizer.max_iterations = 200;
    optimizer.gradient_tolerance = 1e-6;
    optimizer.history_size = 10;

    SUBCASE("Starting from (-1, 1)") {
        Vec2 x0(dp::mat::Vector<double, 2>{-1.0, 1.0});
        auto result = optimizer.optimize(func, x0);

        CHECK(result.converged);
        CHECK(std::abs(result.x[0] - 1.0) < 1e-4);
        CHECK(std::abs(result.x[1] - 1.0) < 1e-4);
        CHECK(result.final_cost < 1e-8);
    }

    SUBCASE("Starting from (0, 0)") {
        Vec2 x0(dp::mat::Vector<double, 2>{0.0, 0.0});
        auto result = optimizer.optimize(func, x0);

        CHECK(result.converged);
        CHECK(std::abs(result.x[0] - 1.0) < 1e-4);
        CHECK(std::abs(result.x[1] - 1.0) < 1e-4);
    }

    SUBCASE("Starting from (-2, 2)") {
        Vec2 x0(dp::mat::Vector<double, 2>{-2.0, 2.0});
        auto result = optimizer.optimize(func, x0);

        CHECK(result.converged);
        CHECK(std::abs(result.x[0] - 1.0) < 1e-4);
        CHECK(std::abs(result.x[1] - 1.0) < 1e-4);
    }
}

TEST_CASE("LBFGS - Booth function") {
    using Vec2 = dp::mat::Vector<double, 2>;
    BoothFunction func;
    LBFGS<double> optimizer;

    optimizer.max_iterations = 100;
    optimizer.gradient_tolerance = 1e-8;

    Vec2 x0(dp::mat::Vector<double, 2>{0.0, 0.0});
    auto result = optimizer.optimize(func, x0);

    CHECK(result.converged);
    CHECK(std::abs(result.x[0] - 1.0) < 1e-6);
    CHECK(std::abs(result.x[1] - 3.0) < 1e-6);
    CHECK(result.final_cost < 1e-12);
}

TEST_CASE("LBFGS - Beale function") {
    using Vec2 = dp::mat::Vector<double, 2>;
    BealeFunction func;
    LBFGS<double> optimizer;

    optimizer.max_iterations = 200;
    optimizer.gradient_tolerance = 1e-6;

    Vec2 x0(dp::mat::Vector<double, 2>{0.0, 0.0});
    auto result = optimizer.optimize(func, x0);

    CHECK(result.converged);
    CHECK(std::abs(result.x[0] - 3.0) < 1e-4);
    CHECK(std::abs(result.x[1] - 0.5) < 1e-4);
    CHECK(result.final_cost < 1e-8);
}

// =============================================================================
// Parameter Sensitivity Tests
// =============================================================================

TEST_CASE("LBFGS - History size variations") {
    using Vec2 = dp::mat::Vector<double, 2>;
    RosenbrockFunction func;

    Vec2 x0(dp::mat::Vector<double, 2>{-1.0, 1.0});

    SUBCASE("Small history (m=3)") {
        LBFGS<double> optimizer;
        optimizer.history_size = 3;
        optimizer.max_iterations = 300;
        optimizer.gradient_tolerance = 1e-6;

        auto result = optimizer.optimize(func, x0);
        CHECK(result.converged);
        CHECK(std::abs(result.x[0] - 1.0) < 1e-4);
    }

    SUBCASE("Medium history (m=10)") {
        LBFGS<double> optimizer;
        optimizer.history_size = 10;
        optimizer.max_iterations = 200;
        optimizer.gradient_tolerance = 1e-6;

        auto result = optimizer.optimize(func, x0);
        CHECK(result.converged);
        CHECK(std::abs(result.x[0] - 1.0) < 1e-4);
    }

    SUBCASE("Large history (m=20)") {
        LBFGS<double> optimizer;
        optimizer.history_size = 20;
        optimizer.max_iterations = 200;
        optimizer.gradient_tolerance = 1e-6;

        auto result = optimizer.optimize(func, x0);
        CHECK(result.converged);
        CHECK(std::abs(result.x[0] - 1.0) < 1e-4);
    }
}

TEST_CASE("LBFGS - Line search type") {
    using Vec2 = dp::mat::Vector<double, 2>;
    RosenbrockFunction func;

    Vec2 x0(dp::mat::Vector<double, 2>{-1.0, 1.0});

    SUBCASE("Wolfe line search (default)") {
        LBFGS<double> optimizer;
        optimizer.line_search_type = "wolfe";
        optimizer.max_iterations = 200;
        optimizer.gradient_tolerance = 1e-6;

        auto result = optimizer.optimize(func, x0);
        CHECK(result.converged);
    }

    SUBCASE("Armijo line search") {
        LBFGS<double> optimizer;
        optimizer.line_search_type = "armijo";
        optimizer.max_iterations = 300; // May need more iterations
        optimizer.gradient_tolerance = 1e-6;

        auto result = optimizer.optimize(func, x0);
        CHECK(result.converged);
    }
}

// =============================================================================
// Dynamic Vector Tests
// =============================================================================

TEST_CASE("LBFGS - Dynamic vectors") {
    using VecDyn = dp::mat::Vector<double, dp::mat::Dynamic>;
    QuadraticFunction<double, dp::mat::Dynamic> func;
    LBFGS<double> optimizer;

    optimizer.gradient_tolerance = 1e-8;

    SUBCASE("5D dynamic") {
        VecDyn x0(5);
        for (std::size_t i = 0; i < 5; ++i) {
            x0[i] = static_cast<double>(i + 1);
        }

        auto result = optimizer.optimize(func, x0);

        CHECK(result.converged);
        for (std::size_t i = 0; i < 5; ++i) {
            CHECK(std::abs(result.x[i]) < 1e-6);
        }
    }

    SUBCASE("20D dynamic") {
        VecDyn x0(20);
        for (std::size_t i = 0; i < 20; ++i) {
            x0[i] = static_cast<double>(i) - 10.0;
        }

        auto result = optimizer.optimize(func, x0);

        CHECK(result.converged);
        for (std::size_t i = 0; i < 20; ++i) {
            CHECK(std::abs(result.x[i]) < 1e-6);
        }
    }
}

TEST_CASE("LBFGS - Dynamic Rosenbrock") {
    using VecDyn = dp::mat::Vector<double, dp::mat::Dynamic>;
    RosenbrockND<double, dp::mat::Dynamic> func;
    LBFGS<double> optimizer;

    optimizer.max_iterations = 500;
    optimizer.gradient_tolerance = 1e-5;
    optimizer.history_size = 15;

    SUBCASE("4D Rosenbrock") {
        VecDyn x0(4);
        x0.fill(-1.0);

        auto result = optimizer.optimize(func, x0);

        CHECK(result.converged);
        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(std::abs(result.x[i] - 1.0) < 1e-3);
        }
    }
}

// =============================================================================
// Float Precision Tests
// =============================================================================

TEST_CASE("LBFGS - Float precision") {
    using Vec2 = dp::mat::Vector<float, 2>;
    QuadraticFunction<float, 2> func;
    LBFGS<float> optimizer;

    optimizer.gradient_tolerance = 1e-5f;

    Vec2 x0(dp::mat::Vector<float, 2>{1.0f, 1.0f});
    auto result = optimizer.optimize(func, x0);

    CHECK(result.converged);
    CHECK(std::abs(result.x[0]) < 1e-4f);
    CHECK(std::abs(result.x[1]) < 1e-4f);
}

// =============================================================================
// Sphere Function Tests (Integration)
// =============================================================================

TEST_CASE("LBFGS - Sphere function") {
    using Vec3 = dp::mat::Vector<double, 3>;
    Sphere<double, 3> sphere;
    LBFGS<double> optimizer;

    optimizer.gradient_tolerance = 1e-8;

    Vec3 x0(dp::mat::Vector<double, 3>{2.0, -1.0, 3.0});
    auto result = optimizer.optimize(sphere, x0);

    CHECK(result.converged);
    CHECK(result.iterations < 20);
    CHECK(std::abs(result.x[0]) < 1e-6);
    CHECK(std::abs(result.x[1]) < 1e-6);
    CHECK(std::abs(result.x[2]) < 1e-6);
    CHECK(result.final_cost < 1e-12);
}

// =============================================================================
// Convergence Criteria Tests
// =============================================================================

TEST_CASE("LBFGS - Convergence criteria") {
    using Vec2 = dp::mat::Vector<double, 2>;
    QuadraticFunction<double, 2> func;

    Vec2 x0(dp::mat::Vector<double, 2>{1.0, 1.0});

    SUBCASE("Gradient tolerance") {
        LBFGS<double> optimizer;
        optimizer.gradient_tolerance = 1e-4;  // Loose tolerance
        optimizer.function_tolerance = 1e-20; // Disable
        optimizer.step_tolerance = 1e-20;     // Disable

        auto result = optimizer.optimize(func, x0);

        CHECK(result.converged);
        CHECK(result.gradient_norm < 1e-4);
    }

    SUBCASE("Function tolerance") {
        LBFGS<double> optimizer;
        optimizer.gradient_tolerance = 1e-20; // Disable
        optimizer.function_tolerance = 1e-6;
        optimizer.step_tolerance = 1e-20; // Disable

        auto result = optimizer.optimize(func, x0);

        CHECK(result.converged);
    }

    SUBCASE("Step tolerance") {
        LBFGS<double> optimizer;
        optimizer.gradient_tolerance = 1e-20; // Disable
        optimizer.function_tolerance = 1e-20; // Disable
        optimizer.step_tolerance = 1e-6;

        auto result = optimizer.optimize(func, x0);

        CHECK(result.converged);
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_CASE("LBFGS - Edge cases") {
    using Vec2 = dp::mat::Vector<double, 2>;
    QuadraticFunction<double, 2> func;
    LBFGS<double> optimizer;

    SUBCASE("Max iterations reached") {
        optimizer.max_iterations = 2;
        optimizer.gradient_tolerance = 1e-20; // Very tight
        optimizer.function_tolerance = 1e-20; // Very tight
        optimizer.step_tolerance = 1e-20;     // Very tight

        Vec2 x0(dp::mat::Vector<double, 2>{100.0, 100.0});
        auto result = optimizer.optimize(func, x0);

        // L-BFGS converges very fast on quadratic, so it may converge in 2 iterations
        // Just check that it ran the expected number of iterations
        CHECK(result.iterations <= 2);
    }

    SUBCASE("Very small initial point") {
        optimizer.gradient_tolerance = 1e-10;

        Vec2 x0(dp::mat::Vector<double, 2>{1e-8, 1e-8});
        auto result = optimizer.optimize(func, x0);

        CHECK(result.converged);
    }

    SUBCASE("Large initial point") {
        optimizer.gradient_tolerance = 1e-6;
        optimizer.max_iterations = 100;

        Vec2 x0(dp::mat::Vector<double, 2>{1000.0, -1000.0});
        auto result = optimizer.optimize(func, x0);

        CHECK(result.converged);
        CHECK(std::abs(result.x[0]) < 1e-4);
        CHECK(std::abs(result.x[1]) < 1e-4);
    }
}

// =============================================================================
// Callback Tests
// =============================================================================

TEST_CASE("LBFGS - Callback functionality") {
    using Vec2 = dp::mat::Vector<double, 2>;
    QuadraticFunction<double, 2> func;
    LBFGS<double> optimizer;

    optimizer.gradient_tolerance = 1e-8;

    SUBCASE("Iteration counting callback") {
        struct CountingCallback {
            std::size_t *count_ptr;
            bool *begin_called_ptr;
            bool *end_called_ptr;

            void on_begin(const Vec2 &) { *begin_called_ptr = true; }

            bool on_iteration(const IterationInfo<double> &, const Vec2 &) {
                ++(*count_ptr);
                return false; // Don't stop
            }

            void on_end(const OptimizationResult<double, 2> &) { *end_called_ptr = true; }
        };

        std::size_t count = 0;
        bool begin_called = false;
        bool end_called = false;

        CountingCallback callback{&count, &begin_called, &end_called};
        Vec2 x0(dp::mat::Vector<double, 2>{5.0, 5.0}); // Farther from optimum
        auto result = optimizer.optimize(func, x0, callback);

        CHECK(begin_called);
        CHECK(end_called);
        // Callback is called at the start of each iteration, so count >= iterations
        // When we break mid-iteration, count may be iterations + 1
        CHECK(count >= result.iterations);
    }

    SUBCASE("Early stopping callback") {
        struct EarlyStopCallback {
            std::size_t *iteration_ptr;

            void on_begin(const Vec2 &) {}

            bool on_iteration(const IterationInfo<double> &info, const Vec2 &) {
                *iteration_ptr = info.iteration;
                return info.iteration >= 3; // Stop after 3 iterations
            }

            void on_end(const OptimizationResult<double, 2> &) {}
        };

        // Use Rosenbrock with a starting point far from optimum
        RosenbrockFunction rosenbrock;
        LBFGS<double> lbfgs_opt;
        lbfgs_opt.gradient_tolerance = 1e-20; // Very tight to prevent early convergence
        lbfgs_opt.function_tolerance = 1e-20;
        lbfgs_opt.step_tolerance = 1e-20;
        lbfgs_opt.max_iterations = 100;

        std::size_t last_iteration = 0;
        EarlyStopCallback callback{&last_iteration};
        // Start far from optimum to ensure we need multiple iterations
        Vec2 x0(dp::mat::Vector<double, 2>{-5.0, 5.0});
        auto result = lbfgs_opt.optimize(rosenbrock, x0, callback);

        // The callback should have stopped the optimizer
        CHECK(result.converged);
        CHECK(last_iteration >= 3); // Callback was called at least 4 times (0, 1, 2, 3)
        CHECK(result.termination_reason == termination::CALLBACK_STOP);
    }
}

// =============================================================================
// Comparison with Gradient Descent
// =============================================================================

TEST_CASE("LBFGS - Faster than gradient descent") {
    using Vec2 = dp::mat::Vector<double, 2>;
    RosenbrockFunction func;

    Vec2 x0(dp::mat::Vector<double, 2>{-1.0, 1.0});

    // L-BFGS should converge in far fewer iterations than gradient descent
    LBFGS<double> lbfgs;
    lbfgs.max_iterations = 200;
    lbfgs.gradient_tolerance = 1e-6;

    auto result = lbfgs.optimize(func, x0);

    CHECK(result.converged);
    // L-BFGS typically converges in 20-50 iterations on Rosenbrock
    // Gradient descent would need thousands
    CHECK(result.iterations < 100);
}
