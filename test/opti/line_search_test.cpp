#include <doctest/doctest.h>

#include <datapod/matrix/vector.hpp>
#include <optinum/opti/line_search/line_search.hpp>
#include <optinum/opti/problem/sphere.hpp>

#include <cmath>

using namespace optinum;
using namespace optinum::opti;
namespace dp = datapod;

// Helper to negate a vector (dp::mat::vector doesn't have unary minus)
template <typename T, std::size_t N> dp::mat::vector<T, N> negate(const dp::mat::vector<T, N> &v) {
    dp::mat::vector<T, N> result;
    if constexpr (N == dp::mat::Dynamic) {
        result.resize(v.size());
    }
    for (std::size_t i = 0; i < v.size(); ++i) {
        result[i] = -v[i];
    }
    return result;
}

// =============================================================================
// Test Functions
// =============================================================================

/**
 * @brief Simple quadratic function for testing: f(x) = 0.5 * x^T * x
 *
 * Minimum at origin, gradient = x
 */
template <typename T, std::size_t N> struct QuadraticFunction {
    using vector_type = dp::mat::vector<T, N>;

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
    using vector_type = dp::mat::vector<double, 2>;

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
 * @brief Shifted quadratic: f(x) = 0.5 * (x - shift)^T * (x - shift)
 */
template <typename T, std::size_t N> struct ShiftedQuadratic {
    using vector_type = dp::mat::vector<T, N>;
    vector_type shift;

    ShiftedQuadratic() {
        if constexpr (N != dp::mat::Dynamic) {
            for (std::size_t i = 0; i < N; ++i) {
                shift[i] = T(i + 1);
            }
        }
    }

    T evaluate(const vector_type &x) const {
        T sum = T(0);
        for (std::size_t i = 0; i < x.size(); ++i) {
            T diff = x[i] - shift[i];
            sum += diff * diff;
        }
        return sum / T(2);
    }

    void gradient(const vector_type &x, vector_type &g) const {
        for (std::size_t i = 0; i < x.size(); ++i) {
            g[i] = x[i] - shift[i];
        }
    }

    T evaluate_with_gradient(const vector_type &x, vector_type &g) const {
        gradient(x, g);
        return evaluate(x);
    }
};

// =============================================================================
// Armijo Line Search Tests
// =============================================================================

TEST_CASE("ArmijoLineSearch - Basic functionality") {
    using Vec2 = dp::mat::vector<double, 2>;
    QuadraticFunction<double, 2> func;
    ArmijoLineSearch<double> ls;

    SUBCASE("Descent direction from (1, 1)") {
        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        Vec2 grad;
        double f0 = func.evaluate_with_gradient(x, grad);

        // Direction = -gradient (steepest descent)
        Vec2 direction = negate(grad);

        auto result = ls.search(func, x, direction, f0, grad);

        CHECK(result.success);
        CHECK(result.alpha > 0.0);
        CHECK(result.function_value < f0);
        CHECK(result.function_evals > 0);
    }

    SUBCASE("Descent direction from (5, -3)") {
        Vec2 x(dp::mat::vector<double, 2>{5.0, -3.0});
        Vec2 grad;
        double f0 = func.evaluate_with_gradient(x, grad);
        Vec2 direction = negate(grad);

        auto result = ls.search(func, x, direction, f0, grad);

        CHECK(result.success);
        CHECK(result.alpha > 0.0);
        CHECK(result.function_value < f0);
    }

    SUBCASE("Non-descent direction should fail") {
        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        Vec2 grad;
        double f0 = func.evaluate_with_gradient(x, grad);

        // Direction = +gradient (ascent direction)
        Vec2 direction = grad;

        auto result = ls.search(func, x, direction, f0, grad);

        CHECK_FALSE(result.success);
        CHECK(result.termination_reason == "Not a descent direction");
    }

    SUBCASE("At minimum, any direction is non-descent") {
        Vec2 x(dp::mat::vector<double, 2>{0.0, 0.0});
        Vec2 grad;
        double f0 = func.evaluate_with_gradient(x, grad);

        // Gradient is zero at minimum
        Vec2 direction(dp::mat::vector<double, 2>{1.0, 0.0});

        // grad^T * direction = 0, so not a descent direction
        auto result = ls.search(func, x, direction, f0, grad);

        CHECK_FALSE(result.success);
    }
}

TEST_CASE("ArmijoLineSearch - Parameter sensitivity") {
    using Vec2 = dp::mat::vector<double, 2>;
    QuadraticFunction<double, 2> func;

    Vec2 x(dp::mat::vector<double, 2>{2.0, 2.0});
    Vec2 grad;
    double f0 = func.evaluate_with_gradient(x, grad);
    Vec2 direction = negate(grad);

    SUBCASE("Default parameters") {
        ArmijoLineSearch<double> ls;
        auto result = ls.search(func, x, direction, f0, grad);
        CHECK(result.success);
    }

    SUBCASE("Strict c1 (larger value)") {
        ArmijoLineSearch<double> ls;
        ls.c1 = 0.1; // More strict
        auto result = ls.search(func, x, direction, f0, grad);
        CHECK(result.success);
    }

    SUBCASE("Aggressive backtracking (small rho)") {
        ArmijoLineSearch<double> ls;
        ls.rho = 0.1; // More aggressive reduction
        auto result = ls.search(func, x, direction, f0, grad);
        CHECK(result.success);
    }

    SUBCASE("Conservative backtracking (large rho)") {
        ArmijoLineSearch<double> ls;
        ls.rho = 0.9; // Less aggressive reduction
        auto result = ls.search(func, x, direction, f0, grad);
        CHECK(result.success);
    }

    SUBCASE("Large initial step") {
        ArmijoLineSearch<double> ls;
        ls.alpha_init = 10.0;
        auto result = ls.search(func, x, direction, f0, grad);
        CHECK(result.success);
    }

    SUBCASE("Small initial step") {
        ArmijoLineSearch<double> ls;
        ls.alpha_init = 0.01;
        auto result = ls.search(func, x, direction, f0, grad);
        CHECK(result.success);
        // Should accept immediately or after few iterations
        CHECK(result.function_evals <= 5);
    }
}

TEST_CASE("ArmijoLineSearch - Higher dimensions") {
    SUBCASE("3D quadratic") {
        using Vec3 = dp::mat::vector<double, 3>;
        QuadraticFunction<double, 3> func;
        ArmijoLineSearch<double> ls;

        Vec3 x(dp::mat::vector<double, 3>{1.0, 2.0, 3.0});
        Vec3 grad;
        double f0 = func.evaluate_with_gradient(x, grad);
        Vec3 direction = negate(grad);

        auto result = ls.search(func, x, direction, f0, grad);

        CHECK(result.success);
        CHECK(result.function_value < f0);
    }

    SUBCASE("10D quadratic") {
        using Vec10 = dp::mat::vector<double, 10>;
        QuadraticFunction<double, 10> func;
        ArmijoLineSearch<double> ls;

        Vec10 x;
        for (std::size_t i = 0; i < 10; ++i) {
            x[i] = static_cast<double>(i) - 5.0;
        }

        Vec10 grad;
        double f0 = func.evaluate_with_gradient(x, grad);
        Vec10 direction = negate(grad);

        auto result = ls.search(func, x, direction, f0, grad);

        CHECK(result.success);
        CHECK(result.function_value < f0);
    }
}

TEST_CASE("ArmijoLineSearch - Float precision") {
    using Vec2 = dp::mat::vector<float, 2>;
    QuadraticFunction<float, 2> func;
    ArmijoLineSearch<float> ls;

    Vec2 x(dp::mat::vector<float, 2>{1.0f, 1.0f});
    Vec2 grad;
    float f0 = func.evaluate_with_gradient(x, grad);
    Vec2 direction = negate(grad);

    auto result = ls.search(func, x, direction, f0, grad);

    CHECK(result.success);
    CHECK(result.function_value < f0);
}

TEST_CASE("ArmijoLineSearch - search_with_gradient") {
    using Vec2 = dp::mat::vector<double, 2>;
    QuadraticFunction<double, 2> func;
    ArmijoLineSearch<double> ls;

    Vec2 x(dp::mat::vector<double, 2>{2.0, 2.0});
    Vec2 grad;
    double f0 = func.evaluate_with_gradient(x, grad);
    Vec2 direction = negate(grad);
    Vec2 grad_new;

    auto result = ls.search_with_gradient(func, x, direction, f0, grad, grad_new);

    CHECK(result.success);
    CHECK(result.function_value < f0);
    // grad_new should be computed at the new point
    CHECK(result.gradient_evals > 0);
}

// =============================================================================
// Wolfe Line Search Tests
// =============================================================================

TEST_CASE("WolfeLineSearch - Basic functionality") {
    using Vec2 = dp::mat::vector<double, 2>;
    QuadraticFunction<double, 2> func;
    WolfeLineSearch<double> ls;

    SUBCASE("Descent direction from (1, 1)") {
        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        Vec2 grad, grad_new;
        double f0 = func.evaluate_with_gradient(x, grad);
        Vec2 direction = negate(grad);

        auto result = ls.search(func, x, direction, f0, grad, grad_new);

        CHECK(result.success);
        CHECK(result.alpha > 0.0);
        CHECK(result.function_value < f0);
    }

    SUBCASE("Descent direction from (5, -3)") {
        Vec2 x(dp::mat::vector<double, 2>{5.0, -3.0});
        Vec2 grad, grad_new;
        double f0 = func.evaluate_with_gradient(x, grad);
        Vec2 direction = negate(grad);

        auto result = ls.search(func, x, direction, f0, grad, grad_new);

        CHECK(result.success);
        CHECK(result.function_value < f0);
    }

    SUBCASE("Non-descent direction should fail") {
        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        Vec2 grad, grad_new;
        double f0 = func.evaluate_with_gradient(x, grad);
        Vec2 direction = grad; // Ascent direction

        auto result = ls.search(func, x, direction, f0, grad, grad_new);

        CHECK_FALSE(result.success);
    }
}

TEST_CASE("WolfeLineSearch - Strong Wolfe conditions") {
    using Vec2 = dp::mat::vector<double, 2>;
    QuadraticFunction<double, 2> func;
    WolfeLineSearch<double> ls;

    Vec2 x(dp::mat::vector<double, 2>{3.0, 4.0});
    Vec2 grad, grad_new;
    double f0 = func.evaluate_with_gradient(x, grad);
    Vec2 direction = negate(grad);

    auto result = ls.search(func, x, direction, f0, grad, grad_new);

    CHECK(result.success);

    // Verify Armijo condition: f(x + alpha*d) <= f(x) + c1*alpha*grad^T*d
    double dphi0 = simd::view(grad).dot(simd::view(direction));
    CHECK(result.function_value <= f0 + ls.c1 * result.alpha * dphi0);

    // Verify curvature condition: |grad_new^T * d| <= c2 * |grad^T * d|
    double dphi_alpha = simd::view(grad_new).dot(simd::view(direction));
    CHECK(std::abs(dphi_alpha) <= ls.c2 * std::abs(dphi0));
}

TEST_CASE("WolfeLineSearch - Parameter variations") {
    using Vec2 = dp::mat::vector<double, 2>;
    QuadraticFunction<double, 2> func;

    Vec2 x(dp::mat::vector<double, 2>{2.0, 2.0});
    Vec2 grad, grad_new;
    double f0 = func.evaluate_with_gradient(x, grad);
    Vec2 direction = negate(grad);

    SUBCASE("Quasi-Newton parameters (c2 = 0.9)") {
        WolfeLineSearch<double> ls(1e-4, 0.9);
        auto result = ls.search(func, x, direction, f0, grad, grad_new);
        CHECK(result.success);
    }

    SUBCASE("Conjugate gradient parameters (c2 = 0.1)") {
        WolfeLineSearch<double> ls(1e-4, 0.1);
        auto result = ls.search(func, x, direction, f0, grad, grad_new);
        CHECK(result.success);
    }
}

TEST_CASE("WolfeLineSearch - Higher dimensions") {
    using Vec5 = dp::mat::vector<double, 5>;
    QuadraticFunction<double, 5> func;
    WolfeLineSearch<double> ls;

    Vec5 x;
    for (std::size_t i = 0; i < 5; ++i) {
        x[i] = static_cast<double>(i + 1);
    }

    Vec5 grad, grad_new;
    double f0 = func.evaluate_with_gradient(x, grad);
    Vec5 direction = negate(grad);

    auto result = ls.search(func, x, direction, f0, grad, grad_new);

    CHECK(result.success);
    CHECK(result.function_value < f0);
}

// =============================================================================
// Weak Wolfe Line Search Tests
// =============================================================================

TEST_CASE("WeakWolfeLineSearch - Basic functionality") {
    using Vec2 = dp::mat::vector<double, 2>;
    QuadraticFunction<double, 2> func;
    WeakWolfeLineSearch<double> ls;

    Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
    Vec2 grad, grad_new;
    double f0 = func.evaluate_with_gradient(x, grad);
    Vec2 direction = negate(grad);

    auto result = ls.search(func, x, direction, f0, grad, grad_new);

    CHECK(result.success);
    CHECK(result.function_value < f0);
}

TEST_CASE("WeakWolfeLineSearch - Weak curvature condition") {
    using Vec2 = dp::mat::vector<double, 2>;
    QuadraticFunction<double, 2> func;
    WeakWolfeLineSearch<double> ls;

    Vec2 x(dp::mat::vector<double, 2>{3.0, 4.0});
    Vec2 grad, grad_new;
    double f0 = func.evaluate_with_gradient(x, grad);
    Vec2 direction = negate(grad);

    auto result = ls.search(func, x, direction, f0, grad, grad_new);

    CHECK(result.success);

    // Verify weak curvature: grad_new^T * d >= c2 * grad^T * d
    double dphi0 = simd::view(grad).dot(simd::view(direction));
    double dphi_alpha = simd::view(grad_new).dot(simd::view(direction));
    CHECK(dphi_alpha >= ls.c2 * dphi0);
}

// =============================================================================
// Goldstein Line Search Tests
// =============================================================================

TEST_CASE("GoldsteinLineSearch - Basic functionality") {
    using Vec2 = dp::mat::vector<double, 2>;
    QuadraticFunction<double, 2> func;
    GoldsteinLineSearch<double> ls;

    Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
    Vec2 grad;
    double f0 = func.evaluate_with_gradient(x, grad);
    Vec2 direction = negate(grad);

    auto result = ls.search(func, x, direction, f0, grad);

    CHECK(result.success);
    CHECK(result.function_value < f0);
}

TEST_CASE("GoldsteinLineSearch - Goldstein conditions") {
    using Vec2 = dp::mat::vector<double, 2>;
    QuadraticFunction<double, 2> func;
    GoldsteinLineSearch<double> ls;
    ls.c = 0.25;

    Vec2 x(dp::mat::vector<double, 2>{3.0, 4.0});
    Vec2 grad;
    double f0 = func.evaluate_with_gradient(x, grad);
    Vec2 direction = negate(grad);

    auto result = ls.search(func, x, direction, f0, grad);

    CHECK(result.success);

    // Verify Goldstein conditions:
    // f(x) + (1-c)*alpha*dphi0 <= f(x+alpha*d) <= f(x) + c*alpha*dphi0
    double dphi0 = simd::view(grad).dot(simd::view(direction));
    double lower_bound = f0 + (1.0 - ls.c) * result.alpha * dphi0;
    double upper_bound = f0 + ls.c * result.alpha * dphi0;

    CHECK(result.function_value >= lower_bound);
    CHECK(result.function_value <= upper_bound);
}

// =============================================================================
// Rosenbrock Function Tests (Challenging)
// =============================================================================

TEST_CASE("Line search on Rosenbrock function") {
    using Vec2 = dp::mat::vector<double, 2>;
    RosenbrockFunction func;

    Vec2 x(dp::mat::vector<double, 2>{-1.0, 1.0});
    Vec2 grad, grad_new;
    double f0 = func.evaluate_with_gradient(x, grad);
    Vec2 direction = negate(grad);

    SUBCASE("Armijo line search") {
        ArmijoLineSearch<double> ls;
        auto result = ls.search(func, x, direction, f0, grad);

        CHECK(result.success);
        CHECK(result.function_value < f0);
    }

    SUBCASE("Wolfe line search") {
        WolfeLineSearch<double> ls;
        auto result = ls.search(func, x, direction, f0, grad, grad_new);

        CHECK(result.success);
        CHECK(result.function_value < f0);
    }

    SUBCASE("Goldstein line search") {
        GoldsteinLineSearch<double> ls;
        auto result = ls.search(func, x, direction, f0, grad);

        CHECK(result.success);
        CHECK(result.function_value < f0);
    }
}

// =============================================================================
// Dynamic Vector Tests
// =============================================================================

TEST_CASE("ArmijoLineSearch - Dynamic vectors") {
    using VecDyn = dp::mat::vector<double, dp::mat::Dynamic>;
    QuadraticFunction<double, dp::mat::Dynamic> func;
    ArmijoLineSearch<double> ls;

    VecDyn x(5);
    for (std::size_t i = 0; i < 5; ++i) {
        x[i] = static_cast<double>(i + 1);
    }

    VecDyn grad(5);
    double f0 = func.evaluate_with_gradient(x, grad);

    VecDyn direction(5);
    for (std::size_t i = 0; i < 5; ++i) {
        direction[i] = -grad[i];
    }

    auto result = ls.search(func, x, direction, f0, grad);

    CHECK(result.success);
    CHECK(result.function_value < f0);
}

TEST_CASE("WolfeLineSearch - Dynamic vectors") {
    using VecDyn = dp::mat::vector<double, dp::mat::Dynamic>;
    QuadraticFunction<double, dp::mat::Dynamic> func;
    WolfeLineSearch<double> ls;

    VecDyn x(5);
    for (std::size_t i = 0; i < 5; ++i) {
        x[i] = static_cast<double>(i + 1);
    }

    VecDyn grad(5), grad_new(5);
    double f0 = func.evaluate_with_gradient(x, grad);

    VecDyn direction(5);
    for (std::size_t i = 0; i < 5; ++i) {
        direction[i] = -grad[i];
    }

    auto result = ls.search(func, x, direction, f0, grad, grad_new);

    CHECK(result.success);
    CHECK(result.function_value < f0);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_CASE("Line search edge cases") {
    using Vec2 = dp::mat::vector<double, 2>;
    QuadraticFunction<double, 2> func;

    SUBCASE("Very small gradient") {
        ArmijoLineSearch<double> ls;

        Vec2 x(dp::mat::vector<double, 2>{1e-10, 1e-10});
        Vec2 grad;
        double f0 = func.evaluate_with_gradient(x, grad);
        Vec2 direction = negate(grad);

        auto result = ls.search(func, x, direction, f0, grad);

        // Should still work, though gradient is tiny
        // May fail if gradient is essentially zero
        if (simd::view(grad).norm() > 1e-15) {
            CHECK(result.success);
        }
    }

    SUBCASE("Large starting point") {
        ArmijoLineSearch<double> ls;
        ls.alpha_init = 0.01; // Start with smaller step for large values

        Vec2 x(dp::mat::vector<double, 2>{1000.0, 1000.0});
        Vec2 grad;
        double f0 = func.evaluate_with_gradient(x, grad);
        Vec2 direction = negate(grad);

        auto result = ls.search(func, x, direction, f0, grad);

        CHECK(result.success);
        CHECK(result.function_value < f0);
    }

    SUBCASE("Zero direction") {
        ArmijoLineSearch<double> ls;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        Vec2 grad;
        double f0 = func.evaluate_with_gradient(x, grad);
        Vec2 direction(dp::mat::vector<double, 2>{0.0, 0.0});

        auto result = ls.search(func, x, direction, f0, grad);

        // Zero direction has zero directional derivative, not a descent direction
        CHECK_FALSE(result.success);
    }
}

// =============================================================================
// Comparison Tests
// =============================================================================

TEST_CASE("Compare line search methods") {
    using Vec2 = dp::mat::vector<double, 2>;
    QuadraticFunction<double, 2> func;

    Vec2 x(dp::mat::vector<double, 2>{3.0, 4.0});
    Vec2 grad, grad_new;
    double f0 = func.evaluate_with_gradient(x, grad);
    Vec2 direction = negate(grad);

    ArmijoLineSearch<double> armijo;
    WolfeLineSearch<double> wolfe;
    GoldsteinLineSearch<double> goldstein;

    auto armijo_result = armijo.search(func, x, direction, f0, grad);
    auto wolfe_result = wolfe.search(func, x, direction, f0, grad, grad_new);
    auto goldstein_result = goldstein.search(func, x, direction, f0, grad);

    // All should succeed
    CHECK(armijo_result.success);
    CHECK(wolfe_result.success);
    CHECK(goldstein_result.success);

    // All should decrease function value
    CHECK(armijo_result.function_value < f0);
    CHECK(wolfe_result.function_value < f0);
    CHECK(goldstein_result.function_value < f0);

    // Armijo typically uses fewer function evaluations (no gradient needed)
    CHECK(armijo_result.gradient_evals == 0);
    CHECK(wolfe_result.gradient_evals > 0);
}

// =============================================================================
// Integration with Sphere function
// =============================================================================

TEST_CASE("Line search with Sphere function") {
    using Vec3 = dp::mat::vector<double, 3>;
    Sphere<double, 3> sphere;
    ArmijoLineSearch<double> ls;

    Vec3 x(dp::mat::vector<double, 3>{2.0, -1.0, 3.0});
    Vec3 grad;
    double f0 = sphere.evaluate_with_gradient(x, grad);
    Vec3 direction = negate(grad);

    auto result = ls.search(sphere, x, direction, f0, grad);

    CHECK(result.success);
    CHECK(result.function_value < f0);

    // For quadratic, optimal step is 1.0 (full Newton step)
    // Armijo should accept alpha close to 1.0
    CHECK(result.alpha >= 0.5);
}
