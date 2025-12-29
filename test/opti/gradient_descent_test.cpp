#include <doctest/doctest.h>

#include <datapod/matrix/vector.hpp>
#include <optinum/opti/opti.hpp>

using namespace optinum;
using namespace optinum::opti;
namespace dp = datapod;

TEST_CASE("GradientDescent - Sphere function 2D") {
    using Vec2 = dp::mat::vector<double, 2>;

    Sphere<double, 2> sphere;
    GradientDescent<> gd;

    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    SUBCASE("Converge from (1, 1)") {
        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.iterations < 1000);
        CHECK(result.final_cost < 1e-5);
        CHECK(std::abs(x[0]) < 1e-3);
        CHECK(std::abs(x[1]) < 1e-3);
    }

    SUBCASE("Converge from (5, -3)") {
        Vec2 x(dp::mat::vector<double, 2>{5.0, -3.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-5);
        CHECK(std::abs(x[0]) < 2e-3);
        CHECK(std::abs(x[1]) < 2e-3);
    }

    SUBCASE("Already at minimum") {
        Vec2 x(dp::mat::vector<double, 2>{0.0, 0.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.iterations < 10); // Should converge very quickly
        CHECK(result.final_cost < 1e-10);
    }
}

TEST_CASE("GradientDescent - Sphere function 3D") {
    using Vec3 = dp::mat::vector<double, 3>;

    Sphere<double, 3> sphere;
    GradientDescent<> gd;

    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    Vec3 x(dp::mat::vector<double, 3>{2.0, -1.0, 3.0});
    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-5);
    CHECK(std::abs(x[0]) < 1e-3);
    CHECK(std::abs(x[1]) < 1e-3);
    CHECK(std::abs(x[2]) < 1e-3);
}

TEST_CASE("GradientDescent - Sphere function 10D") {
    using Vec10 = dp::mat::vector<double, 10>;

    Sphere<double, 10> sphere;
    GradientDescent<> gd;

    gd.step_size = 0.05; // Smaller step for higher dimensions
    gd.max_iterations = 2000;
    gd.tolerance = 1e-6;

    Vec10 x;
    for (std::size_t i = 0; i < 10; ++i) {
        x[i] = static_cast<double>(i) - 5.0; // Initialize to [-5, -4, ..., 4]
    }

    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-5);

    for (std::size_t i = 0; i < 10; ++i) {
        CHECK(std::abs(x[i]) < 2e-3);
    }
}

TEST_CASE("GradientDescent - Step size effects") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    SUBCASE("Small step size (slow convergence)") {
        GradientDescent<> gd;
        gd.step_size = 0.01; // Very small
        gd.max_iterations = 5000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.iterations > 100); // Should take many iterations
    }

    SUBCASE("Large step size (fast convergence)") {
        GradientDescent<> gd;
        gd.step_size = 0.2; // Large but stable
        gd.max_iterations = 1000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.iterations < 100); // Should converge faster
    }
}

TEST_CASE("GradientDescent - Max iterations") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    GradientDescent<> gd;
    gd.step_size = 0.001;   // Very small step
    gd.max_iterations = 10; // Very few iterations
    gd.tolerance = 1e-6;

    Vec2 x;
    x[0] = 10.0;
    x[1] = 10.0; // Far from minimum

    auto result = gd.optimize(sphere, x);

    CHECK_FALSE(result.converged);
    CHECK(result.iterations == 10);
    CHECK(result.termination_reason == std::string(termination::MAX_ITERATIONS));
}

TEST_CASE("GradientDescent - Callback system") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    GradientDescent<> gd;
    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    SUBCASE("No callback") {
        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x, NoCallback{});
        CHECK(result.converged);
    }

    SUBCASE("Early stopping callback") {
        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        EarlyStoppingCallback<double> callback(0.5); // Stop at objective < 0.5

        auto result = gd.optimize(sphere, x, callback);

        CHECK(result.converged);
        CHECK(result.final_cost < 0.5);
        CHECK(result.termination_reason == std::string(termination::CALLBACK_STOP));
    }
}

TEST_CASE("GradientDescent - Float precision") {
    using Vec2 = dp::mat::vector<float, 2>;

    Sphere<float, 2> sphere;
    GradientDescent<> gd;

    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-5f;

    Vec2 x(dp::mat::vector<float, 2>{1.0f, 1.0f});
    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4f);
    CHECK(std::abs(x[0]) < 1e-2f);
    CHECK(std::abs(x[1]) < 1e-2f);
}

TEST_CASE("GradientDescent - Result structure") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    GradientDescent<> gd;
    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
    auto result = gd.optimize(sphere, x);

    // Check all result fields are populated
    CHECK(result.iterations > 0);
    CHECK(result.iterations < 1000);
    CHECK(result.converged == true);
    CHECK_FALSE(result.termination_reason.empty());
    CHECK(result.final_cost >= 0.0);

    // Check solution is stored in result
    CHECK(result.x[0] == doctest::Approx(x[0]).epsilon(1e-10));
    CHECK(result.x[1] == doctest::Approx(x[1]).epsilon(1e-10));
}

TEST_CASE("GradientDescent - Custom function") {
    // Custom quadratic function: f(x, y) = (x - 2)^2 + (y + 3)^2
    // Minimum at (2, -3) with value 0
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

    CustomQuadratic func;
    GradientDescent<> gd;

    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    dp::mat::vector<double, 2> x(dp::mat::vector<double, 2>{0.0, 0.0});
    auto result = gd.optimize(func, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-5);
    CHECK(std::abs(x[0] - 2.0) < 1e-3);
    CHECK(std::abs(x[1] - (-3.0)) < 1e-3);
}

TEST_CASE("GradientDescent - Policy reset") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    GradientDescent<> gd;
    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    SUBCASE("With policy reset (default)") {
        gd.reset_policy = true;

        Vec2 x1(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result1 = gd.optimize(sphere, x1);
        CHECK(result1.converged);

        Vec2 x2(dp::mat::vector<double, 2>{2.0, 2.0});
        auto result2 = gd.optimize(sphere, x2);
        CHECK(result2.converged);

        // Both should converge similarly
        CHECK(std::abs(static_cast<int>(result1.iterations) - static_cast<int>(result2.iterations)) < 10);
    }

    SUBCASE("Without policy reset") {
        gd.reset_policy = false;

        Vec2 x1(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result1 = gd.optimize(sphere, x1);
        CHECK(result1.converged);

        Vec2 x2(dp::mat::vector<double, 2>{2.0, 2.0});
        auto result2 = gd.optimize(sphere, x2);
        CHECK(result2.converged);
    }
}
