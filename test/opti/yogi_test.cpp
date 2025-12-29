#include <doctest/doctest.h>

#include <datapod/matrix/vector.hpp>
#include <optinum/opti/opti.hpp>

using namespace optinum;
using namespace optinum::opti;
namespace dp = datapod;

TEST_CASE("YogiUpdate - Sphere function 2D") {
    using Vec2 = dp::mat::vector<double, 2>;

    Sphere<double, 2> sphere;
    GradientDescent<YogiUpdate> gd;

    gd.step_size = 0.1;
    gd.max_iterations = 2000;
    gd.tolerance = 1e-6;

    SUBCASE("Converge from (1, 1)") {
        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.iterations < 2000);
        CHECK(result.final_cost < 1e-3);
        CHECK(std::abs(x[0]) < 1e-1);
        CHECK(std::abs(x[1]) < 1e-1);
    }

    SUBCASE("Converge from (5, -3)") {
        Vec2 x(dp::mat::vector<double, 2>{5.0, -3.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
        CHECK(std::abs(x[0]) < 1e-2);
        CHECK(std::abs(x[1]) < 1e-2);
    }

    SUBCASE("Already at minimum") {
        Vec2 x(dp::mat::vector<double, 2>{0.0, 0.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.iterations < 10);
        CHECK(result.final_cost < 1e-10);
    }
}

TEST_CASE("YogiUpdate - Sphere function 3D") {
    using Vec3 = dp::mat::vector<double, 3>;

    Sphere<double, 3> sphere;
    GradientDescent<YogiUpdate> gd;

    gd.step_size = 0.1;
    gd.max_iterations = 3000;
    gd.tolerance = 1e-6;

    Vec3 x(dp::mat::vector<double, 3>{2.0, -1.0, 3.0});
    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-3);
    CHECK(std::abs(x[0]) < 5e-2);
    CHECK(std::abs(x[1]) < 5e-2);
    CHECK(std::abs(x[2]) < 5e-2);
}

TEST_CASE("YogiUpdate - Sphere function 10D") {
    using Vec10 = dp::mat::vector<double, 10>;

    Sphere<double, 10> sphere;
    GradientDescent<YogiUpdate> gd;

    gd.step_size = 0.1;
    gd.max_iterations = 3000;
    gd.tolerance = 1e-6;

    Vec10 x;
    for (std::size_t i = 0; i < 10; ++i) {
        x[i] = static_cast<double>(i) - 5.0;
    }

    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4);

    for (std::size_t i = 0; i < 10; ++i) {
        CHECK(std::abs(x[i]) < 1e-2);
    }
}

TEST_CASE("YogiUpdate - Different beta values") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    SUBCASE("Default betas (0.9, 0.999)") {
        GradientDescent<YogiUpdate> gd;
        gd.step_size = 0.1;
        gd.max_iterations = 2000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-3);
    }

    SUBCASE("Lower beta1 (0.8)") {
        GradientDescent<YogiUpdate> gd;
        gd.get_update_policy() = YogiUpdate(0.8, 0.999, 1e-8);
        gd.step_size = 0.1;
        gd.max_iterations = 2000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-3);
    }

    SUBCASE("Lower beta2 (0.99)") {
        GradientDescent<YogiUpdate> gd;
        gd.get_update_policy() = YogiUpdate(0.9, 0.99, 1e-8);
        gd.step_size = 0.1;
        gd.max_iterations = 2000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-3);
    }
}

TEST_CASE("YogiUpdate - Float precision") {
    using Vec2 = dp::mat::vector<float, 2>;

    Sphere<float, 2> sphere;
    GradientDescent<YogiUpdate> gd;

    gd.step_size = 0.1;
    gd.max_iterations = 2000;
    gd.tolerance = 1e-5f;

    Vec2 x(dp::mat::vector<float, 2>{1.0f, 1.0f});
    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-3f);
    CHECK(std::abs(x[0]) < 1e-1f);
    CHECK(std::abs(x[1]) < 1e-1f);
}

TEST_CASE("YogiUpdate - Custom quadratic function") {
    // Custom quadratic: f(x, y) = (x - 2)^2 + (y + 3)^2
    // Minimum at (2, -3)
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
    GradientDescent<YogiUpdate> gd;

    gd.step_size = 0.1;
    gd.max_iterations = 2000;
    gd.tolerance = 1e-6;

    dp::mat::vector<double, 2> x(dp::mat::vector<double, 2>{0.0, 0.0});
    auto result = gd.optimize(func, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4);
    CHECK(std::abs(x[0] - 2.0) < 1e-2);
    CHECK(std::abs(x[1] - (-3.0)) < 1e-2);
}

TEST_CASE("YogiUpdate - Reset behavior") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    GradientDescent<YogiUpdate> gd;
    gd.step_size = 0.1;
    gd.max_iterations = 2000;
    gd.tolerance = 1e-6;
    gd.reset_policy = true;

    // First optimization
    Vec2 x1(dp::mat::vector<double, 2>{1.0, 1.0});
    auto result1 = gd.optimize(sphere, x1);
    CHECK(result1.converged);

    // Second optimization (should behave the same due to reset)
    Vec2 x2(dp::mat::vector<double, 2>{1.0, 1.0});
    auto result2 = gd.optimize(sphere, x2);
    CHECK(result2.converged);

    // Iterations should be similar (within tolerance)
    CHECK(std::abs(static_cast<int>(result1.iterations) - static_cast<int>(result2.iterations)) < 10);
}

TEST_CASE("YogiUpdate - Comparison with Adam") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    // Yogi
    GradientDescent<YogiUpdate> yogi_gd;
    yogi_gd.step_size = 0.1;
    yogi_gd.max_iterations = 2000;
    yogi_gd.tolerance = 1e-6;

    Vec2 x1(dp::mat::vector<double, 2>{5.0, 5.0});
    auto yogi_result = yogi_gd.optimize(sphere, x1);

    // Adam
    GradientDescent<AdamUpdate> adam_gd;
    adam_gd.step_size = 0.1;
    adam_gd.max_iterations = 2000;
    adam_gd.tolerance = 1e-6;

    Vec2 x2(dp::mat::vector<double, 2>{5.0, 5.0});
    auto adam_result = adam_gd.optimize(sphere, x2);

    // Both should converge
    CHECK(yogi_result.converged);
    CHECK(adam_result.converged);
}

TEST_CASE("YogiUpdate - Robustness to initial conditions") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    GradientDescent<YogiUpdate> gd;
    gd.step_size = 0.1;
    gd.max_iterations = 3000;
    gd.tolerance = 1e-6;

    SUBCASE("Far from minimum") {
        Vec2 x(dp::mat::vector<double, 2>{10.0, -10.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
    }

    SUBCASE("Very close to minimum") {
        Vec2 x(dp::mat::vector<double, 2>{0.01, -0.01});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
    }
}

TEST_CASE("YogiUpdate - Additive update behavior") {
    // Test that Yogi's additive update for second moment works correctly
    // The key difference from Adam is that v_t uses sign(g² - v) * g² instead of EMA
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    GradientDescent<YogiUpdate> gd;
    gd.step_size = 0.1;
    gd.max_iterations = 5000;
    gd.tolerance = 1e-6;

    // Test with varying gradient magnitudes
    SUBCASE("Large initial gradients") {
        Vec2 x(dp::mat::vector<double, 2>{100.0, -100.0});
        auto result = gd.optimize(sphere, x);

        // Yogi should handle large gradients well due to additive update
        CHECK(result.converged);
        CHECK(result.final_cost < 1e-2);
    }

    SUBCASE("Small initial gradients") {
        Vec2 x(dp::mat::vector<double, 2>{0.1, -0.1});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-5);
    }
}
