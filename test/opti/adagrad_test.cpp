#include <doctest/doctest.h>

#include <datapod/matrix/vector.hpp>
#include <optinum/opti/opti.hpp>

using namespace optinum;
using namespace optinum::opti;
namespace dp = datapod;

TEST_CASE("AdaGradUpdate - Sphere function 2D") {
    using Vec2 = dp::mat::vector<double, 2>;

    Sphere<double, 2> sphere;
    GradientDescent<AdaGradUpdate> gd;

    gd.step_size = 0.5; // AdaGrad can use larger initial learning rate
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    SUBCASE("Converge from (1, 1)") {
        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.iterations < 1000);
        CHECK(result.final_cost < 1e-4);
        CHECK(std::abs(x[0]) < 1e-2);
        CHECK(std::abs(x[1]) < 1e-2);
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

TEST_CASE("AdaGradUpdate - Sphere function 3D") {
    using Vec3 = dp::mat::vector<double, 3>;

    Sphere<double, 3> sphere;
    GradientDescent<AdaGradUpdate> gd;

    gd.step_size = 0.5;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    Vec3 x(dp::mat::vector<double, 3>{2.0, -1.0, 3.0});
    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4);
    CHECK(std::abs(x[0]) < 1e-2);
    CHECK(std::abs(x[1]) < 1e-2);
    CHECK(std::abs(x[2]) < 1e-2);
}

TEST_CASE("AdaGradUpdate - Sphere function 10D") {
    using Vec10 = dp::mat::vector<double, 10>;

    Sphere<double, 10> sphere;
    GradientDescent<AdaGradUpdate> gd;

    gd.step_size = 0.5;
    gd.max_iterations = 2000;
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

TEST_CASE("AdaGradUpdate - Adaptive learning rate behavior") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    // AdaGrad should work well even with large initial learning rate
    // because it adapts per-parameter
    SUBCASE("Large initial learning rate") {
        GradientDescent<AdaGradUpdate> gd;
        gd.step_size = 1.0; // Large initial rate
        gd.max_iterations = 1000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
    }

    SUBCASE("Small initial learning rate") {
        GradientDescent<AdaGradUpdate> gd;
        gd.step_size = 0.1;
        gd.max_iterations = 2000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
    }
}

TEST_CASE("AdaGradUpdate - Different epsilon values") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    SUBCASE("Default epsilon (1e-8)") {
        GradientDescent<AdaGradUpdate> gd;
        gd.step_size = 0.5;
        gd.max_iterations = 1000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }

    SUBCASE("Larger epsilon (1e-6)") {
        GradientDescent<AdaGradUpdate> gd;
        gd.get_update_policy().epsilon = 1e-6;
        gd.step_size = 0.5;
        gd.max_iterations = 1000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }
}

TEST_CASE("AdaGradUpdate - Float precision") {
    using Vec2 = dp::mat::vector<float, 2>;

    Sphere<float, 2> sphere;
    GradientDescent<AdaGradUpdate> gd;

    gd.step_size = 0.5;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-5f;

    Vec2 x(dp::mat::vector<float, 2>{1.0f, 1.0f});
    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-3f);
    CHECK(std::abs(x[0]) < 1e-1f);
    CHECK(std::abs(x[1]) < 1e-1f);
}

TEST_CASE("AdaGradUpdate - Custom quadratic function") {
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
    GradientDescent<AdaGradUpdate> gd;

    gd.step_size = 0.5;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    dp::mat::vector<double, 2> x(dp::mat::vector<double, 2>{0.0, 0.0});
    auto result = gd.optimize(func, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4);
    CHECK(std::abs(x[0] - 2.0) < 1e-2);
    CHECK(std::abs(x[1] - (-3.0)) < 1e-2);
}

TEST_CASE("AdaGradUpdate - Reset behavior") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    GradientDescent<AdaGradUpdate> gd;
    gd.step_size = 0.5;
    gd.max_iterations = 1000;
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
    CHECK(std::abs(static_cast<int>(result1.iterations) - static_cast<int>(result2.iterations)) < 5);
}

TEST_CASE("AdaGradUpdate - Comparison with vanilla GD") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    // AdaGrad
    GradientDescent<AdaGradUpdate> adagrad_gd;
    adagrad_gd.step_size = 0.5;
    adagrad_gd.max_iterations = 1000;
    adagrad_gd.tolerance = 1e-6;

    Vec2 x1(dp::mat::vector<double, 2>{5.0, 5.0});
    auto adagrad_result = adagrad_gd.optimize(sphere, x1);

    // Vanilla GD
    GradientDescent<VanillaUpdate> vanilla_gd;
    vanilla_gd.step_size = 0.1; // Vanilla needs smaller step size
    vanilla_gd.max_iterations = 1000;
    vanilla_gd.tolerance = 1e-6;

    Vec2 x2(dp::mat::vector<double, 2>{5.0, 5.0});
    auto vanilla_result = vanilla_gd.optimize(sphere, x2);

    // Both should converge
    CHECK(adagrad_result.converged);
    CHECK(vanilla_result.converged);
}

TEST_CASE("AdaGradUpdate - Accumulated gradient behavior") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    // Without reset, accumulated gradients persist
    GradientDescent<AdaGradUpdate> gd;
    gd.step_size = 0.5;
    gd.max_iterations = 500;
    gd.tolerance = 1e-6;
    gd.reset_policy = false;

    // First optimization
    Vec2 x1(dp::mat::vector<double, 2>{1.0, 1.0});
    auto result1 = gd.optimize(sphere, x1);
    CHECK(result1.converged);

    // Second optimization without reset - accumulated gradients affect behavior
    Vec2 x2(dp::mat::vector<double, 2>{1.0, 1.0});
    auto result2 = gd.optimize(sphere, x2);
    CHECK(result2.converged);

    // With accumulated gradients, learning rate is effectively smaller
    // so second run may take more iterations
    CHECK(result2.iterations >= result1.iterations);
}
