#include <doctest/doctest.h>

#include <datapod/matrix/vector.hpp>
#include <optinum/opti/opti.hpp>

using namespace optinum;
using namespace optinum::opti;
namespace dp = datapod;

TEST_CASE("AdaBoundUpdate - Sphere function 2D") {
    using Vec2 = dp::mat::vector<double, 2>;

    Sphere<double, 2> sphere;
    GradientDescent<AdaBoundUpdate> gd;

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

TEST_CASE("AdaBoundUpdate - Sphere function 3D") {
    using Vec3 = dp::mat::vector<double, 3>;

    Sphere<double, 3> sphere;
    GradientDescent<AdaBoundUpdate> gd;

    gd.step_size = 0.1;
    gd.max_iterations = 2000;
    gd.tolerance = 1e-6;

    Vec3 x(dp::mat::vector<double, 3>{2.0, -1.0, 3.0});
    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4);
    CHECK(std::abs(x[0]) < 1e-2);
    CHECK(std::abs(x[1]) < 1e-2);
    CHECK(std::abs(x[2]) < 1e-2);
}

TEST_CASE("AdaBoundUpdate - Sphere function 10D") {
    using Vec10 = dp::mat::vector<double, 10>;

    Sphere<double, 10> sphere;
    GradientDescent<AdaBoundUpdate> gd;

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

TEST_CASE("AdaBoundUpdate - Dynamic bounds behavior") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    // AdaBound should work with different final_lr values
    SUBCASE("Default final_lr (0.1)") {
        GradientDescent<AdaBoundUpdate> gd;
        gd.step_size = 0.1;
        gd.max_iterations = 2000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-3);
    }

    SUBCASE("Smaller final_lr (0.01)") {
        GradientDescent<AdaBoundUpdate> gd;
        gd.get_update_policy().final_lr = 0.01;
        gd.step_size = 0.1;
        gd.max_iterations = 3000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
    }

    SUBCASE("Larger final_lr (0.5)") {
        GradientDescent<AdaBoundUpdate> gd;
        gd.get_update_policy().final_lr = 0.5;
        gd.step_size = 0.1;
        gd.max_iterations = 2000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-3);
    }
}

TEST_CASE("AdaBoundUpdate - Different gamma values") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    SUBCASE("Default gamma (1e-3)") {
        GradientDescent<AdaBoundUpdate> gd;
        gd.step_size = 0.1;
        gd.max_iterations = 2000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }

    SUBCASE("Faster bound convergence (1e-2)") {
        GradientDescent<AdaBoundUpdate> gd;
        gd.get_update_policy().gamma = 1e-2;
        gd.step_size = 0.1;
        gd.max_iterations = 2000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }

    SUBCASE("Slower bound convergence (1e-4)") {
        GradientDescent<AdaBoundUpdate> gd;
        gd.get_update_policy().gamma = 1e-4;
        gd.step_size = 0.1;
        gd.max_iterations = 2000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }
}

TEST_CASE("AdaBoundUpdate - Float precision") {
    using Vec2 = dp::mat::vector<float, 2>;

    Sphere<float, 2> sphere;
    GradientDescent<AdaBoundUpdate> gd;

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

TEST_CASE("AdaBoundUpdate - Custom quadratic function") {
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
    GradientDescent<AdaBoundUpdate> gd;

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

TEST_CASE("AdaBoundUpdate - Reset behavior") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    GradientDescent<AdaBoundUpdate> gd;
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

TEST_CASE("AdaBoundUpdate - Comparison with Adam") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    // AdaBound
    GradientDescent<AdaBoundUpdate> adabound_gd;
    adabound_gd.step_size = 0.1;
    adabound_gd.max_iterations = 2000;
    adabound_gd.tolerance = 1e-6;

    Vec2 x1(dp::mat::vector<double, 2>{5.0, 5.0});
    auto adabound_result = adabound_gd.optimize(sphere, x1);

    // Adam
    GradientDescent<AdamUpdate> adam_gd;
    adam_gd.step_size = 0.1;
    adam_gd.max_iterations = 2000;
    adam_gd.tolerance = 1e-6;

    Vec2 x2(dp::mat::vector<double, 2>{5.0, 5.0});
    auto adam_result = adam_gd.optimize(sphere, x2);

    // Both should converge
    CHECK(adabound_result.converged);
    CHECK(adam_result.converged);
}

TEST_CASE("AdaBoundUpdate - Robustness to initial conditions") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    GradientDescent<AdaBoundUpdate> gd;
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
