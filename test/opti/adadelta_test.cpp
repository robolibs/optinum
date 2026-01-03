#include <doctest/doctest.h>

#include <datapod/matrix/vector.hpp>
#include <optinum/opti/opti.hpp>

using namespace optinum;
using namespace optinum::opti;
namespace dp = datapod;

TEST_CASE("AdaDeltaUpdate - Sphere function 2D") {
    using Vec2 = dp::mat::Vector<double, 2>;

    Sphere<double, 2> sphere;
    GradientDescent<AdaDeltaUpdate> gd;

    // AdaDelta doesn't use step_size, but we set it for API compatibility
    gd.step_size = 1.0; // Ignored by AdaDelta
    gd.max_iterations = 2000;
    gd.tolerance = 1e-6;

    SUBCASE("Converge from (1, 1)") {
        Vec2 x(dp::mat::Vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.iterations < 2000);
        CHECK(result.final_cost < 1e-4);
        CHECK(std::abs(x[0]) < 1e-2);
        CHECK(std::abs(x[1]) < 1e-2);
    }

    SUBCASE("Converge from (5, -3)") {
        Vec2 x(dp::mat::Vector<double, 2>{5.0, -3.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
        CHECK(std::abs(x[0]) < 1e-2);
        CHECK(std::abs(x[1]) < 1e-2);
    }

    SUBCASE("Already at minimum") {
        Vec2 x(dp::mat::Vector<double, 2>{0.0, 0.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.iterations < 10);
        CHECK(result.final_cost < 1e-10);
    }
}

TEST_CASE("AdaDeltaUpdate - Sphere function 3D") {
    using Vec3 = dp::mat::Vector<double, 3>;

    Sphere<double, 3> sphere;
    GradientDescent<AdaDeltaUpdate> gd;

    gd.step_size = 1.0;
    gd.max_iterations = 2000;
    gd.tolerance = 1e-6;

    Vec3 x(dp::mat::Vector<double, 3>{2.0, -1.0, 3.0});
    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4);
    CHECK(std::abs(x[0]) < 1e-2);
    CHECK(std::abs(x[1]) < 1e-2);
    CHECK(std::abs(x[2]) < 1e-2);
}

TEST_CASE("AdaDeltaUpdate - Sphere function 10D") {
    using Vec10 = dp::mat::Vector<double, 10>;

    Sphere<double, 10> sphere;
    GradientDescent<AdaDeltaUpdate> gd;

    gd.step_size = 1.0;
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

TEST_CASE("AdaDeltaUpdate - No learning rate required") {
    using Vec2 = dp::mat::Vector<double, 2>;
    Sphere<double, 2> sphere;

    // AdaDelta should work regardless of step_size value
    // since it computes its own adaptive learning rate
    SUBCASE("step_size = 0.001 (ignored)") {
        GradientDescent<AdaDeltaUpdate> gd;
        gd.step_size = 0.001;
        gd.max_iterations = 2000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::Vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
    }

    SUBCASE("step_size = 100.0 (ignored)") {
        GradientDescent<AdaDeltaUpdate> gd;
        gd.step_size = 100.0;
        gd.max_iterations = 2000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::Vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
    }
}

TEST_CASE("AdaDeltaUpdate - Different rho values") {
    using Vec2 = dp::mat::Vector<double, 2>;
    Sphere<double, 2> sphere;

    SUBCASE("Default rho (0.95)") {
        GradientDescent<AdaDeltaUpdate> gd;
        gd.max_iterations = 2000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::Vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }

    SUBCASE("Lower rho (0.9)") {
        GradientDescent<AdaDeltaUpdate> gd;
        gd.get_update_policy().rho = 0.9;
        gd.max_iterations = 2000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::Vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }

    SUBCASE("Higher rho (0.99)") {
        GradientDescent<AdaDeltaUpdate> gd;
        gd.get_update_policy().rho = 0.99;
        gd.max_iterations = 3000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::Vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }
}

TEST_CASE("AdaDeltaUpdate - Float precision") {
    using Vec2 = dp::mat::Vector<float, 2>;

    Sphere<float, 2> sphere;
    GradientDescent<AdaDeltaUpdate> gd;

    gd.step_size = 1.0;
    gd.max_iterations = 2000;
    gd.tolerance = 1e-5f;

    Vec2 x(dp::mat::Vector<float, 2>{1.0f, 1.0f});
    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-3f);
    CHECK(std::abs(x[0]) < 1e-1f);
    CHECK(std::abs(x[1]) < 1e-1f);
}

TEST_CASE("AdaDeltaUpdate - Custom quadratic function") {
    // Custom quadratic: f(x, y) = (x - 2)^2 + (y + 3)^2
    // Minimum at (2, -3)
    struct CustomQuadratic {
        using tensor_type = dp::mat::Vector<double, 2>;

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
    GradientDescent<AdaDeltaUpdate> gd;

    gd.step_size = 1.0;
    gd.max_iterations = 2000;
    gd.tolerance = 1e-6;

    dp::mat::Vector<double, 2> x(dp::mat::Vector<double, 2>{0.0, 0.0});
    auto result = gd.optimize(func, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4);
    CHECK(std::abs(x[0] - 2.0) < 1e-2);
    CHECK(std::abs(x[1] - (-3.0)) < 1e-2);
}

TEST_CASE("AdaDeltaUpdate - Reset behavior") {
    using Vec2 = dp::mat::Vector<double, 2>;
    Sphere<double, 2> sphere;

    GradientDescent<AdaDeltaUpdate> gd;
    gd.step_size = 1.0;
    gd.max_iterations = 2000;
    gd.tolerance = 1e-6;
    gd.reset_policy = true;

    // First optimization
    Vec2 x1(dp::mat::Vector<double, 2>{1.0, 1.0});
    auto result1 = gd.optimize(sphere, x1);
    CHECK(result1.converged);

    // Second optimization (should behave the same due to reset)
    Vec2 x2(dp::mat::Vector<double, 2>{1.0, 1.0});
    auto result2 = gd.optimize(sphere, x2);
    CHECK(result2.converged);

    // Iterations should be similar (within tolerance)
    CHECK(std::abs(static_cast<int>(result1.iterations) - static_cast<int>(result2.iterations)) < 10);
}

TEST_CASE("AdaDeltaUpdate - Comparison with AdaGrad") {
    using Vec2 = dp::mat::Vector<double, 2>;
    Sphere<double, 2> sphere;

    // AdaDelta
    GradientDescent<AdaDeltaUpdate> adadelta_gd;
    adadelta_gd.step_size = 1.0;
    adadelta_gd.max_iterations = 2000;
    adadelta_gd.tolerance = 1e-6;

    Vec2 x1(dp::mat::Vector<double, 2>{5.0, 5.0});
    auto adadelta_result = adadelta_gd.optimize(sphere, x1);

    // AdaGrad
    GradientDescent<AdaGradUpdate> adagrad_gd;
    adagrad_gd.step_size = 0.5;
    adagrad_gd.max_iterations = 2000;
    adagrad_gd.tolerance = 1e-6;

    Vec2 x2(dp::mat::Vector<double, 2>{5.0, 5.0});
    auto adagrad_result = adagrad_gd.optimize(sphere, x2);

    // Both should converge
    CHECK(adadelta_result.converged);
    CHECK(adagrad_result.converged);
}

TEST_CASE("AdaDeltaUpdate - Robustness to initial conditions") {
    using Vec2 = dp::mat::Vector<double, 2>;
    Sphere<double, 2> sphere;

    GradientDescent<AdaDeltaUpdate> gd;
    gd.step_size = 1.0;
    gd.max_iterations = 3000;
    gd.tolerance = 1e-6;

    SUBCASE("Far from minimum") {
        Vec2 x(dp::mat::Vector<double, 2>{10.0, -10.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
    }

    SUBCASE("Very close to minimum") {
        Vec2 x(dp::mat::Vector<double, 2>{0.01, -0.01});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-6);
    }
}
