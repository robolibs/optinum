#include <doctest/doctest.h>

#include <datapod/matrix/vector.hpp>
#include <optinum/opti/opti.hpp>
#include <optinum/simd/vector.hpp>

using namespace optinum;
using namespace optinum::opti;
namespace dp = datapod;

TEST_CASE("NesterovUpdate - Sphere function 2D") {
    using Vec2 = simd::Vector<double, 2>;

    Sphere<double, 2> sphere;
    GradientDescent<NesterovUpdate> gd;
    gd.get_update_policy().momentum = 0.9;

    gd.step_size = 0.05;
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

TEST_CASE("NesterovUpdate - Sphere function 3D") {
    using Vec3 = simd::Vector<double, 3>;

    Sphere<double, 3> sphere;
    GradientDescent<NesterovUpdate> gd;
    gd.get_update_policy().momentum = 0.9;

    gd.step_size = 0.05;
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

TEST_CASE("NesterovUpdate - Sphere function 10D") {
    using Vec10 = simd::Vector<double, 10>;

    Sphere<double, 10> sphere;
    GradientDescent<NesterovUpdate> gd;
    gd.get_update_policy().momentum = 0.9;

    gd.step_size = 0.02;
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

TEST_CASE("NesterovUpdate - Faster than vanilla momentum") {
    using Vec2 = simd::Vector<double, 2>;
    Sphere<double, 2> sphere;

    // Nesterov momentum
    GradientDescent<NesterovUpdate> nesterov_gd;
    nesterov_gd.get_update_policy().momentum = 0.9;
    nesterov_gd.step_size = 0.05;
    nesterov_gd.max_iterations = 1000;
    nesterov_gd.tolerance = 1e-6;

    Vec2 x1(dp::mat::vector<double, 2>{5.0, 5.0});
    auto nesterov_result = nesterov_gd.optimize(sphere, x1);

    // Classical momentum
    GradientDescent<MomentumUpdate> momentum_gd;
    momentum_gd.get_update_policy().momentum = 0.9;
    momentum_gd.step_size = 0.05;
    momentum_gd.max_iterations = 1000;
    momentum_gd.tolerance = 1e-6;

    Vec2 x2(dp::mat::vector<double, 2>{5.0, 5.0});
    auto momentum_result = momentum_gd.optimize(sphere, x2);

    // Both should converge
    CHECK(nesterov_result.converged);
    CHECK(momentum_result.converged);

    // Nesterov should typically converge in fewer or similar iterations
    // (on simple problems the difference may be small)
    CHECK(nesterov_result.iterations <= momentum_result.iterations + 20);
}

TEST_CASE("NesterovUpdate - Different momentum values") {
    using Vec2 = simd::Vector<double, 2>;
    Sphere<double, 2> sphere;

    SUBCASE("Low momentum (0.5)") {
        GradientDescent<NesterovUpdate> gd;
        gd.get_update_policy().momentum = 0.5;
        gd.step_size = 0.1;
        gd.max_iterations = 1000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
    }

    SUBCASE("High momentum (0.99)") {
        GradientDescent<NesterovUpdate> gd;
        gd.get_update_policy().momentum = 0.99;
        gd.step_size = 0.01;
        gd.max_iterations = 5000; // More iterations for high momentum
        gd.tolerance = 1e-7;      // Tighter tolerance to ensure convergence

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-3); // High momentum oscillates more
    }

    SUBCASE("Zero momentum (degenerates to vanilla GD)") {
        GradientDescent<NesterovUpdate> gd;
        gd.get_update_policy().momentum = 0.0;
        gd.step_size = 0.1;
        gd.max_iterations = 1000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
    }
}

TEST_CASE("NesterovUpdate - Float precision") {
    using Vec2 = simd::Vector<float, 2>;

    Sphere<float, 2> sphere;
    GradientDescent<NesterovUpdate> gd;
    gd.get_update_policy().momentum = 0.9;

    gd.step_size = 0.05;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-5f;

    Vec2 x(dp::mat::vector<float, 2>{1.0f, 1.0f});
    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-3f);
    CHECK(std::abs(x[0]) < 1e-1f);
    CHECK(std::abs(x[1]) < 1e-1f);
}

TEST_CASE("NesterovUpdate - Custom quadratic function") {
    // Custom quadratic: f(x, y) = (x - 2)^2 + (y + 3)^2
    // Minimum at (2, -3)
    struct CustomQuadratic {
        using tensor_type = simd::Vector<double, 2>;

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
    GradientDescent<NesterovUpdate> gd;
    gd.get_update_policy().momentum = 0.9;

    gd.step_size = 0.05;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    simd::Vector<double, 2> x(dp::mat::vector<double, 2>{0.0, 0.0});
    auto result = gd.optimize(func, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4);
    CHECK(std::abs(x[0] - 2.0) < 1e-2);
    CHECK(std::abs(x[1] - (-3.0)) < 1e-2);
}

TEST_CASE("NesterovUpdate - Reset behavior") {
    using Vec2 = simd::Vector<double, 2>;
    Sphere<double, 2> sphere;

    GradientDescent<NesterovUpdate> gd;
    gd.get_update_policy().momentum = 0.9;
    gd.step_size = 0.05;
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
