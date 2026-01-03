#include <doctest/doctest.h>

#include <optinum/optinum.hpp>

using namespace optinum;
namespace dp = datapod;
using namespace optinum::opti;

TEST_CASE("dp::mat::Dynamic Optimization - Sphere function") {
    SUBCASE("Runtime size N=5") {
        std::size_t n = 5;
        Sphere<double, dp::mat::Dynamic> sphere;

        dp::mat::Vector<double, dp::mat::Dynamic> x(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = static_cast<double>(i) + 1.0; // [1, 2, 3, 4, 5]
        }

        GradientDescent<> gd;
        gd.step_size = 0.1;
        gd.max_iterations = 1000;
        gd.tolerance = 1e-6;

        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);

        for (std::size_t i = 0; i < n; ++i) {
            CHECK(std::abs(x[i]) < 1e-2);
        }
    }

    SUBCASE("Runtime size N=10") {
        std::size_t n = 10;
        Sphere<double, dp::mat::Dynamic> sphere;

        dp::mat::Vector<double, dp::mat::Dynamic> x(n, 2.0); // Initialize all to 2.0

        Adam adam;
        adam.step_size = 0.1;
        adam.max_iterations = 1000;
        adam.tolerance = 1e-6;

        auto result = adam.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);

        for (std::size_t i = 0; i < n; ++i) {
            CHECK(std::abs(x[i]) < 1e-2);
        }
    }
}

TEST_CASE("dp::mat::Dynamic vs Fixed-size optimization - Same results") {
    constexpr std::size_t N = 5;

    // Fixed-size optimization
    Sphere<double, N> sphere_fixed;
    dp::mat::Vector<double, N> x_fixed;
    for (std::size_t i = 0; i < N; ++i) {
        x_fixed[i] = static_cast<double>(i) + 1.0;
    }

    Adam adam_fixed;
    adam_fixed.step_size = 0.1;
    adam_fixed.max_iterations = 1000;
    adam_fixed.tolerance = 1e-6;
    auto result_fixed = adam_fixed.optimize(sphere_fixed, x_fixed);

    // dp::mat::Dynamic-size optimization
    Sphere<double, dp::mat::Dynamic> sphere_dynamic;
    dp::mat::Vector<double, dp::mat::Dynamic> x_dynamic(N);
    for (std::size_t i = 0; i < N; ++i) {
        x_dynamic[i] = static_cast<double>(i) + 1.0;
    }

    Adam adam_dynamic;
    adam_dynamic.step_size = 0.1;
    adam_dynamic.max_iterations = 1000;
    adam_dynamic.tolerance = 1e-6;
    auto result_dynamic = adam_dynamic.optimize(sphere_dynamic, x_dynamic);

    // Both should converge
    CHECK(result_fixed.converged);
    CHECK(result_dynamic.converged);

    // Similar final costs
    CHECK(std::abs(result_fixed.final_cost - result_dynamic.final_cost) < 1e-4);

    // Similar solutions
    for (std::size_t i = 0; i < N; ++i) {
        CHECK(std::abs(x_fixed[i] - x_dynamic[i]) < 1e-3);
    }
}

TEST_CASE("All optimizers work with dp::mat::Dynamic") {
    std::size_t n = 3;
    Sphere<double, dp::mat::Dynamic> sphere;

    SUBCASE("Vanilla GD") {
        dp::mat::Vector<double, dp::mat::Dynamic> x(n, 5.0);

        GradientDescent<> gd;
        gd.step_size = 0.1;
        gd.max_iterations = 1000;
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }

    SUBCASE("Momentum") {
        dp::mat::Vector<double, dp::mat::Dynamic> x(n, 5.0);

        Momentum momentum;
        momentum.step_size = 0.1;
        momentum.max_iterations = 1000;
        auto result = momentum.optimize(sphere, x);

        CHECK(result.converged);
    }

    SUBCASE("RMSprop") {
        dp::mat::Vector<double, dp::mat::Dynamic> x(n, 5.0);

        RMSprop rmsprop;
        rmsprop.step_size = 0.1;
        rmsprop.max_iterations = 1000;
        auto result = rmsprop.optimize(sphere, x);

        CHECK(result.converged);
    }

    SUBCASE("Adam") {
        dp::mat::Vector<double, dp::mat::Dynamic> x(n, 5.0);

        Adam adam;
        adam.step_size = 0.1;
        adam.max_iterations = 1000;
        auto result = adam.optimize(sphere, x);

        CHECK(result.converged);
    }
}
