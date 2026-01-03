#include <datapod/matrix.hpp>
#include <doctest/doctest.h>
#include <optinum/meta/pso.hpp>
#include <optinum/opti/problem/ackley.hpp>
#include <optinum/opti/problem/rastrigin.hpp>
#include <optinum/opti/problem/rosenbrock.hpp>
#include <optinum/opti/problem/sphere.hpp>

#include <cmath>

using namespace optinum;
namespace dp = datapod;

TEST_CASE("PSO: Sphere function optimization") {
    meta::PSO<double> pso;
    pso.config.population_size = 30;
    pso.config.max_iterations = 500;
    pso.config.tolerance = 1e-6;

    // 2D Sphere: f(x) = x[0]^2 + x[1]^2, minimum at (0, 0)
    auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i];
        }
        return sum;
    };

    dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    auto result = pso.optimize(sphere, lower, upper);

    CHECK(result.converged);
    CHECK(result.best_value < 1e-4);
    CHECK(std::abs(result.best_position[0]) < 0.1);
    CHECK(std::abs(result.best_position[1]) < 0.1);
}

TEST_CASE("PSO: Higher dimensional Sphere") {
    meta::PSO<double> pso;
    pso.config.population_size = 50;
    pso.config.max_iterations = 1000;
    pso.config.tolerance = 1e-6;

    constexpr std::size_t dim = 5;

    auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i];
        }
        return sum;
    };

    dp::mat::Vector<double, dp::mat::Dynamic> lower(dim);
    dp::mat::Vector<double, dp::mat::Dynamic> upper(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        lower[i] = -10.0;
        upper[i] = 10.0;
    }

    auto result = pso.optimize(sphere, lower, upper);

    CHECK(result.best_value < 0.01);
    for (std::size_t i = 0; i < dim; ++i) {
        CHECK(std::abs(result.best_position[i]) < 0.5);
    }
}

TEST_CASE("PSO: Rosenbrock function") {
    meta::PSO<double> pso;
    pso.config.population_size = 100;
    pso.config.max_iterations = 2000;
    pso.config.tolerance = 1e-8;
    pso.config.inertia_weight = 0.6;
    pso.config.cognitive_coeff = 1.8;
    pso.config.social_coeff = 1.8;

    // 2D Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1, 1)
    auto rosenbrock = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
        double a = 1.0 - x[0];
        double b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    };

    dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    auto result = pso.optimize(rosenbrock, lower, upper);

    // Rosenbrock is harder - just check we get reasonably close
    CHECK(result.best_value < 1.0);
    CHECK(std::abs(result.best_position[0] - 1.0) < 1.0);
    CHECK(std::abs(result.best_position[1] - 1.0) < 1.0);
}

TEST_CASE("PSO: Rastrigin function (multimodal)") {
    meta::PSO<double> pso;
    pso.config.population_size = 100;
    pso.config.max_iterations = 1000;
    pso.config.tolerance = 1e-6;

    // 2D Rastrigin: highly multimodal, global minimum at (0, 0) = 0
    auto rastrigin = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
        const double A = 10.0;
        const double pi = 3.14159265358979323846;
        double sum = A * static_cast<double>(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i] - A * std::cos(2.0 * pi * x[i]);
        }
        return sum;
    };

    dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.12;
    lower[1] = -5.12;
    upper[0] = 5.12;
    upper[1] = 5.12;

    auto result = pso.optimize(rastrigin, lower, upper);

    // Rastrigin is very hard - just check we find a reasonable local minimum
    CHECK(result.best_value < 5.0);
}

TEST_CASE("PSO: Configuration options") {
    SUBCASE("Custom config via constructor") {
        meta::PSO<double>::Config cfg;
        cfg.population_size = 20;
        cfg.max_iterations = 100;
        cfg.inertia_weight = 0.5;

        meta::PSO<double> pso(cfg);

        CHECK(pso.config.population_size == 20);
        CHECK(pso.config.max_iterations == 100);
        CHECK(pso.config.inertia_weight == doctest::Approx(0.5));
    }

    SUBCASE("Track history") {
        meta::PSO<double> pso;
        pso.config.population_size = 20;
        pso.config.max_iterations = 50;
        pso.config.track_history = true;

        auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

        dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
        dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
        lower[0] = -5.0;
        lower[1] = -5.0;
        upper[0] = 5.0;
        upper[1] = 5.0;

        auto result = pso.optimize(sphere, lower, upper);

        CHECK(!result.history.empty());
        // History should be monotonically non-increasing (best value can only improve or stay same)
        for (std::size_t i = 1; i < result.history.size(); ++i) {
            CHECK(result.history[i] <= result.history[i - 1] + 1e-10);
        }
    }
}

TEST_CASE("PSO: Edge cases") {
    meta::PSO<double> pso;

    SUBCASE("Empty bounds returns invalid result") {
        dp::mat::Vector<double, dp::mat::Dynamic> lower(0);
        dp::mat::Vector<double, dp::mat::Dynamic> upper(0);

        auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
            (void)x;
            return 0.0;
        };

        auto result = pso.optimize(sphere, lower, upper);
        CHECK(!result.converged);
    }

    SUBCASE("1D optimization") {
        auto quadratic = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) { return (x[0] - 3.0) * (x[0] - 3.0); };

        dp::mat::Vector<double, dp::mat::Dynamic> lower(1);
        dp::mat::Vector<double, dp::mat::Dynamic> upper(1);
        lower[0] = -10.0;
        upper[0] = 10.0;

        pso.config.population_size = 20;
        pso.config.max_iterations = 200;

        auto result = pso.optimize(quadratic, lower, upper);

        CHECK(result.best_value < 0.01);
        CHECK(std::abs(result.best_position[0] - 3.0) < 0.2);
    }
}

TEST_CASE("PSO: Float type") {
    meta::PSO<float> pso;
    pso.config.population_size = 30;
    pso.config.max_iterations = 300;
    pso.config.tolerance = 1e-4f;

    auto sphere = [](const dp::mat::Vector<float, dp::mat::Dynamic> &x) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i];
        }
        return sum;
    };

    dp::mat::Vector<float, dp::mat::Dynamic> lower(2);
    dp::mat::Vector<float, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0f;
    lower[1] = -5.0f;
    upper[0] = 5.0f;
    upper[1] = 5.0f;

    auto result = pso.optimize(sphere, lower, upper);

    CHECK(result.best_value < 0.01f);
}
