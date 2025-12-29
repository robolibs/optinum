#include <datapod/matrix.hpp>
#include <doctest/doctest.h>
#include <optinum/meta/cem.hpp>

#include <cmath>

using namespace optinum;
namespace dp = datapod;

TEST_CASE("CEM: Sphere function optimization") {
    meta::CEM<double> cem;
    cem.config.population_size = 100;
    cem.config.max_iterations = 100;
    cem.config.elite_fraction = 0.1;
    cem.config.tolerance = 1e-6;

    // 2D Sphere: f(x) = x[0]^2 + x[1]^2, minimum at (0, 0)
    auto sphere = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i];
        }
        return sum;
    };

    dp::mat::vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    auto result = cem.optimize(sphere, lower, upper);

    CHECK(result.converged);
    CHECK(result.best_value < 0.01);
    CHECK(std::abs(result.best_position[0]) < 0.5);
    CHECK(std::abs(result.best_position[1]) < 0.5);
}

TEST_CASE("CEM: Higher dimensional Sphere") {
    meta::CEM<double> cem;
    cem.config.population_size = 200;
    cem.config.max_iterations = 200;
    cem.config.elite_fraction = 0.1;
    cem.config.tolerance = 1e-6;

    constexpr std::size_t dim = 5;

    auto sphere = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i];
        }
        return sum;
    };

    dp::mat::vector<double, dp::mat::Dynamic> lower(dim);
    dp::mat::vector<double, dp::mat::Dynamic> upper(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        lower[i] = -10.0;
        upper[i] = 10.0;
    }

    auto result = cem.optimize(sphere, lower, upper);

    CHECK(result.best_value < 0.5);
    for (std::size_t i = 0; i < dim; ++i) {
        CHECK(std::abs(result.best_position[i]) < 1.0);
    }
}

TEST_CASE("CEM: Rosenbrock function") {
    meta::CEM<double> cem;
    cem.config.population_size = 200;
    cem.config.max_iterations = 300;
    cem.config.elite_fraction = 0.1;
    cem.config.initial_std = 2.0;
    cem.config.std_decay = 0.995;
    cem.config.tolerance = 1e-8;

    // 2D Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1, 1)
    auto rosenbrock = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) {
        double a = 1.0 - x[0];
        double b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    };

    dp::mat::vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    auto result = cem.optimize(rosenbrock, lower, upper);

    // Rosenbrock is harder - just check we get reasonably close
    CHECK(result.best_value < 1.0);
}

TEST_CASE("CEM: Rastrigin function (multimodal)") {
    meta::CEM<double> cem;
    cem.config.population_size = 200;
    cem.config.max_iterations = 200;
    cem.config.elite_fraction = 0.1;
    cem.config.std_decay = 0.99;
    cem.config.tolerance = 1e-6;

    // 2D Rastrigin: highly multimodal, global minimum at (0, 0) = 0
    auto rastrigin = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) {
        const double A = 10.0;
        const double pi = 3.14159265358979323846;
        double sum = A * static_cast<double>(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i] - A * std::cos(2.0 * pi * x[i]);
        }
        return sum;
    };

    dp::mat::vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.12;
    lower[1] = -5.12;
    upper[0] = 5.12;
    upper[1] = 5.12;

    auto result = cem.optimize(rastrigin, lower, upper);

    // Rastrigin is very hard - just check we find a reasonable local minimum
    CHECK(result.best_value < 10.0);
}

TEST_CASE("CEM: Configuration options") {
    SUBCASE("Custom config via constructor") {
        meta::CEM<double>::Config cfg;
        cfg.population_size = 50;
        cfg.max_iterations = 50;
        cfg.elite_fraction = 0.2;
        cfg.initial_std = 2.0;

        meta::CEM<double> cem(cfg);

        CHECK(cem.config.population_size == 50);
        CHECK(cem.config.max_iterations == 50);
        CHECK(cem.config.elite_fraction == doctest::Approx(0.2));
        CHECK(cem.config.initial_std == doctest::Approx(2.0));
    }

    SUBCASE("Track history") {
        meta::CEM<double> cem;
        cem.config.population_size = 50;
        cem.config.max_iterations = 20;
        cem.config.track_history = true;

        auto sphere = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

        dp::mat::vector<double, dp::mat::Dynamic> lower(2);
        dp::mat::vector<double, dp::mat::Dynamic> upper(2);
        lower[0] = -5.0;
        lower[1] = -5.0;
        upper[0] = 5.0;
        upper[1] = 5.0;

        auto result = cem.optimize(sphere, lower, upper);

        CHECK(!result.history.empty());
        // History should be monotonically non-increasing (best value can only improve or stay same)
        for (std::size_t i = 1; i < result.history.size(); ++i) {
            CHECK(result.history[i] <= result.history[i - 1] + 1e-10);
        }
    }

    SUBCASE("Function evaluations tracking") {
        meta::CEM<double> cem;
        cem.config.population_size = 50;
        cem.config.max_iterations = 10;

        auto sphere = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

        dp::mat::vector<double, dp::mat::Dynamic> lower(2);
        dp::mat::vector<double, dp::mat::Dynamic> upper(2);
        lower[0] = -5.0;
        lower[1] = -5.0;
        upper[0] = 5.0;
        upper[1] = 5.0;

        auto result = cem.optimize(sphere, lower, upper);

        // Should have at least population_size * iterations evaluations
        CHECK(result.function_evaluations >= cem.config.population_size * result.iterations);
    }
}

TEST_CASE("CEM: Edge cases") {
    meta::CEM<double> cem;

    SUBCASE("Empty bounds returns invalid result") {
        dp::mat::vector<double, dp::mat::Dynamic> lower(0);
        dp::mat::vector<double, dp::mat::Dynamic> upper(0);

        auto sphere = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) {
            (void)x;
            return 0.0;
        };

        auto result = cem.optimize(sphere, lower, upper);
        CHECK(!result.converged);
    }

    SUBCASE("1D optimization") {
        auto quadratic = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) { return (x[0] - 3.0) * (x[0] - 3.0); };

        dp::mat::vector<double, dp::mat::Dynamic> lower(1);
        dp::mat::vector<double, dp::mat::Dynamic> upper(1);
        lower[0] = -10.0;
        upper[0] = 10.0;

        cem.config.population_size = 50;
        cem.config.max_iterations = 100;

        auto result = cem.optimize(quadratic, lower, upper);

        CHECK(result.best_value < 0.1);
        CHECK(std::abs(result.best_position[0] - 3.0) < 0.5);
    }

    SUBCASE("Mismatched bounds size returns invalid") {
        dp::mat::vector<double, dp::mat::Dynamic> lower(2);
        dp::mat::vector<double, dp::mat::Dynamic> upper(3);
        lower[0] = -5.0;
        lower[1] = -5.0;
        upper[0] = 5.0;
        upper[1] = 5.0;
        upper[2] = 5.0;

        auto sphere = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) {
            (void)x;
            return 0.0;
        };

        auto result = cem.optimize(sphere, lower, upper);
        CHECK(!result.converged);
    }
}

TEST_CASE("CEM: Optimize from initial mean") {
    meta::CEM<double> cem;
    cem.config.population_size = 100;
    cem.config.max_iterations = 100;
    cem.config.initial_std = 1.0;

    auto sphere = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i];
        }
        return sum;
    };

    // Start near the optimum
    dp::mat::vector<double, dp::mat::Dynamic> initial_mean(2);
    initial_mean[0] = 0.5;
    initial_mean[1] = -0.5;

    auto result = cem.optimize(sphere, initial_mean);

    CHECK(result.best_value < 0.1);
}

TEST_CASE("CEM: Float type") {
    meta::CEM<float> cem;
    cem.config.population_size = 100;
    cem.config.max_iterations = 100;
    cem.config.tolerance = 1e-4f;

    auto sphere = [](const dp::mat::vector<float, dp::mat::Dynamic> &x) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i];
        }
        return sum;
    };

    dp::mat::vector<float, dp::mat::Dynamic> lower(2);
    dp::mat::vector<float, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0f;
    lower[1] = -5.0f;
    upper[0] = 5.0f;
    upper[1] = 5.0f;

    auto result = cem.optimize(sphere, lower, upper);

    CHECK(result.best_value < 0.1f);
}

TEST_CASE("CEM: Elite fraction effect") {
    // Test that different elite fractions affect convergence
    auto sphere = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i];
        }
        return sum;
    };

    dp::mat::vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    SUBCASE("Small elite fraction (0.05)") {
        meta::CEM<double> cem;
        cem.config.population_size = 100;
        cem.config.max_iterations = 100;
        cem.config.elite_fraction = 0.05;

        auto result = cem.optimize(sphere, lower, upper);
        CHECK(result.best_value < 0.1);
    }

    SUBCASE("Large elite fraction (0.3)") {
        meta::CEM<double> cem;
        cem.config.population_size = 100;
        cem.config.max_iterations = 100;
        cem.config.elite_fraction = 0.3;

        auto result = cem.optimize(sphere, lower, upper);
        CHECK(result.best_value < 0.1);
    }
}

TEST_CASE("CEM: Std decay effect") {
    auto sphere = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i];
        }
        return sum;
    };

    dp::mat::vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    meta::CEM<double> cem;
    cem.config.population_size = 100;
    cem.config.max_iterations = 100;
    cem.config.std_decay = 0.95; // Faster decay

    auto result = cem.optimize(sphere, lower, upper);

    // Should still converge with faster decay
    CHECK(result.best_value < 0.1);
}
