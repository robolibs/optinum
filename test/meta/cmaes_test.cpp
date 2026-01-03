#include <datapod/matrix.hpp>
#include <doctest/doctest.h>
#include <optinum/meta/cmaes.hpp>

#include <cmath>

using namespace optinum;
namespace dp = datapod;

TEST_CASE("CMAES: Sphere function optimization") {
    meta::CMAES<double> cmaes;
    cmaes.config.max_generations = 500;
    cmaes.config.tolerance = 1e-6;

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

    auto result = cmaes.optimize(sphere, lower, upper);

    CHECK(result.best_value < 1e-4);
    CHECK(std::abs(result.best_position[0]) < 0.1);
    CHECK(std::abs(result.best_position[1]) < 0.1);
}

TEST_CASE("CMAES: Higher dimensional Sphere") {
    meta::CMAES<double> cmaes;
    cmaes.config.max_generations = 1000;
    cmaes.config.tolerance = 1e-6;

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

    auto result = cmaes.optimize(sphere, lower, upper);

    CHECK(result.best_value < 0.1);
    for (std::size_t i = 0; i < dim; ++i) {
        CHECK(std::abs(result.best_position[i]) < 1.0);
    }
}

TEST_CASE("CMAES: Rosenbrock function") {
    meta::CMAES<double> cmaes;
    cmaes.config.max_generations = 2000;
    cmaes.config.tolerance = 1e-8;
    cmaes.config.sigma0 = 0.5;

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

    auto result = cmaes.optimize(rosenbrock, lower, upper);

    // Rosenbrock is harder - just check we get reasonably close
    CHECK(result.best_value <= 1.0);
    CHECK(std::abs(result.best_position[0] - 1.0) <= 1.0);
    CHECK(std::abs(result.best_position[1] - 1.0) <= 1.0);
}

TEST_CASE("CMAES: Rastrigin function (multimodal)") {
    meta::CMAES<double> cmaes;
    cmaes.config.population_size = 50; // Larger population for multimodal
    cmaes.config.max_generations = 1000;
    cmaes.config.tolerance = 1e-6;

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

    auto result = cmaes.optimize(rastrigin, lower, upper);

    // Rastrigin is very hard - just check we find a reasonable local minimum
    CHECK(result.best_value < 10.0);
}

TEST_CASE("CMAES: Ellipsoid function (ill-conditioned)") {
    meta::CMAES<double> cmaes;
    cmaes.config.max_generations = 1000;
    cmaes.config.tolerance = 1e-6;

    // Ellipsoid: f(x) = sum(i * x[i]^2), ill-conditioned
    // CMA-ES should handle this well due to covariance adaptation
    auto ellipsoid = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += static_cast<double>(i + 1) * x[i] * x[i];
        }
        return sum;
    };

    dp::mat::Vector<double, dp::mat::Dynamic> lower(3);
    dp::mat::Vector<double, dp::mat::Dynamic> upper(3);
    for (std::size_t i = 0; i < 3; ++i) {
        lower[i] = -5.0;
        upper[i] = 5.0;
    }

    auto result = cmaes.optimize(ellipsoid, lower, upper);

    CHECK(result.best_value < 0.01);
}

TEST_CASE("CMAES: Configuration options") {
    SUBCASE("Custom config via constructor") {
        meta::CMAES<double>::Config cfg;
        cfg.population_size = 20;
        cfg.max_generations = 100;
        cfg.sigma0 = 0.5;
        cfg.tolerance = 1e-4;

        meta::CMAES<double> cmaes(cfg);

        CHECK(cmaes.config.population_size == 20);
        CHECK(cmaes.config.max_generations == 100);
        CHECK(cmaes.config.sigma0 == doctest::Approx(0.5));
        CHECK(cmaes.config.tolerance == doctest::Approx(1e-4));
    }

    SUBCASE("Track history") {
        meta::CMAES<double> cmaes;
        cmaes.config.max_generations = 50;
        cmaes.config.track_history = true;

        auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

        dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
        dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
        lower[0] = -5.0;
        lower[1] = -5.0;
        upper[0] = 5.0;
        upper[1] = 5.0;

        auto result = cmaes.optimize(sphere, lower, upper);

        CHECK(!result.history.empty());
        // History should be monotonically non-increasing (best value can only improve or stay same)
        for (std::size_t i = 1; i < result.history.size(); ++i) {
            CHECK(result.history[i] <= result.history[i - 1] + 1e-10);
        }
    }

    SUBCASE("Auto population size") {
        meta::CMAES<double> cmaes;
        cmaes.config.population_size = 0; // Auto
        cmaes.config.max_generations = 100;

        auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

        dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
        dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
        lower[0] = -5.0;
        lower[1] = -5.0;
        upper[0] = 5.0;
        upper[1] = 5.0;

        auto result = cmaes.optimize(sphere, lower, upper);

        // Should still work with auto population size
        CHECK(result.function_evaluations > 0);
    }
}

TEST_CASE("CMAES: Edge cases") {
    meta::CMAES<double> cmaes;

    SUBCASE("Empty bounds returns invalid result") {
        dp::mat::Vector<double, dp::mat::Dynamic> lower(0);
        dp::mat::Vector<double, dp::mat::Dynamic> upper(0);

        auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
            (void)x;
            return 0.0;
        };

        auto result = cmaes.optimize(sphere, lower, upper);
        CHECK(!result.converged);
    }

    SUBCASE("1D optimization") {
        auto quadratic = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) { return (x[0] - 3.0) * (x[0] - 3.0); };

        dp::mat::Vector<double, dp::mat::Dynamic> lower(1);
        dp::mat::Vector<double, dp::mat::Dynamic> upper(1);
        lower[0] = -10.0;
        upper[0] = 10.0;

        cmaes.config.max_generations = 200;

        auto result = cmaes.optimize(quadratic, lower, upper);

        CHECK(result.best_value < 0.1);
        CHECK(std::abs(result.best_position[0] - 3.0) < 0.5);
    }

    SUBCASE("Mismatched bounds size returns invalid") {
        dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
        dp::mat::Vector<double, dp::mat::Dynamic> upper(3);
        lower[0] = -5.0;
        lower[1] = -5.0;
        upper[0] = 5.0;
        upper[1] = 5.0;
        upper[2] = 5.0;

        auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
            (void)x;
            return 0.0;
        };

        auto result = cmaes.optimize(sphere, lower, upper);
        CHECK(!result.converged);
    }
}

TEST_CASE("CMAES: Optimize with initial point") {
    meta::CMAES<double> cmaes;
    cmaes.config.max_generations = 500;
    cmaes.config.tolerance = 1e-6;

    auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i];
        }
        return sum;
    };

    // Start near the optimum
    dp::mat::Vector<double, dp::mat::Dynamic> initial(2);
    initial[0] = 0.5;
    initial[1] = 0.5;

    dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    auto result = cmaes.optimize(sphere, initial, lower, upper);

    CHECK(result.best_value < 1e-4);
    CHECK(std::abs(result.best_position[0]) < 0.1);
    CHECK(std::abs(result.best_position[1]) < 0.1);
}

TEST_CASE("CMAES: Float type") {
    meta::CMAES<float> cmaes;
    cmaes.config.max_generations = 300;
    cmaes.config.tolerance = 1e-4f;

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

    auto result = cmaes.optimize(sphere, lower, upper);

    CHECK(result.best_value < 0.1f);
}

TEST_CASE("CMAES: Function evaluations tracking") {
    meta::CMAES<double> cmaes;
    cmaes.config.population_size = 10;
    cmaes.config.max_generations = 5;

    auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

    dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    auto result = cmaes.optimize(sphere, lower, upper);

    // Initial evaluation + lambda evaluations per generation
    // At least: 1 + generations * lambda
    std::size_t expected_min = 1 + result.generations * cmaes.config.population_size;
    CHECK(result.function_evaluations >= expected_min);
    CHECK(result.generations > 0);
}

TEST_CASE("CMAES: Schwefel function") {
    meta::CMAES<double> cmaes;
    cmaes.config.population_size = 30;
    cmaes.config.max_generations = 500;
    cmaes.config.tolerance = 1e-6;

    // Schwefel 2.22: f(x) = sum(|x_i|) + prod(|x_i|), minimum at origin
    auto schwefel = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
        double sum = 0.0;
        double prod = 1.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += std::abs(x[i]);
            prod *= std::abs(x[i]);
        }
        return sum + prod;
    };

    dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -10.0;
    lower[1] = -10.0;
    upper[0] = 10.0;
    upper[1] = 10.0;

    auto result = cmaes.optimize(schwefel, lower, upper);

    CHECK(result.best_value < 0.1);
}

TEST_CASE("CMAES: Ackley function") {
    meta::CMAES<double> cmaes;
    cmaes.config.population_size = 50;
    cmaes.config.max_generations = 500;
    cmaes.config.tolerance = 1e-6;

    // Ackley function: global minimum at origin = 0
    auto ackley = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
        const double a = 20.0;
        const double b = 0.2;
        const double c = 2.0 * 3.14159265358979323846;
        const std::size_t n = x.size();

        double sum1 = 0.0;
        double sum2 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            sum1 += x[i] * x[i];
            sum2 += std::cos(c * x[i]);
        }

        return -a * std::exp(-b * std::sqrt(sum1 / n)) - std::exp(sum2 / n) + a + std::exp(1.0);
    };

    dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    auto result = cmaes.optimize(ackley, lower, upper);

    // Ackley has many local minima, but CMA-ES should find a good solution
    CHECK(result.best_value < 1.0);
}

TEST_CASE("CMAES: Convergence detection") {
    meta::CMAES<double> cmaes;
    cmaes.config.max_generations = 1000;
    cmaes.config.tolerance = 1e-8;
    cmaes.config.patience = 20;

    // Simple quadratic - should converge quickly
    auto quadratic = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

    dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    auto result = cmaes.optimize(quadratic, lower, upper);

    // Should converge before max_generations
    CHECK(result.converged);
    CHECK(result.generations < cmaes.config.max_generations);
}
