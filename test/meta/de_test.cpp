#include <datapod/matrix.hpp>
#include <doctest/doctest.h>
#include <optinum/meta/de.hpp>

#include <cmath>

using namespace optinum;
namespace dp = datapod;

TEST_CASE("DE: Sphere function optimization") {
    meta::DifferentialEvolution<double> de;
    de.config.population_size = 30;
    de.config.max_generations = 500;
    de.config.tolerance = 1e-6;
    de.config.strategy = meta::DEStrategy::Best1;

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

    auto result = de.optimize(sphere, lower, upper);

    CHECK(result.converged);
    CHECK(result.best_value < 1e-4);
    CHECK(std::abs(result.best_position[0]) < 0.1);
    CHECK(std::abs(result.best_position[1]) < 0.1);
}

TEST_CASE("DE: Higher dimensional Sphere") {
    meta::DifferentialEvolution<double> de;
    de.config.population_size = 50;
    de.config.max_generations = 1000;
    de.config.tolerance = 1e-6;
    de.config.strategy = meta::DEStrategy::Best1;

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

    auto result = de.optimize(sphere, lower, upper);

    CHECK(result.best_value < 0.01);
    for (std::size_t i = 0; i < dim; ++i) {
        CHECK(std::abs(result.best_position[i]) < 0.5);
    }
}

TEST_CASE("DE: Rosenbrock function") {
    meta::DifferentialEvolution<double> de;
    de.config.population_size = 100;
    de.config.max_generations = 2000;
    de.config.tolerance = 1e-8;
    de.config.mutation_factor = 0.7;
    de.config.crossover_prob = 0.9;
    de.config.strategy = meta::DEStrategy::CurrentToBest1;

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

    auto result = de.optimize(rosenbrock, lower, upper);

    // Rosenbrock is harder - just check we get reasonably close
    CHECK(result.best_value < 1.0);
    CHECK(std::abs(result.best_position[0] - 1.0) < 1.0);
    CHECK(std::abs(result.best_position[1] - 1.0) < 1.0);
}

TEST_CASE("DE: Rastrigin function (multimodal)") {
    meta::DifferentialEvolution<double> de;
    de.config.population_size = 100;
    de.config.max_generations = 1000;
    de.config.tolerance = 1e-6;
    de.config.strategy = meta::DEStrategy::Rand1;

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

    auto result = de.optimize(rastrigin, lower, upper);

    // Rastrigin is very hard - just check we find a reasonable local minimum
    CHECK(result.best_value < 5.0);
}

TEST_CASE("DE: Configuration options") {
    SUBCASE("Custom config via constructor") {
        meta::DifferentialEvolution<double>::Config cfg;
        cfg.population_size = 20;
        cfg.max_generations = 100;
        cfg.mutation_factor = 0.5;
        cfg.crossover_prob = 0.8;
        cfg.strategy = meta::DEStrategy::Rand1;

        meta::DifferentialEvolution<double> de(cfg);

        CHECK(de.config.population_size == 20);
        CHECK(de.config.max_generations == 100);
        CHECK(de.config.mutation_factor == doctest::Approx(0.5));
        CHECK(de.config.crossover_prob == doctest::Approx(0.8));
        CHECK(de.config.strategy == meta::DEStrategy::Rand1);
    }

    SUBCASE("Track history") {
        meta::DifferentialEvolution<double> de;
        de.config.population_size = 20;
        de.config.max_generations = 50;
        de.config.track_history = true;

        auto sphere = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

        dp::mat::vector<double, dp::mat::Dynamic> lower(2);
        dp::mat::vector<double, dp::mat::Dynamic> upper(2);
        lower[0] = -5.0;
        lower[1] = -5.0;
        upper[0] = 5.0;
        upper[1] = 5.0;

        auto result = de.optimize(sphere, lower, upper);

        CHECK(!result.history.empty());
        // History should be monotonically non-increasing (best value can only improve or stay same)
        for (std::size_t i = 1; i < result.history.size(); ++i) {
            CHECK(result.history[i] <= result.history[i - 1] + 1e-10);
        }
    }
}

TEST_CASE("DE: Mutation strategies") {
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

    SUBCASE("Rand1 strategy") {
        meta::DifferentialEvolution<double> de;
        de.config.population_size = 30;
        de.config.max_generations = 500;
        de.config.strategy = meta::DEStrategy::Rand1;

        auto result = de.optimize(sphere, lower, upper);
        CHECK(result.best_value < 0.1);
    }

    SUBCASE("Best1 strategy") {
        meta::DifferentialEvolution<double> de;
        de.config.population_size = 30;
        de.config.max_generations = 500;
        de.config.strategy = meta::DEStrategy::Best1;

        auto result = de.optimize(sphere, lower, upper);
        CHECK(result.best_value < 0.1);
    }

    SUBCASE("CurrentToBest1 strategy") {
        meta::DifferentialEvolution<double> de;
        de.config.population_size = 30;
        de.config.max_generations = 500;
        de.config.strategy = meta::DEStrategy::CurrentToBest1;

        auto result = de.optimize(sphere, lower, upper);
        CHECK(result.best_value < 0.1);
    }
}

TEST_CASE("DE: Edge cases") {
    meta::DifferentialEvolution<double> de;

    SUBCASE("Empty bounds returns invalid result") {
        dp::mat::vector<double, dp::mat::Dynamic> lower(0);
        dp::mat::vector<double, dp::mat::Dynamic> upper(0);

        auto sphere = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) {
            (void)x;
            return 0.0;
        };

        auto result = de.optimize(sphere, lower, upper);
        CHECK(!result.converged);
    }

    SUBCASE("Population too small returns invalid result") {
        de.config.population_size = 3; // Need at least 4

        dp::mat::vector<double, dp::mat::Dynamic> lower(2);
        dp::mat::vector<double, dp::mat::Dynamic> upper(2);
        lower[0] = -5.0;
        lower[1] = -5.0;
        upper[0] = 5.0;
        upper[1] = 5.0;

        auto sphere = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

        auto result = de.optimize(sphere, lower, upper);
        CHECK(!result.converged);
    }

    SUBCASE("1D optimization") {
        auto quadratic = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) { return (x[0] - 3.0) * (x[0] - 3.0); };

        dp::mat::vector<double, dp::mat::Dynamic> lower(1);
        dp::mat::vector<double, dp::mat::Dynamic> upper(1);
        lower[0] = -10.0;
        upper[0] = 10.0;

        de.config.population_size = 20;
        de.config.max_generations = 200;

        auto result = de.optimize(quadratic, lower, upper);

        CHECK(result.best_value < 0.01);
        CHECK(std::abs(result.best_position[0] - 3.0) < 0.2);
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

        auto result = de.optimize(sphere, lower, upper);
        CHECK(!result.converged);
    }
}

TEST_CASE("DE: Optimize with initial point") {
    meta::DifferentialEvolution<double> de;
    de.config.population_size = 30;
    de.config.max_generations = 500;
    de.config.tolerance = 1e-6;

    auto sphere = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i];
        }
        return sum;
    };

    dp::mat::vector<double, dp::mat::Dynamic> initial(2);
    initial[0] = 1.0;
    initial[1] = 1.0;

    dp::mat::vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    auto result = de.optimize(sphere, initial, lower, upper);

    CHECK(result.best_value < 1e-4);
    CHECK(std::abs(result.best_position[0]) < 0.1);
    CHECK(std::abs(result.best_position[1]) < 0.1);
}

TEST_CASE("DE: Float type") {
    meta::DifferentialEvolution<float> de;
    de.config.population_size = 30;
    de.config.max_generations = 300;
    de.config.tolerance = 1e-4f;

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

    auto result = de.optimize(sphere, lower, upper);

    CHECK(result.best_value < 0.01f);
}

TEST_CASE("DE: Function evaluations tracking") {
    meta::DifferentialEvolution<double> de;
    de.config.population_size = 20;
    de.config.max_generations = 10;

    auto sphere = [](const dp::mat::vector<double, dp::mat::Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

    dp::mat::vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    auto result = de.optimize(sphere, lower, upper);

    // Initial population: pop_size evaluations
    // Each generation: pop_size evaluations (one per trial)
    // Total: pop_size + generations * pop_size
    std::size_t expected_min = de.config.population_size; // At least initial population
    CHECK(result.function_evaluations >= expected_min);
    CHECK(result.generations > 0);
}
