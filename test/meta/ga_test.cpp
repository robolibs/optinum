#include <doctest/doctest.h>
#include <optinum/meta/ga.hpp>

#include <cmath>

using namespace optinum;

TEST_CASE("GA: Sphere function optimization") {
    meta::GeneticAlgorithm<double> ga;
    ga.config.population_size = 50;
    ga.config.max_generations = 500;
    ga.config.tolerance = 1e-6;

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

    auto result = ga.optimize(sphere, lower, upper);

    CHECK(result.best_value < 0.1);
    CHECK(std::abs(result.best_position[0]) < 0.5);
    CHECK(std::abs(result.best_position[1]) < 0.5);
}

TEST_CASE("GA: Higher dimensional Sphere") {
    meta::GeneticAlgorithm<double> ga;
    ga.config.population_size = 100;
    ga.config.max_generations = 1000;
    ga.config.tolerance = 1e-6;

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

    auto result = ga.optimize(sphere, lower, upper);

    CHECK(result.best_value < 1.0);
}

TEST_CASE("GA: Rosenbrock function") {
    meta::GeneticAlgorithm<double> ga;
    ga.config.population_size = 100;
    ga.config.max_generations = 2000;
    ga.config.tolerance = 1e-8;
    ga.config.crossover_prob = 0.9;
    ga.config.mutation_prob = 0.1;

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

    auto result = ga.optimize(rosenbrock, lower, upper);

    // Rosenbrock is harder - just check we get reasonably close
    CHECK(result.best_value < 5.0);
}

TEST_CASE("GA: Selection strategies") {
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

    SUBCASE("Tournament selection") {
        meta::GeneticAlgorithm<double> ga;
        ga.config.population_size = 50;
        ga.config.max_generations = 300;
        ga.config.selection = meta::GASelection::Tournament;
        ga.config.tournament_size = 3;

        auto result = ga.optimize(sphere, lower, upper);
        CHECK(result.best_value < 0.5);
    }

    SUBCASE("Roulette wheel selection") {
        meta::GeneticAlgorithm<double> ga;
        ga.config.population_size = 50;
        ga.config.max_generations = 300;
        ga.config.selection = meta::GASelection::RouletteWheel;

        auto result = ga.optimize(sphere, lower, upper);
        CHECK(result.best_value < 0.5);
    }

    SUBCASE("Rank-based selection") {
        meta::GeneticAlgorithm<double> ga;
        ga.config.population_size = 50;
        ga.config.max_generations = 300;
        ga.config.selection = meta::GASelection::Rank;

        auto result = ga.optimize(sphere, lower, upper);
        CHECK(result.best_value < 0.5);
    }
}

TEST_CASE("GA: Crossover operators") {
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

    SUBCASE("SBX crossover") {
        meta::GeneticAlgorithm<double> ga;
        ga.config.population_size = 50;
        ga.config.max_generations = 300;
        ga.config.crossover = meta::GACrossover::SBX;

        auto result = ga.optimize(sphere, lower, upper);
        CHECK(result.best_value < 0.5);
    }

    SUBCASE("Uniform crossover") {
        meta::GeneticAlgorithm<double> ga;
        ga.config.population_size = 50;
        ga.config.max_generations = 300;
        ga.config.crossover = meta::GACrossover::Uniform;

        auto result = ga.optimize(sphere, lower, upper);
        CHECK(result.best_value < 0.5);
    }

    SUBCASE("Single-point crossover") {
        meta::GeneticAlgorithm<double> ga;
        ga.config.population_size = 50;
        ga.config.max_generations = 300;
        ga.config.crossover = meta::GACrossover::SinglePoint;

        auto result = ga.optimize(sphere, lower, upper);
        CHECK(result.best_value < 0.5);
    }
}

TEST_CASE("GA: Mutation operators") {
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

    SUBCASE("Gaussian mutation") {
        meta::GeneticAlgorithm<double> ga;
        ga.config.population_size = 50;
        ga.config.max_generations = 300;
        ga.config.mutation = meta::GAMutation::Gaussian;
        ga.config.mutation_strength = 0.1;

        auto result = ga.optimize(sphere, lower, upper);
        CHECK(result.best_value < 0.5);
    }

    SUBCASE("Polynomial mutation") {
        meta::GeneticAlgorithm<double> ga;
        ga.config.population_size = 50;
        ga.config.max_generations = 300;
        ga.config.mutation = meta::GAMutation::Polynomial;

        auto result = ga.optimize(sphere, lower, upper);
        CHECK(result.best_value < 0.5);
    }
}

TEST_CASE("GA: Configuration options") {
    SUBCASE("Custom config via constructor") {
        meta::GeneticAlgorithm<double>::Config cfg;
        cfg.population_size = 80;
        cfg.max_generations = 200;
        cfg.crossover_prob = 0.85;
        cfg.mutation_prob = 0.15;
        cfg.elitism = 4;

        meta::GeneticAlgorithm<double> ga(cfg);

        CHECK(ga.config.population_size == 80);
        CHECK(ga.config.max_generations == 200);
        CHECK(ga.config.crossover_prob == doctest::Approx(0.85));
        CHECK(ga.config.mutation_prob == doctest::Approx(0.15));
        CHECK(ga.config.elitism == 4);
    }

    SUBCASE("Track history") {
        meta::GeneticAlgorithm<double> ga;
        ga.config.population_size = 30;
        ga.config.max_generations = 50;
        ga.config.track_history = true;

        auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

        dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
        dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
        lower[0] = -5.0;
        lower[1] = -5.0;
        upper[0] = 5.0;
        upper[1] = 5.0;

        auto result = ga.optimize(sphere, lower, upper);

        CHECK(!result.history.empty());
        // History should be monotonically non-increasing (best value can only improve or stay same)
        for (std::size_t i = 1; i < result.history.size(); ++i) {
            CHECK(result.history[i] <= result.history[i - 1] + 1e-10);
        }
    }

    SUBCASE("Elitism preserves best") {
        meta::GeneticAlgorithm<double> ga;
        ga.config.population_size = 30;
        ga.config.max_generations = 20;
        ga.config.elitism = 5;
        ga.config.track_history = true;

        auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

        dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
        dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
        lower[0] = -5.0;
        lower[1] = -5.0;
        upper[0] = 5.0;
        upper[1] = 5.0;

        auto result = ga.optimize(sphere, lower, upper);

        // With elitism, best should never get worse
        for (std::size_t i = 1; i < result.history.size(); ++i) {
            CHECK(result.history[i] <= result.history[i - 1] + 1e-10);
        }
    }
}

TEST_CASE("GA: Edge cases") {
    meta::GeneticAlgorithm<double> ga;

    SUBCASE("Empty bounds returns invalid result") {
        dp::mat::Vector<double, dp::mat::Dynamic> lower(0);
        dp::mat::Vector<double, dp::mat::Dynamic> upper(0);

        auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
            (void)x;
            return 0.0;
        };

        auto result = ga.optimize(sphere, lower, upper);
        CHECK(!result.converged);
    }

    SUBCASE("Population too small returns invalid result") {
        ga.config.population_size = 3; // Need at least 4

        dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
        dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
        lower[0] = -5.0;
        lower[1] = -5.0;
        upper[0] = 5.0;
        upper[1] = 5.0;

        auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

        auto result = ga.optimize(sphere, lower, upper);
        CHECK(!result.converged);
    }

    SUBCASE("1D optimization") {
        auto quadratic = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) { return (x[0] - 3.0) * (x[0] - 3.0); };

        dp::mat::Vector<double, dp::mat::Dynamic> lower(1);
        dp::mat::Vector<double, dp::mat::Dynamic> upper(1);
        lower[0] = -10.0;
        upper[0] = 10.0;

        ga.config.population_size = 30;
        ga.config.max_generations = 200;

        auto result = ga.optimize(quadratic, lower, upper);

        CHECK(result.best_value < 0.5);
        CHECK(std::abs(result.best_position[0] - 3.0) < 1.0);
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

        auto result = ga.optimize(sphere, lower, upper);
        CHECK(!result.converged);
    }
}

TEST_CASE("GA: Optimize with initial point") {
    meta::GeneticAlgorithm<double> ga;
    ga.config.population_size = 50;
    ga.config.max_generations = 300;
    ga.config.tolerance = 1e-6;

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

    auto result = ga.optimize(sphere, initial, lower, upper);

    CHECK(result.best_value < 0.1);
}

TEST_CASE("GA: Float type") {
    meta::GeneticAlgorithm<float> ga;
    ga.config.population_size = 50;
    ga.config.max_generations = 300;
    ga.config.tolerance = 1e-4f;

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

    auto result = ga.optimize(sphere, lower, upper);

    CHECK(result.best_value < 0.5f);
}

TEST_CASE("GA: Function evaluations tracking") {
    meta::GeneticAlgorithm<double> ga;
    ga.config.population_size = 20;
    ga.config.max_generations = 10;
    ga.config.elitism = 2;

    auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

    dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    auto result = ga.optimize(sphere, lower, upper);

    // Initial population: pop_size evaluations
    // Each generation: (pop_size - elitism) evaluations
    std::size_t expected_min = ga.config.population_size; // At least initial population
    CHECK(result.function_evaluations >= expected_min);
    CHECK(result.generations > 0);
}

TEST_CASE("GA: Rastrigin function (multimodal)") {
    meta::GeneticAlgorithm<double> ga;
    ga.config.population_size = 100;
    ga.config.max_generations = 500;
    ga.config.tolerance = 1e-6;

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

    auto result = ga.optimize(rastrigin, lower, upper);

    // Rastrigin is very hard - just check we find a reasonable local minimum
    CHECK(result.best_value < 10.0);
}
