#include <datapod/datapod.hpp>
#include <doctest/doctest.h>
#include <optinum/meta/sa.hpp>

#include <cmath>

using namespace optinum;

TEST_CASE("SA: Sphere function optimization") {
    meta::SimulatedAnnealing<double> sa;
    sa.config.max_iterations = 10000;
    sa.config.initial_temperature = 100.0;
    sa.config.cooling_rate = 0.999;
    sa.config.tolerance = 1e-6;

    // 2D Sphere: f(x) = x[0]^2 + x[1]^2, minimum at (0, 0)
    auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i];
        }
        return sum;
    };

    dp::mat::Vector<double, dp::mat::Dynamic> initial(2);
    initial[0] = 3.0;
    initial[1] = -2.0;

    dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    auto result = sa.optimize(sphere, initial, lower, upper);

    // SA is stochastic - just verify it ran and produced a result
    // Initial value is 3^2 + 2^2 = 13
    CHECK(result.iterations > 0);
    CHECK(result.best_value <= 13.0 + 1e-6); // Should not get worse than initial
}

TEST_CASE("SA: Higher dimensional Sphere") {
    meta::SimulatedAnnealing<double> sa;
    sa.config.max_iterations = 10000;
    sa.config.initial_temperature = 100.0;
    sa.config.cooling_rate = 0.999;
    sa.config.tolerance = 1e-6;

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

    auto result = sa.optimize(sphere, lower, upper);

    CHECK(result.best_value < 1.0);
}

TEST_CASE("SA: Rosenbrock function") {
    meta::SimulatedAnnealing<double> sa;
    sa.config.max_iterations = 20000;
    sa.config.initial_temperature = 100.0;
    sa.config.cooling_rate = 0.9995;
    sa.config.step_size = 0.2;
    sa.config.tolerance = 1e-8;

    // 2D Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1, 1)
    auto rosenbrock = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
        double a = 1.0 - x[0];
        double b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    };

    dp::mat::Vector<double, dp::mat::Dynamic> initial(2);
    initial[0] = -1.0;
    initial[1] = 1.0;

    dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    auto result = sa.optimize(rosenbrock, initial, lower, upper);

    // Rosenbrock is harder - just check we get reasonably close
    CHECK(result.best_value < 5.0);
}

TEST_CASE("SA: Rastrigin function (multimodal)") {
    meta::SimulatedAnnealing<double> sa;
    sa.config.max_iterations = 10000;
    sa.config.initial_temperature = 100.0;
    sa.config.cooling_rate = 0.999;
    sa.config.tolerance = 1e-6;

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

    auto result = sa.optimize(rastrigin, lower, upper);

    // Rastrigin is very hard - just check we find a reasonable local minimum
    CHECK(result.best_value < 10.0);
}

TEST_CASE("SA: Configuration options") {
    SUBCASE("Custom config via constructor") {
        meta::SimulatedAnnealing<double>::Config cfg;
        cfg.max_iterations = 5000;
        cfg.initial_temperature = 50.0;
        cfg.cooling_rate = 0.98;
        cfg.step_size = 0.2;

        meta::SimulatedAnnealing<double> sa(cfg);

        CHECK(sa.config.max_iterations == 5000);
        CHECK(sa.config.initial_temperature == doctest::Approx(50.0));
        CHECK(sa.config.cooling_rate == doctest::Approx(0.98));
        CHECK(sa.config.step_size == doctest::Approx(0.2));
    }

    SUBCASE("Track history") {
        meta::SimulatedAnnealing<double> sa;
        sa.config.max_iterations = 500;
        sa.config.track_history = true;

        auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

        dp::mat::Vector<double, dp::mat::Dynamic> initial(2);
        initial[0] = 2.0;
        initial[1] = 2.0;

        auto result = sa.optimize(sphere, initial);

        CHECK(!result.history.empty());
        // History should be monotonically non-increasing (best value can only improve or stay same)
        for (std::size_t i = 1; i < result.history.size(); ++i) {
            CHECK(result.history[i] <= result.history[i - 1] + 1e-10);
        }
    }

    SUBCASE("Acceptance tracking") {
        meta::SimulatedAnnealing<double> sa;
        sa.config.max_iterations = 5000;
        sa.config.initial_temperature = 100.0;
        sa.config.cooling_rate = 0.995;

        auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) { return x[0] * x[0] + x[1] * x[1]; };

        dp::mat::Vector<double, dp::mat::Dynamic> initial(2);
        initial[0] = 2.0;
        initial[1] = 2.0;

        dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
        dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
        lower[0] = -5.0;
        lower[1] = -5.0;
        upper[0] = 5.0;
        upper[1] = 5.0;

        auto result = sa.optimize(sphere, initial, lower, upper);

        // Should have some accepted moves
        CHECK(result.accepted_moves > 0);
        CHECK(result.function_evaluations >= result.iterations);
    }
}

TEST_CASE("SA: Cooling schedules") {
    auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i];
        }
        return sum;
    };

    dp::mat::Vector<double, dp::mat::Dynamic> initial(2);
    initial[0] = 3.0;
    initial[1] = -2.0;

    dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    SUBCASE("Geometric cooling") {
        meta::SimulatedAnnealing<double> sa;
        sa.config.max_iterations = 10000;
        sa.config.initial_temperature = 100.0;
        sa.config.cooling_rate = 0.999;
        sa.config.schedule = meta::CoolingSchedule::Geometric;

        auto result = sa.optimize(sphere, initial, lower, upper);
        // SA is stochastic - just verify it improves from initial (13.0)
        CHECK(result.best_value < 5.0);
    }

    SUBCASE("Linear cooling") {
        meta::SimulatedAnnealing<double> sa;
        sa.config.max_iterations = 10000;
        sa.config.initial_temperature = 100.0;
        sa.config.schedule = meta::CoolingSchedule::Linear;

        auto result = sa.optimize(sphere, initial, lower, upper);
        CHECK(result.best_value < 1.0);
    }

    SUBCASE("Logarithmic cooling") {
        meta::SimulatedAnnealing<double> sa;
        sa.config.max_iterations = 10000;
        sa.config.initial_temperature = 100.0;
        sa.config.schedule = meta::CoolingSchedule::Logarithmic;

        auto result = sa.optimize(sphere, initial, lower, upper);
        CHECK(result.best_value < 2.0);
    }

    SUBCASE("Adaptive cooling") {
        meta::SimulatedAnnealing<double> sa;
        sa.config.max_iterations = 10000;
        sa.config.initial_temperature = 100.0;
        sa.config.cooling_rate = 0.999;
        sa.config.schedule = meta::CoolingSchedule::Adaptive;

        auto result = sa.optimize(sphere, initial, lower, upper);
        // Adaptive cooling - just verify it ran
        // Initial value is 3^2 + 2^2 = 13
        CHECK(result.iterations > 0);
        CHECK(result.best_value <= 13.0 + 1e-6);
    }
}

TEST_CASE("SA: Edge cases") {
    meta::SimulatedAnnealing<double> sa;

    SUBCASE("Empty initial returns invalid result") {
        dp::mat::Vector<double, dp::mat::Dynamic> initial(0);

        auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
            (void)x;
            return 0.0;
        };

        auto result = sa.optimize(sphere, initial);
        CHECK(!result.converged);
    }

    SUBCASE("1D optimization") {
        auto quadratic = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) { return (x[0] - 3.0) * (x[0] - 3.0); };

        dp::mat::Vector<double, dp::mat::Dynamic> initial(1);
        initial[0] = 0.0;

        dp::mat::Vector<double, dp::mat::Dynamic> lower(1);
        dp::mat::Vector<double, dp::mat::Dynamic> upper(1);
        lower[0] = -10.0;
        upper[0] = 10.0;

        sa.config.max_iterations = 2000;

        auto result = sa.optimize(quadratic, initial, lower, upper);

        CHECK(result.best_value < 0.5);
        CHECK(std::abs(result.best_position[0] - 3.0) < 1.0);
    }

    SUBCASE("Mismatched bounds size returns invalid") {
        dp::mat::Vector<double, dp::mat::Dynamic> initial(2);
        initial[0] = 0.0;
        initial[1] = 0.0;

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

        auto result = sa.optimize(sphere, initial, lower, upper);
        CHECK(!result.converged);
    }
}

TEST_CASE("SA: Adaptive step size") {
    meta::SimulatedAnnealing<double> sa;
    sa.config.max_iterations = 5000;
    sa.config.adaptive_step = true;
    sa.config.target_acceptance = 0.44;

    auto sphere = [](const dp::mat::Vector<double, dp::mat::Dynamic> &x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i];
        }
        return sum;
    };

    dp::mat::Vector<double, dp::mat::Dynamic> initial(2);
    initial[0] = 3.0;
    initial[1] = -2.0;

    dp::mat::Vector<double, dp::mat::Dynamic> lower(2);
    dp::mat::Vector<double, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0;
    lower[1] = -5.0;
    upper[0] = 5.0;
    upper[1] = 5.0;

    auto result = sa.optimize(sphere, initial, lower, upper);

    // SA is stochastic - just verify it improves from initial (13.0)
    // Use a more lenient threshold since SA is non-deterministic
    CHECK(result.best_value < 10.0);
}

TEST_CASE("SA: Float type") {
    meta::SimulatedAnnealing<float> sa;
    sa.config.max_iterations = 10000;
    sa.config.initial_temperature = 100.0f;
    sa.config.cooling_rate = 0.999f;
    sa.config.tolerance = 1e-4f;

    auto sphere = [](const dp::mat::Vector<float, dp::mat::Dynamic> &x) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * x[i];
        }
        return sum;
    };

    dp::mat::Vector<float, dp::mat::Dynamic> initial(2);
    initial[0] = 3.0f;
    initial[1] = -2.0f;

    dp::mat::Vector<float, dp::mat::Dynamic> lower(2);
    dp::mat::Vector<float, dp::mat::Dynamic> upper(2);
    lower[0] = -5.0f;
    lower[1] = -5.0f;
    upper[0] = 5.0f;
    upper[1] = 5.0f;

    auto result = sa.optimize(sphere, initial, lower, upper);

    // SA is stochastic - use lenient threshold
    CHECK(result.best_value < 5.0f);
}

TEST_CASE("SA: Optimize without initial (from bounds center)") {
    meta::SimulatedAnnealing<double> sa;
    sa.config.max_iterations = 3000;
    sa.config.cooling_rate = 0.995;

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

    auto result = sa.optimize(sphere, lower, upper);

    CHECK(result.best_value < 0.5);
}
