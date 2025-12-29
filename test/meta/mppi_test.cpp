#include <datapod/matrix.hpp>
#include <doctest/doctest.h>
#include <optinum/meta/mppi.hpp>

#include <cmath>

using namespace optinum;
namespace dp = datapod;

TEST_CASE("MPPI: Simple 1D integrator") {
    // Simple 1D system: x_{t+1} = x_t + u_t * dt
    // Goal: drive state to zero from x0 = 5.0

    meta::MPPI<double> mppi;
    mppi.config.num_samples = 500;
    mppi.config.horizon = 20;
    mppi.config.control_dim = 1;
    mppi.config.lambda = 1.0;
    mppi.config.noise_sigma = 2.0;
    mppi.config.dt = 0.1;

    // Set control bounds
    mppi.bounds.lower = dp::mat::vector<double, dp::mat::Dynamic>(1);
    mppi.bounds.upper = dp::mat::vector<double, dp::mat::Dynamic>(1);
    mppi.bounds.lower[0] = -10.0;
    mppi.bounds.upper[0] = 10.0;

    mppi.initialize();

    // Dynamics: simple integrator
    auto dynamics = [](const dp::mat::vector<double, dp::mat::Dynamic> &state,
                       const dp::mat::vector<double, dp::mat::Dynamic> &control) {
        dp::mat::vector<double, dp::mat::Dynamic> next_state(state.size());
        next_state[0] = state[0] + control[0] * 0.1; // dt = 0.1
        return next_state;
    };

    // Cost: quadratic state cost + small control cost
    auto cost = [](const dp::mat::vector<double, dp::mat::Dynamic> &state,
                   const dp::mat::vector<double, dp::mat::Dynamic> &control) {
        return state[0] * state[0] + 0.01 * control[0] * control[0];
    };

    // Initial state
    dp::mat::vector<double, dp::mat::Dynamic> state(1);
    state[0] = 5.0;

    // Run MPPI for several steps
    for (int i = 0; i < 50; ++i) {
        auto result = mppi.step(dynamics, cost, state);
        CHECK(result.valid);

        // Apply control
        state[0] += result.optimal_control[0] * 0.1;
    }

    // State should be close to zero
    CHECK(std::abs(state[0]) < 1.0);
}

TEST_CASE("MPPI: 2D double integrator") {
    // 2D system: position and velocity
    // x = [pos, vel], u = [acceleration]
    // Dynamics: pos' = pos + vel*dt, vel' = vel + u*dt

    meta::MPPI<double> mppi;
    mppi.config.num_samples = 1000;
    mppi.config.horizon = 30;
    mppi.config.control_dim = 1;
    mppi.config.lambda = 0.5;
    mppi.config.noise_sigma = 3.0;
    mppi.config.dt = 0.05;

    mppi.bounds.lower = dp::mat::vector<double, dp::mat::Dynamic>(1);
    mppi.bounds.upper = dp::mat::vector<double, dp::mat::Dynamic>(1);
    mppi.bounds.lower[0] = -5.0;
    mppi.bounds.upper[0] = 5.0;

    mppi.initialize();

    const double dt = 0.05;

    auto dynamics = [dt](const dp::mat::vector<double, dp::mat::Dynamic> &state,
                         const dp::mat::vector<double, dp::mat::Dynamic> &control) {
        dp::mat::vector<double, dp::mat::Dynamic> next_state(2);
        next_state[0] = state[0] + state[1] * dt;   // pos += vel * dt
        next_state[1] = state[1] + control[0] * dt; // vel += acc * dt
        return next_state;
    };

    auto cost = [](const dp::mat::vector<double, dp::mat::Dynamic> &state,
                   const dp::mat::vector<double, dp::mat::Dynamic> &control) {
        // Penalize distance from origin and velocity, plus control effort
        return 10.0 * state[0] * state[0] + 5.0 * state[1] * state[1] + 0.1 * control[0] * control[0];
    };

    // Initial state: position = 3, velocity = 2
    dp::mat::vector<double, dp::mat::Dynamic> state(2);
    state[0] = 3.0;
    state[1] = 2.0;

    // Run MPPI
    for (int i = 0; i < 100; ++i) {
        auto result = mppi.step(dynamics, cost, state);
        CHECK(result.valid);

        // Apply control
        state[0] += state[1] * dt;
        state[1] += result.optimal_control[0] * dt;
    }

    // Should be close to origin with low velocity
    CHECK(std::abs(state[0]) < 1.0);
    CHECK(std::abs(state[1]) < 1.0);
}

TEST_CASE("MPPI: Configuration options") {
    SUBCASE("Custom config via constructor") {
        meta::MPPI<double>::Config cfg;
        cfg.num_samples = 500;
        cfg.horizon = 25;
        cfg.lambda = 2.0;
        cfg.noise_sigma = 1.5;

        meta::MPPI<double> mppi(cfg);

        CHECK(mppi.config.num_samples == 500);
        CHECK(mppi.config.horizon == 25);
        CHECK(mppi.config.lambda == doctest::Approx(2.0));
        CHECK(mppi.config.noise_sigma == doctest::Approx(1.5));
    }

    SUBCASE("Bounds validation") {
        meta::MPPI<double> mppi;

        // Empty bounds should be invalid
        CHECK(!mppi.bounds.valid());

        // Set bounds
        mppi.bounds.lower = dp::mat::vector<double, dp::mat::Dynamic>(2);
        mppi.bounds.upper = dp::mat::vector<double, dp::mat::Dynamic>(2);
        mppi.bounds.lower[0] = -1.0;
        mppi.bounds.lower[1] = -2.0;
        mppi.bounds.upper[0] = 1.0;
        mppi.bounds.upper[1] = 2.0;

        CHECK(mppi.bounds.valid());
    }

    SUBCASE("Track history") {
        meta::MPPI<double> mppi;
        mppi.config.num_samples = 100;
        mppi.config.horizon = 10;
        mppi.config.control_dim = 1;
        mppi.config.track_history = true;
        mppi.initialize();

        auto dynamics = [](const dp::mat::vector<double, dp::mat::Dynamic> &state,
                           const dp::mat::vector<double, dp::mat::Dynamic> &control) {
            dp::mat::vector<double, dp::mat::Dynamic> next(1);
            next[0] = state[0] + control[0] * 0.1;
            return next;
        };

        auto cost = [](const dp::mat::vector<double, dp::mat::Dynamic> &state,
                       const dp::mat::vector<double, dp::mat::Dynamic> &control) {
            return state[0] * state[0] + control[0] * control[0];
        };

        dp::mat::vector<double, dp::mat::Dynamic> state(1);
        state[0] = 1.0;

        // Run several steps
        for (int i = 0; i < 5; ++i) {
            mppi.step(dynamics, cost, state);
        }

        CHECK(mppi.get_history().size() == 5);
    }
}

TEST_CASE("MPPI: Edge cases") {
    meta::MPPI<double> mppi;

    SUBCASE("Zero samples returns invalid") {
        mppi.config.num_samples = 0;
        mppi.config.horizon = 10;
        mppi.config.control_dim = 1;
        mppi.initialize();

        auto dynamics = [](const dp::mat::vector<double, dp::mat::Dynamic> &state,
                           const dp::mat::vector<double, dp::mat::Dynamic> &) { return state; };
        auto cost = [](const dp::mat::vector<double, dp::mat::Dynamic> &,
                       const dp::mat::vector<double, dp::mat::Dynamic> &) { return 0.0; };

        dp::mat::vector<double, dp::mat::Dynamic> state(1);
        state[0] = 0.0;

        auto result = mppi.step(dynamics, cost, state);
        CHECK(!result.valid);
    }

    SUBCASE("Zero horizon returns invalid") {
        mppi.config.num_samples = 100;
        mppi.config.horizon = 0;
        mppi.config.control_dim = 1;
        mppi.initialize();

        auto dynamics = [](const dp::mat::vector<double, dp::mat::Dynamic> &state,
                           const dp::mat::vector<double, dp::mat::Dynamic> &) { return state; };
        auto cost = [](const dp::mat::vector<double, dp::mat::Dynamic> &,
                       const dp::mat::vector<double, dp::mat::Dynamic> &) { return 0.0; };

        dp::mat::vector<double, dp::mat::Dynamic> state(1);
        state[0] = 0.0;

        auto result = mppi.step(dynamics, cost, state);
        CHECK(!result.valid);
    }

    SUBCASE("Reset clears control sequence") {
        mppi.config.num_samples = 100;
        mppi.config.horizon = 10;
        mppi.config.control_dim = 1;
        mppi.initialize();

        auto dynamics = [](const dp::mat::vector<double, dp::mat::Dynamic> &state,
                           const dp::mat::vector<double, dp::mat::Dynamic> &control) {
            dp::mat::vector<double, dp::mat::Dynamic> next(1);
            next[0] = state[0] + control[0];
            return next;
        };
        auto cost = [](const dp::mat::vector<double, dp::mat::Dynamic> &state,
                       const dp::mat::vector<double, dp::mat::Dynamic> &) { return state[0] * state[0]; };

        dp::mat::vector<double, dp::mat::Dynamic> state(1);
        state[0] = 5.0;

        // Run a few steps to build up control sequence
        for (int i = 0; i < 3; ++i) {
            mppi.step(dynamics, cost, state);
        }

        // Reset
        mppi.reset();

        // Control sequence should be zeros
        const auto &controls = mppi.get_control_sequence();
        for (const auto &u : controls) {
            CHECK(std::abs(u[0]) < 1e-10);
        }
    }
}

TEST_CASE("MPPI: Multiple iterations") {
    meta::MPPI<double> mppi;
    mppi.config.num_samples = 200;
    mppi.config.horizon = 15;
    mppi.config.control_dim = 1;
    mppi.config.lambda = 1.0;
    mppi.config.noise_sigma = 2.0;
    mppi.initialize();

    auto dynamics = [](const dp::mat::vector<double, dp::mat::Dynamic> &state,
                       const dp::mat::vector<double, dp::mat::Dynamic> &control) {
        dp::mat::vector<double, dp::mat::Dynamic> next(1);
        next[0] = state[0] + control[0] * 0.1;
        return next;
    };

    auto cost = [](const dp::mat::vector<double, dp::mat::Dynamic> &state,
                   const dp::mat::vector<double, dp::mat::Dynamic> &control) {
        return state[0] * state[0] + 0.01 * control[0] * control[0];
    };

    dp::mat::vector<double, dp::mat::Dynamic> state(1);
    state[0] = 3.0;

    // Use optimize with multiple iterations
    auto result = mppi.optimize(dynamics, cost, state, 3);

    CHECK(result.valid);
    CHECK(result.iterations == 3);
}

TEST_CASE("MPPI: Float type") {
    meta::MPPI<float> mppi;
    mppi.config.num_samples = 200;
    mppi.config.horizon = 15;
    mppi.config.control_dim = 1;
    mppi.config.lambda = 1.0f;
    mppi.config.noise_sigma = 2.0f;
    mppi.config.dt = 0.1f;
    mppi.initialize();

    auto dynamics = [](const dp::mat::vector<float, dp::mat::Dynamic> &state,
                       const dp::mat::vector<float, dp::mat::Dynamic> &control) {
        dp::mat::vector<float, dp::mat::Dynamic> next(1);
        next[0] = state[0] + control[0] * 0.1f;
        return next;
    };

    auto cost = [](const dp::mat::vector<float, dp::mat::Dynamic> &state,
                   const dp::mat::vector<float, dp::mat::Dynamic> &control) {
        return state[0] * state[0] + 0.01f * control[0] * control[0];
    };

    dp::mat::vector<float, dp::mat::Dynamic> state(1);
    state[0] = 3.0f;

    auto result = mppi.step(dynamics, cost, state);
    CHECK(result.valid);
}

TEST_CASE("MPPI: Control bounds enforcement") {
    meta::MPPI<double> mppi;
    mppi.config.num_samples = 500;
    mppi.config.horizon = 20;
    mppi.config.control_dim = 1;
    mppi.config.lambda = 0.1;       // Low temperature = more greedy
    mppi.config.noise_sigma = 10.0; // High noise to test clamping
    mppi.config.dt = 0.1;

    // Tight bounds
    mppi.bounds.lower = dp::mat::vector<double, dp::mat::Dynamic>(1);
    mppi.bounds.upper = dp::mat::vector<double, dp::mat::Dynamic>(1);
    mppi.bounds.lower[0] = -1.0;
    mppi.bounds.upper[0] = 1.0;

    mppi.initialize();

    auto dynamics = [](const dp::mat::vector<double, dp::mat::Dynamic> &state,
                       const dp::mat::vector<double, dp::mat::Dynamic> &control) {
        dp::mat::vector<double, dp::mat::Dynamic> next(1);
        next[0] = state[0] + control[0] * 0.1;
        return next;
    };

    // Cost that would want very large controls
    auto cost = [](const dp::mat::vector<double, dp::mat::Dynamic> &state,
                   const dp::mat::vector<double, dp::mat::Dynamic> &) {
        return 1000.0 * state[0] * state[0]; // Very high state penalty
    };

    dp::mat::vector<double, dp::mat::Dynamic> state(1);
    state[0] = 10.0;

    auto result = mppi.step(dynamics, cost, state);
    CHECK(result.valid);

    // Control should be within bounds
    CHECK(result.optimal_control[0] >= -1.0 - 1e-6);
    CHECK(result.optimal_control[0] <= 1.0 + 1e-6);
}

TEST_CASE("MPPI: Warm start behavior") {
    meta::MPPI<double> mppi;
    mppi.config.num_samples = 100;
    mppi.config.horizon = 10;
    mppi.config.control_dim = 1;
    mppi.config.warm_start = true;
    mppi.initialize();

    auto dynamics = [](const dp::mat::vector<double, dp::mat::Dynamic> &state,
                       const dp::mat::vector<double, dp::mat::Dynamic> &control) {
        dp::mat::vector<double, dp::mat::Dynamic> next(1);
        next[0] = state[0] + control[0] * 0.1;
        return next;
    };

    auto cost = [](const dp::mat::vector<double, dp::mat::Dynamic> &state,
                   const dp::mat::vector<double, dp::mat::Dynamic> &control) {
        return state[0] * state[0] + 0.01 * control[0] * control[0];
    };

    dp::mat::vector<double, dp::mat::Dynamic> state(1);
    state[0] = 5.0;

    // Run first step
    mppi.step(dynamics, cost, state);

    // Get control sequence after first step
    auto controls_after_first = mppi.get_control_sequence();

    // The sequence should have been shifted (warm start)
    // Last element should be zero
    CHECK(std::abs(controls_after_first.back()[0]) < 1e-10);
}
