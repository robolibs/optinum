#include <datapod/matrix.hpp>
#include <doctest/doctest.h>
#include <optinum/meta/lookahead.hpp>
#include <optinum/opti/gradient/gradient_descent.hpp>
#include <optinum/opti/gradient/update_policies/adam_update.hpp>
#include <optinum/opti/gradient/update_policies/vanilla_update.hpp>
#include <optinum/opti/problem/sphere.hpp>

#include <cmath>

using namespace optinum;
namespace dp = datapod;

TEST_CASE("Lookahead: Basic functionality") {
    // Create Lookahead with vanilla SGD
    meta::Lookahead<opti::VanillaUpdate> lookahead;
    lookahead.config.k = 5;
    lookahead.config.alpha = 0.5;

    dp::mat::Vector<double, 3> x;
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;

    dp::mat::Vector<double, 3> grad;
    grad[0] = 0.1;
    grad[1] = 0.2;
    grad[2] = 0.3;

    // First update - should initialize slow weights
    lookahead.update(x, 0.1, grad);

    CHECK(lookahead.step_count() == 1);
    CHECK(lookahead.slow_weights().size() == 3);

    // Do 4 more updates to reach k=5
    for (int i = 0; i < 4; ++i) {
        lookahead.update(x, 0.1, grad);
    }

    // After k steps, step_count should reset to 0
    CHECK(lookahead.step_count() == 0);
}

TEST_CASE("Lookahead: Slow weight interpolation") {
    meta::Lookahead<opti::VanillaUpdate> lookahead;
    lookahead.config.k = 2;
    lookahead.config.alpha = 0.5;

    dp::mat::Vector<double, 2> x;
    x[0] = 10.0;
    x[1] = 10.0;

    dp::mat::Vector<double, 2> grad;
    grad[0] = 1.0;
    grad[1] = 1.0;

    double step_size = 1.0;

    // First update: x becomes [9, 9], slow weights initialized to [10, 10]
    lookahead.update(x, step_size, grad);
    CHECK(lookahead.step_count() == 1);

    // Second update: x becomes [8, 8], then slow weight update triggers
    // slow = slow + 0.5 * (fast - slow) = 10 + 0.5 * (8 - 10) = 9
    // x is reset to slow = 9
    lookahead.update(x, step_size, grad);
    CHECK(lookahead.step_count() == 0);

    // After slow weight update, x should be reset to slow weights
    CHECK(x[0] == doctest::Approx(9.0));
    CHECK(x[1] == doctest::Approx(9.0));

    // Slow weights should be updated
    CHECK(lookahead.slow_weights()[0] == doctest::Approx(9.0));
    CHECK(lookahead.slow_weights()[1] == doctest::Approx(9.0));
}

TEST_CASE("Lookahead: Reset functionality") {
    meta::Lookahead<opti::VanillaUpdate> lookahead;
    lookahead.config.k = 5;

    dp::mat::Vector<double, 2> x;
    x[0] = 1.0;
    x[1] = 2.0;

    dp::mat::Vector<double, 2> grad;
    grad[0] = 0.1;
    grad[1] = 0.2;

    // Do some updates
    lookahead.update(x, 0.1, grad);
    lookahead.update(x, 0.1, grad);

    CHECK(lookahead.step_count() == 2);
    CHECK(lookahead.slow_weights().size() == 2);

    // Reset
    lookahead.reset();

    CHECK(lookahead.step_count() == 0);
    CHECK(lookahead.slow_weights().size() == 0);
}

TEST_CASE("Lookahead: With Adam update policy") {
    meta::Lookahead<opti::AdamUpdate> lookahead_adam;
    lookahead_adam.config.k = 5;
    lookahead_adam.config.alpha = 0.5;
    lookahead_adam.base.beta1 = 0.9;
    lookahead_adam.base.beta2 = 0.999;

    dp::mat::Vector<double, 3> x;
    x[0] = 5.0;
    x[1] = -3.0;
    x[2] = 2.0;

    dp::mat::Vector<double, 3> grad;
    grad[0] = 1.0;
    grad[1] = -0.5;
    grad[2] = 0.3;

    // Run several updates
    for (int i = 0; i < 10; ++i) {
        lookahead_adam.update(x, 0.01, grad);
    }

    // Should have completed 2 full cycles (10 / 5 = 2)
    CHECK(lookahead_adam.step_count() == 0);

    // x should have moved toward the negative gradient direction
    CHECK(x[0] < 5.0);
    CHECK(x[1] > -3.0);
    CHECK(x[2] < 2.0);
}

TEST_CASE("Lookahead: Configuration options") {
    SUBCASE("Custom config via constructor") {
        meta::Lookahead<opti::VanillaUpdate>::Config cfg;
        cfg.k = 10;
        cfg.alpha = 0.8;

        meta::Lookahead<opti::VanillaUpdate> lookahead(cfg);

        CHECK(lookahead.config.k == 10);
        CHECK(lookahead.config.alpha == doctest::Approx(0.8));
    }

    SUBCASE("Constructor with base policy") {
        opti::AdamUpdate adam;
        adam.beta1 = 0.95;

        meta::Lookahead<opti::AdamUpdate> lookahead(adam);

        CHECK(lookahead.base.beta1 == doctest::Approx(0.95));
    }

    SUBCASE("Constructor with base policy and config") {
        opti::AdamUpdate adam;
        adam.beta1 = 0.95;

        meta::Lookahead<opti::AdamUpdate>::Config cfg;
        cfg.k = 10;
        cfg.alpha = 0.8;

        meta::Lookahead<opti::AdamUpdate> lookahead(adam, cfg);

        CHECK(lookahead.config.k == 10);
        CHECK(lookahead.config.alpha == doctest::Approx(0.8));
        CHECK(lookahead.base.beta1 == doctest::Approx(0.95));
    }
}

TEST_CASE("Lookahead: With GradientDescent optimizer") {
    using LookaheadSGD = meta::Lookahead<opti::VanillaUpdate>;

    opti::GradientDescent<LookaheadSGD> gd;
    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    // Configure lookahead
    gd.get_update_policy().config.k = 5;
    gd.get_update_policy().config.alpha = 0.5;

    opti::Sphere<double, 2> sphere;

    dp::mat::Vector<double, 2> x;
    x[0] = 5.0;
    x[1] = -3.0;

    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4);
    CHECK(std::abs(x[0]) < 0.01);
    CHECK(std::abs(x[1]) < 0.01);
}

TEST_CASE("Lookahead: With GradientDescent and Adam") {
    using LookaheadAdam = meta::Lookahead<opti::AdamUpdate>;

    opti::GradientDescent<LookaheadAdam> gd;
    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    // Configure lookahead
    gd.get_update_policy().config.k = 5;
    gd.get_update_policy().config.alpha = 0.5;

    // Configure Adam
    gd.get_update_policy().base.beta1 = 0.9;
    gd.get_update_policy().base.beta2 = 0.999;

    opti::Sphere<double, 3> sphere;

    dp::mat::Vector<double, 3> x;
    x[0] = 5.0;
    x[1] = -3.0;
    x[2] = 2.0;

    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4);
    CHECK(std::abs(x[0]) < 0.01);
    CHECK(std::abs(x[1]) < 0.01);
    CHECK(std::abs(x[2]) < 0.01);
}

TEST_CASE("Lookahead: Dynamic vector support") {
    meta::Lookahead<opti::VanillaUpdate> lookahead;
    lookahead.config.k = 3;
    lookahead.config.alpha = 0.5;

    dp::mat::Vector<double, dp::mat::Dynamic> x(5);
    dp::mat::Vector<double, dp::mat::Dynamic> grad(5);

    for (std::size_t i = 0; i < 5; ++i) {
        x[i] = static_cast<double>(i + 1);
        grad[i] = 0.1 * static_cast<double>(i + 1);
    }

    // Run k updates to trigger slow weight update
    for (int i = 0; i < 3; ++i) {
        lookahead.update(x, 0.1, grad);
    }

    CHECK(lookahead.step_count() == 0);
    CHECK(lookahead.slow_weights().size() == 5);
}

TEST_CASE("Lookahead: Alpha parameter effect") {
    // Test with alpha = 0 (slow weights never change)
    SUBCASE("Alpha = 0") {
        meta::Lookahead<opti::VanillaUpdate> lookahead;
        lookahead.config.k = 2;
        lookahead.config.alpha = 0.0;

        dp::mat::Vector<double, 2> x;
        x[0] = 10.0;
        x[1] = 10.0;

        dp::mat::Vector<double, 2> grad;
        grad[0] = 1.0;
        grad[1] = 1.0;

        // First update initializes slow weights to [10, 10]
        lookahead.update(x, 1.0, grad);
        // Second update triggers slow weight update with alpha=0
        // slow stays at [10, 10], x resets to [10, 10]
        lookahead.update(x, 1.0, grad);

        CHECK(x[0] == doctest::Approx(10.0));
        CHECK(x[1] == doctest::Approx(10.0));
    }

    // Test with alpha = 1 (slow weights fully update to fast weights)
    SUBCASE("Alpha = 1") {
        meta::Lookahead<opti::VanillaUpdate> lookahead;
        lookahead.config.k = 2;
        lookahead.config.alpha = 1.0;

        dp::mat::Vector<double, 2> x;
        x[0] = 10.0;
        x[1] = 10.0;

        dp::mat::Vector<double, 2> grad;
        grad[0] = 1.0;
        grad[1] = 1.0;

        // First update: x becomes [9, 9]
        lookahead.update(x, 1.0, grad);
        // Second update: x becomes [8, 8], then slow = 10 + 1*(8-10) = 8
        lookahead.update(x, 1.0, grad);

        CHECK(x[0] == doctest::Approx(8.0));
        CHECK(x[1] == doctest::Approx(8.0));
    }
}

TEST_CASE("Lookahead: K parameter effect") {
    meta::Lookahead<opti::VanillaUpdate> lookahead;
    lookahead.config.k = 10;
    lookahead.config.alpha = 0.5;

    dp::mat::Vector<double, 2> x;
    x[0] = 10.0;
    x[1] = 10.0;

    dp::mat::Vector<double, 2> grad;
    grad[0] = 0.1;
    grad[1] = 0.1;

    // Run 9 updates - should not trigger slow weight update yet
    for (int i = 0; i < 9; ++i) {
        lookahead.update(x, 1.0, grad);
    }

    CHECK(lookahead.step_count() == 9);

    // 10th update triggers slow weight update
    lookahead.update(x, 1.0, grad);

    CHECK(lookahead.step_count() == 0);
}

TEST_CASE("Lookahead: Float type support") {
    meta::Lookahead<opti::VanillaUpdate> lookahead;
    lookahead.config.k = 3;
    lookahead.config.alpha = 0.5;

    dp::mat::Vector<float, 2> x;
    x[0] = 5.0f;
    x[1] = -3.0f;

    dp::mat::Vector<float, 2> grad;
    grad[0] = 0.5f;
    grad[1] = -0.3f;

    // Run k updates
    for (int i = 0; i < 3; ++i) {
        lookahead.update(x, 0.1f, grad);
    }

    CHECK(lookahead.step_count() == 0);
    // x should have moved
    CHECK(x[0] < 5.0f);
    CHECK(x[1] > -3.0f);
}
