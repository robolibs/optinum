#include <doctest/doctest.h>
#include <optinum/meta/swats.hpp>
#include <optinum/opti/gradient/gradient_descent.hpp>
#include <optinum/opti/problem/sphere.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>

using namespace optinum;

TEST_CASE("SWATS: Basic functionality") {
    meta::SWATS<double> swats;
    swats.config.beta1 = 0.9;
    swats.config.beta2 = 0.999;
    swats.config.min_adam_steps = 10;

    simd::Vector<double, 3> x;
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;

    simd::Vector<double, 3> grad;
    grad[0] = 0.1;
    grad[1] = 0.2;
    grad[2] = 0.3;

    // First update
    swats.update(x, 0.01, grad);

    CHECK(swats.iteration() == 1);
    CHECK(!swats.has_switched());

    // x should have moved in negative gradient direction
    CHECK(x[0] < 1.0);
    CHECK(x[1] < 2.0);
    CHECK(x[2] < 3.0);
}

TEST_CASE("SWATS: Adam phase behavior") {
    meta::SWATS<double> swats;
    swats.config.beta1 = 0.9;
    swats.config.beta2 = 0.999;
    swats.config.min_adam_steps = 100; // High threshold to stay in Adam

    simd::Vector<double, 2> x;
    x[0] = 5.0;
    x[1] = -3.0;

    simd::Vector<double, 2> grad;
    grad[0] = 1.0;
    grad[1] = -0.5;

    // Run several updates
    for (int i = 0; i < 50; ++i) {
        swats.update(x, 0.01, grad);
    }

    CHECK(swats.iteration() == 50);
    CHECK(!swats.has_switched()); // Should still be in Adam phase

    // x should have moved toward minimum
    CHECK(x[0] < 5.0);
    CHECK(x[1] > -3.0);
}

TEST_CASE("SWATS: Switching behavior") {
    meta::SWATS<double> swats;
    swats.config.beta1 = 0.9;
    swats.config.beta2 = 0.999;
    swats.config.min_adam_steps = 10;
    swats.config.switch_threshold = 1.0; // High threshold to trigger switch easily

    simd::Vector<double, 2> x;
    x[0] = 1.0;
    x[1] = 1.0;

    // Use constant gradient to make learning rate stabilize quickly
    simd::Vector<double, 2> grad;
    grad[0] = 0.1;
    grad[1] = 0.1;

    // Run until switch happens or max iterations
    bool switched = false;
    for (int i = 0; i < 200 && !switched; ++i) {
        swats.update(x, 0.01, grad);
        switched = swats.has_switched();
    }

    // With high threshold, should have switched
    CHECK(swats.has_switched());
    CHECK(swats.switch_iteration() > 0);
    CHECK(swats.switch_iteration() >= swats.config.min_adam_steps);
}

TEST_CASE("SWATS: SGD phase after switch") {
    meta::SWATS<double> swats;
    swats.config.beta1 = 0.9;
    swats.config.beta2 = 0.999;
    swats.config.min_adam_steps = 5;
    swats.config.switch_threshold = 10.0; // Very high to switch quickly

    simd::Vector<double, 2> x;
    x[0] = 10.0;
    x[1] = 10.0;

    simd::Vector<double, 2> grad;
    grad[0] = 1.0;
    grad[1] = 1.0;

    // Run until switch
    while (!swats.has_switched() && swats.iteration() < 100) {
        swats.update(x, 0.01, grad);
    }

    CHECK(swats.has_switched());

    // Record position after switch
    double x0_at_switch = x[0];
    double x1_at_switch = x[1];

    // Run more iterations in SGD phase
    for (int i = 0; i < 10; ++i) {
        swats.update(x, 0.01, grad);
    }

    // Should continue to decrease (SGD with learned lr)
    CHECK(x[0] < x0_at_switch);
    CHECK(x[1] < x1_at_switch);

    // SGD learning rate should be positive
    CHECK(swats.sgd_learning_rate() > 0);
}

TEST_CASE("SWATS: Reset functionality") {
    meta::SWATS<double> swats;
    swats.config.min_adam_steps = 5;
    swats.config.switch_threshold = 10.0;

    simd::Vector<double, 2> x;
    x[0] = 1.0;
    x[1] = 2.0;

    simd::Vector<double, 2> grad;
    grad[0] = 0.1;
    grad[1] = 0.2;

    // Run some updates
    for (int i = 0; i < 20; ++i) {
        swats.update(x, 0.01, grad);
    }

    CHECK(swats.iteration() > 0);

    // Reset
    swats.reset();

    CHECK(swats.iteration() == 0);
    CHECK(!swats.has_switched());
    CHECK(swats.switch_iteration() == 0);
}

TEST_CASE("SWATS: Configuration options") {
    SUBCASE("Custom config via constructor") {
        meta::SWATS<double>::Config cfg;
        cfg.beta1 = 0.95;
        cfg.beta2 = 0.9999;
        cfg.epsilon = 1e-7;
        cfg.min_adam_steps = 200;
        cfg.switch_threshold = 1e-10;

        meta::SWATS<double> swats(cfg);

        CHECK(swats.config.beta1 == doctest::Approx(0.95));
        CHECK(swats.config.beta2 == doctest::Approx(0.9999));
        CHECK(swats.config.epsilon == doctest::Approx(1e-7));
        CHECK(swats.config.min_adam_steps == 200);
        CHECK(swats.config.switch_threshold == doctest::Approx(1e-10));
    }

    SUBCASE("Default config values") {
        meta::SWATS<double> swats;

        CHECK(swats.config.beta1 == doctest::Approx(0.9));
        CHECK(swats.config.beta2 == doctest::Approx(0.999));
        CHECK(swats.config.epsilon == doctest::Approx(1e-8));
        CHECK(swats.config.min_adam_steps == 100);
    }
}

TEST_CASE("SWATS: With GradientDescent optimizer") {
    opti::GradientDescent<meta::SWATS<double>> gd;
    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    // Configure SWATS
    gd.get_update_policy().config.beta1 = 0.9;
    gd.get_update_policy().config.beta2 = 0.999;
    gd.get_update_policy().config.min_adam_steps = 50;

    opti::Sphere<double, 2> sphere;

    simd::Vector<double, 2> x;
    x[0] = 5.0;
    x[1] = -3.0;

    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4);
    CHECK(std::abs(x[0]) < 0.01);
    CHECK(std::abs(x[1]) < 0.01);
}

TEST_CASE("SWATS: Higher dimensional optimization") {
    meta::SWATS<double> swats;
    swats.config.beta1 = 0.9;
    swats.config.beta2 = 0.999;
    swats.config.min_adam_steps = 50;

    constexpr std::size_t dim = 10;

    simd::Vector<double, dim> x;
    simd::Vector<double, dim> grad;

    // Compute initial norm squared
    double initial_norm_sq = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        x[i] = static_cast<double>(i) - 5.0;
        grad[i] = 2.0 * x[i]; // Gradient of sphere function
        initial_norm_sq += x[i] * x[i];
    }

    // Run optimization
    for (int iter = 0; iter < 500; ++iter) {
        swats.update(x, 0.01, grad);

        // Update gradient for next iteration
        for (std::size_t i = 0; i < dim; ++i) {
            grad[i] = 2.0 * x[i];
        }
    }

    // Should have made progress toward origin
    double norm_sq = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        norm_sq += x[i] * x[i];
    }
    // Just verify we made significant progress (reduced by at least 50%)
    CHECK(norm_sq < initial_norm_sq * 0.5);
}

TEST_CASE("SWATS: Dynamic vector support") {
    meta::SWATS<double> swats;
    swats.config.min_adam_steps = 10;

    simd::Vector<double, simd::Dynamic> x(5);
    simd::Vector<double, simd::Dynamic> grad(5);

    for (std::size_t i = 0; i < 5; ++i) {
        x[i] = static_cast<double>(i + 1);
        grad[i] = 0.1 * static_cast<double>(i + 1);
    }

    // Run some updates
    for (int i = 0; i < 20; ++i) {
        swats.update(x, 0.01, grad);
    }

    CHECK(swats.iteration() == 20);
}

TEST_CASE("SWATS: Float type support") {
    meta::SWATS<float> swats;
    swats.config.beta1 = 0.9f;
    swats.config.beta2 = 0.999f;
    swats.config.epsilon = 1e-7f;
    swats.config.min_adam_steps = 10;

    simd::Vector<float, 2> x;
    x[0] = 5.0f;
    x[1] = -3.0f;

    simd::Vector<float, 2> grad;
    grad[0] = 0.5f;
    grad[1] = -0.3f;

    // Run some updates
    for (int i = 0; i < 20; ++i) {
        swats.update(x, 0.01f, grad);
    }

    CHECK(swats.iteration() == 20);
    // x should have moved
    CHECK(x[0] < 5.0f);
    CHECK(x[1] > -3.0f);
}

TEST_CASE("SWATS: Min adam steps respected") {
    meta::SWATS<double> swats;
    swats.config.min_adam_steps = 50;
    swats.config.switch_threshold = 1000.0; // Very high to ensure switch would happen

    simd::Vector<double, 2> x;
    x[0] = 1.0;
    x[1] = 1.0;

    simd::Vector<double, 2> grad;
    grad[0] = 0.1;
    grad[1] = 0.1;

    // Run fewer than min_adam_steps
    for (int i = 0; i < 40; ++i) {
        swats.update(x, 0.01, grad);
    }

    // Should not have switched yet
    CHECK(!swats.has_switched());
    CHECK(swats.iteration() == 40);
}

TEST_CASE("SWATS: Learned learning rate is reasonable") {
    meta::SWATS<double> swats;
    swats.config.min_adam_steps = 10;
    swats.config.switch_threshold = 10.0; // High to trigger switch

    simd::Vector<double, 2> x;
    x[0] = 1.0;
    x[1] = 1.0;

    simd::Vector<double, 2> grad;
    grad[0] = 0.1;
    grad[1] = 0.1;

    double base_lr = 0.01;

    // Run until switch
    while (!swats.has_switched() && swats.iteration() < 100) {
        swats.update(x, base_lr, grad);
    }

    if (swats.has_switched()) {
        // Learned SGD learning rate should be positive and reasonable
        CHECK(swats.sgd_learning_rate() > 0);
        // Should be in a reasonable range relative to base learning rate
        CHECK(swats.sgd_learning_rate() < base_lr * 100);
    }
}
