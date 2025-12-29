#include <doctest/doctest.h>

#include <cmath>

#include <optinum/opti/opti.hpp>

using namespace optinum::opti;

// ============================================================================
// NoDecay Tests
// ============================================================================

TEST_CASE("NoDecay - constant learning rate") {
    NoDecay decay;
    double lr = 0.1;

    for (std::size_t i = 0; i < 100; ++i) {
        decay.update(lr, i);
        CHECK(lr == doctest::Approx(0.1));
    }
}

// ============================================================================
// StepDecay Tests
// ============================================================================

TEST_CASE("StepDecay - drops at intervals") {
    StepDecay decay(0.5, 10); // Drop by half every 10 iterations
    double lr = 1.0;

    SUBCASE("Before first drop") {
        for (std::size_t i = 0; i < 10; ++i) {
            decay.update(lr, i);
            CHECK(lr == doctest::Approx(1.0));
        }
    }

    SUBCASE("After first drop") {
        decay.update(lr, 0); // Initialize
        decay.update(lr, 10);
        CHECK(lr == doctest::Approx(0.5));
    }

    SUBCASE("After second drop") {
        decay.update(lr, 0); // Initialize
        decay.update(lr, 20);
        CHECK(lr == doctest::Approx(0.25));
    }

    SUBCASE("After third drop") {
        decay.update(lr, 0); // Initialize
        decay.update(lr, 30);
        CHECK(lr == doctest::Approx(0.125));
    }
}

TEST_CASE("StepDecay - reset") {
    StepDecay decay(0.5, 10);
    double lr = 1.0;

    decay.update(lr, 0);
    decay.update(lr, 20);
    CHECK(lr == doctest::Approx(0.25));

    decay.reset();
    lr = 2.0; // New initial lr
    decay.update(lr, 0);
    decay.update(lr, 10);
    CHECK(lr == doctest::Approx(1.0)); // 2.0 * 0.5
}

// ============================================================================
// ExponentialDecay Tests
// ============================================================================

TEST_CASE("ExponentialDecay - smooth decay") {
    ExponentialDecay decay(0.96, 100);
    double lr = 1.0;

    SUBCASE("At start") {
        decay.update(lr, 0);
        CHECK(lr == doctest::Approx(1.0));
    }

    SUBCASE("After 100 iterations") {
        decay.update(lr, 0); // Initialize
        decay.update(lr, 100);
        CHECK(lr == doctest::Approx(0.96));
    }

    SUBCASE("After 200 iterations") {
        decay.update(lr, 0); // Initialize
        decay.update(lr, 200);
        CHECK(lr == doctest::Approx(0.96 * 0.96));
    }

    SUBCASE("Monotonically decreasing") {
        decay.update(lr, 0);
        double prev_lr = lr;
        for (std::size_t i = 1; i < 500; ++i) {
            decay.update(lr, i);
            CHECK(lr <= prev_lr);
            prev_lr = lr;
        }
    }
}

// ============================================================================
// CosineAnnealing Tests
// ============================================================================

TEST_CASE("CosineAnnealing - cosine curve") {
    CosineAnnealing decay(100, 0.0);
    double lr = 1.0;

    SUBCASE("At start") {
        decay.update(lr, 0);
        CHECK(lr == doctest::Approx(1.0));
    }

    SUBCASE("At middle") {
        decay.update(lr, 0); // Initialize
        decay.update(lr, 50);
        CHECK(lr == doctest::Approx(0.5).epsilon(0.01));
    }

    SUBCASE("At end") {
        decay.update(lr, 0); // Initialize
        decay.update(lr, 100);
        CHECK(lr == doctest::Approx(0.0).epsilon(1e-10));
    }

    SUBCASE("With min_lr") {
        CosineAnnealing decay_min(100, 0.1);
        double lr2 = 1.0;
        decay_min.update(lr2, 0);
        decay_min.update(lr2, 100);
        CHECK(lr2 == doctest::Approx(0.1).epsilon(1e-10));
    }
}

// ============================================================================
// LinearDecay Tests
// ============================================================================

TEST_CASE("LinearDecay - linear decrease") {
    LinearDecay decay(100, 0.0);
    double lr = 1.0;

    SUBCASE("At start") {
        decay.update(lr, 0);
        CHECK(lr == doctest::Approx(1.0));
    }

    SUBCASE("At 25%") {
        decay.update(lr, 0);
        decay.update(lr, 25);
        CHECK(lr == doctest::Approx(0.75));
    }

    SUBCASE("At 50%") {
        decay.update(lr, 0);
        decay.update(lr, 50);
        CHECK(lr == doctest::Approx(0.5));
    }

    SUBCASE("At 75%") {
        decay.update(lr, 0);
        decay.update(lr, 75);
        CHECK(lr == doctest::Approx(0.25));
    }

    SUBCASE("At end") {
        decay.update(lr, 0);
        decay.update(lr, 100);
        CHECK(lr == doctest::Approx(0.0));
    }

    SUBCASE("After end") {
        decay.update(lr, 0);
        decay.update(lr, 150);
        CHECK(lr == doctest::Approx(0.0));
    }

    SUBCASE("With min_lr") {
        LinearDecay decay_min(100, 0.1);
        double lr2 = 1.0;
        decay_min.update(lr2, 0);
        decay_min.update(lr2, 50);
        CHECK(lr2 == doctest::Approx(0.55)); // 1.0 - (1.0 - 0.1) * 0.5
    }
}

// ============================================================================
// InverseTimeDecay Tests
// ============================================================================

TEST_CASE("InverseTimeDecay - inverse decay") {
    InverseTimeDecay decay(0.1, 1, false);
    double lr = 1.0;

    SUBCASE("At start") {
        decay.update(lr, 0);
        CHECK(lr == doctest::Approx(1.0));
    }

    SUBCASE("After 10 iterations") {
        decay.update(lr, 0);
        decay.update(lr, 10);
        // lr = 1.0 / (1 + 0.1 * 10) = 1.0 / 2.0 = 0.5
        CHECK(lr == doctest::Approx(0.5));
    }

    SUBCASE("After 100 iterations") {
        decay.update(lr, 0);
        decay.update(lr, 100);
        // lr = 1.0 / (1 + 0.1 * 100) = 1.0 / 11.0
        CHECK(lr == doctest::Approx(1.0 / 11.0));
    }

    SUBCASE("Staircase mode") {
        InverseTimeDecay stair_decay(0.5, 10, true);
        double lr2 = 1.0;
        stair_decay.update(lr2, 0);

        // Iterations 0-9: step = 0, lr = 1.0 / (1 + 0.5 * 0) = 1.0
        stair_decay.update(lr2, 5);
        CHECK(lr2 == doctest::Approx(1.0));

        // Iterations 10-19: step = 1, lr = 1.0 / (1 + 0.5 * 1) = 0.667
        stair_decay.update(lr2, 15);
        CHECK(lr2 == doctest::Approx(1.0 / 1.5));
    }
}

// ============================================================================
// WarmupDecay Tests
// ============================================================================

TEST_CASE("WarmupDecay - linear warmup") {
    WarmupDecay decay(100);
    double lr = 1.0;

    SUBCASE("At start") {
        decay.update(lr, 0);
        CHECK(lr == doctest::Approx(0.01)); // 1/100
    }

    SUBCASE("At 50%") {
        decay.update(lr, 0);
        decay.update(lr, 49);
        CHECK(lr == doctest::Approx(0.5).epsilon(0.01));
    }

    SUBCASE("At end of warmup") {
        decay.update(lr, 0);
        decay.update(lr, 99);
        CHECK(lr == doctest::Approx(1.0).epsilon(0.01));
    }

    SUBCASE("After warmup") {
        decay.update(lr, 0);
        decay.update(lr, 150);
        CHECK(lr == doctest::Approx(1.0));
    }
}

// ============================================================================
// PolynomialDecay Tests
// ============================================================================

TEST_CASE("PolynomialDecay - polynomial curve") {
    SUBCASE("Linear (power=1)") {
        PolynomialDecay decay(100, 0.0, 1.0);
        double lr = 1.0;

        decay.update(lr, 0);
        CHECK(lr == doctest::Approx(1.0));

        decay.update(lr, 50);
        CHECK(lr == doctest::Approx(0.5));

        decay.update(lr, 100);
        CHECK(lr == doctest::Approx(0.0));
    }

    SUBCASE("Quadratic (power=2)") {
        PolynomialDecay decay(100, 0.0, 2.0);
        double lr = 1.0;

        decay.update(lr, 0);
        CHECK(lr == doctest::Approx(1.0));

        decay.update(lr, 50);
        // (1 - 0.5)^2 = 0.25
        CHECK(lr == doctest::Approx(0.25));

        decay.update(lr, 100);
        CHECK(lr == doctest::Approx(0.0));
    }

    SUBCASE("With end_lr") {
        PolynomialDecay decay(100, 0.1, 1.0);
        double lr = 1.0;

        decay.update(lr, 0);
        decay.update(lr, 100);
        CHECK(lr == doctest::Approx(0.1));
    }

    SUBCASE("Cycle mode") {
        PolynomialDecay decay(100, 0.0, 1.0, true);
        double lr = 1.0;

        decay.update(lr, 0);
        decay.update(lr, 50);
        double lr_at_50 = lr;

        decay.update(lr, 150); // Same as iteration 50 in cycle
        CHECK(lr == doctest::Approx(lr_at_50));
    }
}

// ============================================================================
// Integration with GradientDescent
// ============================================================================

TEST_CASE("Decay policies with GradientDescent") {
    using Vec2 = dp::mat::vector<double, 2>;
    Sphere<double, 2> sphere;

    SUBCASE("StepDecay") {
        GradientDescent<VanillaUpdate, StepDecay> gd;
        gd.step_size = 0.5;
        gd.max_iterations = 100;
        gd.tolerance = 1e-6;
        gd.get_decay_policy() = StepDecay(0.5, 20);

        Vec2 x;
        x[0] = 1.0;
        x[1] = 1.0;
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }

    SUBCASE("ExponentialDecay") {
        GradientDescent<VanillaUpdate, ExponentialDecay> gd;
        gd.step_size = 0.5;
        gd.max_iterations = 200;
        gd.tolerance = 1e-6;
        gd.get_decay_policy() = ExponentialDecay(0.99, 10);

        Vec2 x;
        x[0] = 1.0;
        x[1] = 1.0;
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }

    SUBCASE("CosineAnnealing") {
        GradientDescent<VanillaUpdate, CosineAnnealing> gd;
        gd.step_size = 0.5;
        gd.max_iterations = 200;
        gd.tolerance = 1e-6;
        gd.get_decay_policy() = CosineAnnealing(200, 0.01);

        Vec2 x;
        x[0] = 1.0;
        x[1] = 1.0;
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }

    SUBCASE("LinearDecay") {
        GradientDescent<VanillaUpdate, LinearDecay> gd;
        gd.step_size = 0.5;
        gd.max_iterations = 200;
        gd.tolerance = 1e-6;
        gd.get_decay_policy() = LinearDecay(200, 0.01);

        Vec2 x;
        x[0] = 1.0;
        x[1] = 1.0;
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }
}

TEST_CASE("Float precision") {
    StepDecay decay(0.5f, 10);
    float lr = 1.0f;

    decay.update(lr, 0);
    decay.update(lr, 10);
    CHECK(lr == doctest::Approx(0.5f));
}
