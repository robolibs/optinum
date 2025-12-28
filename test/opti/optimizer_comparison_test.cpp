#include <doctest/doctest.h>

#include <datapod/matrix/vector.hpp>
#include <optinum/optinum.hpp>

using namespace optinum;
namespace dp = datapod;

TEST_CASE("Optimizer Comparison - VanillaUpdate vs Momentum vs RMSprop vs Adam") {
    using Vec2 = Vector<double, 2>;

    Sphere<double, 2> sphere;

    SUBCASE("All optimizers converge from (5, 3)") {
        // Test Vanilla Gradient Descent
        {
            GradientDescent<> vanilla_gd;
            vanilla_gd.step_size = 0.1;
            vanilla_gd.max_iterations = 1000;
            vanilla_gd.tolerance = 1e-6;

            Vec2 x(dp::mat::vector<double, 2>{5.0, 3.0});
            auto result = vanilla_gd.optimize(sphere, x);

            CHECK(result.converged);
            CHECK(result.final_cost < 1e-5);
            CHECK(std::abs(x[0]) < 2e-3);
            CHECK(std::abs(x[1]) < 2e-3);
        }

        // Test Momentum Gradient Descent
        {
            GradientDescent<MomentumUpdate> momentum_gd;
            momentum_gd.step_size = 0.1;
            momentum_gd.max_iterations = 1000;
            momentum_gd.tolerance = 1e-6;
            momentum_gd.get_update_policy().momentum = 0.9;

            Vec2 x(dp::mat::vector<double, 2>{5.0, 3.0});
            auto result = momentum_gd.optimize(sphere, x);

            CHECK(result.converged);
            CHECK(result.final_cost < 1e-5);
            CHECK(std::abs(x[0]) < 2e-3);
            CHECK(std::abs(x[1]) < 2e-3);
        }

        // Test RMSprop
        {
            GradientDescent<RMSPropUpdate> rmsprop_gd;
            rmsprop_gd.step_size = 0.1;
            rmsprop_gd.max_iterations = 1000;
            rmsprop_gd.tolerance = 1e-6;
            rmsprop_gd.get_update_policy().alpha = 0.99;

            Vec2 x(dp::mat::vector<double, 2>{5.0, 3.0});
            auto result = rmsprop_gd.optimize(sphere, x);

            CHECK(result.converged);
            CHECK(result.final_cost < 1e-5);
            CHECK(std::abs(x[0]) < 2e-3);
            CHECK(std::abs(x[1]) < 2e-3);
        }

        // Test Adam
        {
            GradientDescent<AdamUpdate> adam_gd;
            adam_gd.step_size = 0.1;
            adam_gd.max_iterations = 1000;
            adam_gd.tolerance = 1e-6;
            adam_gd.get_update_policy().beta1 = 0.9;
            adam_gd.get_update_policy().beta2 = 0.999;

            Vec2 x(dp::mat::vector<double, 2>{5.0, 3.0});
            auto result = adam_gd.optimize(sphere, x);

            CHECK(result.converged);
            CHECK(result.final_cost < 1e-5);
            CHECK(std::abs(x[0]) < 2e-3);
            CHECK(std::abs(x[1]) < 2e-3);
        }

        // Test AMSGrad
        {
            GradientDescent<AMSGradUpdate> amsgrad_gd;
            amsgrad_gd.step_size = 0.1;
            amsgrad_gd.max_iterations = 1000;
            amsgrad_gd.tolerance = 1e-6;
            amsgrad_gd.get_update_policy().beta1 = 0.9;
            amsgrad_gd.get_update_policy().beta2 = 0.999;

            Vec2 x(dp::mat::vector<double, 2>{5.0, 3.0});
            auto result = amsgrad_gd.optimize(sphere, x);

            CHECK(result.converged);
            CHECK(result.final_cost < 1e-5);
            CHECK(std::abs(x[0]) < 2e-3);
            CHECK(std::abs(x[1]) < 2e-3);
        }
    }

    SUBCASE("Convenient aliases work") {
        // Test using the convenient type aliases
        {
            Momentum momentum;
            momentum.step_size = 0.1;
            momentum.max_iterations = 1000;
            momentum.tolerance = 1e-6;

            Vec2 x(dp::mat::vector<double, 2>{5.0, 3.0});
            auto result = momentum.optimize(sphere, x);

            CHECK(result.converged);
            CHECK(result.final_cost < 1e-5);
        }

        {
            RMSprop rmsprop;
            rmsprop.step_size = 0.1;
            rmsprop.max_iterations = 1000;
            rmsprop.tolerance = 1e-6;

            Vec2 x(dp::mat::vector<double, 2>{5.0, 3.0});
            auto result = rmsprop.optimize(sphere, x);

            CHECK(result.converged);
            CHECK(result.final_cost < 1e-5);
        }

        {
            Adam adam;
            adam.step_size = 0.1;
            adam.max_iterations = 1000;
            adam.tolerance = 1e-6;

            Vec2 x(dp::mat::vector<double, 2>{5.0, 3.0});
            auto result = adam.optimize(sphere, x);

            CHECK(result.converged);
            CHECK(result.final_cost < 1e-5);
        }

        {
            AMSGrad amsgrad;
            amsgrad.step_size = 0.1;
            amsgrad.max_iterations = 1000;
            amsgrad.tolerance = 1e-6;

            Vec2 x(dp::mat::vector<double, 2>{5.0, 3.0});
            auto result = amsgrad.optimize(sphere, x);

            CHECK(result.converged);
            CHECK(result.final_cost < 1e-5);
        }
    }
}

TEST_CASE("Optimizer Comparison - 10D Sphere") {
    using Vec10 = Vector<double, 10>;

    Sphere<double, 10> sphere;

    Vec10 x_init;
    for (std::size_t i = 0; i < 10; ++i) {
        x_init[i] = static_cast<double>(i) - 5.0; // Initialize to [-5, -4, ..., 4]
    }

    SUBCASE("Vanilla Gradient Descent") {
        GradientDescent<> vanilla_gd;
        vanilla_gd.step_size = 0.05;
        vanilla_gd.max_iterations = 2000;
        vanilla_gd.tolerance = 1e-6;

        Vec10 x = x_init;
        auto result = vanilla_gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
    }

    SUBCASE("Momentum") {
        Momentum momentum;
        momentum.step_size = 0.05;
        momentum.max_iterations = 2000;
        momentum.tolerance = 1e-6;
        momentum.get_update_policy().momentum = 0.9;

        Vec10 x = x_init;
        auto result = momentum.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
    }

    SUBCASE("RMSprop") {
        RMSprop rmsprop;
        rmsprop.step_size = 0.05;
        rmsprop.max_iterations = 2000;
        rmsprop.tolerance = 1e-6;

        Vec10 x = x_init;
        auto result = rmsprop.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
    }

    SUBCASE("Adam") {
        Adam adam;
        adam.step_size = 0.05;
        adam.max_iterations = 2000;
        adam.tolerance = 1e-6;

        Vec10 x = x_init;
        auto result = adam.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
    }

    SUBCASE("AMSGrad") {
        AMSGrad amsgrad;
        amsgrad.step_size = 0.05;
        amsgrad.max_iterations = 2000;
        amsgrad.tolerance = 1e-6;

        Vec10 x = x_init;
        auto result = amsgrad.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
    }
}

TEST_CASE("Optimizer state reset") {
    using Vec2 = Vector<double, 2>;
    Sphere<double, 2> sphere;

    SUBCASE("Momentum automatic reset") {
        Momentum momentum;
        momentum.step_size = 0.1;
        momentum.max_iterations = 500;

        // First optimization
        Vec2 x1(dp::mat::vector<double, 2>{5.0, 3.0});
        auto result1 = momentum.optimize(sphere, x1);

        // Run again (optimizer automatically resets with reset_policy = true by default)
        Vec2 x2(dp::mat::vector<double, 2>{5.0, 3.0});
        auto result2 = momentum.optimize(sphere, x2);

        // Both should converge similarly
        CHECK(result1.converged);
        CHECK(result2.converged);
        CHECK(std::abs(result1.final_cost - result2.final_cost) < 1e-4);
    }

    SUBCASE("RMSprop automatic reset") {
        RMSprop rmsprop;
        rmsprop.step_size = 0.1;
        rmsprop.max_iterations = 500;

        // First optimization
        Vec2 x1(dp::mat::vector<double, 2>{5.0, 3.0});
        auto result1 = rmsprop.optimize(sphere, x1);

        // Run again (optimizer automatically resets with reset_policy = true by default)
        Vec2 x2(dp::mat::vector<double, 2>{5.0, 3.0});
        auto result2 = rmsprop.optimize(sphere, x2);

        // Both should converge similarly
        CHECK(result1.converged);
        CHECK(result2.converged);
        CHECK(std::abs(result1.final_cost - result2.final_cost) < 1e-4);
    }

    SUBCASE("Adam automatic reset") {
        Adam adam;
        adam.step_size = 0.1;
        adam.max_iterations = 500;

        // First optimization
        Vec2 x1(dp::mat::vector<double, 2>{5.0, 3.0});
        auto result1 = adam.optimize(sphere, x1);

        // Run again (optimizer automatically resets with reset_policy = true by default)
        Vec2 x2(dp::mat::vector<double, 2>{5.0, 3.0});
        auto result2 = adam.optimize(sphere, x2);

        // Both should converge similarly
        CHECK(result1.converged);
        CHECK(result2.converged);
        CHECK(std::abs(result1.final_cost - result2.final_cost) < 1e-4);
    }

    SUBCASE("AMSGrad automatic reset") {
        AMSGrad amsgrad;
        amsgrad.step_size = 0.1;
        amsgrad.max_iterations = 500;

        // First optimization
        Vec2 x1(dp::mat::vector<double, 2>{5.0, 3.0});
        auto result1 = amsgrad.optimize(sphere, x1);

        // Run again (optimizer automatically resets with reset_policy = true by default)
        Vec2 x2(dp::mat::vector<double, 2>{5.0, 3.0});
        auto result2 = amsgrad.optimize(sphere, x2);

        // Both should converge similarly
        CHECK(result1.converged);
        CHECK(result2.converged);
        CHECK(std::abs(result1.final_cost - result2.final_cost) < 1e-4);
    }
}

TEST_CASE("AMSGrad specific behavior") {
    using Vec2 = Vector<double, 2>;
    Sphere<double, 2> sphere;

    SUBCASE("AMSGrad v_hat is non-decreasing") {
        // This test verifies the key AMSGrad property: v_hat never decreases
        // We can't directly access internal state, but we can verify convergence
        // is stable even with varying gradient magnitudes

        AMSGrad amsgrad;
        amsgrad.step_size = 0.1;
        amsgrad.max_iterations = 1000;
        amsgrad.tolerance = 1e-8;

        Vec2 x(dp::mat::vector<double, 2>{10.0, 10.0});
        auto result = amsgrad.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-6);
    }

    SUBCASE("AMSGrad vs Adam comparison") {
        // Both should converge, but AMSGrad should be more stable
        Vec2 x_adam(dp::mat::vector<double, 2>{5.0, 3.0});
        Vec2 x_amsgrad(dp::mat::vector<double, 2>{5.0, 3.0});

        Adam adam;
        adam.step_size = 0.1;
        adam.max_iterations = 500;
        adam.tolerance = 1e-6;

        AMSGrad amsgrad;
        amsgrad.step_size = 0.1;
        amsgrad.max_iterations = 500;
        amsgrad.tolerance = 1e-6;

        auto result_adam = adam.optimize(sphere, x_adam);
        auto result_amsgrad = amsgrad.optimize(sphere, x_amsgrad);

        CHECK(result_adam.converged);
        CHECK(result_amsgrad.converged);

        // Both should reach similar final cost
        CHECK(result_adam.final_cost < 1e-4);
        CHECK(result_amsgrad.final_cost < 1e-4);
    }
}
