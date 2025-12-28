#include <doctest/doctest.h>

#include <datapod/matrix/vector.hpp>
#include <optinum/opti/opti.hpp>
#include <optinum/simd/vector.hpp>

using namespace optinum;
using namespace optinum::opti;
namespace dp = datapod;

TEST_CASE("NAdamUpdate - Sphere function 2D") {
    using Vec2 = simd::Vector<double, 2>;

    Sphere<double, 2> sphere;
    GradientDescent<NAdamUpdate> gd;

    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    SUBCASE("Converge from (1, 1)") {
        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.iterations < 1000);
        CHECK(result.final_cost < 1e-4);
        CHECK(std::abs(x[0]) < 1e-2);
        CHECK(std::abs(x[1]) < 1e-2);
    }

    SUBCASE("Converge from (5, -3)") {
        Vec2 x(dp::mat::vector<double, 2>{5.0, -3.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.final_cost < 1e-4);
        CHECK(std::abs(x[0]) < 1e-2);
        CHECK(std::abs(x[1]) < 1e-2);
    }

    SUBCASE("Already at minimum") {
        Vec2 x(dp::mat::vector<double, 2>{0.0, 0.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
        CHECK(result.iterations < 10);
        CHECK(result.final_cost < 1e-10);
    }
}

TEST_CASE("NAdamUpdate - Sphere function 3D") {
    using Vec3 = simd::Vector<double, 3>;

    Sphere<double, 3> sphere;
    GradientDescent<NAdamUpdate> gd;

    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    Vec3 x(dp::mat::vector<double, 3>{2.0, -1.0, 3.0});
    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4);
    CHECK(std::abs(x[0]) < 1e-2);
    CHECK(std::abs(x[1]) < 1e-2);
    CHECK(std::abs(x[2]) < 1e-2);
}

TEST_CASE("NAdamUpdate - Sphere function 10D") {
    using Vec10 = simd::Vector<double, 10>;

    Sphere<double, 10> sphere;
    GradientDescent<NAdamUpdate> gd;

    gd.step_size = 0.1;
    gd.max_iterations = 2000;
    gd.tolerance = 1e-6;

    Vec10 x;
    for (std::size_t i = 0; i < 10; ++i) {
        x[i] = static_cast<double>(i) - 5.0;
    }

    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4);

    for (std::size_t i = 0; i < 10; ++i) {
        CHECK(std::abs(x[i]) < 1e-2);
    }
}

TEST_CASE("NAdamUpdate - Different beta values") {
    using Vec2 = simd::Vector<double, 2>;
    Sphere<double, 2> sphere;

    SUBCASE("Default betas (0.9, 0.999)") {
        GradientDescent<NAdamUpdate> gd;
        gd.step_size = 0.1;
        gd.max_iterations = 1000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }

    SUBCASE("Higher beta1 (0.95)") {
        GradientDescent<NAdamUpdate> gd;
        gd.get_update_policy().beta1 = 0.95;
        gd.step_size = 0.1;
        gd.max_iterations = 1000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }

    SUBCASE("Lower beta2 (0.99)") {
        GradientDescent<NAdamUpdate> gd;
        gd.get_update_policy().beta2 = 0.99;
        gd.step_size = 0.1;
        gd.max_iterations = 1000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }
}

TEST_CASE("NAdamUpdate - Different epsilon values") {
    using Vec2 = simd::Vector<double, 2>;
    Sphere<double, 2> sphere;

    SUBCASE("Default epsilon (1e-8)") {
        GradientDescent<NAdamUpdate> gd;
        gd.step_size = 0.1;
        gd.max_iterations = 1000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }

    SUBCASE("Larger epsilon (1e-6)") {
        GradientDescent<NAdamUpdate> gd;
        gd.get_update_policy().epsilon = 1e-6;
        gd.step_size = 0.1;
        gd.max_iterations = 1000;
        gd.tolerance = 1e-6;

        Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
        auto result = gd.optimize(sphere, x);

        CHECK(result.converged);
    }
}

TEST_CASE("NAdamUpdate - Float precision") {
    using Vec2 = simd::Vector<float, 2>;

    Sphere<float, 2> sphere;
    GradientDescent<NAdamUpdate> gd;

    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-5f;

    Vec2 x(dp::mat::vector<float, 2>{1.0f, 1.0f});
    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-2f); // Relaxed for float precision
    CHECK(std::abs(x[0]) < 1e-1f);
    CHECK(std::abs(x[1]) < 1e-1f);
}

TEST_CASE("NAdamUpdate - Custom quadratic function") {
    // Custom quadratic: f(x, y) = (x - 2)^2 + (y + 3)^2
    // Minimum at (2, -3)
    struct CustomQuadratic {
        using tensor_type = simd::Vector<double, 2>;

        double evaluate(const tensor_type &x) const {
            double dx = x[0] - 2.0;
            double dy = x[1] + 3.0;
            return dx * dx + dy * dy;
        }

        void gradient(const tensor_type &x, tensor_type &g) const {
            g[0] = 2.0 * (x[0] - 2.0);
            g[1] = 2.0 * (x[1] + 3.0);
        }

        double evaluate_with_gradient(const tensor_type &x, tensor_type &g) const {
            gradient(x, g);
            return evaluate(x);
        }
    };

    CustomQuadratic func;
    GradientDescent<NAdamUpdate> gd;

    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    simd::Vector<double, 2> x(dp::mat::vector<double, 2>{0.0, 0.0});
    auto result = gd.optimize(func, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4);
    CHECK(std::abs(x[0] - 2.0) < 1e-2);
    CHECK(std::abs(x[1] - (-3.0)) < 1e-2);
}

TEST_CASE("NAdamUpdate - Reset behavior") {
    using Vec2 = simd::Vector<double, 2>;
    Sphere<double, 2> sphere;

    GradientDescent<NAdamUpdate> gd;
    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;
    gd.reset_policy = true;

    // First optimization
    Vec2 x1(dp::mat::vector<double, 2>{1.0, 1.0});
    auto result1 = gd.optimize(sphere, x1);
    CHECK(result1.converged);

    // Second optimization (should behave the same due to reset)
    Vec2 x2(dp::mat::vector<double, 2>{1.0, 1.0});
    auto result2 = gd.optimize(sphere, x2);
    CHECK(result2.converged);

    // Iterations should be similar (within tolerance)
    CHECK(std::abs(static_cast<int>(result1.iterations) - static_cast<int>(result2.iterations)) < 5);
}

TEST_CASE("NAdamUpdate - Comparison with Adam") {
    using Vec2 = simd::Vector<double, 2>;
    Sphere<double, 2> sphere;

    // NAdam
    GradientDescent<NAdamUpdate> nadam_gd;
    nadam_gd.step_size = 0.1;
    nadam_gd.max_iterations = 1000;
    nadam_gd.tolerance = 1e-6;

    Vec2 x1(dp::mat::vector<double, 2>{5.0, 5.0});
    auto nadam_result = nadam_gd.optimize(sphere, x1);

    // Adam
    GradientDescent<AdamUpdate> adam_gd;
    adam_gd.step_size = 0.1;
    adam_gd.max_iterations = 1000;
    adam_gd.tolerance = 1e-6;

    Vec2 x2(dp::mat::vector<double, 2>{5.0, 5.0});
    auto adam_result = adam_gd.optimize(sphere, x2);

    // Both should converge
    CHECK(nadam_result.converged);
    CHECK(adam_result.converged);

    // Both converge to similar quality - iteration count can vary
    // depending on problem conditioning and hyperparameters
    CHECK(nadam_result.final_cost < 1e-2);
    CHECK(adam_result.final_cost < 1e-2);
}

TEST_CASE("NAdamUpdate - Comparison with Nesterov") {
    using Vec2 = simd::Vector<double, 2>;
    Sphere<double, 2> sphere;

    // NAdam
    GradientDescent<NAdamUpdate> nadam_gd;
    nadam_gd.step_size = 0.1;
    nadam_gd.max_iterations = 1000;
    nadam_gd.tolerance = 1e-6;

    Vec2 x1(dp::mat::vector<double, 2>{5.0, 5.0});
    auto nadam_result = nadam_gd.optimize(sphere, x1);

    // Nesterov
    GradientDescent<NesterovUpdate> nesterov_gd;
    nesterov_gd.step_size = 0.01; // Nesterov needs smaller step size
    nesterov_gd.max_iterations = 1000;
    nesterov_gd.tolerance = 1e-6;

    Vec2 x2(dp::mat::vector<double, 2>{5.0, 5.0});
    auto nesterov_result = nesterov_gd.optimize(sphere, x2);

    // Both should converge
    CHECK(nadam_result.converged);
    CHECK(nesterov_result.converged);
}

TEST_CASE("NAdamUpdate - Dynamic vectors") {
    using VecDyn = simd::Vector<double, simd::Dynamic>;

    // Dynamic sphere function
    struct DynamicSphere {
        using tensor_type = VecDyn;

        double evaluate(const tensor_type &x) const {
            double sum = 0.0;
            for (std::size_t i = 0; i < x.size(); ++i) {
                sum += x[i] * x[i];
            }
            return 0.5 * sum;
        }

        void gradient(const tensor_type &x, tensor_type &g) const {
            for (std::size_t i = 0; i < x.size(); ++i) {
                g[i] = x[i];
            }
        }

        double evaluate_with_gradient(const tensor_type &x, tensor_type &g) const {
            gradient(x, g);
            return evaluate(x);
        }
    };

    DynamicSphere sphere;
    GradientDescent<NAdamUpdate> gd;

    gd.step_size = 0.1;
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    VecDyn x(5);
    for (std::size_t i = 0; i < 5; ++i) {
        x[i] = static_cast<double>(i + 1);
    }

    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4);

    for (std::size_t i = 0; i < 5; ++i) {
        CHECK(std::abs(x[i]) < 1e-2);
    }
}

TEST_CASE("NAdamUpdate - Large learning rate stability") {
    using Vec2 = simd::Vector<double, 2>;
    Sphere<double, 2> sphere;

    // NAdam should be stable with larger learning rates due to adaptive scaling
    GradientDescent<NAdamUpdate> gd;
    gd.step_size = 0.5; // Larger learning rate
    gd.max_iterations = 1000;
    gd.tolerance = 1e-6;

    Vec2 x(dp::mat::vector<double, 2>{1.0, 1.0});
    auto result = gd.optimize(sphere, x);

    CHECK(result.converged);
    CHECK(result.final_cost < 1e-4);
}

TEST_CASE("NAdamUpdate - Constructor parameters") {
    // Test explicit constructor
    NAdamUpdate policy1(0.95, 0.99, 1e-7);
    CHECK(policy1.beta1 == 0.95);
    CHECK(policy1.beta2 == 0.99);
    CHECK(policy1.epsilon == 1e-7);

    // Test default constructor
    NAdamUpdate policy2;
    CHECK(policy2.beta1 == 0.9);
    CHECK(policy2.beta2 == 0.999);
    CHECK(policy2.epsilon == 1e-8);
}
