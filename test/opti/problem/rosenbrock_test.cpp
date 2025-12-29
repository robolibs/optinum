#include <doctest/doctest.h>

#include <cmath>

#include <optinum/opti/opti.hpp>

using optinum::opti::Rosenbrock;
using optinum::simd::Dynamic;
namespace dp = datapod;

TEST_CASE("Rosenbrock evaluate") {
    Rosenbrock<double, 2> rosenbrock;
    dp::mat::vector<double, 2> x;

    SUBCASE("at global minimum (1, 1)") {
        x[0] = 1.0;
        x[1] = 1.0;
        CHECK(rosenbrock.evaluate(x) == doctest::Approx(0.0).epsilon(1e-10));
    }

    SUBCASE("at origin (0, 0)") {
        x[0] = 0.0;
        x[1] = 0.0;
        // f(0,0) = 100*(0 - 0)^2 + (1 - 0)^2 = 0 + 1 = 1
        CHECK(rosenbrock.evaluate(x) == doctest::Approx(1.0).epsilon(1e-10));
    }

    SUBCASE("at (-1, 1)") {
        x[0] = -1.0;
        x[1] = 1.0;
        // f(-1,1) = 100*(1 - 1)^2 + (1 - (-1))^2 = 0 + 4 = 4
        CHECK(rosenbrock.evaluate(x) == doctest::Approx(4.0).epsilon(1e-10));
    }

    SUBCASE("at (2, 4)") {
        x[0] = 2.0;
        x[1] = 4.0;
        // f(2,4) = 100*(4 - 4)^2 + (1 - 2)^2 = 0 + 1 = 1
        CHECK(rosenbrock.evaluate(x) == doctest::Approx(1.0).epsilon(1e-10));
    }

    SUBCASE("at (0, 1)") {
        x[0] = 0.0;
        x[1] = 1.0;
        // f(0,1) = 100*(1 - 0)^2 + (1 - 0)^2 = 100 + 1 = 101
        CHECK(rosenbrock.evaluate(x) == doctest::Approx(101.0).epsilon(1e-10));
    }
}

TEST_CASE("Rosenbrock gradient") {
    Rosenbrock<double, 2> rosenbrock;
    dp::mat::vector<double, 2> x;
    dp::mat::vector<double, 2> g;

    SUBCASE("at global minimum (1, 1)") {
        x[0] = 1.0;
        x[1] = 1.0;
        rosenbrock.gradient(x, g);
        // At minimum, gradient should be zero
        CHECK(g[0] == doctest::Approx(0.0).epsilon(1e-10));
        CHECK(g[1] == doctest::Approx(0.0).epsilon(1e-10));
    }

    SUBCASE("at origin (0, 0)") {
        x[0] = 0.0;
        x[1] = 0.0;
        rosenbrock.gradient(x, g);
        // g[0] = -400*0*(0-0) - 2*(1-0) = -2
        // g[1] = 200*(0-0) = 0
        CHECK(g[0] == doctest::Approx(-2.0).epsilon(1e-10));
        CHECK(g[1] == doctest::Approx(0.0).epsilon(1e-10));
    }

    SUBCASE("at (2, 4)") {
        x[0] = 2.0;
        x[1] = 4.0;
        rosenbrock.gradient(x, g);
        // g[0] = -400*2*(4-4) - 2*(1-2) = 0 + 2 = 2
        // g[1] = 200*(4-4) = 0
        CHECK(g[0] == doctest::Approx(2.0).epsilon(1e-10));
        CHECK(g[1] == doctest::Approx(0.0).epsilon(1e-10));
    }
}

TEST_CASE("Rosenbrock evaluate_with_gradient") {
    Rosenbrock<double, 2> rosenbrock;
    dp::mat::vector<double, 2> x;
    dp::mat::vector<double, 2> g;

    x[0] = 1.0;
    x[1] = 1.0;

    double f = rosenbrock.evaluate_with_gradient(x, g);

    CHECK(f == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(g[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(g[1] == doctest::Approx(0.0).epsilon(1e-10));
}

TEST_CASE("Rosenbrock higher dimensions") {
    SUBCASE("3D at minimum") {
        Rosenbrock<double, 3> rosenbrock;
        dp::mat::vector<double, 3> x;
        x[0] = 1.0;
        x[1] = 1.0;
        x[2] = 1.0;
        CHECK(rosenbrock.evaluate(x) == doctest::Approx(0.0).epsilon(1e-10));
    }

    SUBCASE("5D at minimum") {
        Rosenbrock<double, 5> rosenbrock;
        dp::mat::vector<double, 5> x;
        for (std::size_t i = 0; i < 5; ++i)
            x[i] = 1.0;
        CHECK(rosenbrock.evaluate(x) == doctest::Approx(0.0).epsilon(1e-10));
    }

    SUBCASE("10D at minimum") {
        Rosenbrock<double, 10> rosenbrock;
        dp::mat::vector<double, 10> x;
        for (std::size_t i = 0; i < 10; ++i)
            x[i] = 1.0;
        CHECK(rosenbrock.evaluate(x) == doctest::Approx(0.0).epsilon(1e-10));
    }
}

TEST_CASE("Rosenbrock dynamic size") {
    Rosenbrock<double, Dynamic> rosenbrock;
    dp::mat::vector<double, Dynamic> x(5);
    dp::mat::vector<double, Dynamic> g(5);

    // At minimum
    for (std::size_t i = 0; i < 5; ++i)
        x[i] = 1.0;

    double f = rosenbrock.evaluate_with_gradient(x, g);

    CHECK(f == doctest::Approx(0.0).epsilon(1e-10));
    for (std::size_t i = 0; i < 5; ++i) {
        CHECK(g[i] == doctest::Approx(0.0).epsilon(1e-10));
    }
}

TEST_CASE("Rosenbrock float precision") {
    Rosenbrock<float, 2> rosenbrock;
    dp::mat::vector<float, 2> x;

    x[0] = 1.0f;
    x[1] = 1.0f;
    CHECK(rosenbrock.evaluate(x) == doctest::Approx(0.0f).epsilon(1e-5f));
}

TEST_CASE("Rosenbrock minimum_location") {
    auto min_loc = Rosenbrock<double, 3>::minimum_location();
    CHECK(min_loc[0] == 1.0);
    CHECK(min_loc[1] == 1.0);
    CHECK(min_loc[2] == 1.0);

    CHECK(Rosenbrock<double, 3>::minimum_value() == 0.0);
}

TEST_CASE("Rosenbrock gradient numerical check") {
    // Verify gradient using finite differences
    Rosenbrock<double, 2> rosenbrock;
    dp::mat::vector<double, 2> x;
    dp::mat::vector<double, 2> g;

    x[0] = 0.5;
    x[1] = 0.5;
    rosenbrock.gradient(x, g);

    double eps = 1e-7;
    for (std::size_t i = 0; i < 2; ++i) {
        dp::mat::vector<double, 2> x_plus = x;
        dp::mat::vector<double, 2> x_minus = x;
        x_plus[i] += eps;
        x_minus[i] -= eps;

        double numerical_grad = (rosenbrock.evaluate(x_plus) - rosenbrock.evaluate(x_minus)) / (2 * eps);
        CHECK(g[i] == doctest::Approx(numerical_grad).epsilon(1e-5));
    }
}
