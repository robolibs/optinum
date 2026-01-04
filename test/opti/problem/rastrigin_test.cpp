#include <doctest/doctest.h>

#include <cmath>

#include <optinum/opti/opti.hpp>

using optinum::opti::Rastrigin;
using optinum::simd::Dynamic;

TEST_CASE("Rastrigin evaluate") {
    Rastrigin<double, 2> rastrigin;
    dp::mat::Vector<double, 2> x;

    SUBCASE("at global minimum (0, 0)") {
        x[0] = 0.0;
        x[1] = 0.0;
        // f(0,0) = 10*2 + (0 - 10*cos(0)) + (0 - 10*cos(0)) = 20 + (-10) + (-10) = 0
        CHECK(rastrigin.evaluate(x) == doctest::Approx(0.0).epsilon(1e-10));
    }

    SUBCASE("at (1, 1) - local minimum") {
        x[0] = 1.0;
        x[1] = 1.0;
        // f(1,1) = 10*2 + (1 - 10*cos(2*pi)) + (1 - 10*cos(2*pi))
        //        = 20 + (1 - 10) + (1 - 10) = 20 - 9 - 9 = 2
        CHECK(rastrigin.evaluate(x) == doctest::Approx(2.0).epsilon(1e-10));
    }

    SUBCASE("at (0.5, 0.5)") {
        x[0] = 0.5;
        x[1] = 0.5;
        // f(0.5, 0.5) = 10*2 + 2*(0.25 - 10*cos(pi))
        //             = 20 + 2*(0.25 + 10) = 20 + 20.5 = 40.5
        double expected = 20.0 + 2.0 * (0.25 - 10.0 * std::cos(M_PI));
        CHECK(rastrigin.evaluate(x) == doctest::Approx(expected).epsilon(1e-10));
    }
}

TEST_CASE("Rastrigin gradient") {
    Rastrigin<double, 2> rastrigin;
    dp::mat::Vector<double, 2> x;
    dp::mat::Vector<double, 2> g;

    SUBCASE("at global minimum (0, 0)") {
        x[0] = 0.0;
        x[1] = 0.0;
        rastrigin.gradient(x, g);
        // g_i = 2*x_i + 2*pi*10*sin(2*pi*x_i)
        // At x=0: g = 2*0 + 2*pi*10*sin(0) = 0
        CHECK(g[0] == doctest::Approx(0.0).epsilon(1e-10));
        CHECK(g[1] == doctest::Approx(0.0).epsilon(1e-10));
    }

    SUBCASE("at (1, 1) - local minimum") {
        x[0] = 1.0;
        x[1] = 1.0;
        rastrigin.gradient(x, g);
        // At x=1: g = 2*1 + 2*pi*10*sin(2*pi) = 2 + 0 = 2
        CHECK(g[0] == doctest::Approx(2.0).epsilon(1e-10));
        CHECK(g[1] == doctest::Approx(2.0).epsilon(1e-10));
    }
}

TEST_CASE("Rastrigin evaluate_with_gradient") {
    Rastrigin<double, 2> rastrigin;
    dp::mat::Vector<double, 2> x;
    dp::mat::Vector<double, 2> g;

    x[0] = 0.0;
    x[1] = 0.0;

    double f = rastrigin.evaluate_with_gradient(x, g);

    CHECK(f == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(g[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(g[1] == doctest::Approx(0.0).epsilon(1e-10));
}

TEST_CASE("Rastrigin higher dimensions") {
    SUBCASE("3D at minimum") {
        Rastrigin<double, 3> rastrigin;
        dp::mat::Vector<double, 3> x;
        x[0] = 0.0;
        x[1] = 0.0;
        x[2] = 0.0;
        CHECK(rastrigin.evaluate(x) == doctest::Approx(0.0).epsilon(1e-10));
    }

    SUBCASE("5D at minimum") {
        Rastrigin<double, 5> rastrigin;
        dp::mat::Vector<double, 5> x;
        for (std::size_t i = 0; i < 5; ++i)
            x[i] = 0.0;
        CHECK(rastrigin.evaluate(x) == doctest::Approx(0.0).epsilon(1e-10));
    }

    SUBCASE("10D at minimum") {
        Rastrigin<double, 10> rastrigin;
        dp::mat::Vector<double, 10> x;
        for (std::size_t i = 0; i < 10; ++i)
            x[i] = 0.0;
        CHECK(rastrigin.evaluate(x) == doctest::Approx(0.0).epsilon(1e-10));
    }
}

TEST_CASE("Rastrigin dynamic size") {
    Rastrigin<double, Dynamic> rastrigin;
    dp::mat::Vector<double, Dynamic> x(5);
    dp::mat::Vector<double, Dynamic> g(5);

    // At minimum
    for (std::size_t i = 0; i < 5; ++i)
        x[i] = 0.0;

    double f = rastrigin.evaluate_with_gradient(x, g);

    CHECK(f == doctest::Approx(0.0).epsilon(1e-10));
    for (std::size_t i = 0; i < 5; ++i) {
        CHECK(g[i] == doctest::Approx(0.0).epsilon(1e-10));
    }
}

TEST_CASE("Rastrigin float precision") {
    Rastrigin<float, 2> rastrigin;
    dp::mat::Vector<float, 2> x;

    x[0] = 0.0f;
    x[1] = 0.0f;
    CHECK(rastrigin.evaluate(x) == doctest::Approx(0.0f).epsilon(1e-5f));
}

TEST_CASE("Rastrigin minimum_location") {
    auto min_loc = Rastrigin<double, 3>::minimum_location();
    CHECK(min_loc[0] == 0.0);
    CHECK(min_loc[1] == 0.0);
    CHECK(min_loc[2] == 0.0);

    CHECK(Rastrigin<double, 3>::minimum_value() == 0.0);
}

TEST_CASE("Rastrigin local minima") {
    // Rastrigin has local minima at integer coordinates
    Rastrigin<double, 2> rastrigin;
    dp::mat::Vector<double, 2> x;
    dp::mat::Vector<double, 2> g;

    // Check that integer points have near-zero gradient in x direction
    // (they are local minima along each axis)
    for (int i = -2; i <= 2; ++i) {
        x[0] = static_cast<double>(i);
        x[1] = 0.0;
        rastrigin.gradient(x, g);
        // g[0] = 2*i + 2*pi*10*sin(2*pi*i) = 2*i (since sin(2*pi*i) = 0)
        CHECK(g[0] == doctest::Approx(2.0 * i).epsilon(1e-10));
    }
}

TEST_CASE("Rastrigin gradient numerical check") {
    // Verify gradient using finite differences
    Rastrigin<double, 2> rastrigin;
    dp::mat::Vector<double, 2> x;
    dp::mat::Vector<double, 2> g;

    x[0] = 0.3;
    x[1] = -0.7;
    rastrigin.gradient(x, g);

    double eps = 1e-7;
    for (std::size_t i = 0; i < 2; ++i) {
        dp::mat::Vector<double, 2> x_plus = x;
        dp::mat::Vector<double, 2> x_minus = x;
        x_plus[i] += eps;
        x_minus[i] -= eps;

        double numerical_grad = (rastrigin.evaluate(x_plus) - rastrigin.evaluate(x_minus)) / (2 * eps);
        CHECK(g[i] == doctest::Approx(numerical_grad).epsilon(1e-5));
    }
}

TEST_CASE("Rastrigin multimodality") {
    // Verify that the function has multiple local minima
    Rastrigin<double, 1> rastrigin;
    dp::mat::Vector<double, 1> x;

    // Global minimum at 0
    x[0] = 0.0;
    double f_global = rastrigin.evaluate(x);

    // Local minima at integers
    x[0] = 1.0;
    double f_local1 = rastrigin.evaluate(x);

    x[0] = -1.0;
    double f_local2 = rastrigin.evaluate(x);

    CHECK(f_global < f_local1);
    CHECK(f_global < f_local2);
    CHECK(f_local1 == doctest::Approx(f_local2).epsilon(1e-10)); // Symmetric
}
