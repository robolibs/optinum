#include <doctest/doctest.h>

#include <cmath>

#include <optinum/opti/opti.hpp>

using optinum::opti::Ackley;
using optinum::simd::Dynamic;
using optinum::simd::Vector;

TEST_CASE("Ackley evaluate") {
    Ackley<double, 2> ackley;
    Vector<double, 2> x;

    SUBCASE("at global minimum (0, 0)") {
        x[0] = 0.0;
        x[1] = 0.0;
        // f(0,0) = -20*exp(0) - exp(1) + 20 + e = -20 - e + 20 + e = 0
        CHECK(ackley.evaluate(x) == doctest::Approx(0.0).epsilon(1e-10));
    }

    SUBCASE("at (1, 1)") {
        x[0] = 1.0;
        x[1] = 1.0;
        // Should be positive (not at minimum)
        double f = ackley.evaluate(x);
        CHECK(f > 0.0);
        CHECK(f < 10.0); // Reasonable bound
    }

    SUBCASE("far from origin") {
        x[0] = 10.0;
        x[1] = 10.0;
        // Should be close to 20 + e (the asymptotic value)
        double f = ackley.evaluate(x);
        CHECK(f > 15.0);
        CHECK(f < 25.0);
    }
}

TEST_CASE("Ackley gradient") {
    Ackley<double, 2> ackley;
    Vector<double, 2> x;
    Vector<double, 2> g;

    SUBCASE("at global minimum (0, 0)") {
        x[0] = 0.0;
        x[1] = 0.0;
        ackley.gradient(x, g);
        // At minimum, gradient should be zero
        CHECK(g[0] == doctest::Approx(0.0).epsilon(1e-10));
        CHECK(g[1] == doctest::Approx(0.0).epsilon(1e-10));
    }

    SUBCASE("at (1, 0)") {
        x[0] = 1.0;
        x[1] = 0.0;
        ackley.gradient(x, g);
        // Gradient should point toward origin (negative for positive x)
        CHECK(g[0] > 0.0); // Actually positive due to the function shape
    }
}

TEST_CASE("Ackley evaluate_with_gradient") {
    Ackley<double, 2> ackley;
    Vector<double, 2> x;
    Vector<double, 2> g;

    x[0] = 0.0;
    x[1] = 0.0;

    double f = ackley.evaluate_with_gradient(x, g);

    CHECK(f == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(g[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(g[1] == doctest::Approx(0.0).epsilon(1e-10));
}

TEST_CASE("Ackley higher dimensions") {
    SUBCASE("3D at minimum") {
        Ackley<double, 3> ackley;
        Vector<double, 3> x;
        x[0] = 0.0;
        x[1] = 0.0;
        x[2] = 0.0;
        CHECK(ackley.evaluate(x) == doctest::Approx(0.0).epsilon(1e-10));
    }

    SUBCASE("5D at minimum") {
        Ackley<double, 5> ackley;
        Vector<double, 5> x;
        for (std::size_t i = 0; i < 5; ++i)
            x[i] = 0.0;
        CHECK(ackley.evaluate(x) == doctest::Approx(0.0).epsilon(1e-10));
    }

    SUBCASE("10D at minimum") {
        Ackley<double, 10> ackley;
        Vector<double, 10> x;
        for (std::size_t i = 0; i < 10; ++i)
            x[i] = 0.0;
        CHECK(ackley.evaluate(x) == doctest::Approx(0.0).epsilon(1e-10));
    }
}

TEST_CASE("Ackley dynamic size") {
    Ackley<double, Dynamic> ackley;
    Vector<double, Dynamic> x(5);
    Vector<double, Dynamic> g(5);

    // At minimum
    for (std::size_t i = 0; i < 5; ++i)
        x[i] = 0.0;

    double f = ackley.evaluate_with_gradient(x, g);

    CHECK(f == doctest::Approx(0.0).epsilon(1e-10));
    for (std::size_t i = 0; i < 5; ++i) {
        CHECK(g[i] == doctest::Approx(0.0).epsilon(1e-10));
    }
}

TEST_CASE("Ackley float precision") {
    Ackley<float, 2> ackley;
    Vector<float, 2> x;

    x[0] = 0.0f;
    x[1] = 0.0f;
    CHECK(ackley.evaluate(x) == doctest::Approx(0.0f).epsilon(1e-5f));
}

TEST_CASE("Ackley minimum_location") {
    auto min_loc = Ackley<double, 3>::minimum_location();
    CHECK(min_loc[0] == 0.0);
    CHECK(min_loc[1] == 0.0);
    CHECK(min_loc[2] == 0.0);

    CHECK(Ackley<double, 3>::minimum_value() == 0.0);
}

TEST_CASE("Ackley gradient numerical check") {
    // Verify gradient using finite differences
    Ackley<double, 2> ackley;
    Vector<double, 2> x;
    Vector<double, 2> g;

    x[0] = 0.5;
    x[1] = -0.3;
    ackley.gradient(x, g);

    double eps = 1e-7;
    for (std::size_t i = 0; i < 2; ++i) {
        Vector<double, 2> x_plus = x;
        Vector<double, 2> x_minus = x;
        x_plus[i] += eps;
        x_minus[i] -= eps;

        double numerical_grad = (ackley.evaluate(x_plus) - ackley.evaluate(x_minus)) / (2 * eps);
        CHECK(g[i] == doctest::Approx(numerical_grad).epsilon(1e-5));
    }
}

TEST_CASE("Ackley symmetry") {
    // Ackley function is symmetric around the origin
    Ackley<double, 2> ackley;
    Vector<double, 2> x1, x2;

    x1[0] = 1.5;
    x1[1] = 2.0;

    x2[0] = -1.5;
    x2[1] = -2.0;

    CHECK(ackley.evaluate(x1) == doctest::Approx(ackley.evaluate(x2)).epsilon(1e-10));
}

TEST_CASE("Ackley flat outer region") {
    // Ackley has a nearly flat outer region
    Ackley<double, 2> ackley;
    Vector<double, 2> x1, x2;

    // Far from origin, function value should be similar
    x1[0] = 20.0;
    x1[1] = 20.0;

    x2[0] = 30.0;
    x2[1] = 30.0;

    double f1 = ackley.evaluate(x1);
    double f2 = ackley.evaluate(x2);

    // Both should be close to the asymptotic value (~20 + e)
    CHECK(std::abs(f1 - f2) < 1.0);
}

TEST_CASE("Ackley 1D special case") {
    Ackley<double, 1> ackley;
    Vector<double, 1> x;
    Vector<double, 1> g;

    x[0] = 0.0;
    double f = ackley.evaluate_with_gradient(x, g);

    CHECK(f == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(g[0] == doctest::Approx(0.0).epsilon(1e-10));
}

TEST_CASE("Ackley consistency") {
    // evaluate_with_gradient should give same results as separate calls
    Ackley<double, 3> ackley;
    Vector<double, 3> x;
    Vector<double, 3> g1, g2;

    x[0] = 1.2;
    x[1] = -0.8;
    x[2] = 0.5;

    double f1 = ackley.evaluate(x);
    ackley.gradient(x, g1);

    double f2 = ackley.evaluate_with_gradient(x, g2);

    CHECK(f1 == doctest::Approx(f2).epsilon(1e-10));
    for (std::size_t i = 0; i < 3; ++i) {
        CHECK(g1[i] == doctest::Approx(g2[i]).epsilon(1e-10));
    }
}
