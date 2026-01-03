#include <doctest/doctest.h>
#include <optinum/opti/problem/sphere.hpp>

using optinum::opti::Sphere;
namespace dp = datapod;

TEST_CASE("Sphere evaluate") {
    Sphere<double, 3> sphere;
    dp::mat::Vector<double, 3> x;

    SUBCASE("at origin") {
        x[0] = 0.0;
        x[1] = 0.0;
        x[2] = 0.0;
        CHECK(sphere.evaluate(x) == 0.0);
    }

    SUBCASE("at (1, 0, 0)") {
        x[0] = 1.0;
        x[1] = 0.0;
        x[2] = 0.0;
        CHECK(sphere.evaluate(x) == 1.0);
    }

    SUBCASE("at (1, 2, 3)") {
        x[0] = 1.0;
        x[1] = 2.0;
        x[2] = 3.0;
        // 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
        CHECK(sphere.evaluate(x) == 14.0);
    }

    SUBCASE("at (-1, -2, -3)") {
        x[0] = -1.0;
        x[1] = -2.0;
        x[2] = -3.0;
        // (-1)^2 + (-2)^2 + (-3)^2 = 14
        CHECK(sphere.evaluate(x) == 14.0);
    }
}

TEST_CASE("Sphere gradient") {
    Sphere<double, 3> sphere;
    dp::mat::Vector<double, 3> x;
    dp::mat::Vector<double, 3> g;

    SUBCASE("at origin") {
        x[0] = 0.0;
        x[1] = 0.0;
        x[2] = 0.0;
        sphere.gradient(x, g);
        CHECK(g[0] == 0.0);
        CHECK(g[1] == 0.0);
        CHECK(g[2] == 0.0);
    }

    SUBCASE("at (1, 2, 3)") {
        x[0] = 1.0;
        x[1] = 2.0;
        x[2] = 3.0;
        sphere.gradient(x, g);
        // g_i = 2 * x_i
        CHECK(g[0] == 2.0);
        CHECK(g[1] == 4.0);
        CHECK(g[2] == 6.0);
    }

    SUBCASE("at (-1, -2, -3)") {
        x[0] = -1.0;
        x[1] = -2.0;
        x[2] = -3.0;
        sphere.gradient(x, g);
        CHECK(g[0] == -2.0);
        CHECK(g[1] == -4.0);
        CHECK(g[2] == -6.0);
    }
}

TEST_CASE("Sphere evaluate_with_gradient") {
    Sphere<double, 3> sphere;
    dp::mat::Vector<double, 3> x;
    dp::mat::Vector<double, 3> g;

    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;

    double f = sphere.evaluate_with_gradient(x, g);

    CHECK(f == 14.0);
    CHECK(g[0] == 2.0);
    CHECK(g[1] == 4.0);
    CHECK(g[2] == 6.0);
}

TEST_CASE("Sphere different dimensions") {
    SUBCASE("1D") {
        Sphere<float, 1> sphere;
        dp::mat::Vector<float, 1> x;
        x[0] = 5.0f;
        CHECK(sphere.evaluate(x) == 25.0f);
    }

    SUBCASE("5D") {
        Sphere<float, 5> sphere;
        dp::mat::Vector<float, 5> x;
        x[0] = 1.0f;
        x[1] = 1.0f;
        x[2] = 1.0f;
        x[3] = 1.0f;
        x[4] = 1.0f;
        CHECK(sphere.evaluate(x) == 5.0f);
    }
}

TEST_CASE("Sphere global minimum") {
    Sphere<double, 4> sphere;
    dp::mat::Vector<double, 4> origin;
    origin.fill(0.0);

    // Global minimum is at origin with value 0
    CHECK(sphere.evaluate(origin) == 0.0);

    // Any other point has higher value
    dp::mat::Vector<double, 4> other;
    other[0] = 0.001;
    other[1] = 0.0;
    other[2] = 0.0;
    other[3] = 0.0;
    CHECK(sphere.evaluate(other) > 0.0);
}
