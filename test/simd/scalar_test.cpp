#include <doctest/doctest.h>
#include <optinum/simd/simd.hpp>

namespace dp = datapod;

TEST_CASE("Scalar construction") {
    SUBCASE("default construction") {
        dp::mat::scalar<float> s;
        CHECK(s.value == 0.0f);
    }

    SUBCASE("value construction") {
        dp::mat::scalar<double> s{3.14};
        CHECK(s.value == 3.14);
    }

    SUBCASE("from value") {
        dp::mat::scalar<int> s{42};
        CHECK(s.value == 42);
    }
}

TEST_CASE("Scalar value access") {
    dp::mat::scalar<float> s{2.5f};

    CHECK(s.value == 2.5f);

    s.value = 10.0f;
    CHECK(s.value == 10.0f);
}

TEST_CASE("Scalar arithmetic") {
    dp::mat::scalar<float> a{10.0f};
    dp::mat::scalar<float> b{3.0f};

    SUBCASE("addition") {
        dp::mat::scalar<float> c{a.value + b.value};
        CHECK(c.value == 13.0f);
    }

    SUBCASE("subtraction") {
        dp::mat::scalar<float> c{a.value - b.value};
        CHECK(c.value == 7.0f);
    }

    SUBCASE("multiplication") {
        dp::mat::scalar<float> c{a.value * b.value};
        CHECK(c.value == 30.0f);
    }

    SUBCASE("division") {
        dp::mat::scalar<float> c{a.value / b.value};
        CHECK(c.value == doctest::Approx(3.333333f));
    }
}

TEST_CASE("Scalar compound assignment") {
    dp::mat::scalar<float> a{10.0f};
    dp::mat::scalar<float> b{2.0f};

    SUBCASE("+=") {
        a.value += b.value;
        CHECK(a.value == 12.0f);
    }

    SUBCASE("-=") {
        a.value -= b.value;
        CHECK(a.value == 8.0f);
    }

    SUBCASE("*=") {
        a.value *= b.value;
        CHECK(a.value == 20.0f);
    }

    SUBCASE("/=") {
        a.value /= b.value;
        CHECK(a.value == 5.0f);
    }
}

TEST_CASE("Scalar unary operators") {
    dp::mat::scalar<float> a{5.0f};

    CHECK((-a.value) == -5.0f);
    CHECK((+a.value) == 5.0f);
}

TEST_CASE("Scalar comparison") {
    dp::mat::scalar<int> a{10};
    dp::mat::scalar<int> b{10};
    dp::mat::scalar<int> c{20};

    CHECK(a.value == b.value);
    CHECK(a.value != c.value);
    CHECK(a.value < c.value);
    CHECK(a.value <= b.value);
    CHECK(c.value > a.value);
    CHECK(c.value >= b.value);
}

TEST_CASE("Scalar implicit conversion") {
    dp::mat::scalar<float> s{3.14f};
    float val = s.value;
    CHECK(val == 3.14f);
}
