#include <doctest/doctest.h>
#include <optinum/simd/scalar.hpp>

using optinum::simd::Scalar;

TEST_CASE("Scalar construction") {
    SUBCASE("default construction") {
        Scalar<float> s;
        CHECK(s.get() == 0.0f);
    }

    SUBCASE("value construction") {
        Scalar<double> s(3.14);
        CHECK(s.get() == 3.14);
    }

    SUBCASE("from datapod") {
        datapod::scalar<int> pod{42};
        Scalar<int> s(pod);
        CHECK(s.get() == 42);
    }
}

TEST_CASE("Scalar pod access") {
    Scalar<float> s(2.5f);

    CHECK(s.pod().value == 2.5f);

    s.pod().value = 10.0f;
    CHECK(s.get() == 10.0f);
}

TEST_CASE("Scalar arithmetic") {
    Scalar<float> a(10.0f);
    Scalar<float> b(3.0f);

    SUBCASE("addition") {
        auto c = a + b;
        CHECK(c.get() == 13.0f);
    }

    SUBCASE("subtraction") {
        auto c = a - b;
        CHECK(c.get() == 7.0f);
    }

    SUBCASE("multiplication") {
        auto c = a * b;
        CHECK(c.get() == 30.0f);
    }

    SUBCASE("division") {
        auto c = a / b;
        CHECK(c.get() == doctest::Approx(3.333333f));
    }
}

TEST_CASE("Scalar compound assignment") {
    Scalar<float> a(10.0f);
    Scalar<float> b(2.0f);

    SUBCASE("+=") {
        a += b;
        CHECK(a.get() == 12.0f);
    }

    SUBCASE("-=") {
        a -= b;
        CHECK(a.get() == 8.0f);
    }

    SUBCASE("*=") {
        a *= b;
        CHECK(a.get() == 20.0f);
    }

    SUBCASE("/=") {
        a /= b;
        CHECK(a.get() == 5.0f);
    }
}

TEST_CASE("Scalar unary operators") {
    Scalar<float> a(5.0f);

    CHECK((-a).get() == -5.0f);
    CHECK((+a).get() == 5.0f);
}

TEST_CASE("Scalar comparison") {
    Scalar<int> a(10);
    Scalar<int> b(10);
    Scalar<int> c(20);

    CHECK(a == b);
    CHECK(a != c);
    CHECK(a < c);
    CHECK(a <= b);
    CHECK(c > a);
    CHECK(c >= b);
}

TEST_CASE("Scalar implicit conversion") {
    Scalar<float> s(3.14f);
    float val = s;
    CHECK(val == 3.14f);
}
