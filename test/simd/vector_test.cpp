#include <doctest/doctest.h>
#include <optinum/simd/vector.hpp>

using optinum::simd::Vector;

TEST_CASE("Tensor construction") {
    SUBCASE("default construction") {
        Vector<float, 3> t;
        CHECK(t.size() == 3);
    }

    SUBCASE("from datapod") {
        datapod::mat::vector<float, 3> pod{1.0f, 2.0f, 3.0f};
        Vector<float, 3> t(pod);
        CHECK(t[0] == 1.0f);
        CHECK(t[1] == 2.0f);
        CHECK(t[2] == 3.0f);
    }
}

TEST_CASE("Tensor element access") {
    Vector<float, 4> t;
    t[0] = 1.0f;
    t[1] = 2.0f;
    t[2] = 3.0f;
    t[3] = 4.0f;

    CHECK(t[0] == 1.0f);
    CHECK(t[3] == 4.0f);
    CHECK(t.front() == 1.0f);
    CHECK(t.back() == 4.0f);
}

TEST_CASE("Tensor fill") {
    Vector<double, 5> t;
    t.fill(3.14);

    for (std::size_t i = 0; i < t.size(); ++i) {
        CHECK(t[i] == 3.14);
    }
}

TEST_CASE("Tensor arithmetic") {
    Vector<float, 3> a;
    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;

    Vector<float, 3> b;
    b[0] = 4.0f;
    b[1] = 5.0f;
    b[2] = 6.0f;

    SUBCASE("addition") {
        auto c = a + b;
        CHECK(c[0] == 5.0f);
        CHECK(c[1] == 7.0f);
        CHECK(c[2] == 9.0f);
    }

    SUBCASE("subtraction") {
        auto c = b - a;
        CHECK(c[0] == 3.0f);
        CHECK(c[1] == 3.0f);
        CHECK(c[2] == 3.0f);
    }

    SUBCASE("element-wise multiplication") {
        auto c = a * b;
        CHECK(c[0] == 4.0f);
        CHECK(c[1] == 10.0f);
        CHECK(c[2] == 18.0f);
    }

    SUBCASE("scalar multiplication") {
        auto c = a * 2.0f;
        CHECK(c[0] == 2.0f);
        CHECK(c[1] == 4.0f);
        CHECK(c[2] == 6.0f);
    }

    SUBCASE("scalar multiplication (commutative)") {
        auto c = 2.0f * a;
        CHECK(c[0] == 2.0f);
        CHECK(c[1] == 4.0f);
        CHECK(c[2] == 6.0f);
    }
}

TEST_CASE("Tensor compound assignment") {
    Vector<float, 3> a;
    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;

    Vector<float, 3> b;
    b[0] = 1.0f;
    b[1] = 1.0f;
    b[2] = 1.0f;

    SUBCASE("+=") {
        a += b;
        CHECK(a[0] == 2.0f);
        CHECK(a[1] == 3.0f);
        CHECK(a[2] == 4.0f);
    }

    SUBCASE("*= scalar") {
        a *= 3.0f;
        CHECK(a[0] == 3.0f);
        CHECK(a[1] == 6.0f);
        CHECK(a[2] == 9.0f);
    }
}

TEST_CASE("Tensor dot product") {
    Vector<float, 3> a;
    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;

    Vector<float, 3> b;
    b[0] = 4.0f;
    b[1] = 5.0f;
    b[2] = 6.0f;

    float d = optinum::simd::dot(a, b);
    CHECK(d == 32.0f); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}

TEST_CASE("Tensor sum") {
    Vector<int, 4> t;
    t[0] = 1;
    t[1] = 2;
    t[2] = 3;
    t[3] = 4;

    CHECK(optinum::simd::sum(t) == 10);
}

TEST_CASE("Tensor norm") {
    Vector<float, 3> t;
    t[0] = 3.0f;
    t[1] = 0.0f;
    t[2] = 4.0f;

    CHECK(optinum::simd::norm(t) == doctest::Approx(5.0f));
}

TEST_CASE("Tensor normalized") {
    Vector<float, 3> t;
    t[0] = 3.0f;
    t[1] = 0.0f;
    t[2] = 4.0f;

    auto n = optinum::simd::normalized(t);
    CHECK(n[0] == doctest::Approx(0.6f));
    CHECK(n[1] == doctest::Approx(0.0f));
    CHECK(n[2] == doctest::Approx(0.8f));
    CHECK(optinum::simd::norm(n) == doctest::Approx(1.0f));
}

TEST_CASE("Tensor comparison") {
    Vector<int, 3> a;
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;

    Vector<int, 3> b;
    b[0] = 1;
    b[1] = 2;
    b[2] = 3;

    Vector<int, 3> c;
    c[0] = 1;
    c[1] = 2;
    c[2] = 4;

    CHECK(a == b);
    CHECK(a != c);
}

TEST_CASE("Tensor iteration") {
    Vector<int, 4> t;
    t[0] = 10;
    t[1] = 20;
    t[2] = 30;
    t[3] = 40;

    int sum = 0;
    for (auto val : t) {
        sum += val;
    }
    CHECK(sum == 100);
}

TEST_CASE("Tensor pod access") {
    Vector<float, 3> t;
    t[0] = 1.0f;
    t[1] = 2.0f;
    t[2] = 3.0f;

    datapod::mat::vector<float, 3> &pod = t.pod();
    CHECK(pod[0] == 1.0f);

    pod[0] = 99.0f;
    CHECK(t[0] == 99.0f);
}
