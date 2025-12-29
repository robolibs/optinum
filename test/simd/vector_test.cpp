#include <doctest/doctest.h>
#include <optinum/simd/simd.hpp>

namespace dp = datapod;
namespace simd = optinum::simd;

TEST_CASE("Tensor construction") {
    SUBCASE("default construction") {
        dp::mat::vector<float, 3> t;
        CHECK(t.size() == 3);
    }

    SUBCASE("from initializer") {
        dp::mat::vector<float, 3> t{1.0f, 2.0f, 3.0f};
        CHECK(t[0] == 1.0f);
        CHECK(t[1] == 2.0f);
        CHECK(t[2] == 3.0f);
    }
}

TEST_CASE("Tensor element access") {
    dp::mat::vector<float, 4> t;
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
    dp::mat::vector<double, 5> t;
    t.fill(3.14);

    for (std::size_t i = 0; i < t.size(); ++i) {
        CHECK(t[i] == 3.14);
    }
}

TEST_CASE("Tensor arithmetic via views") {
    dp::mat::vector<float, 3> a{1.0f, 2.0f, 3.0f};
    dp::mat::vector<float, 3> b{4.0f, 5.0f, 6.0f};

    SUBCASE("addition") {
        dp::mat::vector<float, 3> c;
        simd::backend::add<float, 3>(c.data(), a.data(), b.data());
        CHECK(c[0] == 5.0f);
        CHECK(c[1] == 7.0f);
        CHECK(c[2] == 9.0f);
    }

    SUBCASE("subtraction") {
        dp::mat::vector<float, 3> c;
        simd::backend::sub<float, 3>(c.data(), b.data(), a.data());
        CHECK(c[0] == 3.0f);
        CHECK(c[1] == 3.0f);
        CHECK(c[2] == 3.0f);
    }

    SUBCASE("element-wise multiplication") {
        dp::mat::vector<float, 3> c;
        simd::backend::mul<float, 3>(c.data(), a.data(), b.data());
        CHECK(c[0] == 4.0f);
        CHECK(c[1] == 10.0f);
        CHECK(c[2] == 18.0f);
    }

    SUBCASE("scalar multiplication") {
        dp::mat::vector<float, 3> c;
        simd::backend::mul_scalar<float, 3>(c.data(), a.data(), 2.0f);
        CHECK(c[0] == 2.0f);
        CHECK(c[1] == 4.0f);
        CHECK(c[2] == 6.0f);
    }
}

TEST_CASE("Tensor compound assignment via backend") {
    dp::mat::vector<float, 3> a{1.0f, 2.0f, 3.0f};
    dp::mat::vector<float, 3> b{1.0f, 1.0f, 1.0f};

    SUBCASE("+=") {
        simd::backend::add<float, 3>(a.data(), a.data(), b.data());
        CHECK(a[0] == 2.0f);
        CHECK(a[1] == 3.0f);
        CHECK(a[2] == 4.0f);
    }

    SUBCASE("*= scalar") {
        simd::backend::mul_scalar<float, 3>(a.data(), a.data(), 3.0f);
        CHECK(a[0] == 3.0f);
        CHECK(a[1] == 6.0f);
        CHECK(a[2] == 9.0f);
    }
}

TEST_CASE("Tensor dot product") {
    dp::mat::vector<float, 3> a{1.0f, 2.0f, 3.0f};
    dp::mat::vector<float, 3> b{4.0f, 5.0f, 6.0f};

    float d = simd::backend::dot<float, 3>(a.data(), b.data());
    CHECK(d == 32.0f); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}

TEST_CASE("Tensor sum") {
    dp::mat::vector<int, 4> t{1, 2, 3, 4};

    CHECK(simd::backend::reduce_sum<int, 4>(t.data()) == 10);
}

TEST_CASE("Tensor norm") {
    dp::mat::vector<float, 3> t{3.0f, 0.0f, 4.0f};

    CHECK(simd::backend::norm_l2<float, 3>(t.data()) == doctest::Approx(5.0f));
}

TEST_CASE("Tensor normalized") {
    dp::mat::vector<float, 3> t{3.0f, 0.0f, 4.0f};
    dp::mat::vector<float, 3> n;

    simd::backend::normalize<float, 3>(n.data(), t.data());
    CHECK(n[0] == doctest::Approx(0.6f));
    CHECK(n[1] == doctest::Approx(0.0f));
    CHECK(n[2] == doctest::Approx(0.8f));
    CHECK(simd::backend::norm_l2<float, 3>(n.data()) == doctest::Approx(1.0f));
}

TEST_CASE("Tensor comparison") {
    dp::mat::vector<int, 3> a{1, 2, 3};
    dp::mat::vector<int, 3> b{1, 2, 3};
    dp::mat::vector<int, 3> c{1, 2, 4};

    CHECK(a == b);
    CHECK(a != c);
}

TEST_CASE("Tensor iteration") {
    dp::mat::vector<int, 4> t{10, 20, 30, 40};

    int sum = 0;
    for (auto val : t) {
        sum += val;
    }
    CHECK(sum == 100);
}

TEST_CASE("Tensor data access") {
    dp::mat::vector<float, 3> t{1.0f, 2.0f, 3.0f};

    float *data = t.data();
    CHECK(data[0] == 1.0f);

    data[0] = 99.0f;
    CHECK(t[0] == 99.0f);
}
