#include <doctest/doctest.h>
#include <optinum/simd/math/simd_math.hpp>

using optinum::simd::SIMDVec;

TEST_CASE("simd special: erf/erfc") {
    double in[4] = {0.0, -0.5, 0.5, 1.0};
    auto v = SIMDVec<double, 4>::loadu(in);

    auto e = optinum::simd::erf(v);
    auto ec = optinum::simd::erfc(v);

    double oe[4]{}, oec[4]{};
    e.storeu(oe);
    ec.storeu(oec);

    for (int i = 0; i < 4; ++i) {
        CHECK(oe[i] == doctest::Approx(std::erf(in[i])));
        CHECK(oec[i] == doctest::Approx(std::erfc(in[i])));
    }
}

TEST_CASE("simd special: hypot/gamma") {
    double a_in[4] = {0.0, 3.0, -4.0, 5.0};
    double b_in[4] = {0.0, 4.0, 3.0, -12.0};
    auto a = SIMDVec<double, 4>::loadu(a_in);
    auto b = SIMDVec<double, 4>::loadu(b_in);
    auto h = optinum::simd::hypot(a, b);
    double oh[4]{};
    h.storeu(oh);
    for (int i = 0; i < 4; ++i) {
        CHECK(oh[i] == doctest::Approx(std::hypot(a_in[i], b_in[i])));
    }

    double g_in[4] = {0.5, 1.0, 2.0, 3.5};
    auto g = SIMDVec<double, 4>::loadu(g_in);
    auto tg = optinum::simd::tgamma(g);
    auto lg = optinum::simd::lgamma(g);
    double otg[4]{}, olg[4]{};
    tg.storeu(otg);
    lg.storeu(olg);
    for (int i = 0; i < 4; ++i) {
        CHECK(otg[i] == doctest::Approx(std::tgamma(g_in[i])));
        CHECK(olg[i] == doctest::Approx(std::lgamma(g_in[i])));
    }
}
