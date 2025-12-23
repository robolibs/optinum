#include <doctest/doctest.h>
#include <optinum/simd/math/simd_math.hpp>

using optinum::simd::SIMDVec;

TEST_CASE("simd hyperbolic: sinh/cosh/tanh") {
    double in[4] = {-1.0, -0.25, 0.25, 1.0};
    auto v = SIMDVec<double, 4>::loadu(in);

    auto sh = optinum::simd::sinh(v);
    auto ch = optinum::simd::cosh(v);
    auto th = optinum::simd::tanh(v);

    double osh[4]{}, och[4]{}, oth[4]{};
    sh.storeu(osh);
    ch.storeu(och);
    th.storeu(oth);

    for (int i = 0; i < 4; ++i) {
        CHECK(osh[i] == doctest::Approx(std::sinh(in[i])));
        CHECK(och[i] == doctest::Approx(std::cosh(in[i])));
        CHECK(oth[i] == doctest::Approx(std::tanh(in[i])));
    }
}

TEST_CASE("simd hyperbolic: asinh/acosh/atanh") {
    double in_asinh[4] = {-2.0, -0.5, 0.5, 2.0};
    auto vasinh = SIMDVec<double, 4>::loadu(in_asinh);
    auto as = optinum::simd::asinh(vasinh);
    double oas[4]{};
    as.storeu(oas);
    for (int i = 0; i < 4; ++i) {
        CHECK(oas[i] == doctest::Approx(std::asinh(in_asinh[i])));
    }

    double in_acosh[4] = {1.0, 1.5, 2.0, 10.0};
    auto vacosh = SIMDVec<double, 4>::loadu(in_acosh);
    auto ac = optinum::simd::acosh(vacosh);
    double oac[4]{};
    ac.storeu(oac);
    for (int i = 0; i < 4; ++i) {
        CHECK(oac[i] == doctest::Approx(std::acosh(in_acosh[i])));
    }

    double in_atanh[4] = {-0.75, -0.25, 0.25, 0.75};
    auto vatanh = SIMDVec<double, 4>::loadu(in_atanh);
    auto at = optinum::simd::atanh(vatanh);
    double oat[4]{};
    at.storeu(oat);
    for (int i = 0; i < 4; ++i) {
        CHECK(oat[i] == doctest::Approx(std::atanh(in_atanh[i])));
    }
}

