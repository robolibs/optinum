#include <doctest/doctest.h>
#include <optinum/simd/math/simd_math.hpp>

using optinum::simd::SIMDVec;

TEST_CASE("simd trig: sin/cos/tan") {
    float in[4] = {0.0f, 0.25f, -0.5f, 1.0f};
    auto v = SIMDVec<float, 4>::loadu(in);

    auto s = optinum::simd::sin(v);
    auto c = optinum::simd::cos(v);
    auto t = optinum::simd::tan(v);

    float os[4]{}, oc[4]{}, ot[4]{};
    s.storeu(os);
    c.storeu(oc);
    t.storeu(ot);

    for (int i = 0; i < 4; ++i) {
        CHECK(os[i] == doctest::Approx(std::sin(in[i])));
        CHECK(oc[i] == doctest::Approx(std::cos(in[i])));
        CHECK(ot[i] == doctest::Approx(std::tan(in[i])));
    }
}

TEST_CASE("simd trig: inverse and atan2") {
    float in[4] = {-0.75f, -0.25f, 0.25f, 0.75f};
    auto v = SIMDVec<float, 4>::loadu(in);

    auto as = optinum::simd::asin(v);
    auto ac = optinum::simd::acos(v);
    auto at = optinum::simd::atan(v);

    float oas[4]{}, oac[4]{}, oat[4]{};
    as.storeu(oas);
    ac.storeu(oac);
    at.storeu(oat);

    for (int i = 0; i < 4; ++i) {
        CHECK(oas[i] == doctest::Approx(std::asin(in[i])));
        CHECK(oac[i] == doctest::Approx(std::acos(in[i])));
        CHECK(oat[i] == doctest::Approx(std::atan(in[i])));
    }

    float yv[4] = {0.0f, 1.0f, -1.0f, 1.0f};
    float xv[4] = {1.0f, 1.0f, 1.0f, -1.0f};
    auto y = SIMDVec<float, 4>::loadu(yv);
    auto x = SIMDVec<float, 4>::loadu(xv);
    auto a2 = optinum::simd::atan2(y, x);
    float oa2[4]{};
    a2.storeu(oa2);
    for (int i = 0; i < 4; ++i) {
        CHECK(oa2[i] == doctest::Approx(std::atan2(yv[i], xv[i])));
    }
}

