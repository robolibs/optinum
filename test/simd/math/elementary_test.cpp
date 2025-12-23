#include <doctest/doctest.h>
#include <optinum/simd/math/simd_math.hpp>

using optinum::simd::SIMDVec;

TEST_CASE("simd elementary: abs/sign/clamp") {
    float in[4] = {-1.5f, 0.0f, 2.25f, -0.0f};
    auto v = SIMDVec<float, 4>::loadu(in);

    {
        auto a = optinum::simd::abs(v);
        float out[4]{};
        a.storeu(out);
        CHECK(out[0] == doctest::Approx(1.5f));
        CHECK(out[1] == doctest::Approx(0.0f));
        CHECK(out[2] == doctest::Approx(2.25f));
        CHECK(out[3] == doctest::Approx(0.0f));
    }

    {
        auto s = optinum::simd::sign(v);
        float out[4]{};
        s.storeu(out);
        CHECK(out[0] == doctest::Approx(-1.0f));
        CHECK(out[1] == doctest::Approx(0.0f));
        CHECK(out[2] == doctest::Approx(1.0f));
        CHECK(out[3] == doctest::Approx(0.0f));
    }

    {
        auto c = optinum::simd::clamp(v, -1.0f, 1.0f);
        float out[4]{};
        c.storeu(out);
        CHECK(out[0] == doctest::Approx(-1.0f));
        CHECK(out[1] == doctest::Approx(0.0f));
        CHECK(out[2] == doctest::Approx(1.0f));
        CHECK(out[3] == doctest::Approx(-0.0f));
    }
}

TEST_CASE("simd elementary: floor/ceil/round/trunc") {
    float in[4] = {-1.6f, -1.2f, 1.2f, 1.6f};
    auto v = SIMDVec<float, 4>::loadu(in);

    auto vf = optinum::simd::floor(v);
    auto vc = optinum::simd::ceil(v);
    auto vr = optinum::simd::round(v);
    auto vt = optinum::simd::trunc(v);

    float of[4]{}, oc[4]{}, orr[4]{}, ot[4]{};
    vf.storeu(of);
    vc.storeu(oc);
    vr.storeu(orr);
    vt.storeu(ot);

    CHECK(of[0] == doctest::Approx(-2.0f));
    CHECK(of[1] == doctest::Approx(-2.0f));
    CHECK(of[2] == doctest::Approx(1.0f));
    CHECK(of[3] == doctest::Approx(1.0f));

    CHECK(oc[0] == doctest::Approx(-1.0f));
    CHECK(oc[1] == doctest::Approx(-1.0f));
    CHECK(oc[2] == doctest::Approx(2.0f));
    CHECK(oc[3] == doctest::Approx(2.0f));

    CHECK(orr[0] == doctest::Approx(-2.0f));
    CHECK(orr[1] == doctest::Approx(-1.0f));
    CHECK(orr[2] == doctest::Approx(1.0f));
    CHECK(orr[3] == doctest::Approx(2.0f));

    CHECK(ot[0] == doctest::Approx(-1.0f));
    CHECK(ot[1] == doctest::Approx(-1.0f));
    CHECK(ot[2] == doctest::Approx(1.0f));
    CHECK(ot[3] == doctest::Approx(1.0f));
}

