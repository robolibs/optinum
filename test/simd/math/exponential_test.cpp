#include <doctest/doctest.h>
#include <optinum/simd/math/simd_math.hpp>

using optinum::simd::SIMDVec;

TEST_CASE("simd exponential: exp/log family") {
    double in[4] = {0.0, 0.1, 1.0, 2.0};
    auto v = SIMDVec<double, 4>::loadu(in);

    auto e = optinum::simd::exp(v);
    auto e2 = optinum::simd::exp2(v);
    auto em1 = optinum::simd::expm1(v);

    double oe[4]{}, oe2[4]{}, oem1[4]{};
    e.storeu(oe);
    e2.storeu(oe2);
    em1.storeu(oem1);

    for (int i = 0; i < 4; ++i) {
        CHECK(oe[i] == doctest::Approx(std::exp(in[i])));
        CHECK(oe2[i] == doctest::Approx(std::exp2(in[i])));
        CHECK(oem1[i] == doctest::Approx(std::expm1(in[i])));
    }

    double pos[4] = {0.5, 1.0, 2.0, 10.0};
    auto p = SIMDVec<double, 4>::loadu(pos);
    auto l = optinum::simd::log(p);
    auto l2 = optinum::simd::log2(p);
    auto l10 = optinum::simd::log10(p);
    auto l1p = optinum::simd::log1p(v);

    double ol[4]{}, ol2[4]{}, ol10[4]{}, ol1p[4]{};
    l.storeu(ol);
    l2.storeu(ol2);
    l10.storeu(ol10);
    l1p.storeu(ol1p);

    for (int i = 0; i < 4; ++i) {
        CHECK(ol[i] == doctest::Approx(std::log(pos[i])));
        CHECK(ol2[i] == doctest::Approx(std::log2(pos[i])));
        CHECK(ol10[i] == doctest::Approx(std::log10(pos[i])));
        CHECK(ol1p[i] == doctest::Approx(std::log1p(in[i])));
    }
}

