#include <doctest/doctest.h>
#include <optinum/simd/math/simd_math.hpp>

using optinum::simd::SIMDVec;

TEST_CASE("simd pow/cbrt") {
    float in[4] = {0.25f, 1.0f, 2.0f, 4.0f};
    auto v = SIMDVec<float, 4>::loadu(in);

    auto p = optinum::simd::pow(v, 2.0f);
    float op[4]{};
    p.storeu(op);
    for (int i = 0; i < 4; ++i) {
        CHECK(op[i] == doctest::Approx(std::pow(in[i], 2.0f)));
    }

    auto cb = optinum::simd::cbrt(v);
    float ocb[4]{};
    cb.storeu(ocb);
    for (int i = 0; i < 4; ++i) {
        CHECK(ocb[i] == doctest::Approx(std::cbrt(in[i])));
    }
}

