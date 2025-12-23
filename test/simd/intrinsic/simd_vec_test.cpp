#include <doctest/doctest.h>
#include <optinum/simd/intrinsic/simd_vec.hpp>

using optinum::simd::SIMDVec;

TEST_CASE("SIMDVec scalar fallback basic ops") {
    SIMDVec<int, 3> a(2);
    SIMDVec<int, 3> b(3);

    auto c = a + b;
    CHECK(c[0] == 5);
    CHECK(c[1] == 5);
    CHECK(c[2] == 5);

    auto d = b - a;
    CHECK(d[0] == 1);
    CHECK(d[1] == 1);
    CHECK(d[2] == 1);

    auto e = a * b;
    CHECK(e[0] == 6);
    CHECK(e[1] == 6);
    CHECK(e[2] == 6);

    CHECK(a.hsum() == 6);
    CHECK(b.hmin() == 3);
    CHECK(b.hmax() == 3);
}

TEST_CASE("SIMDVec load/store roundtrip") {
    float in[4] = {1.f, 2.f, 3.f, 4.f};
    float out[4] = {0.f, 0.f, 0.f, 0.f};

    auto v = SIMDVec<float, 4>::loadu(in);
    v.storeu(out);

    CHECK(out[0] == doctest::Approx(1.f));
    CHECK(out[1] == doctest::Approx(2.f));
    CHECK(out[2] == doctest::Approx(3.f));
    CHECK(out[3] == doctest::Approx(4.f));
}

