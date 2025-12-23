#include <doctest/doctest.h>
#include <optinum/simd/intrinsic/neon.hpp>

using optinum::simd::SIMDVec;

#if defined(OPTINUM_HAS_NEON)

TEST_CASE("NEON SIMDVec<float,4> arithmetic") {
    float a_data[4] = {1.f, 2.f, 3.f, 4.f};
    float b_data[4] = {4.f, 3.f, 2.f, 1.f};

    auto a = SIMDVec<float, 4>::loadu(a_data);
    auto b = SIMDVec<float, 4>::loadu(b_data);
    auto c = a + b;

    float out[4];
    c.storeu(out);

    CHECK(out[0] == doctest::Approx(5.f));
    CHECK(out[3] == doctest::Approx(5.f));
    CHECK(c.hsum() == doctest::Approx(20.f));
}

#else

TEST_CASE("NEON intrinsics header compiles without NEON") { CHECK(true); }

#endif

