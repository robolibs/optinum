#include <doctest/doctest.h>
#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/intrinsic/sse.hpp>

using optinum::simd::SIMDVec;

#if defined(OPTINUM_HAS_SSE2)

TEST_CASE("SSE SIMDVec<float,4> arithmetic") {
    float a_data[4] = {1.f, 2.f, 3.f, 4.f};
    float b_data[4] = {5.f, 6.f, 7.f, 8.f};

    auto a = SIMDVec<float, 4>::loadu(a_data);
    auto b = SIMDVec<float, 4>::loadu(b_data);
    auto c = a + b;

    float out[4];
    c.storeu(out);

    CHECK(out[0] == doctest::Approx(6.f));
    CHECK(out[1] == doctest::Approx(8.f));
    CHECK(out[2] == doctest::Approx(10.f));
    CHECK(out[3] == doctest::Approx(12.f));
    CHECK(c.hsum() == doctest::Approx(36.f));
}

TEST_CASE("SSE SIMDVec<double,2> dot") {
    double a_data[2] = {1.0, 2.0};
    double b_data[2] = {3.0, 4.0};

    auto a = SIMDVec<double, 2>::loadu(a_data);
    auto b = SIMDVec<double, 2>::loadu(b_data);

    CHECK(a.dot(b) == doctest::Approx(11.0));
}

#else

TEST_CASE("SSE intrinsics header compiles without SSE2") { CHECK(true); }

#endif

