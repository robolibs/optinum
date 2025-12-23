#include <doctest/doctest.h>
#include <optinum/simd/intrinsic/avx.hpp>

using optinum::simd::SIMDVec;

#if defined(OPTINUM_HAS_AVX)

TEST_CASE("AVX SIMDVec<float,8> arithmetic") {
    float a_data[8] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    float b_data[8] = {8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f};

    auto a = SIMDVec<float, 8>::loadu(a_data);
    auto b = SIMDVec<float, 8>::loadu(b_data);
    auto c = a + b;

    float out[8];
    c.storeu(out);
    CHECK(out[0] == doctest::Approx(9.f));
    CHECK(out[7] == doctest::Approx(9.f));
    CHECK(c.hsum() == doctest::Approx(72.f));
}

TEST_CASE("AVX SIMDVec<double,4> dot") {
    double a_data[4] = {1.0, 2.0, 3.0, 4.0};
    double b_data[4] = {5.0, 6.0, 7.0, 8.0};
    auto a = SIMDVec<double, 4>::loadu(a_data);
    auto b = SIMDVec<double, 4>::loadu(b_data);
    CHECK(a.dot(b) == doctest::Approx(70.0));
}

#else

TEST_CASE("AVX intrinsics header compiles without AVX") { CHECK(true); }

#endif

