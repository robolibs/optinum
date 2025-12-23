#include <doctest/doctest.h>
#include <optinum/simd/intrinsic/avx512.hpp>

using optinum::simd::SIMDVec;

#if defined(OPTINUM_HAS_AVX512F)

TEST_CASE("AVX512 SIMDVec<float,16> hsum") {
    alignas(64) float a[16];
    for (int i = 0; i < 16; ++i)
        a[i] = 1.f;
    auto v = SIMDVec<float, 16>::load(a);
    CHECK(v.hsum() == doctest::Approx(16.f));
}

TEST_CASE("AVX512 SIMDVec<double,8> hsum") {
    alignas(64) double a[8];
    for (int i = 0; i < 8; ++i)
        a[i] = 2.0;
    auto v = SIMDVec<double, 8>::load(a);
    CHECK(v.hsum() == doctest::Approx(16.0));
}

#else

TEST_CASE("AVX512 intrinsics header compiles without AVX512") { CHECK(true); }

#endif

