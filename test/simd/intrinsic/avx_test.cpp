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

#endif

#if defined(OPTINUM_HAS_AVX2)

TEST_CASE("AVX2 SIMDVec<int32_t,8> arithmetic") {
    alignas(32) int32_t a_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(32) int32_t b_data[8] = {8, 7, 6, 5, 4, 3, 2, 1};

    auto a = SIMDVec<int32_t, 8>::loadu(a_data);
    auto b = SIMDVec<int32_t, 8>::loadu(b_data);
    auto c = a + b;

    alignas(32) int32_t out[8];
    c.storeu(out);
    for (int i = 0; i < 8; ++i) {
        CHECK(out[i] == 9);
    }
    CHECK(c.hsum() == 72);
}

TEST_CASE("AVX2 SIMDVec<int32_t,8> multiply") {
    alignas(32) int32_t a_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(32) int32_t b_data[8] = {2, 2, 2, 2, 2, 2, 2, 2};

    auto a = SIMDVec<int32_t, 8>::loadu(a_data);
    auto b = SIMDVec<int32_t, 8>::loadu(b_data);
    auto c = a * b;

    alignas(32) int32_t out[8];
    c.storeu(out);
    CHECK(out[0] == 2);
    CHECK(out[1] == 4);
    CHECK(out[7] == 16);
}

TEST_CASE("AVX2 SIMDVec<int32_t,8> bitwise") {
    alignas(32) int32_t a_data[8] = {0xFF, 0xF0, 0x0F, 0x00, 0xFF, 0xF0, 0x0F, 0x00};
    alignas(32) int32_t b_data[8] = {0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F};

    auto a = SIMDVec<int32_t, 8>::loadu(a_data);
    auto b = SIMDVec<int32_t, 8>::loadu(b_data);

    auto and_result = a & b;
    auto or_result = a | b;

    alignas(32) int32_t out[8];
    and_result.storeu(out);
    CHECK(out[0] == 0x0F);
    CHECK(out[1] == 0x00);
    CHECK(out[2] == 0x0F);
    CHECK(out[3] == 0x00);

    or_result.storeu(out);
    CHECK(out[0] == 0xFF);
    CHECK(out[1] == 0xFF);
}

TEST_CASE("AVX2 SIMDVec<int32_t,8> reductions") {
    alignas(32) int32_t data[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    auto a = SIMDVec<int32_t, 8>::loadu(data);

    CHECK(a.hsum() == 36);
    CHECK(a.hmin() == 1);
    CHECK(a.hmax() == 8);
}

TEST_CASE("AVX2 SIMDVec<int32_t,8> abs and min/max") {
    alignas(32) int32_t a_data[8] = {-1, 2, -3, 4, -5, 6, -7, 8};
    alignas(32) int32_t b_data[8] = {1, -2, 3, -4, 5, -6, 7, -8};

    auto a = SIMDVec<int32_t, 8>::loadu(a_data);
    auto b = SIMDVec<int32_t, 8>::loadu(b_data);

    auto abs_a = a.abs();
    auto min_ab = SIMDVec<int32_t, 8>::min(a, b);
    auto max_ab = SIMDVec<int32_t, 8>::max(a, b);

    alignas(32) int32_t out[8];
    abs_a.storeu(out);
    CHECK(out[0] == 1);
    CHECK(out[2] == 3);
    CHECK(out[6] == 7);

    min_ab.storeu(out);
    CHECK(out[0] == -1);
    CHECK(out[1] == -2);

    max_ab.storeu(out);
    CHECK(out[0] == 1);
    CHECK(out[1] == 2);
}

TEST_CASE("AVX2 SIMDVec<int64_t,4> arithmetic") {
    alignas(32) int64_t a_data[4] = {100, 200, 300, 400};
    alignas(32) int64_t b_data[4] = {10, 20, 30, 40};

    auto a = SIMDVec<int64_t, 4>::loadu(a_data);
    auto b = SIMDVec<int64_t, 4>::loadu(b_data);

    auto sum = a + b;
    auto diff = a - b;
    auto prod = a * b;

    alignas(32) int64_t out[4];
    sum.storeu(out);
    CHECK(out[0] == 110);
    CHECK(out[1] == 220);
    CHECK(out[2] == 330);
    CHECK(out[3] == 440);

    diff.storeu(out);
    CHECK(out[0] == 90);
    CHECK(out[3] == 360);

    prod.storeu(out);
    CHECK(out[0] == 1000);
    CHECK(out[3] == 16000);
}

TEST_CASE("AVX2 SIMDVec<int64_t,4> reductions") {
    alignas(32) int64_t data[4] = {10, 20, 30, 40};

    auto a = SIMDVec<int64_t, 4>::loadu(data);

    CHECK(a.hsum() == 100);
    CHECK(a.hmin() == 10);
    CHECK(a.hmax() == 40);
}

#endif

#if !defined(OPTINUM_HAS_AVX) && !defined(OPTINUM_HAS_AVX2)
TEST_CASE("AVX intrinsics header compiles without AVX") { CHECK(true); }
#endif
