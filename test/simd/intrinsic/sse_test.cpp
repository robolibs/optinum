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

TEST_CASE("SSE SIMDVec<int32_t,4> arithmetic") {
    alignas(16) int32_t a_data[4] = {1, 2, 3, 4};
    alignas(16) int32_t b_data[4] = {5, 6, 7, 8};

    auto a = SIMDVec<int32_t, 4>::loadu(a_data);
    auto b = SIMDVec<int32_t, 4>::loadu(b_data);
    auto c = a + b;

    alignas(16) int32_t out[4];
    c.storeu(out);

    CHECK(out[0] == 6);
    CHECK(out[1] == 8);
    CHECK(out[2] == 10);
    CHECK(out[3] == 12);
    CHECK(c.hsum() == 36);
}

TEST_CASE("SSE SIMDVec<int32_t,4> multiply") {
    alignas(16) int32_t a_data[4] = {1, 2, 3, 4};
    alignas(16) int32_t b_data[4] = {5, 6, 7, 8};

    auto a = SIMDVec<int32_t, 4>::loadu(a_data);
    auto b = SIMDVec<int32_t, 4>::loadu(b_data);
    auto c = a * b;

    alignas(16) int32_t out[4];
    c.storeu(out);

    CHECK(out[0] == 5);
    CHECK(out[1] == 12);
    CHECK(out[2] == 21);
    CHECK(out[3] == 32);
}

TEST_CASE("SSE SIMDVec<int32_t,4> bitwise") {
    alignas(16) int32_t a_data[4] = {0xFF, 0xF0, 0x0F, 0x00};
    alignas(16) int32_t b_data[4] = {0x0F, 0x0F, 0x0F, 0x0F};

    auto a = SIMDVec<int32_t, 4>::loadu(a_data);
    auto b = SIMDVec<int32_t, 4>::loadu(b_data);

    auto and_result = a & b;
    auto or_result = a | b;
    auto xor_result = a ^ b;

    alignas(16) int32_t out[4];
    and_result.storeu(out);
    CHECK(out[0] == 0x0F);
    CHECK(out[1] == 0x00);
    CHECK(out[2] == 0x0F);
    CHECK(out[3] == 0x00);

    or_result.storeu(out);
    CHECK(out[0] == 0xFF);
    CHECK(out[1] == 0xFF);
    CHECK(out[2] == 0x0F);
    CHECK(out[3] == 0x0F);
}

TEST_CASE("SSE SIMDVec<int32_t,4> shifts") {
    alignas(16) int32_t data[4] = {1, 2, 4, 8};

    auto a = SIMDVec<int32_t, 4>::loadu(data);
    auto left = a << 2;
    auto right = a >> 1;

    alignas(16) int32_t out[4];
    left.storeu(out);
    CHECK(out[0] == 4);
    CHECK(out[1] == 8);
    CHECK(out[2] == 16);
    CHECK(out[3] == 32);

    right.storeu(out);
    CHECK(out[0] == 0);
    CHECK(out[1] == 1);
    CHECK(out[2] == 2);
    CHECK(out[3] == 4);
}

TEST_CASE("SSE SIMDVec<int32_t,4> reductions") {
    alignas(16) int32_t data[4] = {3, 1, 4, 2};

    auto a = SIMDVec<int32_t, 4>::loadu(data);

    CHECK(a.hsum() == 10);
    CHECK(a.hmin() == 1);
    CHECK(a.hmax() == 4);
    CHECK(a.hprod() == 24);
}

TEST_CASE("SSE SIMDVec<int32_t,4> abs and min/max") {
    alignas(16) int32_t a_data[4] = {-3, 1, -4, 2};
    alignas(16) int32_t b_data[4] = {1, 2, -2, -1};

    auto a = SIMDVec<int32_t, 4>::loadu(a_data);
    auto b = SIMDVec<int32_t, 4>::loadu(b_data);

    auto abs_a = a.abs();
    auto min_ab = SIMDVec<int32_t, 4>::min(a, b);
    auto max_ab = SIMDVec<int32_t, 4>::max(a, b);

    alignas(16) int32_t out[4];
    abs_a.storeu(out);
    CHECK(out[0] == 3);
    CHECK(out[1] == 1);
    CHECK(out[2] == 4);
    CHECK(out[3] == 2);

    min_ab.storeu(out);
    CHECK(out[0] == -3);
    CHECK(out[1] == 1);
    CHECK(out[2] == -4);
    CHECK(out[3] == -1);

    max_ab.storeu(out);
    CHECK(out[0] == 1);
    CHECK(out[1] == 2);
    CHECK(out[2] == -2);
    CHECK(out[3] == 2);
}

TEST_CASE("SSE SIMDVec<int64_t,2> arithmetic") {
    alignas(16) int64_t a_data[2] = {100, 200};
    alignas(16) int64_t b_data[2] = {10, 20};

    auto a = SIMDVec<int64_t, 2>::loadu(a_data);
    auto b = SIMDVec<int64_t, 2>::loadu(b_data);

    auto sum = a + b;
    auto diff = a - b;
    auto prod = a * b;

    alignas(16) int64_t out[2];
    sum.storeu(out);
    CHECK(out[0] == 110);
    CHECK(out[1] == 220);

    diff.storeu(out);
    CHECK(out[0] == 90);
    CHECK(out[1] == 180);

    prod.storeu(out);
    CHECK(out[0] == 1000);
    CHECK(out[1] == 4000);
}

TEST_CASE("SSE SIMDVec<int64_t,2> reductions") {
    alignas(16) int64_t data[2] = {100, 50};

    auto a = SIMDVec<int64_t, 2>::loadu(data);

    CHECK(a.hsum() == 150);
    CHECK(a.hmin() == 50);
    CHECK(a.hmax() == 100);
    CHECK(a.hprod() == 5000);
}

#else

TEST_CASE("SSE intrinsics header compiles without SSE2") { CHECK(true); }

#endif
