#include <doctest/doctest.h>
#include <optinum/simd/backend/matmul.hpp>

TEST_CASE("backend matmul column-major 2x2") {
    // A = [1 2; 3 4] (2x2), column-major => [1,3,2,4]
    float A[4] = {1.f, 3.f, 2.f, 4.f};
    // B = [5 6; 7 8] (2x2), column-major => [5,7,6,8]
    float B[4] = {5.f, 7.f, 6.f, 8.f};

    float C[4] = {};
    optinum::simd::backend::matmul<float, 2, 2, 2>(C, A, B);

    // C = [19 22; 43 50], column-major => [19,43,22,50]
    CHECK(C[0] == doctest::Approx(19.f));
    CHECK(C[1] == doctest::Approx(43.f));
    CHECK(C[2] == doctest::Approx(22.f));
    CHECK(C[3] == doctest::Approx(50.f));
}

TEST_CASE("backend matvec column-major 2x3") {
    // M = [1 2 3; 4 5 6] (2x3), column-major => [1,4,2,5,3,6]
    float M[6] = {1.f, 4.f, 2.f, 5.f, 3.f, 6.f};
    float x[3] = {1.f, 2.f, 3.f};
    float y[2] = {};

    optinum::simd::backend::matvec<float, 2, 3>(y, M, x);
    CHECK(y[0] == doctest::Approx(14.f));
    CHECK(y[1] == doctest::Approx(32.f));
}

