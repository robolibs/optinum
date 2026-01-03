#include <doctest/doctest.h>
#include <optinum/simd/backend/matmul.hpp>

#include <cmath>

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

TEST_CASE("backend matmul column-major 4x4") {
    // Identity * A = A
    float I[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; // column-major identity
    float A[16], C[16] = {};

    for (int i = 0; i < 16; ++i) {
        A[i] = static_cast<float>(i + 1);
    }

    optinum::simd::backend::matmul<float, 4, 4, 4>(C, I, A);

    for (int i = 0; i < 16; ++i) {
        CHECK(C[i] == doctest::Approx(A[i]));
    }
}

TEST_CASE("backend matmul column-major 8x8") {
    double A[64], B[64], C[64] = {};

    // Fill with simple pattern
    for (int i = 0; i < 64; ++i) {
        A[i] = static_cast<double>(i % 8 + 1);
        B[i] = static_cast<double>((i / 8) + 1);
    }

    optinum::simd::backend::matmul<double, 8, 8, 8>(C, A, B);

    // Verify a few elements manually
    // C(0,0) = sum of A(0,k) * B(k,0) for k=0..7
    // In column-major: A(0,k) = A[k*8], B(k,0) = B[k]
    double c00 = 0.0;
    for (int k = 0; k < 8; ++k) {
        c00 += A[k * 8] * B[k];
    }
    CHECK(C[0] == doctest::Approx(c00));
}

TEST_CASE("backend matmul non-square 4x8 * 8x4") {
    float A[32], B[32], C[16] = {};

    for (int i = 0; i < 32; ++i) {
        A[i] = static_cast<float>(i + 1);
        B[i] = static_cast<float>(32 - i);
    }

    optinum::simd::backend::matmul<float, 4, 8, 4>(C, A, B);

    // Verify result dimensions and a sample element
    // C is 4x4, so 16 elements
    // C(0,0) = sum of A(0,k) * B(k,0) for k=0..7
    float c00 = 0.f;
    for (int k = 0; k < 8; ++k) {
        c00 += A[k * 4] * B[k]; // column-major access
    }
    CHECK(C[0] == doctest::Approx(c00));
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

TEST_CASE("backend matvec column-major 8x8") {
    double M[64], x[8], y[8] = {};

    // M = identity
    for (int i = 0; i < 64; ++i)
        M[i] = 0.0;
    for (int i = 0; i < 8; ++i)
        M[i * 8 + i] = 1.0; // diagonal

    for (int i = 0; i < 8; ++i) {
        x[i] = static_cast<double>(i + 1);
    }

    optinum::simd::backend::matvec<double, 8, 8>(y, M, x);

    // y should equal x
    for (int i = 0; i < 8; ++i) {
        CHECK(y[i] == doctest::Approx(x[i]));
    }
}

TEST_CASE("backend matvec column-major 16x16") {
    float M[256], x[16], y[16] = {};

    // Fill M with row index + 1
    for (int col = 0; col < 16; ++col) {
        for (int row = 0; row < 16; ++row) {
            M[col * 16 + row] = static_cast<float>(row + 1);
        }
    }

    // x = all ones
    for (int i = 0; i < 16; ++i) {
        x[i] = 1.f;
    }

    optinum::simd::backend::matvec<float, 16, 16>(y, M, x);

    // y[i] = sum of M(i, j) for j=0..15 = 16 * (i+1)
    for (int i = 0; i < 16; ++i) {
        CHECK(y[i] == doctest::Approx(16.f * (i + 1)));
    }
}

TEST_CASE("backend matvec non-square 32x17") {
    float M[32 * 17], x[17], y[32] = {};

    for (int i = 0; i < 32 * 17; ++i) {
        M[i] = static_cast<float>(i % 10);
    }
    for (int i = 0; i < 17; ++i) {
        x[i] = 1.f;
    }

    optinum::simd::backend::matvec<float, 32, 17>(y, M, x);

    // Verify first element
    float y0 = 0.f;
    for (int j = 0; j < 17; ++j) {
        y0 += M[j * 32]; // M(0, j) in column-major
    }
    CHECK(y[0] == doctest::Approx(y0));
}
