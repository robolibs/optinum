// =============================================================================
// test/simd/backend/det_inverse_small_test.cpp
// Tests for specialized small matrix determinant and inverse kernels
// =============================================================================

#include <cmath>
#include <doctest/doctest.h>
#include <optinum/simd/backend/det_small.hpp>
#include <optinum/simd/backend/inverse_small.hpp>

namespace backend = optinum::simd::backend;

// =============================================================================
// 2x2 Tests
// =============================================================================

TEST_CASE("Backend: det_2x2 - Identity matrix") {
    // | 1  0 |
    // | 0  1 |
    float data[4] = {1.0f, 0.0f, 0.0f, 1.0f}; // column-major

    float det = backend::det_2x2(data);
    CHECK(det == doctest::Approx(1.0f));
}

TEST_CASE("Backend: det_2x2 - General matrix") {
    // | 3   5 |
    // | 2  -1 |
    float data[4] = {3.0f, 2.0f, 5.0f, -1.0f};

    float det = backend::det_2x2(data);
    // det = 3*(-1) - 5*2 = -3 - 10 = -13
    CHECK(det == doctest::Approx(-13.0f));
}

TEST_CASE("Backend: det_2x2 - Singular matrix") {
    // | 2  4 |
    // | 1  2 |
    float data[4] = {2.0f, 1.0f, 4.0f, 2.0f};

    float det = backend::det_2x2(data);
    // det = 2*2 - 4*1 = 4 - 4 = 0
    CHECK(det == doctest::Approx(0.0f));
}

TEST_CASE("Backend: inverse_2x2 - Identity matrix") {
    // | 1  0 |
    // | 0  1 |
    float data[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float result[4];

    bool success = backend::inverse_2x2(data, result);
    CHECK(success);

    // Inverse of identity is identity
    CHECK(result[0] == doctest::Approx(1.0f));
    CHECK(result[1] == doctest::Approx(0.0f));
    CHECK(result[2] == doctest::Approx(0.0f));
    CHECK(result[3] == doctest::Approx(1.0f));
}

TEST_CASE("Backend: inverse_2x2 - General matrix") {
    // | 4  7 |
    // | 2  6 |
    float data[4] = {4.0f, 2.0f, 7.0f, 6.0f};
    float result[4];

    bool success = backend::inverse_2x2(data, result);
    CHECK(success);

    // det = 4*6 - 7*2 = 24 - 14 = 10
    // inv = (1/10) * | 6  -7 |
    //                | -2   4 |
    CHECK(result[0] == doctest::Approx(0.6f));  // 6/10
    CHECK(result[1] == doctest::Approx(-0.2f)); // -2/10
    CHECK(result[2] == doctest::Approx(-0.7f)); // -7/10
    CHECK(result[3] == doctest::Approx(0.4f));  // 4/10
}

TEST_CASE("Backend: inverse_2x2 - Singular matrix fails") {
    // | 2  4 |
    // | 1  2 |
    float data[4] = {2.0f, 1.0f, 4.0f, 2.0f};
    float result[4];

    bool success = backend::inverse_2x2(data, result);
    CHECK(!success); // Should fail for singular matrix
}

// =============================================================================
// 3x3 Tests
// =============================================================================

TEST_CASE("Backend: det_3x3 - Identity matrix") {
    // | 1  0  0 |
    // | 0  1  0 |
    // | 0  0  1 |
    float data[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};

    float det = backend::det_3x3(data);
    CHECK(det == doctest::Approx(1.0f));
}

TEST_CASE("Backend: det_3x3 - General matrix") {
    // | 1  2  3 |
    // | 0  1  4 |
    // | 5  6  0 |
    float data[9] = {1.0f, 0.0f, 5.0f, 2.0f, 1.0f, 6.0f, 3.0f, 4.0f, 0.0f};

    float det = backend::det_3x3(data);
    // det = 1*(1*0 - 4*6) - 2*(0*0 - 4*5) + 3*(0*6 - 1*5)
    //     = 1*(-24) - 2*(-20) + 3*(-5)
    //     = -24 + 40 - 15 = 1
    CHECK(det == doctest::Approx(1.0f));
}

TEST_CASE("Backend: det_3x3 - Singular matrix") {
    // | 1  2  3 |
    // | 2  4  6 |  (second row is 2x first row)
    // | 0  0  1 |
    float data[9] = {1.0f, 2.0f, 0.0f, 2.0f, 4.0f, 0.0f, 3.0f, 6.0f, 1.0f};

    float det = backend::det_3x3(data);
    CHECK(std::abs(det) < 1e-6f); // Should be near zero
}

TEST_CASE("Backend: inverse_3x3 - Identity matrix") {
    // | 1  0  0 |
    // | 0  1  0 |
    // | 0  0  1 |
    float data[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    float result[9];

    bool success = backend::inverse_3x3(data, result);
    CHECK(success);

    // Inverse of identity is identity
    for (int i = 0; i < 9; ++i) {
        CHECK(result[i] == doctest::Approx(data[i]));
    }
}

TEST_CASE("Backend: inverse_3x3 - General matrix") {
    // | 1  0  2 |
    // | 2  1  0 |
    // | 0  1  1 |
    float data[9] = {1.0f, 2.0f, 0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 0.0f, 1.0f};
    float result[9];

    bool success = backend::inverse_3x3(data, result);
    CHECK(success);

    // Verify A * A^-1 = I
    // Multiply result matrix by original
    float product[9] = {0};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            for (int k = 0; k < 3; ++k) {
                product[col * 3 + row] += data[k * 3 + row] * result[col * 3 + k];
            }
        }
    }

    // Check if product is identity
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            CHECK(product[j * 3 + i] == doctest::Approx(expected).epsilon(1e-5));
        }
    }
}

TEST_CASE("Backend: inverse_3x3 - Singular matrix fails") {
    // | 1  2  3 |
    // | 2  4  6 |
    // | 0  0  1 |
    float data[9] = {1.0f, 2.0f, 0.0f, 2.0f, 4.0f, 0.0f, 3.0f, 6.0f, 1.0f};
    float result[9];

    bool success = backend::inverse_3x3(data, result);
    CHECK(!success);
}

// =============================================================================
// 4x4 Tests
// =============================================================================

TEST_CASE("Backend: det_4x4 - Identity matrix") {
    // 4x4 identity
    float data[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

    float det = backend::det_4x4(data);
    CHECK(det == doctest::Approx(1.0f));
}

TEST_CASE("Backend: det_4x4 - Diagonal matrix") {
    // | 2  0  0  0 |
    // | 0  3  0  0 |
    // | 0  0  4  0 |
    // | 0  0  0  5 |
    float data[16] = {2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f};

    float det = backend::det_4x4(data);
    // det = 2 * 3 * 4 * 5 = 120
    CHECK(det == doctest::Approx(120.0f));
}

TEST_CASE("Backend: det_4x4 - General matrix") {
    // A well-conditioned test matrix
    float data[16] = {4.0f, 3.0f, 2.0f, 1.0f, 3.0f, 4.0f, 3.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 1.0f, 2.0f, 3.0f, 4.0f};

    float det = backend::det_4x4(data);
    // Actual determinant for this matrix: 20
    CHECK(det == doctest::Approx(20.0f).epsilon(1e-4));

    // Non-zero determinant means invertible
    CHECK(std::abs(det) > 1e-6f);
}

TEST_CASE("Backend: inverse_4x4 - Identity matrix") {
    float data[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    float result[16];

    bool success = backend::inverse_4x4(data, result);
    CHECK(success);

    // Inverse of identity is identity
    for (int i = 0; i < 16; ++i) {
        CHECK(result[i] == doctest::Approx(data[i]));
    }
}

TEST_CASE("Backend: inverse_4x4 - General matrix") {
    // A well-conditioned matrix
    float data[16] = {4.0f, 3.0f, 2.0f, 1.0f, 3.0f, 4.0f, 3.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    float result[16];

    bool success = backend::inverse_4x4(data, result);
    CHECK(success);

    // Verify A * A^-1 = I
    float product[16] = {0};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            for (int k = 0; k < 4; ++k) {
                product[col * 4 + row] += data[k * 4 + row] * result[col * 4 + k];
            }
        }
    }

    // Check if product is identity
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            CHECK(product[j * 4 + i] == doctest::Approx(expected).epsilon(1e-4));
        }
    }
}

TEST_CASE("Backend: inverse_4x4 - Singular matrix fails") {
    // Matrix with dependent rows
    float data[16] = {1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 4.0f, 6.0f, 8.0f, // 2x first row
                      0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    float result[16];

    bool success = backend::inverse_4x4(data, result);
    CHECK(!success);
}

// =============================================================================
// Double Precision Tests
// =============================================================================

TEST_CASE("Backend: det_2x2 - Double precision") {
    double data[4] = {3.0, 2.0, 5.0, -1.0};
    double det = backend::det_2x2(data);
    CHECK(det == doctest::Approx(-13.0));
}

TEST_CASE("Backend: det_3x3 - Double precision") {
    double data[9] = {1.0, 0.0, 5.0, 2.0, 1.0, 6.0, 3.0, 4.0, 0.0};
    double det = backend::det_3x3(data);
    CHECK(det == doctest::Approx(1.0));
}

TEST_CASE("Backend: det_4x4 - Double precision") {
    double data[16] = {2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0};
    double det = backend::det_4x4(data);
    CHECK(det == doctest::Approx(120.0));
}
