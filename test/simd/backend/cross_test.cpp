#include <doctest/doctest.h>
#include <optinum/simd/backend/cross.hpp>

#include <cmath>

TEST_CASE("backend cross - basic") {
    double a[3] = {1.0, 0.0, 0.0};
    double b[3] = {0.0, 1.0, 0.0};
    double out[3] = {};

    optinum::simd::backend::cross(a, b, out);

    // i × j = k
    CHECK(out[0] == doctest::Approx(0.0));
    CHECK(out[1] == doctest::Approx(0.0));
    CHECK(out[2] == doctest::Approx(1.0));
}

TEST_CASE("backend cross - unit vectors") {
    SUBCASE("i × j = k") {
        float a[3] = {1.f, 0.f, 0.f};
        float b[3] = {0.f, 1.f, 0.f};
        float out[3] = {};
        optinum::simd::backend::cross(a, b, out);
        CHECK(out[0] == doctest::Approx(0.f));
        CHECK(out[1] == doctest::Approx(0.f));
        CHECK(out[2] == doctest::Approx(1.f));
    }

    SUBCASE("j × k = i") {
        float a[3] = {0.f, 1.f, 0.f};
        float b[3] = {0.f, 0.f, 1.f};
        float out[3] = {};
        optinum::simd::backend::cross(a, b, out);
        CHECK(out[0] == doctest::Approx(1.f));
        CHECK(out[1] == doctest::Approx(0.f));
        CHECK(out[2] == doctest::Approx(0.f));
    }

    SUBCASE("k × i = j") {
        float a[3] = {0.f, 0.f, 1.f};
        float b[3] = {1.f, 0.f, 0.f};
        float out[3] = {};
        optinum::simd::backend::cross(a, b, out);
        CHECK(out[0] == doctest::Approx(0.f));
        CHECK(out[1] == doctest::Approx(1.f));
        CHECK(out[2] == doctest::Approx(0.f));
    }
}

TEST_CASE("backend cross - anticommutativity") {
    double a[3] = {1.0, 2.0, 3.0};
    double b[3] = {4.0, 5.0, 6.0};
    double ab[3] = {}, ba[3] = {};

    optinum::simd::backend::cross(a, b, ab);
    optinum::simd::backend::cross(b, a, ba);

    // a × b = -(b × a)
    CHECK(ab[0] == doctest::Approx(-ba[0]));
    CHECK(ab[1] == doctest::Approx(-ba[1]));
    CHECK(ab[2] == doctest::Approx(-ba[2]));
}

TEST_CASE("backend cross - self cross is zero") {
    double a[3] = {1.0, 2.0, 3.0};
    double out[3] = {};

    optinum::simd::backend::cross(a, a, out);

    CHECK(out[0] == doctest::Approx(0.0));
    CHECK(out[1] == doctest::Approx(0.0));
    CHECK(out[2] == doctest::Approx(0.0));
}

TEST_CASE("backend cross_add") {
    float a[3] = {1.f, 0.f, 0.f};
    float b[3] = {0.f, 1.f, 0.f};
    float out[3] = {10.f, 20.f, 30.f};

    optinum::simd::backend::cross_add(a, b, out);

    // out += i × j = out + k
    CHECK(out[0] == doctest::Approx(10.f));
    CHECK(out[1] == doctest::Approx(20.f));
    CHECK(out[2] == doctest::Approx(31.f));
}

TEST_CASE("backend cross_scale") {
    double a[3] = {1.0, 0.0, 0.0};
    double b[3] = {0.0, 1.0, 0.0};
    double out[3] = {};

    optinum::simd::backend::cross_scale(a, b, 2.5, out);

    // 2.5 * (i × j) = 2.5 * k
    CHECK(out[0] == doctest::Approx(0.0));
    CHECK(out[1] == doctest::Approx(0.0));
    CHECK(out[2] == doctest::Approx(2.5));
}

TEST_CASE("backend cross_scale_add") {
    float a[3] = {1.f, 0.f, 0.f};
    float b[3] = {0.f, 1.f, 0.f};
    float out[3] = {1.f, 2.f, 3.f};

    optinum::simd::backend::cross_scale_add(a, b, 0.5f, out);

    // out += 0.5 * (i × j) = out + 0.5 * k
    CHECK(out[0] == doctest::Approx(1.f));
    CHECK(out[1] == doctest::Approx(2.f));
    CHECK(out[2] == doctest::Approx(3.5f));
}

TEST_CASE("backend cross_batch") {
    using pack_t = optinum::simd::pack<float, 4>;

    // 4 pairs of vectors
    alignas(16) float ax_arr[4] = {1.f, 0.f, 0.f, 1.f};
    alignas(16) float ay_arr[4] = {0.f, 1.f, 0.f, 1.f};
    alignas(16) float az_arr[4] = {0.f, 0.f, 1.f, 1.f};
    alignas(16) float bx_arr[4] = {0.f, 0.f, 1.f, 2.f};
    alignas(16) float by_arr[4] = {1.f, 0.f, 0.f, 2.f};
    alignas(16) float bz_arr[4] = {0.f, 1.f, 0.f, 2.f};

    pack_t ax = pack_t::loadu(ax_arr);
    pack_t ay = pack_t::loadu(ay_arr);
    pack_t az = pack_t::loadu(az_arr);
    pack_t bx = pack_t::loadu(bx_arr);
    pack_t by = pack_t::loadu(by_arr);
    pack_t bz = pack_t::loadu(bz_arr);

    pack_t ox, oy, oz;
    optinum::simd::backend::cross_batch(ax, ay, az, bx, by, bz, ox, oy, oz);

    // Check first result: i × j = k
    CHECK(ox[0] == doctest::Approx(0.f));
    CHECK(oy[0] == doctest::Approx(0.f));
    CHECK(oz[0] == doctest::Approx(1.f));

    // Check second result: j × k = i
    CHECK(ox[1] == doctest::Approx(1.f));
    CHECK(oy[1] == doctest::Approx(0.f));
    CHECK(oz[1] == doctest::Approx(0.f));

    // Check third result: k × i = j
    CHECK(ox[2] == doctest::Approx(0.f));
    CHECK(oy[2] == doctest::Approx(1.f));
    CHECK(oz[2] == doctest::Approx(0.f));
}

TEST_CASE("backend cross_batch_fma") {
    using pack_t = optinum::simd::pack<double, 4>;

    alignas(32) double ax_arr[4] = {1.0, 2.0, 3.0, 4.0};
    alignas(32) double ay_arr[4] = {5.0, 6.0, 7.0, 8.0};
    alignas(32) double az_arr[4] = {9.0, 10.0, 11.0, 12.0};
    alignas(32) double bx_arr[4] = {12.0, 11.0, 10.0, 9.0};
    alignas(32) double by_arr[4] = {8.0, 7.0, 6.0, 5.0};
    alignas(32) double bz_arr[4] = {4.0, 3.0, 2.0, 1.0};

    pack_t ax = pack_t::loadu(ax_arr);
    pack_t ay = pack_t::loadu(ay_arr);
    pack_t az = pack_t::loadu(az_arr);
    pack_t bx = pack_t::loadu(bx_arr);
    pack_t by = pack_t::loadu(by_arr);
    pack_t bz = pack_t::loadu(bz_arr);

    pack_t ox, oy, oz;
    optinum::simd::backend::cross_batch_fma(ax, ay, az, bx, by, bz, ox, oy, oz);

    // Verify against scalar cross product
    for (int i = 0; i < 4; ++i) {
        double a[3] = {ax_arr[i], ay_arr[i], az_arr[i]};
        double b[3] = {bx_arr[i], by_arr[i], bz_arr[i]};
        double expected[3];
        optinum::simd::backend::cross(a, b, expected);

        CHECK(ox[i] == doctest::Approx(expected[0]));
        CHECK(oy[i] == doctest::Approx(expected[1]));
        CHECK(oz[i] == doctest::Approx(expected[2]));
    }
}

TEST_CASE("backend skew") {
    double omega[3] = {1.0, 2.0, 3.0};
    double mat[9] = {};

    optinum::simd::backend::skew(omega, mat);

    // Expected skew-symmetric matrix (column-major):
    // [ 0  -3   2]
    // [ 3   0  -1]
    // [-2   1   0]
    CHECK(mat[0] == doctest::Approx(0.0));  // (0,0)
    CHECK(mat[1] == doctest::Approx(3.0));  // (1,0)
    CHECK(mat[2] == doctest::Approx(-2.0)); // (2,0)
    CHECK(mat[3] == doctest::Approx(-3.0)); // (0,1)
    CHECK(mat[4] == doctest::Approx(0.0));  // (1,1)
    CHECK(mat[5] == doctest::Approx(1.0));  // (2,1)
    CHECK(mat[6] == doctest::Approx(2.0));  // (0,2)
    CHECK(mat[7] == doctest::Approx(-1.0)); // (1,2)
    CHECK(mat[8] == doctest::Approx(0.0));  // (2,2)
}

TEST_CASE("backend vee") {
    // Create a skew-symmetric matrix
    double omega_in[3] = {1.5, 2.5, 3.5};
    double mat[9] = {};
    optinum::simd::backend::skew(omega_in, mat);

    // Extract back
    double omega_out[3] = {};
    optinum::simd::backend::vee(mat, omega_out);

    CHECK(omega_out[0] == doctest::Approx(omega_in[0]));
    CHECK(omega_out[1] == doctest::Approx(omega_in[1]));
    CHECK(omega_out[2] == doctest::Approx(omega_in[2]));
}

TEST_CASE("backend skew-vee roundtrip") {
    float omega[3] = {-1.f, 0.5f, 2.f};
    float mat[9] = {};
    float recovered[3] = {};

    optinum::simd::backend::skew(omega, mat);
    optinum::simd::backend::vee(mat, recovered);

    CHECK(recovered[0] == doctest::Approx(omega[0]));
    CHECK(recovered[1] == doctest::Approx(omega[1]));
    CHECK(recovered[2] == doctest::Approx(omega[2]));
}

TEST_CASE("backend cross matches skew-matrix multiply") {
    // For any vectors a, b: a × b = [a]× * b
    double a[3] = {1.0, 2.0, 3.0};
    double b[3] = {4.0, 5.0, 6.0};

    // Compute cross product directly
    double cross_result[3] = {};
    optinum::simd::backend::cross(a, b, cross_result);

    // Compute via skew matrix multiplication
    double skew_a[9] = {};
    optinum::simd::backend::skew(a, skew_a);

    // mat * b (column-major)
    double mat_result[3] = {};
    for (int i = 0; i < 3; ++i) {
        mat_result[i] = 0.0;
        for (int j = 0; j < 3; ++j) {
            mat_result[i] += skew_a[j * 3 + i] * b[j];
        }
    }

    CHECK(cross_result[0] == doctest::Approx(mat_result[0]));
    CHECK(cross_result[1] == doctest::Approx(mat_result[1]));
    CHECK(cross_result[2] == doctest::Approx(mat_result[2]));
}
