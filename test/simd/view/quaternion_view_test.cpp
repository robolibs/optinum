// =============================================================================
// test/simd/view/quaternion_view_test.cpp
// Tests for quaternion_view - transparent SIMD access to quaternion arrays
// =============================================================================

#include <doctest/doctest.h>
#include <optinum/simd/bridge.hpp>
#include <optinum/simd/quaternion.hpp>

#include <cmath>

namespace on = optinum::simd;
namespace dp = datapod;

// =============================================================================
// quaternion_view Basic Tests
// =============================================================================

TEST_CASE("quaternion_view - Construction and size queries") {
    using quat_t = dp::mat::quaternion<double>;

    quat_t data[8];
    for (std::size_t i = 0; i < 8; ++i) {
        data[i] = quat_t::identity();
    }

    SUBCASE("From raw pointer") {
        auto qv = on::view(data, 8);
        CHECK(qv.size() == 8);
        CHECK_FALSE(qv.empty());
    }

    SUBCASE("From C-style array") {
        auto qv = on::view(data);
        CHECK(qv.size() == 8);
    }

    SUBCASE("Const view") {
        const quat_t *const_ptr = data;
        auto qv = on::view(const_ptr, 8);
        CHECK(qv.size() == 8);
    }
}

TEST_CASE("quaternion_view - Element access") {
    using quat_t = dp::mat::quaternion<double>;

    quat_t data[4] = {
        {1.0, 0.0, 0.0, 0.0},     // identity
        {0.707, 0.0, 0.707, 0.0}, // 90 deg Y
        {0.5, 0.5, 0.5, 0.5},     // some rotation
        {0.0, 1.0, 0.0, 0.0}      // 180 deg X
    };

    auto qv = on::view(data);

    SUBCASE("Operator[]") {
        CHECK(qv[0].w == doctest::Approx(1.0));
        CHECK(qv[1].y == doctest::Approx(0.707));
        CHECK(qv[3].x == doctest::Approx(1.0));
    }

    SUBCASE("Data pointer") { CHECK(qv.data() == data); }

    SUBCASE("Iteration") {
        int count = 0;
        for (const auto &q : qv) {
            (void)q;
            count++;
        }
        CHECK(count == 4);
    }
}

// =============================================================================
// In-place Operations Tests
// =============================================================================

TEST_CASE("quaternion_view - Conjugate in-place") {
    using quat_t = dp::mat::quaternion<double>;

    quat_t data[4] = {{1.0, 0.0, 0.0, 0.0}, {0.5, 0.5, 0.5, 0.5}, {0.707, 0.0, 0.707, 0.0}, {0.0, 1.0, 0.0, 0.0}};

    auto qv = on::view(data);
    qv.conjugate_inplace();

    // Conjugate negates imaginary parts
    CHECK(data[0].w == doctest::Approx(1.0));
    CHECK(data[0].x == doctest::Approx(0.0));

    CHECK(data[1].w == doctest::Approx(0.5));
    CHECK(data[1].x == doctest::Approx(-0.5));
    CHECK(data[1].y == doctest::Approx(-0.5));
    CHECK(data[1].z == doctest::Approx(-0.5));

    CHECK(data[3].x == doctest::Approx(-1.0));
}

TEST_CASE("quaternion_view - Normalize in-place") {
    using quat_t = dp::mat::quaternion<double>;

    quat_t data[3] = {
        {3.0, 4.0, 0.0, 0.0}, // norm = 5
        {1.0, 1.0, 1.0, 1.0}, // norm = 2
        {0.0, 0.0, 3.0, 4.0}  // norm = 5
    };

    auto qv = on::view(data);
    qv.normalize_inplace();

    // Check normalized
    CHECK(data[0].w == doctest::Approx(0.6)); // 3/5
    CHECK(data[0].x == doctest::Approx(0.8)); // 4/5

    CHECK(data[1].w == doctest::Approx(0.5));
    CHECK(data[1].x == doctest::Approx(0.5));
    CHECK(data[1].y == doctest::Approx(0.5));
    CHECK(data[1].z == doctest::Approx(0.5));

    CHECK(data[2].y == doctest::Approx(0.6)); // 3/5
    CHECK(data[2].z == doctest::Approx(0.8)); // 4/5

    // Verify unit norms
    for (int i = 0; i < 3; ++i) {
        double norm =
            std::sqrt(data[i].w * data[i].w + data[i].x * data[i].x + data[i].y * data[i].y + data[i].z * data[i].z);
        CHECK(norm == doctest::Approx(1.0));
    }
}

TEST_CASE("quaternion_view - Inverse in-place") {
    using quat_t = dp::mat::quaternion<double>;

    quat_t data[2] = {
        {3.0, 4.0, 0.0, 0.0}, // norm = 5, inverse = (3, -4, 0, 0) / 25
        {1.0, 0.0, 0.0, 0.0}  // identity, inverse = identity
    };

    auto qv = on::view(data);
    qv.inverse_inplace();

    CHECK(data[0].w == doctest::Approx(0.12));  // 3/25
    CHECK(data[0].x == doctest::Approx(-0.16)); // -4/25

    CHECK(data[1].w == doctest::Approx(1.0));
    CHECK(data[1].x == doctest::Approx(0.0));
}

// =============================================================================
// Operations Returning New Array Tests
// =============================================================================

TEST_CASE("quaternion_view - Conjugate to output") {
    using quat_t = dp::mat::quaternion<double>;

    quat_t input[2] = {{0.5, 0.5, 0.5, 0.5}, {0.707, 0.0, 0.707, 0.0}};
    quat_t output[2];

    auto qv_in = on::view(input);
    (void)qv_in.conjugate_to(output);

    // Input unchanged
    CHECK(input[0].x == doctest::Approx(0.5));

    // Output is conjugate
    CHECK(output[0].w == doctest::Approx(0.5));
    CHECK(output[0].x == doctest::Approx(-0.5));
    CHECK(output[0].y == doctest::Approx(-0.5));
    CHECK(output[0].z == doctest::Approx(-0.5));

    CHECK(output[1].w == doctest::Approx(0.707));
    CHECK(output[1].y == doctest::Approx(-0.707));
}

TEST_CASE("quaternion_view - Normalized to output") {
    using quat_t = dp::mat::quaternion<double>;

    quat_t input[2] = {{3.0, 4.0, 0.0, 0.0}, {0.0, 0.0, 3.0, 4.0}};
    quat_t output[2];

    auto qv_in = on::view(input);
    (void)qv_in.normalized_to(output);

    CHECK(output[0].w == doctest::Approx(0.6));
    CHECK(output[0].x == doctest::Approx(0.8));
    CHECK(output[1].y == doctest::Approx(0.6));
    CHECK(output[1].z == doctest::Approx(0.8));
}

// =============================================================================
// Binary Operations Tests
// =============================================================================

TEST_CASE("quaternion_view - Hamilton product") {
    using quat_t = dp::mat::quaternion<double>;

    // Test i * j = k
    quat_t a[2] = {
        {0.0, 1.0, 0.0, 0.0}, // pure i
        {1.0, 0.0, 0.0, 0.0}  // identity
    };
    quat_t b[2] = {
        {0.0, 0.0, 1.0, 0.0},    // pure j
        {0.707, 0.0, 0.707, 0.0} // 90 deg Y
    };
    quat_t result[2];

    auto qv_a = on::view(a);
    auto qv_b = on::view(b);
    (void)qv_a.multiply_to(qv_b, result);

    // i * j = k
    CHECK(result[0].w == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(result[0].x == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(result[0].y == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(result[0].z == doctest::Approx(1.0));

    // identity * q = q
    CHECK(result[1].w == doctest::Approx(0.707));
    CHECK(result[1].y == doctest::Approx(0.707));
}

TEST_CASE("quaternion_view - SLERP interpolation") {
    using quat_t = dp::mat::quaternion<double>;

    // Interpolate between identity and 90-deg Z rotation
    const double angle = M_PI / 2.0;
    quat_t a[2] = {quat_t::identity(), quat_t::identity()};
    quat_t b[2] = {{std::cos(angle / 2.0), 0.0, 0.0, std::sin(angle / 2.0)},
                   {std::cos(angle / 2.0), 0.0, 0.0, std::sin(angle / 2.0)}};
    quat_t result[2];

    auto qv_a = on::view(a);
    auto qv_b = on::view(b);

    SUBCASE("t=0 gives first quaternion") {
        (void)qv_a.slerp_to(qv_b, 0.0, result);
        CHECK(result[0].w == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(result[0].z == doctest::Approx(0.0).epsilon(1e-6));
    }

    SUBCASE("t=1 gives second quaternion") {
        (void)qv_a.slerp_to(qv_b, 1.0, result);
        CHECK(result[0].w == doctest::Approx(std::cos(angle / 2.0)).epsilon(1e-6));
        CHECK(result[0].z == doctest::Approx(std::sin(angle / 2.0)).epsilon(1e-6));
    }

    SUBCASE("t=0.5 gives halfway rotation") {
        (void)qv_a.slerp_to(qv_b, 0.5, result);
        // Halfway between 0 and 90 deg = 45 deg
        CHECK(result[0].w == doctest::Approx(std::cos(M_PI / 8.0)).epsilon(1e-3));
        CHECK(result[0].z == doctest::Approx(std::sin(M_PI / 8.0)).epsilon(1e-3));
    }
}

TEST_CASE("quaternion_view - NLERP interpolation") {
    using quat_t = dp::mat::quaternion<double>;

    quat_t a[2] = {quat_t::identity(), quat_t::identity()};
    const double angle = M_PI / 2.0;
    quat_t b[2] = {{std::cos(angle / 2.0), 0.0, 0.0, std::sin(angle / 2.0)},
                   {std::cos(angle / 2.0), 0.0, 0.0, std::sin(angle / 2.0)}};
    quat_t result[2];

    auto qv_a = on::view(a);
    auto qv_b = on::view(b);
    (void)qv_a.nlerp_to(qv_b, 0.5, result);

    // NLERP should give normalized result
    double norm = std::sqrt(result[0].w * result[0].w + result[0].x * result[0].x + result[0].y * result[0].y +
                            result[0].z * result[0].z);
    CHECK(norm == doctest::Approx(1.0));
}

// =============================================================================
// Reduction Operations Tests
// =============================================================================

TEST_CASE("quaternion_view - Norms") {
    using quat_t = dp::mat::quaternion<double>;

    quat_t data[3] = {
        {1.0, 0.0, 0.0, 0.0}, // norm = 1
        {3.0, 4.0, 0.0, 0.0}, // norm = 5
        {1.0, 1.0, 1.0, 1.0}  // norm = 2
    };
    double norms[3];

    auto qv = on::view(data);
    qv.norms_to(norms);

    CHECK(norms[0] == doctest::Approx(1.0));
    CHECK(norms[1] == doctest::Approx(5.0));
    CHECK(norms[2] == doctest::Approx(2.0));
}

TEST_CASE("quaternion_view - Dot product") {
    using quat_t = dp::mat::quaternion<double>;

    quat_t a[2] = {
        {0.5, 0.5, 0.5, 0.5}, {0.0, 1.0, 0.0, 0.0} // pure i
    };
    quat_t b[2] = {
        {0.5, 0.5, 0.5, 0.5}, {0.0, 0.0, 1.0, 0.0} // pure j
    };
    double dots[2];

    auto qv_a = on::view(a);
    auto qv_b = on::view(b);
    qv_a.dot_to(qv_b, dots);

    CHECK(dots[0] == doctest::Approx(1.0)); // same quaternion
    CHECK(dots[1] == doctest::Approx(0.0)); // orthogonal
}

// =============================================================================
// Rotation Operations Tests
// =============================================================================

TEST_CASE("quaternion_view - Rotate vectors") {
    using quat_t = dp::mat::quaternion<double>;

    // 90 degree rotation about Z axis
    const double angle = M_PI / 2.0;
    const double c = std::cos(angle / 2.0);
    const double s = std::sin(angle / 2.0);

    quat_t quats[2] = {{c, 0.0, 0.0, s}, {c, 0.0, 0.0, s}};

    // Rotate (1, 0, 0) by 90 deg about Z -> (0, 1, 0)
    double vx[2] = {1.0, 0.0};
    double vy[2] = {0.0, 1.0};
    double vz[2] = {0.0, 0.0};

    auto qv = on::view(quats);
    qv.rotate_vectors(vx, vy, vz);

    CHECK(vx[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(vy[0] == doctest::Approx(1.0));
    CHECK(vz[0] == doctest::Approx(0.0).epsilon(1e-10));

    // (0, 1, 0) rotated 90 deg about Z -> (-1, 0, 0)
    CHECK(vx[1] == doctest::Approx(-1.0));
    CHECK(vy[1] == doctest::Approx(0.0).epsilon(1e-10));
}

// =============================================================================
// Conversion Operations Tests
// =============================================================================

TEST_CASE("quaternion_view - To/From Euler angles") {
    using quat_t = dp::mat::quaternion<double>;

    double roll[3] = {0.0, M_PI / 4.0, 0.0};
    double pitch[3] = {0.0, 0.0, M_PI / 4.0};
    double yaw[3] = {0.0, 0.0, 0.0};

    quat_t quats[3];

    // Create from Euler
    on::quaternion_view<double, 4>::from_euler(roll, pitch, yaw, quats, 3);

    // Identity case
    CHECK(quats[0].w == doctest::Approx(1.0));
    CHECK(quats[0].x == doctest::Approx(0.0).epsilon(1e-10));

    // Convert back to Euler
    double out_roll[3], out_pitch[3], out_yaw[3];
    auto qv = on::view(quats);
    qv.to_euler(out_roll, out_pitch, out_yaw);

    CHECK(out_roll[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(out_pitch[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(out_yaw[0] == doctest::Approx(0.0).epsilon(1e-10));

    CHECK(out_roll[1] == doctest::Approx(M_PI / 4.0).epsilon(1e-3));
    CHECK(out_pitch[2] == doctest::Approx(M_PI / 4.0).epsilon(1e-3));
}

TEST_CASE("quaternion_view - To/From axis-angle") {
    using quat_t = dp::mat::quaternion<double>;

    // 90 deg rotation about Z
    const double angle = M_PI / 2.0;
    quat_t quats[2] = {
        {std::cos(angle / 2.0), 0.0, 0.0, std::sin(angle / 2.0)}, {1.0, 0.0, 0.0, 0.0} // identity
    };

    double ax[2], ay[2], az[2], angles[2];

    auto qv = on::view(quats);
    qv.to_axis_angle(ax, ay, az, angles);

    // First quaternion: 90 deg about Z
    CHECK(ax[0] == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(ay[0] == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(az[0] == doctest::Approx(1.0));
    CHECK(angles[0] == doctest::Approx(M_PI / 2.0).epsilon(1e-6));

    // Second quaternion: identity (angle = 0)
    CHECK(angles[1] == doctest::Approx(0.0).epsilon(1e-6));
}

// =============================================================================
// Subview Tests
// =============================================================================

TEST_CASE("quaternion_view - Subview") {
    using quat_t = dp::mat::quaternion<double>;

    quat_t data[8];
    for (std::size_t i = 0; i < 8; ++i) {
        data[i] = {static_cast<double>(i), 0.0, 0.0, 0.0};
    }

    auto qv = on::view(data);
    auto sub = qv.subview(2, 4); // elements [2, 3, 4, 5]

    CHECK(sub.size() == 4);
    CHECK(sub[0].w == doctest::Approx(2.0));
    CHECK(sub[3].w == doctest::Approx(5.0));
}

// =============================================================================
// Tail Handling Tests (non-multiple of SIMD width)
// =============================================================================

TEST_CASE("quaternion_view - Tail handling") {
    using quat_t = dp::mat::quaternion<double>;

    // 5 quaternions (not a multiple of typical SIMD width 4)
    quat_t data[5] = {
        {3.0, 4.0, 0.0, 0.0}, {0.0, 3.0, 4.0, 0.0}, {0.0, 0.0, 3.0, 4.0}, {1.0, 1.0, 1.0, 1.0}, {5.0, 0.0, 0.0, 0.0}};

    auto qv = on::view(data);

    SUBCASE("Normalize handles tail correctly") {
        qv.normalize_inplace();

        for (int i = 0; i < 5; ++i) {
            double norm = std::sqrt(data[i].w * data[i].w + data[i].x * data[i].x + data[i].y * data[i].y +
                                    data[i].z * data[i].z);
            CHECK(norm == doctest::Approx(1.0));
        }
    }
}

// =============================================================================
// Spatial Quaternion (dp::Quaternion) Tests
// =============================================================================

TEST_CASE("quaternion_view - Spatial Quaternion support") {
    dp::Quaternion data[4] = {
        {1.0, 0.0, 0.0, 0.0}, {0.707, 0.0, 0.707, 0.0}, {0.5, 0.5, 0.5, 0.5}, {0.0, 1.0, 0.0, 0.0}};

    auto qv = on::view(data);

    CHECK(qv.size() == 4);

    // Normalize should work
    qv.normalize_inplace();

    for (int i = 0; i < 4; ++i) {
        double norm =
            std::sqrt(data[i].w * data[i].w + data[i].x * data[i].x + data[i].y * data[i].y + data[i].z * data[i].z);
        CHECK(norm == doctest::Approx(1.0));
    }
}

// =============================================================================
// Quaternion View (non-owning) Tests
// =============================================================================

TEST_CASE("Quaternion view - Basic operations") {
    // Storage for the view
    dp::mat::vector<dp::mat::quaternion<double>, 4> storage;
    on::Quaternion<double, 4> quats(storage);

    SUBCASE("Default construction - null view") {
        on::Quaternion<double, 4> null_view;
        CHECK(!null_view.valid());
        CHECK(!null_view);
    }

    SUBCASE("Construction from storage") {
        CHECK(quats.valid());
        CHECK(quats.size() == 4);
    }

    SUBCASE("Fill with identity") {
        quats.fill(dp::mat::quaternion<double>::identity());
        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(quats[i].w == doctest::Approx(1.0));
            CHECK(quats[i].x == doctest::Approx(0.0));
        }
    }

    SUBCASE("Normalize in-place") {
        quats[0] = {3.0, 4.0, 0.0, 0.0};
        quats[1] = {1.0, 1.0, 1.0, 1.0};
        quats[2] = {0.0, 3.0, 4.0, 0.0};
        quats[3] = {5.0, 0.0, 0.0, 0.0};

        quats.normalize_inplace();

        CHECK(quats[0].w == doctest::Approx(0.6));
        CHECK(quats[0].x == doctest::Approx(0.8));
    }

    SUBCASE("Hamilton product - multiply_to") {
        dp::mat::vector<dp::mat::quaternion<double>, 2> a_storage, b_storage, result_storage;
        on::Quaternion<double, 2> a(a_storage);
        on::Quaternion<double, 2> b(b_storage);
        on::Quaternion<double, 2> result(result_storage);

        // i * j = k
        a[0] = {0.0, 1.0, 0.0, 0.0};
        a[1] = {1.0, 0.0, 0.0, 0.0};
        b[0] = {0.0, 0.0, 1.0, 0.0};
        b[1] = {0.707, 0.0, 0.707, 0.0};

        a.multiply_to(b, result.data());

        CHECK(result[0].z == doctest::Approx(1.0));
        CHECK(result[1].w == doctest::Approx(0.707));
    }

    SUBCASE("Hamilton product - multiply_inplace") {
        dp::mat::vector<dp::mat::quaternion<double>, 2> a_storage, b_storage;
        on::Quaternion<double, 2> a(a_storage);
        on::Quaternion<double, 2> b(b_storage);

        // i * j = k
        a[0] = {0.0, 1.0, 0.0, 0.0};
        a[1] = {1.0, 0.0, 0.0, 0.0};
        b[0] = {0.0, 0.0, 1.0, 0.0};
        b[1] = {0.707, 0.0, 0.707, 0.0};

        a.multiply_inplace(b);

        CHECK(a[0].z == doctest::Approx(1.0));
        CHECK(a[1].w == doctest::Approx(0.707));
    }

    SUBCASE("Iteration") {
        quats.fill(dp::mat::quaternion<double>::identity());
        int count = 0;
        for (const auto &q : quats) {
            CHECK(q.w == doctest::Approx(1.0));
            count++;
        }
        CHECK(count == 4);
    }
}
