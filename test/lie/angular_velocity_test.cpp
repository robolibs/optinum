#include <doctest/doctest.h>
#include <optinum/lie/lie.hpp>

#include <cmath>
#include <numbers>
#include <random>

using namespace optinum::lie;
using namespace optinum;

namespace dp = ::datapod;

// ===== HELPER FUNCTIONS =====

template <typename T> bool approx_equal(T a, T b, T tol = T(1e-10)) { return std::abs(a - b) < tol; }

template <typename T, std::size_t N>
bool vec_approx_equal(const dp::mat::vector<T, N> &a, const dp::mat::vector<T, N> &b, T tol = T(1e-10)) {
    for (std::size_t i = 0; i < N; ++i) {
        if (std::abs(a[i] - b[i]) >= tol)
            return false;
    }
    return true;
}

// ===== LocalAngularVelocity CONSTRUCTION TESTS =====

TEST_CASE("LocalAngularVelocity default construction is zero") {
    LocalAngularVelocityd omega;

    CHECK(omega.is_zero());
    CHECK(approx_equal(omega.x(), 0.0));
    CHECK(approx_equal(omega.y(), 0.0));
    CHECK(approx_equal(omega.z(), 0.0));
}

TEST_CASE("LocalAngularVelocity construction from components") {
    LocalAngularVelocityd omega(1.0, 2.0, 3.0);

    CHECK(approx_equal(omega.x(), 1.0));
    CHECK(approx_equal(omega.y(), 2.0));
    CHECK(approx_equal(omega.z(), 3.0));
}

TEST_CASE("LocalAngularVelocity construction from vector") {
    dp::mat::vector<double, 3> v{{1.0, 2.0, 3.0}};
    LocalAngularVelocityd omega(v);

    CHECK(vec_approx_equal(omega.vector(), v));
}

TEST_CASE("LocalAngularVelocity norm") {
    LocalAngularVelocityd omega(3.0, 4.0, 0.0);

    CHECK(approx_equal(omega.norm(), 5.0));
    CHECK(approx_equal(omega.norm_squared(), 25.0));
}

// ===== GlobalAngularVelocity CONSTRUCTION TESTS =====

TEST_CASE("GlobalAngularVelocity default construction is zero") {
    GlobalAngularVelocityd omega;

    CHECK(omega.is_zero());
    CHECK(approx_equal(omega.x(), 0.0));
    CHECK(approx_equal(omega.y(), 0.0));
    CHECK(approx_equal(omega.z(), 0.0));
}

TEST_CASE("GlobalAngularVelocity construction from components") {
    GlobalAngularVelocityd omega(1.0, 2.0, 3.0);

    CHECK(approx_equal(omega.x(), 1.0));
    CHECK(approx_equal(omega.y(), 2.0));
    CHECK(approx_equal(omega.z(), 3.0));
}

// ===== FRAME CONVERSION TESTS =====

TEST_CASE("Frame conversion: local to global and back") {
    SUBCASE("identity rotation") {
        LocalAngularVelocityd omega_local(1.0, 2.0, 3.0);
        SO3d R = SO3d::identity();

        auto omega_global = omega_local.to_global(R);
        auto omega_local2 = omega_global.to_local(R);

        CHECK(omega_local.is_approx(omega_local2, 1e-10));
    }

    SUBCASE("90 degree rotation around z") {
        LocalAngularVelocityd omega_local(1.0, 0.0, 0.0); // rotation around body x-axis
        SO3d R = SO3d::rot_z(std::numbers::pi / 2);       // body rotated 90° around z

        auto omega_global = omega_local.to_global(R);

        // Body x-axis is now aligned with world y-axis
        CHECK(approx_equal(omega_global.x(), 0.0, 1e-10));
        CHECK(approx_equal(omega_global.y(), 1.0, 1e-10));
        CHECK(approx_equal(omega_global.z(), 0.0, 1e-10));
    }

    SUBCASE("round-trip with random rotation") {
        std::mt19937 rng(42);

        for (int i = 0; i < 100; ++i) {
            auto R = SO3d::sample_uniform(rng);
            LocalAngularVelocityd omega_local(0.5, 1.0, -0.3);

            auto omega_global = omega_local.to_global(R);
            auto omega_local2 = omega_global.to_local(R);

            CHECK(omega_local.is_approx(omega_local2, 1e-9));
        }
    }
}

TEST_CASE("Frame conversion: global to local and back") {
    SUBCASE("round-trip with random rotation") {
        std::mt19937 rng(42);

        for (int i = 0; i < 100; ++i) {
            auto R = SO3d::sample_uniform(rng);
            GlobalAngularVelocityd omega_global(0.5, 1.0, -0.3);

            auto omega_local = omega_global.to_local(R);
            auto omega_global2 = omega_local.to_global(R);

            CHECK(omega_global.is_approx(omega_global2, 1e-9));
        }
    }
}

TEST_CASE("Frame conversion preserves magnitude") {
    std::mt19937 rng(42);

    for (int i = 0; i < 100; ++i) {
        auto R = SO3d::sample_uniform(rng);
        LocalAngularVelocityd omega_local(1.0, 2.0, 3.0);

        auto omega_global = omega_local.to_global(R);

        CHECK(approx_equal(omega_local.norm(), omega_global.norm(), 1e-10));
    }
}

// ===== INTEGRATION TESTS =====

TEST_CASE("LocalAngularVelocity integration") {
    SUBCASE("zero angular velocity gives identity") {
        LocalAngularVelocityd omega;
        auto dR = omega.integrate(1.0);

        CHECK(dR.is_identity(1e-10));
    }

    SUBCASE("rotation around z-axis") {
        const double rate = 1.0;                // 1 rad/s
        const double dt = std::numbers::pi / 2; // π/2 seconds
        LocalAngularVelocityd omega(0.0, 0.0, rate);

        auto dR = omega.integrate(dt);

        // Should be 90° rotation around z
        auto expected = SO3d::rot_z(rate * dt);
        CHECK(dR.is_approx(expected, 1e-10));
    }

    SUBCASE("apply_to rotates correctly") {
        SO3d R_initial = SO3d::rot_x(0.3);
        LocalAngularVelocityd omega(0.0, 0.0, 1.0); // rotate around body z
        const double dt = 0.1;

        auto R_new = omega.apply_to(R_initial, dt);

        // R_new = R_initial * exp(omega * dt)
        auto expected = R_initial * SO3d::exp(dp::mat::vector<double, 3>{{0.0, 0.0, 0.1}});
        CHECK(R_new.is_approx(expected, 1e-10));
    }
}

TEST_CASE("GlobalAngularVelocity integration") {
    SUBCASE("zero angular velocity gives identity") {
        GlobalAngularVelocityd omega;
        auto dR = omega.integrate(1.0);

        CHECK(dR.is_identity(1e-10));
    }

    SUBCASE("rotation around z-axis") {
        const double rate = 1.0;
        const double dt = std::numbers::pi / 2;
        GlobalAngularVelocityd omega(0.0, 0.0, rate);

        auto dR = omega.integrate(dt);

        auto expected = SO3d::rot_z(rate * dt);
        CHECK(dR.is_approx(expected, 1e-10));
    }

    SUBCASE("apply_to rotates correctly") {
        SO3d R_initial = SO3d::rot_x(0.3);
        GlobalAngularVelocityd omega(0.0, 0.0, 1.0); // rotate around world z
        const double dt = 0.1;

        auto R_new = omega.apply_to(R_initial, dt);

        // R_new = exp(omega * dt) * R_initial
        auto expected = SO3d::exp(dp::mat::vector<double, 3>{{0.0, 0.0, 0.1}}) * R_initial;
        CHECK(R_new.is_approx(expected, 1e-10));
    }
}

TEST_CASE("Local vs Global integration difference") {
    // When body is rotated, local and global angular velocities produce different results
    // Use a rotation that clearly distinguishes the frames:
    // Rotate body 90° around z, so body's x-axis points along world's y-axis
    SO3d R = SO3d::rot_z(std::numbers::pi / 2);

    // Angular velocity around x-axis (1 rad/s)
    LocalAngularVelocityd omega_local(1.0, 0.0, 0.0);
    GlobalAngularVelocityd omega_global(1.0, 0.0, 0.0);

    const double dt = 0.5;

    auto R_from_local = omega_local.apply_to(R, dt);
    auto R_from_global = omega_global.apply_to(R, dt);

    // Results should be different because:
    // - Local omega_x rotates around body's x-axis = world's y-axis
    // - Global omega_x rotates around world's x-axis
    // These are perpendicular axes, so results must differ significantly

    // Compute the angular difference between the two results
    auto diff = R_from_local.inverse() * R_from_global;
    double angle_diff = diff.angle();

    // The difference should be significant (not near zero)
    CHECK(angle_diff > 0.1);
}

// ===== ARITHMETIC TESTS =====

TEST_CASE("LocalAngularVelocity arithmetic") {
    LocalAngularVelocityd a(1.0, 2.0, 3.0);
    LocalAngularVelocityd b(0.5, 1.0, 1.5);

    SUBCASE("addition") {
        auto c = a + b;
        CHECK(approx_equal(c.x(), 1.5));
        CHECK(approx_equal(c.y(), 3.0));
        CHECK(approx_equal(c.z(), 4.5));
    }

    SUBCASE("subtraction") {
        auto c = a - b;
        CHECK(approx_equal(c.x(), 0.5));
        CHECK(approx_equal(c.y(), 1.0));
        CHECK(approx_equal(c.z(), 1.5));
    }

    SUBCASE("scalar multiplication") {
        auto c = a * 2.0;
        CHECK(approx_equal(c.x(), 2.0));
        CHECK(approx_equal(c.y(), 4.0));
        CHECK(approx_equal(c.z(), 6.0));

        auto d = 2.0 * a;
        CHECK(approx_equal(d.x(), 2.0));
        CHECK(approx_equal(d.y(), 4.0));
        CHECK(approx_equal(d.z(), 6.0));
    }

    SUBCASE("scalar division") {
        auto c = a / 2.0;
        CHECK(approx_equal(c.x(), 0.5));
        CHECK(approx_equal(c.y(), 1.0));
        CHECK(approx_equal(c.z(), 1.5));
    }

    SUBCASE("negation") {
        auto c = -a;
        CHECK(approx_equal(c.x(), -1.0));
        CHECK(approx_equal(c.y(), -2.0));
        CHECK(approx_equal(c.z(), -3.0));
    }

    SUBCASE("compound assignment") {
        LocalAngularVelocityd c = a;
        c += b;
        CHECK(approx_equal(c.x(), 1.5));

        c = a;
        c -= b;
        CHECK(approx_equal(c.x(), 0.5));

        c = a;
        c *= 2.0;
        CHECK(approx_equal(c.x(), 2.0));

        c = a;
        c /= 2.0;
        CHECK(approx_equal(c.x(), 0.5));
    }
}

TEST_CASE("GlobalAngularVelocity arithmetic") {
    GlobalAngularVelocityd a(1.0, 2.0, 3.0);
    GlobalAngularVelocityd b(0.5, 1.0, 1.5);

    SUBCASE("addition") {
        auto c = a + b;
        CHECK(approx_equal(c.x(), 1.5));
        CHECK(approx_equal(c.y(), 3.0));
        CHECK(approx_equal(c.z(), 4.5));
    }

    SUBCASE("scalar multiplication") {
        auto c = a * 2.0;
        CHECK(approx_equal(c.x(), 2.0));

        auto d = 2.0 * a;
        CHECK(approx_equal(d.x(), 2.0));
    }
}

// ===== COMPARISON TESTS =====

TEST_CASE("LocalAngularVelocity comparison") {
    LocalAngularVelocityd a(1.0, 2.0, 3.0);
    LocalAngularVelocityd b(1.0, 2.0, 3.0);
    LocalAngularVelocityd c(1.0, 2.0, 3.1);

    CHECK(a == b);
    CHECK(a != c);
    CHECK(a.is_approx(b));
    CHECK_FALSE(a.is_approx(c, 0.01));
}

TEST_CASE("GlobalAngularVelocity comparison") {
    GlobalAngularVelocityd a(1.0, 2.0, 3.0);
    GlobalAngularVelocityd b(1.0, 2.0, 3.0);
    GlobalAngularVelocityd c(1.0, 2.0, 3.1);

    CHECK(a == b);
    CHECK(a != c);
}

// ===== TYPE CONVERSION =====

TEST_CASE("LocalAngularVelocity cast to different scalar type") {
    LocalAngularVelocityd omega_d(1.0, 2.0, 3.0);
    auto omega_f = omega_d.cast<float>();

    CHECK(approx_equal(static_cast<double>(omega_f.x()), omega_d.x(), 1e-5));
    CHECK(approx_equal(static_cast<double>(omega_f.y()), omega_d.y(), 1e-5));
    CHECK(approx_equal(static_cast<double>(omega_f.z()), omega_d.z(), 1e-5));
}

TEST_CASE("GlobalAngularVelocity cast to different scalar type") {
    GlobalAngularVelocityd omega_d(1.0, 2.0, 3.0);
    auto omega_f = omega_d.cast<float>();

    CHECK(approx_equal(static_cast<double>(omega_f.x()), omega_d.x(), 1e-5));
}

// ===== STATIC FACTORY METHODS =====

TEST_CASE("Static factory methods") {
    SUBCASE("LocalAngularVelocity::zero") {
        auto omega = LocalAngularVelocityd::zero();
        CHECK(omega.is_zero());
    }

    SUBCASE("GlobalAngularVelocity::zero") {
        auto omega = GlobalAngularVelocityd::zero();
        CHECK(omega.is_zero());
    }

    SUBCASE("LocalAngularVelocity::from_global") {
        GlobalAngularVelocityd omega_global(1.0, 0.0, 0.0);
        SO3d R = SO3d::rot_z(std::numbers::pi / 2);

        auto omega_local = LocalAngularVelocityd::from_global(omega_global, R);
        auto omega_global2 = omega_local.to_global(R);

        CHECK(omega_global.is_approx(omega_global2, 1e-10));
    }

    SUBCASE("GlobalAngularVelocity::from_local") {
        LocalAngularVelocityd omega_local(1.0, 0.0, 0.0);
        SO3d R = SO3d::rot_z(std::numbers::pi / 2);

        auto omega_global = GlobalAngularVelocityd::from_local(omega_local, R);
        auto omega_local2 = omega_global.to_local(R);

        CHECK(omega_local.is_approx(omega_local2, 1e-10));
    }
}

// ===== TYPE ALIASES =====

TEST_CASE("Type aliases work correctly") {
    LocalAngularVelocityf local_f(1.0f, 2.0f, 3.0f);
    LocalAngularVelocityd local_d(1.0, 2.0, 3.0);

    GlobalAngularVelocityf global_f(1.0f, 2.0f, 3.0f);
    GlobalAngularVelocityd global_d(1.0, 2.0, 3.0);

    // Alternative names
    BodyAngularVelocity<double> body_omega(1.0, 2.0, 3.0);
    SpatialAngularVelocity<double> spatial_omega(1.0, 2.0, 3.0);

    CHECK(approx_equal(body_omega.x(), local_d.x()));
    CHECK(approx_equal(spatial_omega.x(), global_d.x()));
}

// ===== MUTATORS =====

TEST_CASE("LocalAngularVelocity mutators") {
    LocalAngularVelocityd omega;

    omega.set_x(1.0);
    omega.set_y(2.0);
    omega.set_z(3.0);

    CHECK(approx_equal(omega.x(), 1.0));
    CHECK(approx_equal(omega.y(), 2.0));
    CHECK(approx_equal(omega.z(), 3.0));

    omega.set_zero();
    CHECK(omega.is_zero());
}

TEST_CASE("GlobalAngularVelocity mutators") {
    GlobalAngularVelocityd omega;

    omega.set_x(1.0);
    omega.set_y(2.0);
    omega.set_z(3.0);

    CHECK(approx_equal(omega.x(), 1.0));
    CHECK(approx_equal(omega.y(), 2.0));
    CHECK(approx_equal(omega.z(), 3.0));

    omega.set_zero();
    CHECK(omega.is_zero());
}
