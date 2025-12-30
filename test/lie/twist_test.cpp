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

// ===== LocalTwist CONSTRUCTION TESTS =====

TEST_CASE("LocalTwist default construction is zero") {
    LocalTwistd twist;

    CHECK(twist.is_zero());
    CHECK(approx_equal(twist.vx(), 0.0));
    CHECK(approx_equal(twist.vy(), 0.0));
    CHECK(approx_equal(twist.vz(), 0.0));
    CHECK(approx_equal(twist.wx(), 0.0));
    CHECK(approx_equal(twist.wy(), 0.0));
    CHECK(approx_equal(twist.wz(), 0.0));
}

TEST_CASE("LocalTwist construction from vectors") {
    dp::mat::vector<double, 3> linear{{1.0, 2.0, 3.0}};
    dp::mat::vector<double, 3> angular{{0.1, 0.2, 0.3}};
    LocalTwistd twist(linear, angular);

    CHECK(vec_approx_equal(twist.linear(), linear));
    CHECK(vec_approx_equal(twist.angular(), angular));
}

TEST_CASE("LocalTwist construction from components") {
    LocalTwistd twist(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);

    CHECK(approx_equal(twist.vx(), 1.0));
    CHECK(approx_equal(twist.vy(), 2.0));
    CHECK(approx_equal(twist.vz(), 3.0));
    CHECK(approx_equal(twist.wx(), 0.1));
    CHECK(approx_equal(twist.wy(), 0.2));
    CHECK(approx_equal(twist.wz(), 0.3));
}

TEST_CASE("LocalTwist construction from Vector6") {
    dp::mat::vector<double, 6> vec{{1.0, 2.0, 3.0, 0.1, 0.2, 0.3}};
    LocalTwistd twist(vec);

    CHECK(vec_approx_equal(twist.vector(), vec));
}

TEST_CASE("LocalTwist construction from LocalAngularVelocity") {
    LocalAngularVelocityd omega(0.1, 0.2, 0.3);
    LocalTwistd twist(omega);

    CHECK(approx_equal(twist.vx(), 0.0));
    CHECK(approx_equal(twist.vy(), 0.0));
    CHECK(approx_equal(twist.vz(), 0.0));
    CHECK(approx_equal(twist.wx(), 0.1));
    CHECK(approx_equal(twist.wy(), 0.2));
    CHECK(approx_equal(twist.wz(), 0.3));
}

// ===== GlobalTwist CONSTRUCTION TESTS =====

TEST_CASE("GlobalTwist default construction is zero") {
    GlobalTwistd twist;

    CHECK(twist.is_zero());
}

TEST_CASE("GlobalTwist construction from vectors") {
    dp::mat::vector<double, 3> linear{{1.0, 2.0, 3.0}};
    dp::mat::vector<double, 3> angular{{0.1, 0.2, 0.3}};
    GlobalTwistd twist(linear, angular);

    CHECK(vec_approx_equal(twist.linear(), linear));
    CHECK(vec_approx_equal(twist.angular(), angular));
}

// ===== FRAME CONVERSION TESTS =====

TEST_CASE("Frame conversion: local to global and back") {
    SUBCASE("identity transform") {
        LocalTwistd twist_local(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        SE3d T = SE3d::identity();

        auto twist_global = twist_local.to_global(T);
        auto twist_local2 = twist_global.to_local(T);

        CHECK(twist_local.is_approx(twist_local2, 1e-10));
    }

    SUBCASE("pure rotation") {
        LocalTwistd twist_local(1.0, 0.0, 0.0, 0.0, 0.0, 0.0); // linear velocity along body x
        SE3d T(SO3d::rot_z(std::numbers::pi / 2), dp::mat::vector<double, 3>{{0.0, 0.0, 0.0}});

        auto twist_global = twist_local.to_global(T);

        // Body x-axis is now aligned with world y-axis
        // Linear velocity should be along world y
        CHECK(approx_equal(twist_global.vx(), 0.0, 1e-10));
        CHECK(approx_equal(twist_global.vy(), 1.0, 1e-10));
        CHECK(approx_equal(twist_global.vz(), 0.0, 1e-10));
    }

    SUBCASE("round-trip with random transform") {
        std::mt19937 rng(42);

        for (int i = 0; i < 50; ++i) {
            auto T = SE3d::sample_uniform(rng);
            LocalTwistd twist_local(0.5, 1.0, -0.3, 0.1, -0.2, 0.15);

            auto twist_global = twist_local.to_global(T);
            auto twist_local2 = twist_global.to_local(T);

            CHECK(twist_local.is_approx(twist_local2, 1e-9));
        }
    }
}

TEST_CASE("Frame conversion: global to local and back") {
    SUBCASE("round-trip with random transform") {
        std::mt19937 rng(42);

        for (int i = 0; i < 50; ++i) {
            auto T = SE3d::sample_uniform(rng);
            GlobalTwistd twist_global(0.5, 1.0, -0.3, 0.1, -0.2, 0.15);

            auto twist_local = twist_global.to_local(T);
            auto twist_global2 = twist_local.to_global(T);

            CHECK(twist_global.is_approx(twist_global2, 1e-9));
        }
    }
}

// ===== INTEGRATION TESTS =====

TEST_CASE("LocalTwist integration") {
    SUBCASE("zero twist gives identity") {
        LocalTwistd twist;
        auto dT = twist.integrate(1.0);

        CHECK(dT.is_identity(1e-10));
    }

    SUBCASE("pure linear velocity") {
        LocalTwistd twist(1.0, 0.0, 0.0, 0.0, 0.0, 0.0); // 1 m/s along x
        const double dt = 2.0;

        auto dT = twist.integrate(dt);

        // Should translate 2m along x
        CHECK(dT.so3().is_identity(1e-10));
        CHECK(approx_equal(dT.x(), 2.0, 1e-10));
        CHECK(approx_equal(dT.y(), 0.0, 1e-10));
        CHECK(approx_equal(dT.z(), 0.0, 1e-10));
    }

    SUBCASE("pure angular velocity") {
        LocalTwistd twist(0.0, 0.0, 0.0, 0.0, 0.0, 1.0); // 1 rad/s around z
        const double dt = std::numbers::pi / 2;

        auto dT = twist.integrate(dt);

        // Should rotate 90° around z
        auto expected_R = SO3d::rot_z(dt);
        CHECK(dT.so3().is_approx(expected_R, 1e-10));
        CHECK(approx_equal(dT.x(), 0.0, 1e-10));
        CHECK(approx_equal(dT.y(), 0.0, 1e-10));
        CHECK(approx_equal(dT.z(), 0.0, 1e-10));
    }

    SUBCASE("apply_to works correctly") {
        SE3d T_initial = SE3d::trans(1.0, 0.0, 0.0);
        LocalTwistd twist(0.0, 1.0, 0.0, 0.0, 0.0, 0.0); // move along body y
        const double dt = 1.0;

        auto T_new = twist.apply_to(T_initial, dt);

        // T_new = T_initial * exp(twist * dt)
        auto expected = T_initial * twist.integrate(dt);
        CHECK(T_new.is_approx(expected, 1e-10));
    }
}

TEST_CASE("GlobalTwist integration") {
    SUBCASE("zero twist gives identity") {
        GlobalTwistd twist;
        auto dT = twist.integrate(1.0);

        CHECK(dT.is_identity(1e-10));
    }

    SUBCASE("apply_to works correctly") {
        SE3d T_initial = SE3d::rot_z(0.5);
        GlobalTwistd twist(1.0, 0.0, 0.0, 0.0, 0.0, 0.0); // move along world x
        const double dt = 1.0;

        auto T_new = twist.apply_to(T_initial, dt);

        // T_new = exp(twist * dt) * T_initial
        auto expected = twist.integrate(dt) * T_initial;
        CHECK(T_new.is_approx(expected, 1e-10));
    }
}

TEST_CASE("Local vs Global integration difference") {
    // When body is rotated, local and global twists produce different results
    SE3d T = SE3d(SO3d::rot_z(std::numbers::pi / 2), dp::mat::vector<double, 3>{{1.0, 0.0, 0.0}});

    // Same numerical values, but different frames
    LocalTwistd twist_local(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    GlobalTwistd twist_global(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    const double dt = 1.0;

    auto T_from_local = twist_local.apply_to(T, dt);
    auto T_from_global = twist_global.apply_to(T, dt);

    // Results should be different
    CHECK_FALSE(T_from_local.is_approx(T_from_global, 0.1));
}

// ===== ADJOINT TRANSFORMATION TESTS =====

TEST_CASE("LocalTwist adjoint transformation") {
    SUBCASE("identity transform") {
        LocalTwistd twist(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        SE3d T = SE3d::identity();

        auto twist_adj = twist.adjoint(T);

        CHECK(twist.is_approx(twist_adj, 1e-10));
    }

    SUBCASE("adjoint equals to_global for body-to-world transform") {
        std::mt19937 rng(42);

        for (int i = 0; i < 20; ++i) {
            auto T = SE3d::sample_uniform(rng);
            LocalTwistd twist(0.5, 1.0, -0.3, 0.1, -0.2, 0.15);

            auto twist_adj = twist.adjoint(T);
            auto twist_global = twist.to_global(T);

            // adjoint(T) should give the same result as to_global(T)
            CHECK(vec_approx_equal(twist_adj.vector(), twist_global.vector(), 1e-9));
        }
    }
}

// ===== ARITHMETIC TESTS =====

TEST_CASE("LocalTwist arithmetic") {
    LocalTwistd a(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    LocalTwistd b(0.5, 1.0, 1.5, 0.05, 0.1, 0.15);

    SUBCASE("addition") {
        auto c = a + b;
        CHECK(approx_equal(c.vx(), 1.5));
        CHECK(approx_equal(c.vy(), 3.0));
        CHECK(approx_equal(c.vz(), 4.5));
        CHECK(approx_equal(c.wx(), 0.15));
        CHECK(approx_equal(c.wy(), 0.3));
        CHECK(approx_equal(c.wz(), 0.45));
    }

    SUBCASE("subtraction") {
        auto c = a - b;
        CHECK(approx_equal(c.vx(), 0.5));
        CHECK(approx_equal(c.vy(), 1.0));
        CHECK(approx_equal(c.vz(), 1.5));
    }

    SUBCASE("scalar multiplication") {
        auto c = a * 2.0;
        CHECK(approx_equal(c.vx(), 2.0));
        CHECK(approx_equal(c.wx(), 0.2));

        auto d = 2.0 * a;
        CHECK(approx_equal(d.vx(), 2.0));
    }

    SUBCASE("scalar division") {
        auto c = a / 2.0;
        CHECK(approx_equal(c.vx(), 0.5));
        CHECK(approx_equal(c.wx(), 0.05));
    }

    SUBCASE("negation") {
        auto c = -a;
        CHECK(approx_equal(c.vx(), -1.0));
        CHECK(approx_equal(c.wx(), -0.1));
    }

    SUBCASE("compound assignment") {
        LocalTwistd c = a;
        c += b;
        CHECK(approx_equal(c.vx(), 1.5));

        c = a;
        c -= b;
        CHECK(approx_equal(c.vx(), 0.5));

        c = a;
        c *= 2.0;
        CHECK(approx_equal(c.vx(), 2.0));
    }
}

TEST_CASE("GlobalTwist arithmetic") {
    GlobalTwistd a(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    GlobalTwistd b(0.5, 1.0, 1.5, 0.05, 0.1, 0.15);

    SUBCASE("addition") {
        auto c = a + b;
        CHECK(approx_equal(c.vx(), 1.5));
    }

    SUBCASE("scalar multiplication") {
        auto c = a * 2.0;
        CHECK(approx_equal(c.vx(), 2.0));

        auto d = 2.0 * a;
        CHECK(approx_equal(d.vx(), 2.0));
    }
}

// ===== COMPARISON TESTS =====

TEST_CASE("LocalTwist comparison") {
    LocalTwistd a(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    LocalTwistd b(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    LocalTwistd c(1.0, 2.0, 3.0, 0.1, 0.2, 0.31);

    CHECK(a == b);
    CHECK(a != c);
    CHECK(a.is_approx(b));
    CHECK_FALSE(a.is_approx(c, 0.001));
}

// ===== TYPE CONVERSION =====

TEST_CASE("LocalTwist cast to different scalar type") {
    LocalTwistd twist_d(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    auto twist_f = twist_d.cast<float>();

    CHECK(approx_equal(static_cast<double>(twist_f.vx()), twist_d.vx(), 1e-5));
    CHECK(approx_equal(static_cast<double>(twist_f.wx()), twist_d.wx(), 1e-5));
}

// ===== STATIC FACTORY METHODS =====

TEST_CASE("Static factory methods") {
    SUBCASE("LocalTwist::zero") {
        auto twist = LocalTwistd::zero();
        CHECK(twist.is_zero());
    }

    SUBCASE("GlobalTwist::zero") {
        auto twist = GlobalTwistd::zero();
        CHECK(twist.is_zero());
    }

    SUBCASE("LocalTwist::from_vector") {
        dp::mat::vector<double, 6> vec{{1.0, 2.0, 3.0, 0.1, 0.2, 0.3}};
        auto twist = LocalTwistd::from_vector(vec);
        CHECK(vec_approx_equal(twist.vector(), vec));
    }

    SUBCASE("GlobalTwist::from_local") {
        LocalTwistd twist_local(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        SE3d T = SE3d::rot_z(0.5);

        auto twist_global = GlobalTwistd::from_local(twist_local, T);
        auto twist_local2 = twist_global.to_local(T);

        CHECK(twist_local.is_approx(twist_local2, 1e-10));
    }

    SUBCASE("LocalTwist::from_global") {
        GlobalTwistd twist_global(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        SE3d T = SE3d::rot_z(0.5);

        auto twist_local = LocalTwistd::from_global(twist_global, T);
        auto twist_global2 = twist_local.to_global(T);

        CHECK(twist_global.is_approx(twist_global2, 1e-10));
    }
}

// ===== MUTATORS =====

TEST_CASE("LocalTwist mutators") {
    LocalTwistd twist;

    twist.set_linear(1.0, 2.0, 3.0);
    twist.set_angular(0.1, 0.2, 0.3);

    CHECK(approx_equal(twist.vx(), 1.0));
    CHECK(approx_equal(twist.wx(), 0.1));

    dp::mat::vector<double, 3> v{{4.0, 5.0, 6.0}};
    twist.set_linear(v);
    CHECK(approx_equal(twist.vx(), 4.0));

    dp::mat::vector<double, 6> vec{{7.0, 8.0, 9.0, 0.7, 0.8, 0.9}};
    twist.set_vector(vec);
    CHECK(approx_equal(twist.vx(), 7.0));
    CHECK(approx_equal(twist.wx(), 0.7));

    twist.set_zero();
    CHECK(twist.is_zero());
}

// ===== TYPE ALIASES =====

TEST_CASE("Type aliases work correctly") {
    LocalTwistf local_f(1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f);
    LocalTwistd local_d(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);

    GlobalTwistf global_f(1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f);
    GlobalTwistd global_d(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);

    // Alternative names
    BodyTwist<double> body_twist(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    SpatialTwist<double> spatial_twist(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);

    CHECK(approx_equal(body_twist.vx(), local_d.vx()));
    CHECK(approx_equal(spatial_twist.vx(), global_d.vx()));
}

// ===== ANGULAR VELOCITY ACCESSOR =====

TEST_CASE("Angular velocity accessor") {
    LocalTwistd local_twist(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    auto omega = local_twist.angular_velocity();

    CHECK(approx_equal(omega.x(), 0.1));
    CHECK(approx_equal(omega.y(), 0.2));
    CHECK(approx_equal(omega.z(), 0.3));

    GlobalTwistd global_twist(1.0, 2.0, 3.0, 0.4, 0.5, 0.6);
    auto omega_global = global_twist.angular_velocity();

    CHECK(approx_equal(omega_global.x(), 0.4));
    CHECK(approx_equal(omega_global.y(), 0.5));
    CHECK(approx_equal(omega_global.z(), 0.6));
}
