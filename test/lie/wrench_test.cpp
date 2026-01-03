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
bool vec_approx_equal(const dp::mat::Vector<T, N> &a, const dp::mat::Vector<T, N> &b, T tol = T(1e-10)) {
    for (std::size_t i = 0; i < N; ++i) {
        if (std::abs(a[i] - b[i]) >= tol)
            return false;
    }
    return true;
}

// ===== WRENCH CONSTRUCTION TESTS =====

TEST_CASE("Wrench default construction is zero") {
    Wrenchd wrench;

    CHECK(wrench.is_zero());
    CHECK(approx_equal(wrench.fx(), 0.0));
    CHECK(approx_equal(wrench.fy(), 0.0));
    CHECK(approx_equal(wrench.fz(), 0.0));
    CHECK(approx_equal(wrench.tx(), 0.0));
    CHECK(approx_equal(wrench.ty(), 0.0));
    CHECK(approx_equal(wrench.tz(), 0.0));
}

TEST_CASE("Wrench construction from vectors") {
    dp::mat::Vector<double, 3> force{{1.0, 2.0, 3.0}};
    dp::mat::Vector<double, 3> torque{{0.1, 0.2, 0.3}};
    Wrenchd wrench(force, torque);

    CHECK(vec_approx_equal(wrench.force(), force));
    CHECK(vec_approx_equal(wrench.torque(), torque));
}

TEST_CASE("Wrench construction from components") {
    Wrenchd wrench(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);

    CHECK(approx_equal(wrench.fx(), 1.0));
    CHECK(approx_equal(wrench.fy(), 2.0));
    CHECK(approx_equal(wrench.fz(), 3.0));
    CHECK(approx_equal(wrench.tx(), 0.1));
    CHECK(approx_equal(wrench.ty(), 0.2));
    CHECK(approx_equal(wrench.tz(), 0.3));
}

TEST_CASE("Wrench construction from Vector6") {
    dp::mat::Vector<double, 6> vec{{1.0, 2.0, 3.0, 0.1, 0.2, 0.3}};
    Wrenchd wrench(vec);

    CHECK(vec_approx_equal(wrench.vector(), vec));
}

// ===== STATIC FACTORY METHODS =====

TEST_CASE("Wrench static factory methods") {
    SUBCASE("zero") {
        auto wrench = Wrenchd::zero();
        CHECK(wrench.is_zero());
    }

    SUBCASE("from_vector") {
        dp::mat::Vector<double, 6> vec{{1.0, 2.0, 3.0, 0.1, 0.2, 0.3}};
        auto wrench = Wrenchd::from_vector(vec);
        CHECK(vec_approx_equal(wrench.vector(), vec));
    }

    SUBCASE("pure_force from Vector3") {
        dp::mat::Vector<double, 3> force{{1.0, 2.0, 3.0}};
        auto wrench = Wrenchd::pure_force(force);

        CHECK(vec_approx_equal(wrench.force(), force));
        CHECK(approx_equal(wrench.tx(), 0.0));
        CHECK(approx_equal(wrench.ty(), 0.0));
        CHECK(approx_equal(wrench.tz(), 0.0));
    }

    SUBCASE("pure_force from components") {
        auto wrench = Wrenchd::pure_force(1.0, 2.0, 3.0);

        CHECK(approx_equal(wrench.fx(), 1.0));
        CHECK(approx_equal(wrench.fy(), 2.0));
        CHECK(approx_equal(wrench.fz(), 3.0));
        CHECK(approx_equal(wrench.tx(), 0.0));
        CHECK(approx_equal(wrench.ty(), 0.0));
        CHECK(approx_equal(wrench.tz(), 0.0));
    }

    SUBCASE("pure_torque from Vector3") {
        dp::mat::Vector<double, 3> torque{{0.1, 0.2, 0.3}};
        auto wrench = Wrenchd::pure_torque(torque);

        CHECK(approx_equal(wrench.fx(), 0.0));
        CHECK(approx_equal(wrench.fy(), 0.0));
        CHECK(approx_equal(wrench.fz(), 0.0));
        CHECK(vec_approx_equal(wrench.torque(), torque));
    }

    SUBCASE("pure_torque from components") {
        auto wrench = Wrenchd::pure_torque(0.1, 0.2, 0.3);

        CHECK(approx_equal(wrench.fx(), 0.0));
        CHECK(approx_equal(wrench.fy(), 0.0));
        CHECK(approx_equal(wrench.fz(), 0.0));
        CHECK(approx_equal(wrench.tx(), 0.1));
        CHECK(approx_equal(wrench.ty(), 0.2));
        CHECK(approx_equal(wrench.tz(), 0.3));
    }
}

// ===== MUTATORS =====

TEST_CASE("Wrench mutators") {
    Wrenchd wrench;

    SUBCASE("set_force from components") {
        wrench.set_force(1.0, 2.0, 3.0);
        CHECK(approx_equal(wrench.fx(), 1.0));
        CHECK(approx_equal(wrench.fy(), 2.0));
        CHECK(approx_equal(wrench.fz(), 3.0));
    }

    SUBCASE("set_force from Vector3") {
        dp::mat::Vector<double, 3> force{{4.0, 5.0, 6.0}};
        wrench.set_force(force);
        CHECK(vec_approx_equal(wrench.force(), force));
    }

    SUBCASE("set_torque from components") {
        wrench.set_torque(0.1, 0.2, 0.3);
        CHECK(approx_equal(wrench.tx(), 0.1));
        CHECK(approx_equal(wrench.ty(), 0.2));
        CHECK(approx_equal(wrench.tz(), 0.3));
    }

    SUBCASE("set_torque from Vector3") {
        dp::mat::Vector<double, 3> torque{{0.4, 0.5, 0.6}};
        wrench.set_torque(torque);
        CHECK(vec_approx_equal(wrench.torque(), torque));
    }

    SUBCASE("set_vector") {
        dp::mat::Vector<double, 6> vec{{7.0, 8.0, 9.0, 0.7, 0.8, 0.9}};
        wrench.set_vector(vec);
        CHECK(vec_approx_equal(wrench.vector(), vec));
    }

    SUBCASE("set_zero") {
        wrench.set_force(1.0, 2.0, 3.0);
        wrench.set_torque(0.1, 0.2, 0.3);
        wrench.set_zero();
        CHECK(wrench.is_zero());
    }

    SUBCASE("mutable accessors") {
        wrench.force()[0] = 10.0;
        wrench.torque()[1] = 20.0;
        CHECK(approx_equal(wrench.fx(), 10.0));
        CHECK(approx_equal(wrench.ty(), 20.0));
    }
}

// ===== FRAME TRANSFORMATION TESTS =====

TEST_CASE("Wrench frame transformation") {
    SUBCASE("identity transform") {
        Wrenchd wrench(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        SE3d T = SE3d::identity();

        auto wrench_new = wrench.transform(T);

        CHECK(wrench.is_approx(wrench_new, 1e-10));
    }

    SUBCASE("pure rotation") {
        // Force along x-axis
        Wrenchd wrench = Wrenchd::pure_force(1.0, 0.0, 0.0);
        // Rotate 90° around z
        SE3d T(SO3d::rot_z(std::numbers::pi / 2), dp::mat::Vector<double, 3>{{0.0, 0.0, 0.0}});

        auto wrench_new = wrench.transform(T);

        // Force should now be along y-axis (approximately)
        CHECK(approx_equal(wrench_new.fx(), 0.0, 1e-10));
        CHECK(approx_equal(wrench_new.fy(), 1.0, 1e-10));
        CHECK(approx_equal(wrench_new.fz(), 0.0, 1e-10));
    }

    SUBCASE("pure translation with force") {
        // Pure force along z
        Wrenchd wrench = Wrenchd::pure_force(0.0, 0.0, 1.0);
        // Translate along x
        SE3d T = SE3d::trans(1.0, 0.0, 0.0);

        auto wrench_new = wrench.transform(T);

        // Force should be unchanged
        CHECK(approx_equal(wrench_new.fx(), 0.0, 1e-10));
        CHECK(approx_equal(wrench_new.fy(), 0.0, 1e-10));
        CHECK(approx_equal(wrench_new.fz(), 1.0, 1e-10));

        // But torque should appear due to moment arm: τ = r × f
        // r = (1, 0, 0), f = (0, 0, 1) => τ = (0, -1, 0)
        CHECK(approx_equal(wrench_new.tx(), 0.0, 1e-10));
        CHECK(approx_equal(wrench_new.ty(), -1.0, 1e-10));
        CHECK(approx_equal(wrench_new.tz(), 0.0, 1e-10));
    }

    SUBCASE("round-trip with random transform") {
        std::mt19937 rng(42);

        for (int i = 0; i < 50; ++i) {
            auto T = SE3d::sample_uniform(rng);
            Wrenchd wrench(0.5, 1.0, -0.3, 0.1, -0.2, 0.15);

            // Transform forward then backward
            auto wrench_new = wrench.transform(T);
            auto wrench_back = wrench_new.transform(T.inverse());

            CHECK(wrench.is_approx(wrench_back, 1e-9));
        }
    }
}

// ===== POWER COMPUTATION TESTS =====

TEST_CASE("Wrench power computation") {
    SUBCASE("zero twist gives zero power") {
        Wrenchd wrench(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        LocalTwistd twist;

        CHECK(approx_equal(wrench.power(twist), 0.0));
    }

    SUBCASE("zero wrench gives zero power") {
        Wrenchd wrench;
        LocalTwistd twist(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);

        CHECK(approx_equal(wrench.power(twist), 0.0));
    }

    SUBCASE("pure force with linear velocity") {
        // Force of 10N along x, velocity of 2 m/s along x
        Wrenchd wrench = Wrenchd::pure_force(10.0, 0.0, 0.0);
        LocalTwistd twist(2.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        // Power = F · v = 10 * 2 = 20 W
        CHECK(approx_equal(wrench.power(twist), 20.0));
    }

    SUBCASE("pure torque with angular velocity") {
        // Torque of 5 Nm around z, angular velocity of 3 rad/s around z
        Wrenchd wrench = Wrenchd::pure_torque(0.0, 0.0, 5.0);
        LocalTwistd twist(0.0, 0.0, 0.0, 0.0, 0.0, 3.0);

        // Power = τ · ω = 5 * 3 = 15 W
        CHECK(approx_equal(wrench.power(twist), 15.0));
    }

    SUBCASE("general case") {
        // P = v·f + ω·τ
        Wrenchd wrench(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        LocalTwistd twist(0.5, 1.0, 1.5, 0.05, 0.1, 0.15);

        // v·f = 0.5*1 + 1*2 + 1.5*3 = 0.5 + 2 + 4.5 = 7.0
        // ω·τ = 0.05*0.1 + 0.1*0.2 + 0.15*0.3 = 0.005 + 0.02 + 0.045 = 0.07
        // Total = 7.07
        CHECK(approx_equal(wrench.power(twist), 7.07, 1e-10));
    }

    SUBCASE("power with GlobalTwist") {
        Wrenchd wrench(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        GlobalTwistd twist(0.5, 1.0, 1.5, 0.05, 0.1, 0.15);

        CHECK(approx_equal(wrench.power(twist), 7.07, 1e-10));
    }

    SUBCASE("dot product with Vector6") {
        Wrenchd wrench(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        dp::mat::Vector<double, 6> v{{0.5, 1.0, 1.5, 0.05, 0.1, 0.15}};

        CHECK(approx_equal(wrench.dot(v), 7.07, 1e-10));
    }
}

TEST_CASE("Power invariance under frame transformation") {
    // Power should be invariant: twist_new · wrench_new = twist_old · wrench_old
    std::mt19937 rng(42);

    for (int i = 0; i < 50; ++i) {
        auto T = SE3d::sample_uniform(rng);
        Wrenchd wrench(0.5, 1.0, -0.3, 0.1, -0.2, 0.15);
        LocalTwistd twist(0.3, -0.5, 0.8, 0.05, -0.1, 0.2);

        // Transform both to new frame
        auto wrench_new = wrench.transform(T);
        auto twist_new = twist.to_global(T);

        // Power should be preserved
        double power_old = wrench.power(twist);
        double power_new = wrench_new.power(twist_new);

        CHECK(approx_equal(power_old, power_new, 1e-9));
    }
}

// ===== ARITHMETIC TESTS =====

TEST_CASE("Wrench arithmetic") {
    Wrenchd a(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    Wrenchd b(0.5, 1.0, 1.5, 0.05, 0.1, 0.15);

    SUBCASE("addition") {
        auto c = a + b;
        CHECK(approx_equal(c.fx(), 1.5));
        CHECK(approx_equal(c.fy(), 3.0));
        CHECK(approx_equal(c.fz(), 4.5));
        CHECK(approx_equal(c.tx(), 0.15));
        CHECK(approx_equal(c.ty(), 0.3));
        CHECK(approx_equal(c.tz(), 0.45));
    }

    SUBCASE("subtraction") {
        auto c = a - b;
        CHECK(approx_equal(c.fx(), 0.5));
        CHECK(approx_equal(c.fy(), 1.0));
        CHECK(approx_equal(c.fz(), 1.5));
        CHECK(approx_equal(c.tx(), 0.05));
        CHECK(approx_equal(c.ty(), 0.1));
        CHECK(approx_equal(c.tz(), 0.15));
    }

    SUBCASE("scalar multiplication (right)") {
        auto c = a * 2.0;
        CHECK(approx_equal(c.fx(), 2.0));
        CHECK(approx_equal(c.fy(), 4.0));
        CHECK(approx_equal(c.fz(), 6.0));
        CHECK(approx_equal(c.tx(), 0.2));
        CHECK(approx_equal(c.ty(), 0.4));
        CHECK(approx_equal(c.tz(), 0.6));
    }

    SUBCASE("scalar multiplication (left)") {
        auto c = 2.0 * a;
        CHECK(approx_equal(c.fx(), 2.0));
        CHECK(approx_equal(c.fy(), 4.0));
        CHECK(approx_equal(c.fz(), 6.0));
    }

    SUBCASE("scalar division") {
        auto c = a / 2.0;
        CHECK(approx_equal(c.fx(), 0.5));
        CHECK(approx_equal(c.fy(), 1.0));
        CHECK(approx_equal(c.fz(), 1.5));
        CHECK(approx_equal(c.tx(), 0.05));
        CHECK(approx_equal(c.ty(), 0.1));
        CHECK(approx_equal(c.tz(), 0.15));
    }

    SUBCASE("negation") {
        auto c = -a;
        CHECK(approx_equal(c.fx(), -1.0));
        CHECK(approx_equal(c.fy(), -2.0));
        CHECK(approx_equal(c.fz(), -3.0));
        CHECK(approx_equal(c.tx(), -0.1));
        CHECK(approx_equal(c.ty(), -0.2));
        CHECK(approx_equal(c.tz(), -0.3));
    }

    SUBCASE("compound addition") {
        Wrenchd c = a;
        c += b;
        CHECK(approx_equal(c.fx(), 1.5));
        CHECK(approx_equal(c.tx(), 0.15));
    }

    SUBCASE("compound subtraction") {
        Wrenchd c = a;
        c -= b;
        CHECK(approx_equal(c.fx(), 0.5));
        CHECK(approx_equal(c.tx(), 0.05));
    }

    SUBCASE("compound multiplication") {
        Wrenchd c = a;
        c *= 2.0;
        CHECK(approx_equal(c.fx(), 2.0));
        CHECK(approx_equal(c.tx(), 0.2));
    }

    SUBCASE("compound division") {
        Wrenchd c = a;
        c /= 2.0;
        CHECK(approx_equal(c.fx(), 0.5));
        CHECK(approx_equal(c.tx(), 0.05));
    }
}

// ===== COMPARISON TESTS =====

TEST_CASE("Wrench comparison") {
    Wrenchd a(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    Wrenchd b(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    Wrenchd c(1.0, 2.0, 3.0, 0.1, 0.2, 0.31);

    SUBCASE("equality") {
        CHECK(a == b);
        CHECK_FALSE(a == c);
    }

    SUBCASE("inequality") {
        CHECK_FALSE(a != b);
        CHECK(a != c);
    }

    SUBCASE("is_approx") {
        CHECK(a.is_approx(b));
        CHECK_FALSE(a.is_approx(c, 0.001));
        CHECK(a.is_approx(c, 0.1)); // With larger tolerance
    }

    SUBCASE("is_zero") {
        Wrenchd zero;
        CHECK(zero.is_zero());
        CHECK_FALSE(a.is_zero());

        Wrenchd almost_zero(1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15);
        CHECK(almost_zero.is_zero(1e-10));
    }
}

// ===== NORM TESTS =====

TEST_CASE("Wrench norms") {
    SUBCASE("force_norm") {
        Wrenchd wrench = Wrenchd::pure_force(3.0, 4.0, 0.0);
        CHECK(approx_equal(wrench.force_norm(), 5.0));
    }

    SUBCASE("torque_norm") {
        Wrenchd wrench = Wrenchd::pure_torque(0.0, 3.0, 4.0);
        CHECK(approx_equal(wrench.torque_norm(), 5.0));
    }

    SUBCASE("general case") {
        Wrenchd wrench(1.0, 2.0, 2.0, 0.1, 0.2, 0.2);
        CHECK(approx_equal(wrench.force_norm(), 3.0));
        CHECK(approx_equal(wrench.torque_norm(), 0.3));
    }
}

// ===== TYPE CONVERSION =====

TEST_CASE("Wrench cast to different scalar type") {
    Wrenchd wrench_d(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    auto wrench_f = wrench_d.cast<float>();

    CHECK(approx_equal(static_cast<double>(wrench_f.fx()), wrench_d.fx(), 1e-5));
    CHECK(approx_equal(static_cast<double>(wrench_f.fy()), wrench_d.fy(), 1e-5));
    CHECK(approx_equal(static_cast<double>(wrench_f.fz()), wrench_d.fz(), 1e-5));
    CHECK(approx_equal(static_cast<double>(wrench_f.tx()), wrench_d.tx(), 1e-5));
    CHECK(approx_equal(static_cast<double>(wrench_f.ty()), wrench_d.ty(), 1e-5));
    CHECK(approx_equal(static_cast<double>(wrench_f.tz()), wrench_d.tz(), 1e-5));
}

// ===== TYPE ALIASES =====

TEST_CASE("Type aliases work correctly") {
    Wrenchf wrench_f(1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f);
    Wrenchd wrench_d(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);

    CHECK(approx_equal(static_cast<double>(wrench_f.fx()), wrench_d.fx(), 1e-5));
    CHECK(approx_equal(static_cast<double>(wrench_f.tx()), wrench_d.tx(), 1e-5));
}

// ===== PHYSICAL SCENARIOS =====

TEST_CASE("Physical scenario: gravity force") {
    // 1 kg mass at position (1, 0, 0) from origin
    // Gravity force: F = (0, 0, -9.81) N
    // Torque about origin: τ = r × F = (1, 0, 0) × (0, 0, -9.81) = (0, 9.81, 0)

    Wrenchd wrench_at_mass = Wrenchd::pure_force(0.0, 0.0, -9.81);

    // Transform to origin (mass is at x=1 from origin)
    SE3d T_origin_mass = SE3d::trans(1.0, 0.0, 0.0);
    auto wrench_at_origin = wrench_at_mass.transform(T_origin_mass);

    CHECK(approx_equal(wrench_at_origin.fx(), 0.0, 1e-10));
    CHECK(approx_equal(wrench_at_origin.fy(), 0.0, 1e-10));
    CHECK(approx_equal(wrench_at_origin.fz(), -9.81, 1e-10));
    CHECK(approx_equal(wrench_at_origin.tx(), 0.0, 1e-10));
    CHECK(approx_equal(wrench_at_origin.ty(), 9.81, 1e-10));
    CHECK(approx_equal(wrench_at_origin.tz(), 0.0, 1e-10));
}

TEST_CASE("Physical scenario: robot arm") {
    // End-effector applying force to environment
    // Force at end-effector: 10N along z
    // End-effector is at (0.5, 0, 0.3) from base, rotated 45° around z

    Wrenchd wrench_ee = Wrenchd::pure_force(0.0, 0.0, 10.0);

    // Transform from end-effector to base
    SE3d T_base_ee(SO3d::rot_z(std::numbers::pi / 4), dp::mat::Vector<double, 3>{{0.5, 0.0, 0.3}});

    auto wrench_base = wrench_ee.transform(T_base_ee);

    // Force should still be along z (rotation around z doesn't change z-force)
    CHECK(approx_equal(wrench_base.fz(), 10.0, 1e-10));

    // Torque should appear due to moment arm
    // τ = r × F where r = (0.5, 0, 0.3) and F = (0, 0, 10)
    // τ = (0*10 - 0.3*0, 0.3*0 - 0.5*10, 0.5*0 - 0*0) = (0, -5, 0)
    CHECK(approx_equal(wrench_base.tx(), 0.0, 1e-10));
    CHECK(approx_equal(wrench_base.ty(), -5.0, 1e-10));
    CHECK(approx_equal(wrench_base.tz(), 0.0, 1e-10));
}
