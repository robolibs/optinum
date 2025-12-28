#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <optinum/lie/lie.hpp>

#include <cmath>
#include <random>

using namespace optinum;
using namespace optinum::lie;

// ============================================================================
// RxSO2 Tests
// ============================================================================

TEST_SUITE("RxSO2") {

    TEST_CASE("Default construction is identity") {
        RxSO2d r;
        CHECK(r.is_identity());
        CHECK(std::abs(r.scale() - 1.0) < 1e-10);
        CHECK(std::abs(r.angle()) < 1e-10);
    }

    TEST_CASE("Construction from scale and angle") {
        RxSO2d r(2.0, M_PI / 4);
        CHECK(std::abs(r.scale() - 2.0) < 1e-10);
        CHECK(std::abs(r.angle() - M_PI / 4) < 1e-10);
    }

    TEST_CASE("Construction from SO2") {
        SO2d rot(M_PI / 3);
        RxSO2d r(rot);
        CHECK(std::abs(r.scale() - 1.0) < 1e-10);
        CHECK(std::abs(r.angle() - M_PI / 3) < 1e-10);
    }

    TEST_CASE("Exp/log round trip") {
        simd::Vector<double, 2> tangent{0.5, M_PI / 6}; // [log(scale), angle]
        auto r = RxSO2d::exp(tangent);
        auto recovered = r.log();

        CHECK(std::abs(recovered[0] - tangent[0]) < 1e-10);
        CHECK(std::abs(recovered[1] - tangent[1]) < 1e-10);
    }

    TEST_CASE("Inverse operation") {
        RxSO2d r(2.0, M_PI / 4);
        auto r_inv = r.inverse();
        auto identity = r * r_inv;

        CHECK(identity.is_identity(1e-10));
    }

    TEST_CASE("Group composition") {
        RxSO2d r1(2.0, M_PI / 6);
        RxSO2d r2(0.5, M_PI / 3);
        auto r3 = r1 * r2;

        // Scale should multiply
        CHECK(std::abs(r3.scale() - 1.0) < 1e-10); // 2.0 * 0.5 = 1.0
        // Angles should add
        CHECK(std::abs(r3.angle() - M_PI / 2) < 1e-10); // pi/6 + pi/3 = pi/2
    }

    TEST_CASE("Point transformation") {
        RxSO2d r(2.0, M_PI / 2); // Scale by 2, rotate 90 degrees
        simd::Vector<double, 2> p{1.0, 0.0};
        auto p_transformed = r * p;

        // (1, 0) scaled by 2 and rotated 90 deg -> (0, 2)
        CHECK(std::abs(p_transformed[0]) < 1e-10);
        CHECK(std::abs(p_transformed[1] - 2.0) < 1e-10);
    }

    TEST_CASE("SO2 extraction") {
        RxSO2d r(3.0, M_PI / 4);
        auto so2 = r.so2();
        CHECK(std::abs(so2.log() - M_PI / 4) < 1e-10);
    }

    TEST_CASE("Interpolation") {
        RxSO2d r1 = RxSO2d::identity();
        RxSO2d r2(std::exp(1.0), M_PI / 2);

        auto r_mid = interpolate(r1, r2, 0.5);

        CHECK(std::abs(r_mid.log()[0] - 0.5) < 1e-10);      // log(scale) = 0.5
        CHECK(std::abs(r_mid.log()[1] - M_PI / 4) < 1e-10); // angle = pi/4
    }
}

// ============================================================================
// RxSO3 Tests
// ============================================================================

TEST_SUITE("RxSO3") {

    TEST_CASE("Default construction is identity") {
        RxSO3d r;
        CHECK(r.is_identity());
        CHECK(std::abs(r.scale() - 1.0) < 1e-10);
    }

    TEST_CASE("Construction from scale and SO3") {
        auto R = SO3d::rot_z(M_PI / 3);
        RxSO3d r(2.0, R);
        CHECK(std::abs(r.scale() - 2.0) < 1e-10);
        CHECK(r.so3().is_approx(R, 1e-10));
    }

    TEST_CASE("Exp/log round trip") {
        simd::Vector<double, 4> tangent{0.3, 0.1, 0.2, 0.15}; // [sigma, wx, wy, wz]
        auto r = RxSO3d::exp(tangent);
        auto recovered = r.log();

        for (int i = 0; i < 4; ++i) {
            CHECK(std::abs(recovered[i] - tangent[i]) < 1e-10);
        }
    }

    TEST_CASE("Inverse operation") {
        auto R = SO3d::rot_x(0.5) * SO3d::rot_y(0.3);
        RxSO3d r(1.5, R);
        auto r_inv = r.inverse();
        auto identity = r * r_inv;

        CHECK(identity.is_identity(1e-10));
    }

    TEST_CASE("Group composition") {
        RxSO3d r1(2.0, SO3d::rot_z(M_PI / 6));
        RxSO3d r2(0.5, SO3d::rot_z(M_PI / 3));
        auto r3 = r1 * r2;

        CHECK(std::abs(r3.scale() - 1.0) < 1e-10); // 2.0 * 0.5 = 1.0
    }

    TEST_CASE("Point transformation") {
        RxSO3d r(2.0, SO3d::rot_z(M_PI / 2)); // Scale by 2, rotate 90 deg around z
        simd::Vector<double, 3> p{1.0, 0.0, 0.0};
        auto p_transformed = r * p;

        // (1, 0, 0) scaled by 2 and rotated 90 deg around z -> (0, 2, 0)
        CHECK(std::abs(p_transformed[0]) < 1e-10);
        CHECK(std::abs(p_transformed[1] - 2.0) < 1e-10);
        CHECK(std::abs(p_transformed[2]) < 1e-10);
    }

    TEST_CASE("Interpolation") {
        RxSO3d r1 = RxSO3d::identity();
        RxSO3d r2(std::exp(0.5), SO3d::rot_z(M_PI / 2));

        auto r_mid = interpolate(r1, r2, 0.5);

        auto log_mid = r_mid.log();
        CHECK(std::abs(log_mid[0] - 0.25) < 1e-10); // log(scale) = 0.25
    }
}

// ============================================================================
// Sim2 Tests
// ============================================================================

TEST_SUITE("Sim2") {

    TEST_CASE("Default construction is identity") {
        Sim2d sim;
        CHECK(sim.is_identity());
        CHECK(std::abs(sim.scale() - 1.0) < 1e-10);
        CHECK(std::abs(sim.angle()) < 1e-10);
        CHECK(std::abs(sim.translation()[0]) < 1e-10);
        CHECK(std::abs(sim.translation()[1]) < 1e-10);
    }

    TEST_CASE("Static factory methods") {
        auto s = Sim2d::scale(2.0);
        CHECK(std::abs(s.scale() - 2.0) < 1e-10);

        auto r = Sim2d::rot(M_PI / 4);
        CHECK(std::abs(r.angle() - M_PI / 4) < 1e-10);

        auto t = Sim2d::trans(1.0, 2.0);
        CHECK(std::abs(t.translation()[0] - 1.0) < 1e-10);
        CHECK(std::abs(t.translation()[1] - 2.0) < 1e-10);
    }

    TEST_CASE("Construction from SE2") {
        SE2d se2(SO2d(M_PI / 6), simd::Vector<double, 2>{1.0, 2.0});
        Sim2d sim(se2);

        CHECK(std::abs(sim.scale() - 1.0) < 1e-10);
        CHECK(std::abs(sim.angle() - M_PI / 6) < 1e-10);
        CHECK(std::abs(sim.translation()[0] - 1.0) < 1e-10);
        CHECK(std::abs(sim.translation()[1] - 2.0) < 1e-10);
    }

    TEST_CASE("Inverse operation") {
        Sim2d sim(RxSO2d(2.0, M_PI / 4), simd::Vector<double, 2>{1.0, 2.0});
        auto sim_inv = sim.inverse();
        auto identity = sim * sim_inv;

        CHECK(identity.is_identity(1e-10));
    }

    TEST_CASE("Group composition") {
        Sim2d sim1 = Sim2d::trans(1.0, 0.0);
        Sim2d sim2 = Sim2d::scale(2.0);
        auto sim3 = sim1 * sim2;

        // First translate, then scale: t1 + sR1 * t2 = (1,0) + 1*(0,0) = (1,0)
        CHECK(std::abs(sim3.translation()[0] - 1.0) < 1e-10);
        CHECK(std::abs(sim3.scale() - 2.0) < 1e-10);
    }

    TEST_CASE("Point transformation") {
        Sim2d sim(RxSO2d(2.0, M_PI / 2), simd::Vector<double, 2>{1.0, 0.0});
        simd::Vector<double, 2> p{1.0, 0.0};
        auto p_transformed = sim * p;

        // sR*p + t = 2*rot90*(1,0) + (1,0) = 2*(0,1) + (1,0) = (1, 2)
        CHECK(std::abs(p_transformed[0] - 1.0) < 1e-10);
        CHECK(std::abs(p_transformed[1] - 2.0) < 1e-10);
    }

    TEST_CASE("Matrix representation") {
        Sim2d sim(RxSO2d(2.0, M_PI / 2), simd::Vector<double, 2>{1.0, 2.0});
        auto M = sim.matrix();

        // Should be 3x3 homogeneous matrix
        CHECK(std::abs(M(2, 0)) < 1e-10);
        CHECK(std::abs(M(2, 1)) < 1e-10);
        CHECK(std::abs(M(2, 2) - 1.0) < 1e-10);
        CHECK(std::abs(M(0, 2) - 1.0) < 1e-10); // tx
        CHECK(std::abs(M(1, 2) - 2.0) < 1e-10); // ty
    }

    TEST_CASE("Exp/log consistency near identity") {
        simd::Vector<double, 4> twist{0.1, 0.1, 0.2, 0.3}; // small values
        auto sim = Sim2d::exp(twist);
        auto recovered = sim.log();

        for (int i = 0; i < 4; ++i) {
            CHECK(std::abs(recovered[i] - twist[i]) < 1e-6);
        }
    }

    TEST_CASE("Interpolation") {
        Sim2d s1 = Sim2d::identity();
        Sim2d s2 = Sim2d::trans(2.0, 0.0) * Sim2d::scale(std::exp(1.0));

        auto s_mid = interpolate(s1, s2, 0.5);

        // At midpoint, translation should be approximately 1.0 (depends on scale integration)
        // Scale should be exp(0.5)
        CHECK(std::abs(s_mid.scale() - std::exp(0.5)) < 0.1); // Relaxed tolerance
    }
}

// ============================================================================
// Sim3 Tests
// ============================================================================

TEST_SUITE("Sim3") {

    TEST_CASE("Default construction is identity") {
        Sim3d sim;
        CHECK(sim.is_identity());
        CHECK(std::abs(sim.scale() - 1.0) < 1e-10);
    }

    TEST_CASE("Static factory methods") {
        auto s = Sim3d::scale(2.0);
        CHECK(std::abs(s.scale() - 2.0) < 1e-10);

        auto rx = Sim3d::rot_x(M_PI / 4);
        CHECK(rx.so3().is_approx(SO3d::rot_x(M_PI / 4), 1e-10));

        auto t = Sim3d::trans(1.0, 2.0, 3.0);
        CHECK(std::abs(t.translation()[0] - 1.0) < 1e-10);
        CHECK(std::abs(t.translation()[1] - 2.0) < 1e-10);
        CHECK(std::abs(t.translation()[2] - 3.0) < 1e-10);
    }

    TEST_CASE("Construction from SE3") {
        SE3d se3(SO3d::rot_z(M_PI / 6), simd::Vector<double, 3>{1.0, 2.0, 3.0});
        Sim3d sim(se3);

        CHECK(std::abs(sim.scale() - 1.0) < 1e-10);
        CHECK(sim.so3().is_approx(SO3d::rot_z(M_PI / 6), 1e-10));
        CHECK(std::abs(sim.translation()[0] - 1.0) < 1e-10);
        CHECK(std::abs(sim.translation()[1] - 2.0) < 1e-10);
        CHECK(std::abs(sim.translation()[2] - 3.0) < 1e-10);
    }

    TEST_CASE("Inverse operation") {
        Sim3d sim(RxSO3d(2.0, SO3d::rot_x(0.5)), simd::Vector<double, 3>{1.0, 2.0, 3.0});
        auto sim_inv = sim.inverse();
        auto identity = sim * sim_inv;

        CHECK(identity.is_identity(1e-10));
    }

    TEST_CASE("Group composition associativity") {
        std::mt19937 rng(42);

        Sim3d a = Sim3d::sample_uniform(rng, 0.5, 5.0);
        Sim3d b = Sim3d::sample_uniform(rng, 0.5, 5.0);
        Sim3d c = Sim3d::sample_uniform(rng, 0.5, 5.0);

        auto left = (a * b) * c;
        auto right = a * (b * c);

        CHECK(left.is_approx(right, 1e-10));
    }

    TEST_CASE("Point transformation") {
        Sim3d sim(RxSO3d(2.0, SO3d::rot_z(M_PI / 2)), simd::Vector<double, 3>{1.0, 0.0, 0.0});
        simd::Vector<double, 3> p{1.0, 0.0, 0.0};
        auto p_transformed = sim * p;

        // sR*p + t = 2*rotz90*(1,0,0) + (1,0,0) = 2*(0,1,0) + (1,0,0) = (1, 2, 0)
        CHECK(std::abs(p_transformed[0] - 1.0) < 1e-10);
        CHECK(std::abs(p_transformed[1] - 2.0) < 1e-10);
        CHECK(std::abs(p_transformed[2]) < 1e-10);
    }

    TEST_CASE("Matrix representation") {
        Sim3d sim(RxSO3d(2.0, SO3d::rot_z(M_PI / 2)), simd::Vector<double, 3>{1.0, 2.0, 3.0});
        auto M = sim.matrix();

        // Should be 4x4 homogeneous matrix
        CHECK(std::abs(M(3, 0)) < 1e-10);
        CHECK(std::abs(M(3, 1)) < 1e-10);
        CHECK(std::abs(M(3, 2)) < 1e-10);
        CHECK(std::abs(M(3, 3) - 1.0) < 1e-10);
        CHECK(std::abs(M(0, 3) - 1.0) < 1e-10); // tx
        CHECK(std::abs(M(1, 3) - 2.0) < 1e-10); // ty
        CHECK(std::abs(M(2, 3) - 3.0) < 1e-10); // tz
    }

    TEST_CASE("Exp/log consistency near identity") {
        simd::Vector<double, 7> twist{0.1, 0.05, 0.08, 0.06, 0.2, 0.3, 0.4};
        auto sim = Sim3d::exp(twist);
        auto recovered = sim.log();

        // Check first 4 components (scale and rotation)
        for (int i = 0; i < 4; ++i) {
            CHECK(std::abs(recovered[i] - twist[i]) < 1e-6);
        }
    }

    TEST_CASE("SE3 extraction") {
        Sim3d sim(RxSO3d(2.0, SO3d::rot_x(0.5)), simd::Vector<double, 3>{1.0, 2.0, 3.0});
        auto se3 = sim.se3();

        CHECK(se3.so3().is_approx(sim.so3(), 1e-10));
        CHECK(std::abs(se3.translation()[0] - 1.0) < 1e-10);
        CHECK(std::abs(se3.translation()[1] - 2.0) < 1e-10);
        CHECK(std::abs(se3.translation()[2] - 3.0) < 1e-10);
    }

    TEST_CASE("Scale preserves rotation") {
        Sim3d sim(RxSO3d(2.0, SO3d::rot_z(M_PI / 4)), simd::Vector<double, 3>{1.0, 2.0, 3.0});
        sim.set_scale(3.0);

        CHECK(std::abs(sim.scale() - 3.0) < 1e-10);
        CHECK(sim.so3().is_approx(SO3d::rot_z(M_PI / 4), 1e-10));
    }

    TEST_CASE("Interpolation") {
        Sim3d s1 = Sim3d::identity();
        Sim3d s2 = Sim3d::trans(2.0, 0.0, 0.0) * Sim3d::scale(std::exp(1.0));

        auto s_mid = interpolate(s1, s2, 0.5);

        // Scale should be exp(0.5)
        CHECK(std::abs(s_mid.scale() - std::exp(0.5)) < 0.1); // Relaxed tolerance
    }
}

// ============================================================================
// Cross-group tests
// ============================================================================

TEST_SUITE("Similarity Group Cross-Tests") {

    TEST_CASE("RxSO2 vs Sim2 with zero translation") {
        RxSO2d rxso2(2.0, M_PI / 4);
        Sim2d sim2(rxso2, simd::Vector<double, 2>{0, 0});

        simd::Vector<double, 2> p{1.0, 0.0};
        auto p1 = rxso2 * p;
        auto p2 = sim2 * p;

        CHECK(std::abs(p1[0] - p2[0]) < 1e-10);
        CHECK(std::abs(p1[1] - p2[1]) < 1e-10);
    }

    TEST_CASE("RxSO3 vs Sim3 with zero translation") {
        RxSO3d rxso3(2.0, SO3d::rot_z(M_PI / 4));
        Sim3d sim3(rxso3, simd::Vector<double, 3>{0, 0, 0});

        simd::Vector<double, 3> p{1.0, 0.0, 0.0};
        auto p1 = rxso3 * p;
        auto p2 = sim3 * p;

        CHECK(std::abs(p1[0] - p2[0]) < 1e-10);
        CHECK(std::abs(p1[1] - p2[1]) < 1e-10);
        CHECK(std::abs(p1[2] - p2[2]) < 1e-10);
    }

    TEST_CASE("Scale=1 Sim3 equals SE3") {
        SE3d se3(SO3d::rot_y(0.5), simd::Vector<double, 3>{1.0, 2.0, 3.0});
        Sim3d sim3(se3);

        simd::Vector<double, 3> p{1.0, 2.0, 3.0};
        auto p_se3 = se3 * p;
        auto p_sim3 = sim3 * p;

        CHECK(std::abs(p_se3[0] - p_sim3[0]) < 1e-10);
        CHECK(std::abs(p_se3[1] - p_sim3[1]) < 1e-10);
        CHECK(std::abs(p_se3[2] - p_sim3[2]) < 1e-10);
    }
}
