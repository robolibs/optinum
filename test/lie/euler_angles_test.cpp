#include <doctest/doctest.h>
#include <optinum/lie/lie.hpp>

#include <cmath>
#include <numbers>
#include <random>

using namespace optinum::lie;
using namespace optinum;

// ===== HELPER FUNCTIONS =====

template <typename T> bool approx_equal(T a, T b, T tol = T(1e-10)) { return std::abs(a - b) < tol; }

namespace dp = ::datapod;

template <typename T, std::size_t N>
bool vec_approx_equal(const dp::mat::Vector<T, N> &a, const dp::mat::Vector<T, N> &b, T tol = T(1e-10)) {
    for (std::size_t i = 0; i < N; ++i) {
        if (std::abs(a[i] - b[i]) >= tol)
            return false;
    }
    return true;
}

template <typename T, std::size_t R, std::size_t C>
bool mat_approx_equal(const dp::mat::Matrix<T, R, C> &A, const dp::mat::Matrix<T, R, C> &B, T tol = T(1e-10)) {
    for (std::size_t i = 0; i < R; ++i) {
        for (std::size_t j = 0; j < C; ++j) {
            if (std::abs(A(i, j) - B(i, j)) >= tol)
                return false;
        }
    }
    return true;
}

// ===== EulerAnglesZYX CONSTRUCTION TESTS =====

TEST_CASE("EulerAnglesZYX default construction is identity") {
    EulerAnglesZYXd euler;

    CHECK(euler.is_identity());
    CHECK(approx_equal(euler.yaw(), 0.0));
    CHECK(approx_equal(euler.pitch(), 0.0));
    CHECK(approx_equal(euler.roll(), 0.0));
}

TEST_CASE("EulerAnglesZYX construction from angles") {
    const double yaw = 0.1, pitch = 0.2, roll = 0.3;
    EulerAnglesZYXd euler(yaw, pitch, roll);

    CHECK(approx_equal(euler.yaw(), yaw));
    CHECK(approx_equal(euler.pitch(), pitch));
    CHECK(approx_equal(euler.roll(), roll));

    // Check aliases
    CHECK(approx_equal(euler.z(), yaw));
    CHECK(approx_equal(euler.y(), pitch));
    CHECK(approx_equal(euler.x(), roll));
}

TEST_CASE("EulerAnglesZYX construction from vector") {
    dp::mat::Vector<double, 3> v{{0.1, 0.2, 0.3}};
    EulerAnglesZYXd euler(v);

    CHECK(approx_equal(euler.yaw(), 0.1));
    CHECK(approx_equal(euler.pitch(), 0.2));
    CHECK(approx_equal(euler.roll(), 0.3));
}

// ===== EulerAnglesZYX CONVERSION TESTS =====

TEST_CASE("EulerAnglesZYX to/from SO3 round-trip") {
    SUBCASE("small angles") {
        const double yaw = 0.1, pitch = 0.2, roll = 0.3;
        EulerAnglesZYXd euler1(yaw, pitch, roll);

        auto R = euler1.to_rotation();
        EulerAnglesZYXd euler2(R);

        CHECK(approx_equal(euler1.yaw(), euler2.yaw(), 1e-10));
        CHECK(approx_equal(euler1.pitch(), euler2.pitch(), 1e-10));
        CHECK(approx_equal(euler1.roll(), euler2.roll(), 1e-10));
    }

    SUBCASE("larger angles") {
        const double yaw = 1.0, pitch = 0.5, roll = -0.7;
        EulerAnglesZYXd euler1(yaw, pitch, roll);

        auto R = euler1.to_rotation();
        EulerAnglesZYXd euler2(R);

        CHECK(approx_equal(euler1.yaw(), euler2.yaw(), 1e-10));
        CHECK(approx_equal(euler1.pitch(), euler2.pitch(), 1e-10));
        CHECK(approx_equal(euler1.roll(), euler2.roll(), 1e-10));
    }

    SUBCASE("random angles") {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 0; i < 100; ++i) {
            // Keep pitch away from gimbal lock
            const double yaw = dist(rng) * std::numbers::pi;
            const double pitch = dist(rng) * 0.4 * std::numbers::pi; // [-0.4π, 0.4π]
            const double roll = dist(rng) * std::numbers::pi;

            EulerAnglesZYXd euler1(yaw, pitch, roll);
            auto R = euler1.to_rotation();
            EulerAnglesZYXd euler2(R);

            CHECK(approx_equal(euler1.yaw(), euler2.yaw(), 1e-9));
            CHECK(approx_equal(euler1.pitch(), euler2.pitch(), 1e-9));
            CHECK(approx_equal(euler1.roll(), euler2.roll(), 1e-9));
        }
    }
}

TEST_CASE("EulerAnglesZYX rotation matches SO3 axis rotations") {
    SUBCASE("pure yaw") {
        const double angle = 0.5;
        EulerAnglesZYXd euler(angle, 0.0, 0.0);
        auto R_euler = euler.to_rotation();
        auto R_so3 = SO3d::rot_z(angle);

        CHECK(R_euler.is_approx(R_so3, 1e-10));
    }

    SUBCASE("pure pitch") {
        const double angle = 0.5;
        EulerAnglesZYXd euler(0.0, angle, 0.0);
        auto R_euler = euler.to_rotation();
        auto R_so3 = SO3d::rot_y(angle);

        CHECK(R_euler.is_approx(R_so3, 1e-10));
    }

    SUBCASE("pure roll") {
        const double angle = 0.5;
        EulerAnglesZYXd euler(0.0, 0.0, angle);
        auto R_euler = euler.to_rotation();
        auto R_so3 = SO3d::rot_x(angle);

        CHECK(R_euler.is_approx(R_so3, 1e-10));
    }
}

// ===== GIMBAL LOCK TESTS =====

TEST_CASE("EulerAnglesZYX gimbal lock detection") {
    SUBCASE("not at gimbal lock") {
        EulerAnglesZYXd euler(0.1, 0.2, 0.3);
        CHECK_FALSE(euler.is_near_gimbal_lock());
        CHECK(euler.gimbal_lock_distance() > 1.0);
    }

    SUBCASE("near gimbal lock (pitch = π/2)") {
        EulerAnglesZYXd euler(0.1, std::numbers::pi / 2 - 0.001, 0.3);
        CHECK(euler.is_near_gimbal_lock());
        CHECK(euler.gimbal_lock_distance() < 0.01);
    }

    SUBCASE("near gimbal lock (pitch = -π/2)") {
        EulerAnglesZYXd euler(0.1, -std::numbers::pi / 2 + 0.001, 0.3);
        CHECK(euler.is_near_gimbal_lock());
        CHECK(euler.gimbal_lock_distance() < 0.01);
    }
}

TEST_CASE("EulerAnglesZYX conversion at gimbal lock") {
    // At gimbal lock, yaw and roll become coupled
    // The rotation should still be correct even if angles differ
    SUBCASE("pitch = π/2") {
        EulerAnglesZYXd euler1(0.5, std::numbers::pi / 2, 0.3);
        auto R = euler1.to_rotation();
        EulerAnglesZYXd euler2(R);

        // Rotations should match even if angles differ
        CHECK(euler1.to_rotation().is_approx(euler2.to_rotation(), 1e-9));
    }

    SUBCASE("pitch = -π/2") {
        EulerAnglesZYXd euler1(0.5, -std::numbers::pi / 2, 0.3);
        auto R = euler1.to_rotation();
        EulerAnglesZYXd euler2(R);

        // Rotations should match even if angles differ
        CHECK(euler1.to_rotation().is_approx(euler2.to_rotation(), 1e-9));
    }
}

// ===== JACOBIAN TESTS =====

TEST_CASE("EulerAnglesZYX Jacobians") {
    SUBCASE("J * J_inv = I for non-gimbal-lock") {
        EulerAnglesZYXd euler(0.3, 0.2, 0.1);
        auto J = euler.euler_rates_to_angular_velocity();
        auto J_inv = euler.angular_velocity_to_euler_rates();

        // Compute J * J_inv
        dp::mat::Matrix<double, 3, 3> I;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double sum = 0;
                for (int k = 0; k < 3; ++k) {
                    sum += J(i, k) * J_inv(k, j);
                }
                I(i, j) = sum;
            }
        }

        CHECK(approx_equal(I(0, 0), 1.0, 1e-10));
        CHECK(approx_equal(I(1, 1), 1.0, 1e-10));
        CHECK(approx_equal(I(2, 2), 1.0, 1e-10));
        CHECK(approx_equal(I(0, 1), 0.0, 1e-10));
        CHECK(approx_equal(I(0, 2), 0.0, 1e-10));
        CHECK(approx_equal(I(1, 0), 0.0, 1e-10));
        CHECK(approx_equal(I(1, 2), 0.0, 1e-10));
        CHECK(approx_equal(I(2, 0), 0.0, 1e-10));
        CHECK(approx_equal(I(2, 1), 0.0, 1e-10));
    }

    SUBCASE("Jacobian at identity") {
        EulerAnglesZYXd euler;
        auto J = euler.euler_rates_to_angular_velocity();

        // At identity (all angles = 0):
        // J = [0, 0, 1; 0, 1, 0; 1, 0, 0]
        CHECK(approx_equal(J(0, 0), 0.0, 1e-10));
        CHECK(approx_equal(J(0, 1), 0.0, 1e-10));
        CHECK(approx_equal(J(0, 2), 1.0, 1e-10));
        CHECK(approx_equal(J(1, 0), 0.0, 1e-10));
        CHECK(approx_equal(J(1, 1), 1.0, 1e-10));
        CHECK(approx_equal(J(1, 2), 0.0, 1e-10));
        CHECK(approx_equal(J(2, 0), 1.0, 1e-10));
        CHECK(approx_equal(J(2, 1), 0.0, 1e-10));
        CHECK(approx_equal(J(2, 2), 0.0, 1e-10));
    }

    SUBCASE("random angles (away from gimbal lock)") {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 0; i < 50; ++i) {
            const double yaw = dist(rng) * std::numbers::pi;
            const double pitch = dist(rng) * 0.4 * std::numbers::pi;
            const double roll = dist(rng) * std::numbers::pi;

            EulerAnglesZYXd euler(yaw, pitch, roll);
            auto J = euler.euler_rates_to_angular_velocity();
            auto J_inv = euler.angular_velocity_to_euler_rates();

            // Compute J * J_inv
            dp::mat::Matrix<double, 3, 3> I;
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    double sum = 0;
                    for (int k = 0; k < 3; ++k) {
                        sum += J(r, k) * J_inv(k, c);
                    }
                    I(r, c) = sum;
                }
            }

            CHECK(approx_equal(I(0, 0), 1.0, 1e-9));
            CHECK(approx_equal(I(1, 1), 1.0, 1e-9));
            CHECK(approx_equal(I(2, 2), 1.0, 1e-9));
        }
    }
}

// ===== EulerAnglesXYZ TESTS =====

TEST_CASE("EulerAnglesXYZ default construction is identity") {
    EulerAnglesXYZd euler;

    CHECK(euler.is_identity());
    CHECK(approx_equal(euler.roll(), 0.0));
    CHECK(approx_equal(euler.pitch(), 0.0));
    CHECK(approx_equal(euler.yaw(), 0.0));
}

TEST_CASE("EulerAnglesXYZ construction from angles") {
    const double roll = 0.1, pitch = 0.2, yaw = 0.3;
    EulerAnglesXYZd euler(roll, pitch, yaw);

    CHECK(approx_equal(euler.roll(), roll));
    CHECK(approx_equal(euler.pitch(), pitch));
    CHECK(approx_equal(euler.yaw(), yaw));

    // Check aliases
    CHECK(approx_equal(euler.x(), roll));
    CHECK(approx_equal(euler.y(), pitch));
    CHECK(approx_equal(euler.z(), yaw));
}

TEST_CASE("EulerAnglesXYZ to/from SO3 round-trip") {
    SUBCASE("small angles") {
        const double roll = 0.1, pitch = 0.2, yaw = 0.3;
        EulerAnglesXYZd euler1(roll, pitch, yaw);

        auto R = euler1.to_rotation();
        EulerAnglesXYZd euler2(R);

        CHECK(approx_equal(euler1.roll(), euler2.roll(), 1e-10));
        CHECK(approx_equal(euler1.pitch(), euler2.pitch(), 1e-10));
        CHECK(approx_equal(euler1.yaw(), euler2.yaw(), 1e-10));
    }

    SUBCASE("random angles") {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 0; i < 100; ++i) {
            const double roll = dist(rng) * std::numbers::pi;
            const double pitch = dist(rng) * 0.4 * std::numbers::pi;
            const double yaw = dist(rng) * std::numbers::pi;

            EulerAnglesXYZd euler1(roll, pitch, yaw);
            auto R = euler1.to_rotation();
            EulerAnglesXYZd euler2(R);

            CHECK(approx_equal(euler1.roll(), euler2.roll(), 1e-9));
            CHECK(approx_equal(euler1.pitch(), euler2.pitch(), 1e-9));
            CHECK(approx_equal(euler1.yaw(), euler2.yaw(), 1e-9));
        }
    }
}

TEST_CASE("EulerAnglesXYZ rotation matches SO3 axis rotations") {
    SUBCASE("pure roll") {
        const double angle = 0.5;
        EulerAnglesXYZd euler(angle, 0.0, 0.0);
        auto R_euler = euler.to_rotation();
        auto R_so3 = SO3d::rot_x(angle);

        CHECK(R_euler.is_approx(R_so3, 1e-10));
    }

    SUBCASE("pure pitch") {
        const double angle = 0.5;
        EulerAnglesXYZd euler(0.0, angle, 0.0);
        auto R_euler = euler.to_rotation();
        auto R_so3 = SO3d::rot_y(angle);

        CHECK(R_euler.is_approx(R_so3, 1e-10));
    }

    SUBCASE("pure yaw") {
        const double angle = 0.5;
        EulerAnglesXYZd euler(0.0, 0.0, angle);
        auto R_euler = euler.to_rotation();
        auto R_so3 = SO3d::rot_z(angle);

        CHECK(R_euler.is_approx(R_so3, 1e-10));
    }
}

TEST_CASE("EulerAnglesXYZ Jacobians") {
    SUBCASE("J * J_inv = I for non-gimbal-lock") {
        EulerAnglesXYZd euler(0.1, 0.2, 0.3);
        auto J = euler.euler_rates_to_angular_velocity();
        auto J_inv = euler.angular_velocity_to_euler_rates();

        // Compute J * J_inv
        dp::mat::Matrix<double, 3, 3> I;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double sum = 0;
                for (int k = 0; k < 3; ++k) {
                    sum += J(i, k) * J_inv(k, j);
                }
                I(i, j) = sum;
            }
        }

        CHECK(approx_equal(I(0, 0), 1.0, 1e-10));
        CHECK(approx_equal(I(1, 1), 1.0, 1e-10));
        CHECK(approx_equal(I(2, 2), 1.0, 1e-10));
    }
}

// ===== CONVERSION BETWEEN CONVENTIONS =====

TEST_CASE("Conversion between ZYX and XYZ") {
    SUBCASE("round-trip ZYX -> XYZ -> ZYX") {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 0; i < 50; ++i) {
            const double yaw = dist(rng) * std::numbers::pi;
            const double pitch = dist(rng) * 0.4 * std::numbers::pi;
            const double roll = dist(rng) * std::numbers::pi;

            EulerAnglesZYXd zyx1(yaw, pitch, roll);
            auto xyz = to_xyz(zyx1);
            auto zyx2 = to_zyx(xyz);

            // Rotations should match
            CHECK(zyx1.to_rotation().is_approx(zyx2.to_rotation(), 1e-9));
        }
    }

    SUBCASE("both conventions produce same rotation for same physical rotation") {
        // Create a rotation using ZYX
        EulerAnglesZYXd zyx(0.3, 0.2, 0.1);
        auto R_zyx = zyx.to_rotation();

        // Convert to XYZ
        auto xyz = to_xyz(zyx);
        auto R_xyz = xyz.to_rotation();

        // Both should produce the same rotation
        CHECK(R_zyx.is_approx(R_xyz, 1e-10));
    }
}

// ===== NORMALIZATION TESTS =====

TEST_CASE("EulerAnglesZYX normalization") {
    SUBCASE("angles already in range") {
        EulerAnglesZYXd euler(0.1, 0.2, 0.3);
        auto unique = euler.get_unique();

        CHECK(approx_equal(euler.yaw(), unique.yaw(), 1e-10));
        CHECK(approx_equal(euler.pitch(), unique.pitch(), 1e-10));
        CHECK(approx_equal(euler.roll(), unique.roll(), 1e-10));
    }

    SUBCASE("yaw outside range") {
        EulerAnglesZYXd euler(4.0, 0.2, 0.3); // yaw > π
        auto unique = euler.get_unique();

        // Rotation should be the same
        CHECK(euler.to_rotation().is_approx(unique.to_rotation(), 1e-9));
        // Yaw should be in [-π, π)
        CHECK(unique.yaw() >= -std::numbers::pi);
        CHECK(unique.yaw() < std::numbers::pi);
    }
}

// ===== TYPE CONVERSION =====

TEST_CASE("EulerAnglesZYX cast to different scalar type") {
    EulerAnglesZYXd euler_d(0.1, 0.2, 0.3);
    auto euler_f = euler_d.cast<float>();

    CHECK(approx_equal(static_cast<double>(euler_f.yaw()), euler_d.yaw(), 1e-5));
    CHECK(approx_equal(static_cast<double>(euler_f.pitch()), euler_d.pitch(), 1e-5));
    CHECK(approx_equal(static_cast<double>(euler_f.roll()), euler_d.roll(), 1e-5));
}

TEST_CASE("EulerAnglesXYZ cast to different scalar type") {
    EulerAnglesXYZd euler_d(0.1, 0.2, 0.3);
    auto euler_f = euler_d.cast<float>();

    CHECK(approx_equal(static_cast<double>(euler_f.roll()), euler_d.roll(), 1e-5));
    CHECK(approx_equal(static_cast<double>(euler_f.pitch()), euler_d.pitch(), 1e-5));
    CHECK(approx_equal(static_cast<double>(euler_f.yaw()), euler_d.yaw(), 1e-5));
}

// ===== TYPE ALIASES =====

TEST_CASE("Type aliases work correctly") {
    // ZYX aliases
    EulerAnglesZYXf zyx_f(0.1f, 0.2f, 0.3f);
    EulerAnglesZYXd zyx_d(0.1, 0.2, 0.3);
    EulerAnglesYPRd ypr_d(0.1, 0.2, 0.3);

    CHECK(approx_equal(zyx_d.yaw(), ypr_d.yaw(), 1e-10));

    // XYZ aliases
    EulerAnglesXYZf xyz_f(0.1f, 0.2f, 0.3f);
    EulerAnglesXYZd xyz_d(0.1, 0.2, 0.3);
    EulerAnglesRPYd rpy_d(0.1, 0.2, 0.3);

    CHECK(approx_equal(xyz_d.roll(), rpy_d.roll(), 1e-10));
}
