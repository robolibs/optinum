#include <doctest/doctest.h>
#include <optinum/lie/lie.hpp>

#include <cmath>
#include <numbers>
#include <random>

using namespace optinum::lie;
using namespace optinum;

// ===== HELPER FUNCTIONS =====

template <typename T> bool approx_equal(T a, T b, T tol = T(1e-10)) { return std::abs(a - b) < tol; }

template <typename T, std::size_t N>
bool vec_approx_equal(const simd::Vector<T, N> &a, const simd::Vector<T, N> &b, T tol = T(1e-10)) {
    for (std::size_t i = 0; i < N; ++i) {
        if (std::abs(a[i] - b[i]) >= tol)
            return false;
    }
    return true;
}

template <typename T, std::size_t R, std::size_t C>
bool mat_approx_equal(const simd::Matrix<T, R, C> &A, const simd::Matrix<T, R, C> &B, T tol = T(1e-10)) {
    for (std::size_t i = 0; i < R; ++i) {
        for (std::size_t j = 0; j < C; ++j) {
            if (std::abs(A(i, j) - B(i, j)) >= tol)
                return false;
        }
    }
    return true;
}

// ===== CONSTRUCTION TESTS =====

TEST_CASE("SO3 default construction is identity") {
    SO3d R;

    CHECK(R.is_identity());
    CHECK(approx_equal(R.w(), 1.0));
    CHECK(approx_equal(R.x(), 0.0));
    CHECK(approx_equal(R.y(), 0.0));
    CHECK(approx_equal(R.z(), 0.0));
}

TEST_CASE("SO3 construction from quaternion normalizes") {
    // Non-unit quaternion
    SO3d R(2.0, 0.0, 0.0, 0.0);

    // Should be normalized
    const auto &q = R.unit_quaternion();
    const double norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    CHECK(approx_equal(norm, 1.0, 1e-10));
}

TEST_CASE("SO3 axis rotations") {
    SUBCASE("rot_x") {
        auto R = SO3d::rot_x(std::numbers::pi / 2);
        simd::Vector<double, 3> p{0.0, 1.0, 0.0};
        auto p2 = R * p;

        CHECK(approx_equal(p2[0], 0.0, 1e-10));
        CHECK(approx_equal(p2[1], 0.0, 1e-10));
        CHECK(approx_equal(p2[2], 1.0, 1e-10));
    }

    SUBCASE("rot_y") {
        auto R = SO3d::rot_y(std::numbers::pi / 2);
        simd::Vector<double, 3> p{0.0, 0.0, 1.0};
        auto p2 = R * p;

        CHECK(approx_equal(p2[0], 1.0, 1e-10));
        CHECK(approx_equal(p2[1], 0.0, 1e-10));
        CHECK(approx_equal(p2[2], 0.0, 1e-10));
    }

    SUBCASE("rot_z") {
        auto R = SO3d::rot_z(std::numbers::pi / 2);
        simd::Vector<double, 3> p{1.0, 0.0, 0.0};
        auto p2 = R * p;

        CHECK(approx_equal(p2[0], 0.0, 1e-10));
        CHECK(approx_equal(p2[1], 1.0, 1e-10));
        CHECK(approx_equal(p2[2], 0.0, 1e-10));
    }
}

// ===== EXP AND LOG TESTS =====

TEST_CASE("SO3 exp and log are inverses") {
    SUBCASE("exp(log(R)) = R") {
        std::mt19937 rng(42);
        for (int i = 0; i < 100; ++i) {
            auto R = SO3d::sample_uniform(rng);
            auto omega = R.log();
            auto R2 = SO3d::exp(omega);
            CHECK(R.is_approx(R2, 1e-9));
        }
    }

    SUBCASE("log(exp(omega)) = omega for small omega") {
        simd::Vector<double, 3> omega{0.1, 0.2, 0.3};
        auto R = SO3d::exp(omega);
        auto omega2 = R.log();
        CHECK(vec_approx_equal(omega, omega2, 1e-10));
    }
}

TEST_CASE("SO3 exp of specific values") {
    SUBCASE("exp(0) = identity") {
        simd::Vector<double, 3> omega{0.0, 0.0, 0.0};
        auto R = SO3d::exp(omega);
        CHECK(R.is_identity(1e-10));
    }

    SUBCASE("exp around z-axis") {
        const double theta = std::numbers::pi / 4;
        simd::Vector<double, 3> omega{0.0, 0.0, theta};
        auto R = SO3d::exp(omega);

        // Should equal rot_z(theta)
        auto R2 = SO3d::rot_z(theta);
        CHECK(R.is_approx(R2, 1e-10));
    }
}

TEST_CASE("SO3 exp small angle") {
    // Very small angle
    simd::Vector<double, 3> omega{1e-12, 2e-12, 3e-12};
    auto R = SO3d::exp(omega);
    auto omega2 = R.log();
    CHECK(vec_approx_equal(omega, omega2, 1e-10));
}

// ===== INVERSE TESTS =====

TEST_CASE("SO3 inverse") {
    SUBCASE("R * R^-1 = identity") {
        std::mt19937 rng(42);
        for (int i = 0; i < 100; ++i) {
            auto R = SO3d::sample_uniform(rng);
            auto R_inv = R.inverse();
            auto I = R * R_inv;
            CHECK(I.is_identity(1e-9));
        }
    }

    SUBCASE("(R^-1)^-1 = R") {
        auto R = SO3d::rot_x(0.5);
        auto R_inv_inv = R.inverse().inverse();
        CHECK(R.is_approx(R_inv_inv, 1e-10));
    }
}

// ===== COMPOSITION TESTS =====

TEST_CASE("SO3 composition") {
    SUBCASE("identity * R = R") {
        auto R = SO3d::rot_x(0.5);
        auto result = SO3d::identity() * R;
        CHECK(result.is_approx(R, 1e-10));
    }

    SUBCASE("R * identity = R") {
        auto R = SO3d::rot_x(0.5);
        auto result = R * SO3d::identity();
        CHECK(result.is_approx(R, 1e-10));
    }

    SUBCASE("associativity") {
        auto A = SO3d::rot_x(0.3);
        auto B = SO3d::rot_y(0.5);
        auto C = SO3d::rot_z(0.7);

        auto AB_C = (A * B) * C;
        auto A_BC = A * (B * C);
        CHECK(AB_C.is_approx(A_BC, 1e-10));
    }
}

// ===== POINT ROTATION TESTS =====

TEST_CASE("SO3 rotates points correctly") {
    SUBCASE("identity doesn't change point") {
        simd::Vector<double, 3> p{1.0, 2.0, 3.0};
        auto p2 = SO3d::identity() * p;
        CHECK(vec_approx_equal(p, p2, 1e-10));
    }

    SUBCASE("rotation preserves norm") {
        std::mt19937 rng(42);
        for (int i = 0; i < 100; ++i) {
            auto R = SO3d::sample_uniform(rng);
            simd::Vector<double, 3> p{1.0, 2.0, 3.0};
            auto p2 = R * p;

            const double norm_before = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            const double norm_after = std::sqrt(p2[0] * p2[0] + p2[1] * p2[1] + p2[2] * p2[2]);
            CHECK(approx_equal(norm_before, norm_after, 1e-10));
        }
    }

    SUBCASE("180 degree rotation around z") {
        auto R = SO3d::rot_z(std::numbers::pi);
        simd::Vector<double, 3> p{1.0, 0.0, 0.0};
        auto p2 = R * p;

        CHECK(approx_equal(p2[0], -1.0, 1e-10));
        CHECK(approx_equal(p2[1], 0.0, 1e-10));
        CHECK(approx_equal(p2[2], 0.0, 1e-10));
    }
}

// ===== ROTATION MATRIX TESTS =====

TEST_CASE("SO3 rotation matrix") {
    SUBCASE("matrix is orthogonal (R^T * R = I)") {
        std::mt19937 rng(42);
        for (int i = 0; i < 20; ++i) {
            auto R = SO3d::sample_uniform(rng);
            auto M = R.matrix();

            // Compute M^T * M
            simd::Matrix<double, 3, 3> I;
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    double sum = 0;
                    for (int k = 0; k < 3; ++k) {
                        sum += M(k, r) * M(k, c); // M^T * M
                    }
                    I(r, c) = sum;
                }
            }

            CHECK(approx_equal(I(0, 0), 1.0, 1e-10));
            CHECK(approx_equal(I(1, 1), 1.0, 1e-10));
            CHECK(approx_equal(I(2, 2), 1.0, 1e-10));
            CHECK(approx_equal(I(0, 1), 0.0, 1e-10));
            CHECK(approx_equal(I(0, 2), 0.0, 1e-10));
            CHECK(approx_equal(I(1, 2), 0.0, 1e-10));
        }
    }

    SUBCASE("determinant is 1") {
        auto R = SO3d::rot_x(0.5) * SO3d::rot_y(0.3);
        auto M = R.matrix();

        // 3x3 determinant
        const double det = M(0, 0) * (M(1, 1) * M(2, 2) - M(1, 2) * M(2, 1)) -
                           M(0, 1) * (M(1, 0) * M(2, 2) - M(1, 2) * M(2, 0)) +
                           M(0, 2) * (M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0));
        CHECK(approx_equal(det, 1.0, 1e-10));
    }

    SUBCASE("matrix rotation equals quaternion rotation") {
        std::mt19937 rng(42);
        for (int i = 0; i < 20; ++i) {
            auto R = SO3d::sample_uniform(rng);
            auto M = R.matrix();

            simd::Vector<double, 3> p{1.0, 2.0, 3.0};

            // Rotate with quaternion
            auto p_quat = R * p;

            // Rotate with matrix
            simd::Vector<double, 3> p_mat;
            p_mat[0] = M(0, 0) * p[0] + M(0, 1) * p[1] + M(0, 2) * p[2];
            p_mat[1] = M(1, 0) * p[0] + M(1, 1) * p[1] + M(1, 2) * p[2];
            p_mat[2] = M(2, 0) * p[0] + M(2, 1) * p[1] + M(2, 2) * p[2];

            CHECK(vec_approx_equal(p_quat, p_mat, 1e-10));
        }
    }
}

// ===== CONSTRUCTION FROM MATRIX =====

TEST_CASE("SO3 from rotation matrix") {
    SUBCASE("round-trip: SO3 -> matrix -> SO3") {
        std::mt19937 rng(42);
        for (int i = 0; i < 20; ++i) {
            auto R1 = SO3d::sample_uniform(rng);
            auto M = R1.matrix();
            SO3d R2(M);
            CHECK(R1.is_approx(R2, 1e-9));
        }
    }
}

// ===== LIE ALGEBRA TESTS =====

TEST_CASE("SO3 hat and vee") {
    SUBCASE("vee(hat(omega)) = omega") {
        simd::Vector<double, 3> omega{0.1, 0.2, 0.3};
        auto Omega = SO3d::hat(omega);
        auto omega2 = SO3d::vee(Omega);
        CHECK(vec_approx_equal(omega, omega2, 1e-10));
    }

    SUBCASE("hat produces skew-symmetric matrix") {
        simd::Vector<double, 3> omega{1.0, 2.0, 3.0};
        auto Omega = SO3d::hat(omega);

        // Diagonal is zero
        CHECK(approx_equal(Omega(0, 0), 0.0, 1e-10));
        CHECK(approx_equal(Omega(1, 1), 0.0, 1e-10));
        CHECK(approx_equal(Omega(2, 2), 0.0, 1e-10));

        // Skew-symmetric: Omega^T = -Omega
        CHECK(approx_equal(Omega(0, 1), -Omega(1, 0), 1e-10));
        CHECK(approx_equal(Omega(0, 2), -Omega(2, 0), 1e-10));
        CHECK(approx_equal(Omega(1, 2), -Omega(2, 1), 1e-10));
    }
}

TEST_CASE("SO3 Adjoint equals rotation matrix") {
    auto R = SO3d::rot_x(0.5) * SO3d::rot_y(0.3);
    auto Adj = R.Adj();
    auto M = R.matrix();
    CHECK(mat_approx_equal(Adj, M, 1e-10));
}

TEST_CASE("SO3 Lie bracket is cross product") {
    simd::Vector<double, 3> a{1.0, 0.0, 0.0};
    simd::Vector<double, 3> b{0.0, 1.0, 0.0};

    auto c = SO3d::lie_bracket(a, b);

    // a x b = [0, 0, 1]
    CHECK(approx_equal(c[0], 0.0, 1e-10));
    CHECK(approx_equal(c[1], 0.0, 1e-10));
    CHECK(approx_equal(c[2], 1.0, 1e-10));
}

// ===== JACOBIAN TESTS =====

TEST_CASE("SO3 left Jacobian") {
    SUBCASE("identity for zero omega") {
        simd::Vector<double, 3> omega{0.0, 0.0, 0.0};
        auto J = SO3d::left_jacobian(omega);

        CHECK(approx_equal(J(0, 0), 1.0, 1e-10));
        CHECK(approx_equal(J(1, 1), 1.0, 1e-10));
        CHECK(approx_equal(J(2, 2), 1.0, 1e-10));
    }

    SUBCASE("J * J_inv = I") {
        simd::Vector<double, 3> omega{0.3, 0.5, 0.7};
        auto J = SO3d::left_jacobian(omega);
        auto J_inv = SO3d::left_jacobian_inverse(omega);

        // Compute J * J_inv
        simd::Matrix<double, 3, 3> I;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double sum = 0;
                for (int k = 0; k < 3; ++k) {
                    sum += J(i, k) * J_inv(k, j);
                }
                I(i, j) = sum;
            }
        }

        CHECK(approx_equal(I(0, 0), 1.0, 1e-9));
        CHECK(approx_equal(I(1, 1), 1.0, 1e-9));
        CHECK(approx_equal(I(2, 2), 1.0, 1e-9));
        CHECK(approx_equal(I(0, 1), 0.0, 1e-9));
        CHECK(approx_equal(I(0, 2), 0.0, 1e-9));
        CHECK(approx_equal(I(1, 0), 0.0, 1e-9));
        CHECK(approx_equal(I(1, 2), 0.0, 1e-9));
        CHECK(approx_equal(I(2, 0), 0.0, 1e-9));
        CHECK(approx_equal(I(2, 1), 0.0, 1e-9));
    }
}

// ===== INTERPOLATION TESTS =====

TEST_CASE("SO3 interpolation") {
    SUBCASE("interpolate(a, b, 0) = a") {
        auto a = SO3d::rot_x(0.2);
        auto b = SO3d::rot_x(0.8);
        auto result = interpolate(a, b, 0.0);
        CHECK(result.is_approx(a, 1e-10));
    }

    SUBCASE("interpolate(a, b, 1) = b") {
        auto a = SO3d::rot_x(0.2);
        auto b = SO3d::rot_x(0.8);
        auto result = interpolate(a, b, 1.0);
        CHECK(result.is_approx(b, 1e-10));
    }

    SUBCASE("interpolate is monotonic") {
        auto a = SO3d::identity();
        auto b = SO3d::rot_z(1.0);

        auto r1 = interpolate(a, b, 0.25);
        auto r2 = interpolate(a, b, 0.5);
        auto r3 = interpolate(a, b, 0.75);

        // Angles should be monotonically increasing
        CHECK(r1.angle() < r2.angle());
        CHECK(r2.angle() < r3.angle());
    }
}

TEST_CASE("SO3 slerp") {
    SUBCASE("slerp(a, b, 0) = a") {
        auto a = SO3d::rot_y(0.3);
        auto b = SO3d::rot_y(0.9);
        auto result = slerp(a, b, 0.0);
        CHECK(result.is_approx(a, 1e-10));
    }

    SUBCASE("slerp(a, b, 1) = b") {
        auto a = SO3d::rot_y(0.3);
        auto b = SO3d::rot_y(0.9);
        auto result = slerp(a, b, 1.0);
        CHECK(result.is_approx(b, 1e-10));
    }
}

// ===== EULER ANGLES =====

TEST_CASE("SO3 Euler angles") {
    SUBCASE("round-trip for small angles") {
        const double roll = 0.1, pitch = 0.2, yaw = 0.3;
        auto q = datapod::mat::quaternion<double>::from_euler(roll, pitch, yaw);
        SO3d R(q);

        auto euler = R.to_euler();
        CHECK(approx_equal(euler[0], roll, 1e-10));
        CHECK(approx_equal(euler[1], pitch, 1e-10));
        CHECK(approx_equal(euler[2], yaw, 1e-10));
    }
}

// ===== ACCESSORS =====

TEST_CASE("SO3 accessors") {
    auto R = SO3d::rot_z(std::numbers::pi / 4);

    CHECK(R.angle() > 0);
    CHECK(R.angle() < std::numbers::pi);

    auto axis = R.axis();
    // Should be approximately [0, 0, 1]
    CHECK(approx_equal(axis[0], 0.0, 1e-10));
    CHECK(approx_equal(axis[1], 0.0, 1e-10));
    CHECK(approx_equal(axis[2], 1.0, 1e-10));
}

// ===== TYPE CONVERSION =====

TEST_CASE("SO3 cast to different scalar type") {
    SO3d Rd = SO3d::rot_x(0.5);
    SO3f Rf = Rd.cast<float>();

    CHECK(Rd.is_approx(Rf.cast<double>(), 1e-5));
}

// ===== RANDOM SAMPLING =====

TEST_CASE("SO3 sample_uniform produces valid rotations") {
    std::mt19937 rng(42);

    for (int i = 0; i < 100; ++i) {
        auto R = SO3d::sample_uniform(rng);

        // Check quaternion is unit
        const auto &q = R.unit_quaternion();
        const double norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
        CHECK(approx_equal(norm, 1.0, 1e-10));

        // Check rotation matrix is orthogonal
        auto M = R.matrix();
        const double det = M(0, 0) * (M(1, 1) * M(2, 2) - M(1, 2) * M(2, 1)) -
                           M(0, 1) * (M(1, 0) * M(2, 2) - M(1, 2) * M(2, 0)) +
                           M(0, 2) * (M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0));
        CHECK(approx_equal(det, 1.0, 1e-10));
    }
}

// ===== FROM TWO VECTORS =====

TEST_CASE("SO3 from_two_vectors") {
    SUBCASE("basic rotation from x to y axis") {
        simd::Vector<double, 3> v1{1.0, 0.0, 0.0};
        simd::Vector<double, 3> v2{0.0, 1.0, 0.0};

        auto R = SO3d::from_two_vectors(v1, v2);
        auto rotated = R * v1;

        CHECK(approx_equal(rotated[0], 0.0, 1e-10));
        CHECK(approx_equal(rotated[1], 1.0, 1e-10));
        CHECK(approx_equal(rotated[2], 0.0, 1e-10));
    }

    SUBCASE("basic rotation from y to z axis") {
        simd::Vector<double, 3> v1{0.0, 1.0, 0.0};
        simd::Vector<double, 3> v2{0.0, 0.0, 1.0};

        auto R = SO3d::from_two_vectors(v1, v2);
        auto rotated = R * v1;

        CHECK(approx_equal(rotated[0], 0.0, 1e-10));
        CHECK(approx_equal(rotated[1], 0.0, 1e-10));
        CHECK(approx_equal(rotated[2], 1.0, 1e-10));
    }

    SUBCASE("basic rotation from z to x axis") {
        simd::Vector<double, 3> v1{0.0, 0.0, 1.0};
        simd::Vector<double, 3> v2{1.0, 0.0, 0.0};

        auto R = SO3d::from_two_vectors(v1, v2);
        auto rotated = R * v1;

        CHECK(approx_equal(rotated[0], 1.0, 1e-10));
        CHECK(approx_equal(rotated[1], 0.0, 1e-10));
        CHECK(approx_equal(rotated[2], 0.0, 1e-10));
    }

    SUBCASE("parallel vectors (same direction) returns identity") {
        simd::Vector<double, 3> v1{1.0, 2.0, 3.0};
        simd::Vector<double, 3> v2{2.0, 4.0, 6.0}; // Same direction, different magnitude

        auto R = SO3d::from_two_vectors(v1, v2);
        CHECK(R.is_identity(1e-10));
    }

    SUBCASE("anti-parallel vectors (opposite direction) - x axis") {
        simd::Vector<double, 3> v1{1.0, 0.0, 0.0};
        simd::Vector<double, 3> v2{-1.0, 0.0, 0.0};

        auto R = SO3d::from_two_vectors(v1, v2);
        auto rotated = R * v1;

        // Should rotate to opposite direction
        CHECK(approx_equal(rotated[0], -1.0, 1e-10));
        CHECK(approx_equal(rotated[1], 0.0, 1e-10));
        CHECK(approx_equal(rotated[2], 0.0, 1e-10));

        // Rotation angle should be pi
        CHECK(approx_equal(R.angle(), std::numbers::pi, 1e-10));
    }

    SUBCASE("anti-parallel vectors (opposite direction) - y axis") {
        simd::Vector<double, 3> v1{0.0, 1.0, 0.0};
        simd::Vector<double, 3> v2{0.0, -1.0, 0.0};

        auto R = SO3d::from_two_vectors(v1, v2);
        auto rotated = R * v1;

        CHECK(approx_equal(rotated[0], 0.0, 1e-10));
        CHECK(approx_equal(rotated[1], -1.0, 1e-10));
        CHECK(approx_equal(rotated[2], 0.0, 1e-10));
    }

    SUBCASE("anti-parallel vectors (opposite direction) - z axis") {
        simd::Vector<double, 3> v1{0.0, 0.0, 1.0};
        simd::Vector<double, 3> v2{0.0, 0.0, -1.0};

        auto R = SO3d::from_two_vectors(v1, v2);
        auto rotated = R * v1;

        CHECK(approx_equal(rotated[0], 0.0, 1e-10));
        CHECK(approx_equal(rotated[1], 0.0, 1e-10));
        CHECK(approx_equal(rotated[2], -1.0, 1e-10));
    }

    SUBCASE("anti-parallel vectors (opposite direction) - arbitrary") {
        simd::Vector<double, 3> v1{1.0, 1.0, 1.0};
        simd::Vector<double, 3> v2{-1.0, -1.0, -1.0};

        auto R = SO3d::from_two_vectors(v1, v2);
        auto rotated = R * v1;

        // Normalize v1 for comparison
        const double norm = std::sqrt(3.0);
        CHECK(approx_equal(rotated[0], -1.0 / norm * norm, 1e-10));
        CHECK(approx_equal(rotated[1], -1.0 / norm * norm, 1e-10));
        CHECK(approx_equal(rotated[2], -1.0 / norm * norm, 1e-10));
    }

    SUBCASE("arbitrary vectors") {
        simd::Vector<double, 3> v1{1.0, 2.0, 3.0};
        simd::Vector<double, 3> v2{-2.0, 1.0, 0.5};

        auto R = SO3d::from_two_vectors(v1, v2);
        auto rotated = R * v1;

        // Normalize both for comparison
        const double norm1 = std::sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
        const double norm2 = std::sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);

        // Rotated v1 should be parallel to v2 (same direction)
        const double rotated_norm =
            std::sqrt(rotated[0] * rotated[0] + rotated[1] * rotated[1] + rotated[2] * rotated[2]);
        CHECK(approx_equal(rotated[0] / rotated_norm, v2[0] / norm2, 1e-10));
        CHECK(approx_equal(rotated[1] / rotated_norm, v2[1] / norm2, 1e-10));
        CHECK(approx_equal(rotated[2] / rotated_norm, v2[2] / norm2, 1e-10));

        // Magnitude should be preserved
        CHECK(approx_equal(rotated_norm, norm1, 1e-10));
    }

    SUBCASE("non-unit vectors are handled correctly") {
        simd::Vector<double, 3> v1{10.0, 0.0, 0.0};
        simd::Vector<double, 3> v2{0.0, 5.0, 0.0};

        auto R = SO3d::from_two_vectors(v1, v2);
        auto rotated = R * v1;

        // Should rotate to y direction, preserving magnitude
        CHECK(approx_equal(rotated[0], 0.0, 1e-10));
        CHECK(approx_equal(rotated[1], 10.0, 1e-10));
        CHECK(approx_equal(rotated[2], 0.0, 1e-10));
    }

    SUBCASE("result is valid SO3") {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-10.0, 10.0);

        for (int i = 0; i < 100; ++i) {
            simd::Vector<double, 3> v1{dist(rng), dist(rng), dist(rng)};
            simd::Vector<double, 3> v2{dist(rng), dist(rng), dist(rng)};

            auto R = SO3d::from_two_vectors(v1, v2);

            // Check quaternion is unit
            const auto &q = R.unit_quaternion();
            const double norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
            CHECK(approx_equal(norm, 1.0, 1e-10));

            // Check rotation matrix is orthogonal with det = 1
            auto M = R.matrix();
            const double det = M(0, 0) * (M(1, 1) * M(2, 2) - M(1, 2) * M(2, 1)) -
                               M(0, 1) * (M(1, 0) * M(2, 2) - M(1, 2) * M(2, 0)) +
                               M(0, 2) * (M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0));
            CHECK(approx_equal(det, 1.0, 1e-10));
        }
    }

    SUBCASE("rotation maps v1 direction to v2 direction") {
        std::mt19937 rng(123);
        std::uniform_real_distribution<double> dist(-10.0, 10.0);

        for (int i = 0; i < 100; ++i) {
            simd::Vector<double, 3> v1{dist(rng), dist(rng), dist(rng)};
            simd::Vector<double, 3> v2{dist(rng), dist(rng), dist(rng)};

            // Skip if either vector is too small
            const double norm1 = std::sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
            const double norm2 = std::sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);
            if (norm1 < 1e-6 || norm2 < 1e-6)
                continue;

            auto R = SO3d::from_two_vectors(v1, v2);
            auto rotated = R * v1;

            // Normalize for direction comparison
            const double rotated_norm =
                std::sqrt(rotated[0] * rotated[0] + rotated[1] * rotated[1] + rotated[2] * rotated[2]);

            // Dot product of normalized vectors should be 1 (same direction)
            const double dot = (rotated[0] / rotated_norm) * (v2[0] / norm2) +
                               (rotated[1] / rotated_norm) * (v2[1] / norm2) +
                               (rotated[2] / rotated_norm) * (v2[2] / norm2);
            CHECK(approx_equal(dot, 1.0, 1e-9));
        }
    }

    SUBCASE("zero vector returns identity") {
        simd::Vector<double, 3> v1{0.0, 0.0, 0.0};
        simd::Vector<double, 3> v2{1.0, 0.0, 0.0};

        auto R = SO3d::from_two_vectors(v1, v2);
        CHECK(R.is_identity(1e-10));

        R = SO3d::from_two_vectors(v2, v1);
        CHECK(R.is_identity(1e-10));
    }

    SUBCASE("float type works correctly") {
        simd::Vector<float, 3> v1{1.0f, 0.0f, 0.0f};
        simd::Vector<float, 3> v2{0.0f, 1.0f, 0.0f};

        auto R = SO3f::from_two_vectors(v1, v2);
        auto rotated = R * v1;

        CHECK(approx_equal(rotated[0], 0.0f, 1e-5f));
        CHECK(approx_equal(rotated[1], 1.0f, 1e-5f));
        CHECK(approx_equal(rotated[2], 0.0f, 1e-5f));
    }
}

// ===== EDGE CASES =====

TEST_CASE("SO3 edge cases") {
    SUBCASE("rotation near 180 degrees") {
        // Rotation by pi around z-axis
        simd::Vector<double, 3> omega{0.0, 0.0, std::numbers::pi - 1e-10};
        auto R = SO3d::exp(omega);
        auto omega2 = R.log();

        // Should recover the same rotation
        auto R2 = SO3d::exp(omega2);
        CHECK(R.is_approx(R2, 1e-8));
    }

    SUBCASE("very small rotation") {
        simd::Vector<double, 3> omega{1e-15, 1e-15, 1e-15};
        auto R = SO3d::exp(omega);
        CHECK(R.is_identity(1e-10));
    }
}
