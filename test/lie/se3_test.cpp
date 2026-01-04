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
bool vec_approx_equal(const dp::mat::Vector<T, N> &a, const dp::mat::Vector<T, N> &b, T tol = T(1e-10)) {
    for (std::size_t i = 0; i < N; ++i) {
        if (std::abs(a[i] - b[i]) >= tol)
            return false;
    }
    return true;
}

// ===== CONSTRUCTION TESTS =====

TEST_CASE("SE3 default construction is identity") {
    SE3d T;

    CHECK(T.is_identity());
    CHECK(approx_equal(T.x(), 0.0));
    CHECK(approx_equal(T.y(), 0.0));
    CHECK(approx_equal(T.z(), 0.0));
    CHECK(T.so3().is_identity());
}

TEST_CASE("SE3 construction from SO3 and translation") {
    SO3d R = SO3d::rot_x(0.5);
    dp::mat::Vector<double, 3> t{1.0, 2.0, 3.0};

    SE3d T(R, t);

    CHECK(T.so3().is_approx(R, 1e-10));
    CHECK(approx_equal(T.x(), 1.0));
    CHECK(approx_equal(T.y(), 2.0));
    CHECK(approx_equal(T.z(), 3.0));
}

TEST_CASE("SE3 construction from homogeneous matrix") {
    SE3d T_orig = SE3d::rot_z(0.5) * SE3d::trans(1.0, 2.0, 3.0);
    auto M = T_orig.matrix();

    SE3d T_from_mat(M);
    CHECK(T_from_mat.is_approx(T_orig, 1e-9));
}

// ===== FACTORY METHODS =====

TEST_CASE("SE3 factory methods") {
    SUBCASE("identity") {
        auto T = SE3d::identity();
        CHECK(T.is_identity());
    }

    SUBCASE("pure rotation") {
        auto T = SE3d::rot_x(0.5);
        CHECK(approx_equal(T.x(), 0.0));
        CHECK(approx_equal(T.y(), 0.0));
        CHECK(approx_equal(T.z(), 0.0));
    }

    SUBCASE("pure translation") {
        auto T = SE3d::trans(1.0, 2.0, 3.0);
        CHECK(T.so3().is_identity());
        CHECK(approx_equal(T.x(), 1.0));
        CHECK(approx_equal(T.y(), 2.0));
        CHECK(approx_equal(T.z(), 3.0));
    }

    SUBCASE("trans_x, trans_y, trans_z") {
        auto Tx = SE3d::trans_x(1.0);
        CHECK(approx_equal(Tx.x(), 1.0));
        CHECK(approx_equal(Tx.y(), 0.0));
        CHECK(approx_equal(Tx.z(), 0.0));

        auto Ty = SE3d::trans_y(2.0);
        CHECK(approx_equal(Ty.x(), 0.0));
        CHECK(approx_equal(Ty.y(), 2.0));
        CHECK(approx_equal(Ty.z(), 0.0));

        auto Tz = SE3d::trans_z(3.0);
        CHECK(approx_equal(Tz.x(), 0.0));
        CHECK(approx_equal(Tz.y(), 0.0));
        CHECK(approx_equal(Tz.z(), 3.0));
    }
}

// ===== EXP AND LOG TESTS =====

TEST_CASE("SE3 exp and log are inverses") {
    SUBCASE("exp(log(T)) = T") {
        std::mt19937 rng(42);
        for (int i = 0; i < 50; ++i) {
            auto T = SE3d::sample_uniform(rng, 10.0);
            auto twist = T.log();
            auto T2 = SE3d::exp(twist);
            CHECK(T.is_approx(T2, 1e-8));
        }
    }

    SUBCASE("log(exp(twist)) = twist for small twists") {
        dp::mat::Vector<double, 6> twist{0.1, 0.2, 0.3, 0.05, 0.1, 0.15};
        auto T = SE3d::exp(twist);
        auto twist2 = T.log();
        CHECK(vec_approx_equal(twist, twist2, 1e-9));
    }
}

TEST_CASE("SE3 exp of specific values") {
    SUBCASE("exp(0) = identity") {
        dp::mat::Vector<double, 6> twist{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        auto T = SE3d::exp(twist);
        CHECK(T.is_identity(1e-10));
    }

    SUBCASE("exp with zero rotation = pure translation") {
        dp::mat::Vector<double, 6> twist{1.0, 2.0, 3.0, 0.0, 0.0, 0.0};
        auto T = SE3d::exp(twist);
        CHECK(T.so3().is_identity(1e-10));
        CHECK(approx_equal(T.x(), 1.0, 1e-10));
        CHECK(approx_equal(T.y(), 2.0, 1e-10));
        CHECK(approx_equal(T.z(), 3.0, 1e-10));
    }

    SUBCASE("exp with zero translation = pure rotation") {
        dp::mat::Vector<double, 6> twist{0.0, 0.0, 0.0, 0.0, 0.0, 0.5};
        auto T = SE3d::exp(twist);
        CHECK(approx_equal(T.x(), 0.0, 1e-10));
        CHECK(approx_equal(T.y(), 0.0, 1e-10));
        CHECK(approx_equal(T.z(), 0.0, 1e-10));

        // Should be rotation around z-axis by 0.5 rad
        auto R = SO3d::rot_z(0.5);
        CHECK(T.so3().is_approx(R, 1e-10));
    }
}

// ===== INVERSE TESTS =====

TEST_CASE("SE3 inverse") {
    SUBCASE("T * T^-1 = identity") {
        std::mt19937 rng(42);
        for (int i = 0; i < 50; ++i) {
            auto T = SE3d::sample_uniform(rng, 10.0);
            auto T_inv = T.inverse();
            auto I = T * T_inv;
            CHECK(I.is_identity(1e-9));
        }
    }

    SUBCASE("T^-1 * T = identity") {
        std::mt19937 rng(42);
        for (int i = 0; i < 50; ++i) {
            auto T = SE3d::sample_uniform(rng, 10.0);
            auto T_inv = T.inverse();
            auto I = T_inv * T;
            CHECK(I.is_identity(1e-9));
        }
    }

    SUBCASE("(T^-1)^-1 = T") {
        SE3d T = SE3d::rot_x(0.5) * SE3d::trans(1.0, 2.0, 3.0);
        auto T_inv_inv = T.inverse().inverse();
        CHECK(T.is_approx(T_inv_inv, 1e-10));
    }
}

// ===== COMPOSITION TESTS =====

TEST_CASE("SE3 composition") {
    SUBCASE("identity * T = T") {
        SE3d T = SE3d::rot_y(0.5) * SE3d::trans(1.0, 2.0, 3.0);
        auto result = SE3d::identity() * T;
        CHECK(result.is_approx(T, 1e-10));
    }

    SUBCASE("T * identity = T") {
        SE3d T = SE3d::rot_y(0.5) * SE3d::trans(1.0, 2.0, 3.0);
        auto result = T * SE3d::identity();
        CHECK(result.is_approx(T, 1e-10));
    }

    SUBCASE("associativity") {
        auto A = SE3d::rot_x(0.3) * SE3d::trans(1.0, 0.0, 0.0);
        auto B = SE3d::rot_y(0.5) * SE3d::trans(0.0, 1.0, 0.0);
        auto C = SE3d::rot_z(0.7) * SE3d::trans(0.0, 0.0, 1.0);

        auto AB_C = (A * B) * C;
        auto A_BC = A * (B * C);
        CHECK(AB_C.is_approx(A_BC, 1e-10));
    }
}

// ===== POINT TRANSFORMATION TESTS =====

TEST_CASE("SE3 transforms points correctly") {
    SUBCASE("identity doesn't change point") {
        dp::mat::Vector<double, 3> p{1.0, 2.0, 3.0};
        auto p2 = SE3d::identity() * p;
        CHECK(vec_approx_equal(p, p2, 1e-10));
    }

    SUBCASE("pure rotation rotates point") {
        dp::mat::Vector<double, 3> p{1.0, 0.0, 0.0};
        auto T = SE3d::rot_z(std::numbers::pi / 2);
        auto p2 = T * p;

        CHECK(approx_equal(p2[0], 0.0, 1e-10));
        CHECK(approx_equal(p2[1], 1.0, 1e-10));
        CHECK(approx_equal(p2[2], 0.0, 1e-10));
    }

    SUBCASE("pure translation translates point") {
        dp::mat::Vector<double, 3> p{1.0, 2.0, 3.0};
        auto T = SE3d::trans(4.0, 5.0, 6.0);
        auto p2 = T * p;

        CHECK(approx_equal(p2[0], 5.0, 1e-10));
        CHECK(approx_equal(p2[1], 7.0, 1e-10));
        CHECK(approx_equal(p2[2], 9.0, 1e-10));
    }

    SUBCASE("transform preserves distances between points") {
        std::mt19937 rng(42);
        auto T = SE3d::sample_uniform(rng, 10.0);

        dp::mat::Vector<double, 3> p1{1.0, 2.0, 3.0};
        dp::mat::Vector<double, 3> p2{4.0, 5.0, 6.0};

        auto p1_t = T * p1;
        auto p2_t = T * p2;

        auto diff_before = dp::mat::Vector<double, 3>{p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
        auto diff_after = dp::mat::Vector<double, 3>{p2_t[0] - p1_t[0], p2_t[1] - p1_t[1], p2_t[2] - p1_t[2]};

        const double dist_before = std::sqrt(diff_before[0] * diff_before[0] + diff_before[1] * diff_before[1] +
                                             diff_before[2] * diff_before[2]);
        const double dist_after =
            std::sqrt(diff_after[0] * diff_after[0] + diff_after[1] * diff_after[1] + diff_after[2] * diff_after[2]);

        CHECK(approx_equal(dist_before, dist_after, 1e-10));
    }
}

// ===== MATRIX REPRESENTATION TESTS =====

TEST_CASE("SE3 matrix representations") {
    SE3d T = SE3d::rot_x(0.5) * SE3d::trans(1.0, 2.0, 3.0);

    SUBCASE("4x4 homogeneous matrix structure") {
        auto M = T.matrix();

        // Bottom row should be [0, 0, 0, 1]
        CHECK(approx_equal(M(3, 0), 0.0, 1e-10));
        CHECK(approx_equal(M(3, 1), 0.0, 1e-10));
        CHECK(approx_equal(M(3, 2), 0.0, 1e-10));
        CHECK(approx_equal(M(3, 3), 1.0, 1e-10));

        // Last column should be translation + 1
        CHECK(approx_equal(M(0, 3), T.x(), 1e-10));
        CHECK(approx_equal(M(1, 3), T.y(), 1e-10));
        CHECK(approx_equal(M(2, 3), T.z(), 1e-10));
    }

    SUBCASE("3x4 compact matrix") {
        auto M = T.matrix3x4();

        CHECK(approx_equal(M(0, 3), T.x(), 1e-10));
        CHECK(approx_equal(M(1, 3), T.y(), 1e-10));
        CHECK(approx_equal(M(2, 3), T.z(), 1e-10));
    }

    SUBCASE("matrix transformation equals direct transformation") {
        auto M = T.matrix();
        dp::mat::Vector<double, 3> p{1.0, 2.0, 3.0};

        auto p_direct = T * p;

        // Matrix multiply: p_h = [p, 1], result = M * p_h
        double p_mat_x = M(0, 0) * p[0] + M(0, 1) * p[1] + M(0, 2) * p[2] + M(0, 3);
        double p_mat_y = M(1, 0) * p[0] + M(1, 1) * p[1] + M(1, 2) * p[2] + M(1, 3);
        double p_mat_z = M(2, 0) * p[0] + M(2, 1) * p[1] + M(2, 2) * p[2] + M(2, 3);

        CHECK(approx_equal(p_direct[0], p_mat_x, 1e-10));
        CHECK(approx_equal(p_direct[1], p_mat_y, 1e-10));
        CHECK(approx_equal(p_direct[2], p_mat_z, 1e-10));
    }
}

// ===== LIE ALGEBRA TESTS =====

TEST_CASE("SE3 hat and vee") {
    SUBCASE("vee(hat(twist)) = twist") {
        dp::mat::Vector<double, 6> twist{1.0, 2.0, 3.0, 0.1, 0.2, 0.3};
        auto Omega = SE3d::hat(twist);
        auto twist2 = SE3d::vee(Omega);
        CHECK(vec_approx_equal(twist, twist2, 1e-10));
    }

    SUBCASE("hat produces correct structure") {
        dp::mat::Vector<double, 6> twist{1.0, 2.0, 3.0, 0.1, 0.2, 0.3};
        auto Omega = SE3d::hat(twist);

        // Bottom row is zeros
        CHECK(approx_equal(Omega(3, 0), 0.0, 1e-10));
        CHECK(approx_equal(Omega(3, 1), 0.0, 1e-10));
        CHECK(approx_equal(Omega(3, 2), 0.0, 1e-10));
        CHECK(approx_equal(Omega(3, 3), 0.0, 1e-10));

        // Right column is v
        CHECK(approx_equal(Omega(0, 3), twist[0], 1e-10));
        CHECK(approx_equal(Omega(1, 3), twist[1], 1e-10));
        CHECK(approx_equal(Omega(2, 3), twist[2], 1e-10));
    }
}

TEST_CASE("SE3 Adjoint") {
    SE3d T = SE3d::rot_z(0.5) * SE3d::trans(1.0, 2.0, 3.0);
    auto Adj = T.Adj();

    // Adj is 6x6
    // Top-left 3x3 should be rotation matrix
    auto R = T.rotation_matrix();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            CHECK(approx_equal(Adj(i, j), R(i, j), 1e-10));
        }
    }

    // Bottom-left 3x3 should be zeros
    for (int i = 3; i < 6; ++i) {
        for (int j = 0; j < 3; ++j) {
            CHECK(approx_equal(Adj(i, j), 0.0, 1e-10));
        }
    }

    // Bottom-right 3x3 should be rotation matrix
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            CHECK(approx_equal(Adj(i + 3, j + 3), R(i, j), 1e-10));
        }
    }
}

// ===== JACOBIAN TESTS =====

TEST_CASE("SE3 left Jacobian") {
    SUBCASE("identity for zero twist") {
        dp::mat::Vector<double, 6> twist{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        auto J = SE3d::left_jacobian(twist);

        // Should be approximately identity
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                if (i == j) {
                    CHECK(approx_equal(J(i, j), 1.0, 1e-10));
                } else {
                    CHECK(approx_equal(J(i, j), 0.0, 1e-10));
                }
            }
        }
    }
}

// ===== INTERPOLATION TESTS =====

TEST_CASE("SE3 interpolation") {
    SUBCASE("interpolate(a, b, 0) = a") {
        auto a = SE3d::rot_x(0.2) * SE3d::trans(1.0, 0.0, 0.0);
        auto b = SE3d::rot_x(0.8) * SE3d::trans(2.0, 1.0, 0.5);
        auto result = interpolate(a, b, 0.0);
        CHECK(result.is_approx(a, 1e-10));
    }

    SUBCASE("interpolate(a, b, 1) = b") {
        auto a = SE3d::rot_x(0.2) * SE3d::trans(1.0, 0.0, 0.0);
        auto b = SE3d::rot_x(0.8) * SE3d::trans(2.0, 1.0, 0.5);
        auto result = interpolate(a, b, 1.0);
        CHECK(result.is_approx(b, 1e-10));
    }

    SUBCASE("interpolate is smooth") {
        auto a = SE3d::identity();
        auto b = SE3d::trans(2.0, 0.0, 0.0);

        auto r1 = interpolate(a, b, 0.25);
        auto r2 = interpolate(a, b, 0.5);
        auto r3 = interpolate(a, b, 0.75);

        // Translations should increase monotonically
        CHECK(r1.x() < r2.x());
        CHECK(r2.x() < r3.x());
    }
}

// ===== ACCESSORS =====

TEST_CASE("SE3 accessors") {
    // Use direct construction instead of composition
    SE3d T(SO3d::rot_z(0.5), dp::mat::Vector<double, 3>{1.0, 2.0, 3.0});

    CHECK(approx_equal(T.x(), 1.0));
    CHECK(approx_equal(T.y(), 2.0));
    CHECK(approx_equal(T.z(), 3.0));

    auto params = T.params();
    CHECK(params.size() == 7);
}

// ===== TYPE CONVERSION =====

TEST_CASE("SE3 cast to different scalar type") {
    // Use direct construction
    SE3d Td(SO3d::rot_x(0.5), dp::mat::Vector<double, 3>{1.0, 2.0, 3.0});
    SE3f Tf = Td.cast<float>();

    CHECK(approx_equal(static_cast<double>(Tf.x()), 1.0, 1e-5));
    CHECK(approx_equal(static_cast<double>(Tf.y()), 2.0, 1e-5));
    CHECK(approx_equal(static_cast<double>(Tf.z()), 3.0, 1e-5));
}

// ===== RANDOM SAMPLING =====

TEST_CASE("SE3 sample_uniform produces valid transforms") {
    std::mt19937 rng(42);

    for (int i = 0; i < 50; ++i) {
        auto T = SE3d::sample_uniform(rng, 10.0);

        // Check rotation is valid
        const auto &q = T.unit_quaternion();
        const double norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
        CHECK(approx_equal(norm, 1.0, 1e-10));

        // Translation should be bounded
        CHECK(std::abs(T.x()) <= 10.0);
        CHECK(std::abs(T.y()) <= 10.0);
        CHECK(std::abs(T.z()) <= 10.0);
    }
}

// ===== EDGE CASES =====

TEST_CASE("SE3 edge cases") {
    SUBCASE("very small rotation") {
        dp::mat::Vector<double, 6> twist{1.0, 2.0, 3.0, 1e-15, 1e-15, 1e-15};
        auto T = SE3d::exp(twist);
        auto twist2 = T.log();

        // Translation part should be preserved
        CHECK(approx_equal(twist2[0], twist[0], 1e-10));
        CHECK(approx_equal(twist2[1], twist[1], 1e-10));
        CHECK(approx_equal(twist2[2], twist[2], 1e-10));
    }
}
