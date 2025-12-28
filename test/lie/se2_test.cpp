#include <doctest/doctest.h>
#include <optinum/lie/lie.hpp>

#include <cmath>
#include <numbers>
#include <random>
#include <vector>

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

// ===== CONSTRUCTION TESTS =====

TEST_CASE("SE2 default construction is identity") {
    SE2d T;

    CHECK(T.is_identity());
    CHECK(approx_equal(T.angle(), 0.0));
    CHECK(approx_equal(T.x(), 0.0));
    CHECK(approx_equal(T.y(), 0.0));
}

TEST_CASE("SE2 construction from angle and translation") {
    SUBCASE("pure rotation") {
        SE2d T(0.5, 0.0, 0.0);
        CHECK(approx_equal(T.angle(), 0.5, 1e-10));
        CHECK(approx_equal(T.x(), 0.0));
        CHECK(approx_equal(T.y(), 0.0));
    }

    SUBCASE("pure translation") {
        SE2d T(0.0, 1.0, 2.0);
        CHECK(approx_equal(T.angle(), 0.0, 1e-10));
        CHECK(approx_equal(T.x(), 1.0));
        CHECK(approx_equal(T.y(), 2.0));
    }

    SUBCASE("rotation and translation") {
        SE2d T(std::numbers::pi / 4, 1.0, 2.0);
        CHECK(approx_equal(T.angle(), std::numbers::pi / 4, 1e-10));
        CHECK(approx_equal(T.x(), 1.0));
        CHECK(approx_equal(T.y(), 2.0));
    }
}

TEST_CASE("SE2 construction from SO2 and translation") {
    SO2d R(0.5);
    simd::Vector<double, 2> t{1.0, 2.0};

    SE2d T(R, t);
    CHECK(approx_equal(T.angle(), 0.5, 1e-10));
    CHECK(approx_equal(T.x(), 1.0));
    CHECK(approx_equal(T.y(), 2.0));
}

TEST_CASE("SE2 construction from homogeneous matrix") {
    const double theta = 0.5;
    simd::Matrix<double, 3, 3> T_mat;

    T_mat(0, 0) = std::cos(theta);
    T_mat(0, 1) = -std::sin(theta);
    T_mat(0, 2) = 1.0;
    T_mat(1, 0) = std::sin(theta);
    T_mat(1, 1) = std::cos(theta);
    T_mat(1, 2) = 2.0;
    T_mat(2, 0) = 0.0;
    T_mat(2, 1) = 0.0;
    T_mat(2, 2) = 1.0;

    SE2d T(T_mat);
    CHECK(approx_equal(T.angle(), theta, 1e-10));
    CHECK(approx_equal(T.x(), 1.0, 1e-10));
    CHECK(approx_equal(T.y(), 2.0, 1e-10));
}

// ===== FACTORY METHODS =====

TEST_CASE("SE2 factory methods") {
    SUBCASE("identity") {
        auto T = SE2d::identity();
        CHECK(T.is_identity());
    }

    SUBCASE("pure rotation") {
        auto T = SE2d::rot(0.5);
        CHECK(approx_equal(T.angle(), 0.5, 1e-10));
        CHECK(approx_equal(T.x(), 0.0));
        CHECK(approx_equal(T.y(), 0.0));
    }

    SUBCASE("pure translation") {
        auto T = SE2d::trans(1.0, 2.0);
        CHECK(approx_equal(T.angle(), 0.0, 1e-10));
        CHECK(approx_equal(T.x(), 1.0));
        CHECK(approx_equal(T.y(), 2.0));
    }

    SUBCASE("trans_x") {
        auto T = SE2d::trans_x(3.0);
        CHECK(approx_equal(T.x(), 3.0));
        CHECK(approx_equal(T.y(), 0.0));
    }

    SUBCASE("trans_y") {
        auto T = SE2d::trans_y(4.0);
        CHECK(approx_equal(T.x(), 0.0));
        CHECK(approx_equal(T.y(), 4.0));
    }
}

// ===== EXP AND LOG TESTS =====

TEST_CASE("SE2 exp and log are inverses") {
    SUBCASE("exp(log(T)) = T") {
        std::mt19937 rng(42);
        for (int i = 0; i < 100; ++i) {
            auto T = SE2d::sample_uniform(rng, 10.0);
            auto twist = T.log();
            auto T2 = SE2d::exp(twist);
            CHECK(T.is_approx(T2, 1e-9));
        }
    }

    SUBCASE("log(exp(twist)) = twist for small twists") {
        simd::Vector<double, 3> twist{0.5, -0.3, 0.2};
        auto T = SE2d::exp(twist);
        auto twist2 = T.log();
        CHECK(vec_approx_equal(twist, twist2, 1e-10));
    }
}

TEST_CASE("SE2 exp of specific twists") {
    SUBCASE("exp(0) = identity") {
        simd::Vector<double, 3> twist{0.0, 0.0, 0.0};
        auto T = SE2d::exp(twist);
        CHECK(T.is_identity(1e-10));
    }

    SUBCASE("exp with zero rotation = pure translation") {
        simd::Vector<double, 3> twist{1.0, 2.0, 0.0};
        auto T = SE2d::exp(twist);
        CHECK(approx_equal(T.angle(), 0.0, 1e-10));
        CHECK(approx_equal(T.x(), 1.0, 1e-10));
        CHECK(approx_equal(T.y(), 2.0, 1e-10));
    }

    SUBCASE("exp with zero translation = pure rotation") {
        simd::Vector<double, 3> twist{0.0, 0.0, 0.5};
        auto T = SE2d::exp(twist);
        CHECK(approx_equal(T.angle(), 0.5, 1e-10));
        // Translation should be 0 since v = [0, 0]
        CHECK(approx_equal(T.x(), 0.0, 1e-10));
        CHECK(approx_equal(T.y(), 0.0, 1e-10));
    }
}

TEST_CASE("SE2 exp small angle approximation") {
    // Very small angle should use Taylor expansion
    simd::Vector<double, 3> twist{1.0, 2.0, 1e-12};
    auto T = SE2d::exp(twist);

    // For very small theta, translation should be approximately [vx, vy]
    CHECK(approx_equal(T.x(), 1.0, 1e-8));
    CHECK(approx_equal(T.y(), 2.0, 1e-8));
}

// ===== INVERSE TESTS =====

TEST_CASE("SE2 inverse") {
    SUBCASE("T * T^-1 = identity") {
        std::mt19937 rng(42);
        for (int i = 0; i < 100; ++i) {
            auto T = SE2d::sample_uniform(rng, 10.0);
            auto T_inv = T.inverse();
            auto I = T * T_inv;
            CHECK(I.is_identity(1e-9));
        }
    }

    SUBCASE("T^-1 * T = identity") {
        std::mt19937 rng(42);
        for (int i = 0; i < 100; ++i) {
            auto T = SE2d::sample_uniform(rng, 10.0);
            auto T_inv = T.inverse();
            auto I = T_inv * T;
            CHECK(I.is_identity(1e-9));
        }
    }

    SUBCASE("(T^-1)^-1 = T") {
        SE2d T(0.5, 1.0, 2.0);
        auto T_inv_inv = T.inverse().inverse();
        CHECK(T.is_approx(T_inv_inv, 1e-10));
    }

    SUBCASE("inverse of identity is identity") {
        auto I = SE2d::identity();
        CHECK(I.inverse().is_identity());
    }
}

// ===== COMPOSITION TESTS =====

TEST_CASE("SE2 composition") {
    SUBCASE("identity * T = T") {
        SE2d T(0.5, 1.0, 2.0);
        auto result = SE2d::identity() * T;
        CHECK(result.is_approx(T, 1e-10));
    }

    SUBCASE("T * identity = T") {
        SE2d T(0.5, 1.0, 2.0);
        auto result = T * SE2d::identity();
        CHECK(result.is_approx(T, 1e-10));
    }

    SUBCASE("associativity: (A*B)*C = A*(B*C)") {
        SE2d A(0.3, 1.0, 0.5);
        SE2d B(0.5, -0.5, 1.0);
        SE2d C(0.7, 2.0, -1.0);

        auto AB_C = (A * B) * C;
        auto A_BC = A * (B * C);
        CHECK(AB_C.is_approx(A_BC, 1e-10));
    }

    SUBCASE("pure rotations compose correctly") {
        auto R1 = SE2d::rot(0.3);
        auto R2 = SE2d::rot(0.5);
        auto R12 = R1 * R2;
        CHECK(approx_equal(R12.angle(), 0.8, 1e-10));
    }

    SUBCASE("pure translations compose correctly") {
        auto T1 = SE2d::trans(1.0, 2.0);
        auto T2 = SE2d::trans(3.0, 4.0);
        auto T12 = T1 * T2;
        CHECK(approx_equal(T12.x(), 4.0, 1e-10));
        CHECK(approx_equal(T12.y(), 6.0, 1e-10));
    }
}

// ===== POINT TRANSFORMATION TESTS =====

TEST_CASE("SE2 transforms points correctly") {
    SUBCASE("identity doesn't change point") {
        simd::Vector<double, 2> p{1.0, 2.0};
        auto p2 = SE2d::identity() * p;
        CHECK(approx_equal(p2[0], 1.0));
        CHECK(approx_equal(p2[1], 2.0));
    }

    SUBCASE("pure rotation rotates point") {
        simd::Vector<double, 2> p{1.0, 0.0};
        auto T = SE2d::rot(std::numbers::pi / 2);
        auto p2 = T * p;
        CHECK(approx_equal(p2[0], 0.0, 1e-10));
        CHECK(approx_equal(p2[1], 1.0, 1e-10));
    }

    SUBCASE("pure translation translates point") {
        simd::Vector<double, 2> p{1.0, 2.0};
        auto T = SE2d::trans(3.0, 4.0);
        auto p2 = T * p;
        CHECK(approx_equal(p2[0], 4.0, 1e-10));
        CHECK(approx_equal(p2[1], 6.0, 1e-10));
    }

    SUBCASE("rotation then translation") {
        simd::Vector<double, 2> p{1.0, 0.0};
        SE2d T(std::numbers::pi / 2, 0.0, 1.0); // Rotate 90 deg, translate y by 1
        auto p2 = T * p;
        // After rotation: (0, 1), then translation: (0, 2)
        CHECK(approx_equal(p2[0], 0.0, 1e-10));
        CHECK(approx_equal(p2[1], 2.0, 1e-10));
    }
}

// ===== MATRIX REPRESENTATION TESTS =====

TEST_CASE("SE2 matrix representations") {
    SE2d T(0.5, 1.0, 2.0);

    SUBCASE("3x3 homogeneous matrix") {
        auto M = T.matrix();

        // Check structure: [[R, t], [0, 0, 1]]
        const double c = std::cos(0.5);
        const double s = std::sin(0.5);

        CHECK(approx_equal(M(0, 0), c, 1e-10));
        CHECK(approx_equal(M(0, 1), -s, 1e-10));
        CHECK(approx_equal(M(0, 2), 1.0, 1e-10));

        CHECK(approx_equal(M(1, 0), s, 1e-10));
        CHECK(approx_equal(M(1, 1), c, 1e-10));
        CHECK(approx_equal(M(1, 2), 2.0, 1e-10));

        CHECK(approx_equal(M(2, 0), 0.0, 1e-10));
        CHECK(approx_equal(M(2, 1), 0.0, 1e-10));
        CHECK(approx_equal(M(2, 2), 1.0, 1e-10));
    }

    SUBCASE("2x3 compact matrix") {
        auto M = T.matrix2x3();

        const double c = std::cos(0.5);
        const double s = std::sin(0.5);

        CHECK(approx_equal(M(0, 0), c, 1e-10));
        CHECK(approx_equal(M(0, 1), -s, 1e-10));
        CHECK(approx_equal(M(0, 2), 1.0, 1e-10));

        CHECK(approx_equal(M(1, 0), s, 1e-10));
        CHECK(approx_equal(M(1, 1), c, 1e-10));
        CHECK(approx_equal(M(1, 2), 2.0, 1e-10));
    }
}

// ===== LIE ALGEBRA TESTS =====

TEST_CASE("SE2 hat and vee") {
    SUBCASE("vee(hat(twist)) = twist") {
        simd::Vector<double, 3> twist{1.0, 2.0, 0.5};
        auto Omega = SE2d::hat(twist);
        auto twist2 = SE2d::vee(Omega);
        CHECK(vec_approx_equal(twist, twist2, 1e-10));
    }

    SUBCASE("hat produces correct structure") {
        simd::Vector<double, 3> twist{1.0, 2.0, 0.5};
        auto Omega = SE2d::hat(twist);

        // [[0, -theta, vx], [theta, 0, vy], [0, 0, 0]]
        CHECK(approx_equal(Omega(0, 0), 0.0, 1e-10));
        CHECK(approx_equal(Omega(0, 1), -0.5, 1e-10));
        CHECK(approx_equal(Omega(0, 2), 1.0, 1e-10));

        CHECK(approx_equal(Omega(1, 0), 0.5, 1e-10));
        CHECK(approx_equal(Omega(1, 1), 0.0, 1e-10));
        CHECK(approx_equal(Omega(1, 2), 2.0, 1e-10));

        CHECK(approx_equal(Omega(2, 0), 0.0, 1e-10));
        CHECK(approx_equal(Omega(2, 1), 0.0, 1e-10));
        CHECK(approx_equal(Omega(2, 2), 0.0, 1e-10));
    }
}

TEST_CASE("SE2 Adjoint") {
    SE2d T(0.5, 1.0, 2.0);
    auto Adj = T.Adj();

    // Check it's 3x3
    const double c = std::cos(0.5);
    const double s = std::sin(0.5);

    // Top-left 2x2 is rotation
    CHECK(approx_equal(Adj(0, 0), c, 1e-10));
    CHECK(approx_equal(Adj(0, 1), -s, 1e-10));
    CHECK(approx_equal(Adj(1, 0), s, 1e-10));
    CHECK(approx_equal(Adj(1, 1), c, 1e-10));

    // Right column: [-ty, tx]
    CHECK(approx_equal(Adj(0, 2), -2.0, 1e-10));
    CHECK(approx_equal(Adj(1, 2), 1.0, 1e-10));

    // Bottom row: [0, 0, 1]
    CHECK(approx_equal(Adj(2, 0), 0.0, 1e-10));
    CHECK(approx_equal(Adj(2, 1), 0.0, 1e-10));
    CHECK(approx_equal(Adj(2, 2), 1.0, 1e-10));
}

TEST_CASE("SE2 Lie bracket") {
    simd::Vector<double, 3> a{1.0, 2.0, 0.3};
    simd::Vector<double, 3> b{3.0, 4.0, 0.5};

    auto c = SE2d::lie_bracket(a, b);

    // [a, b]_0 = a_theta * b_y - b_theta * a_y = 0.3*4.0 - 0.5*2.0 = 1.2 - 1.0 = 0.2
    // [a, b]_1 = -a_theta * b_x + b_theta * a_x = -0.3*3.0 + 0.5*1.0 = -0.9 + 0.5 = -0.4
    // [a, b]_2 = 0

    CHECK(approx_equal(c[0], 0.2, 1e-10));
    CHECK(approx_equal(c[1], -0.4, 1e-10));
    CHECK(approx_equal(c[2], 0.0, 1e-10));
}

// ===== JACOBIAN TESTS =====

TEST_CASE("SE2 left Jacobian") {
    SUBCASE("identity for zero twist") {
        simd::Vector<double, 3> twist{0.0, 0.0, 0.0};
        auto J = SE2d::left_jacobian(twist);

        // Should be approximately identity
        CHECK(approx_equal(J(0, 0), 1.0, 1e-10));
        CHECK(approx_equal(J(1, 1), 1.0, 1e-10));
        CHECK(approx_equal(J(2, 2), 1.0, 1e-10));
    }

    SUBCASE("J * J_inv = I") {
        simd::Vector<double, 3> twist{1.0, 2.0, 0.5};
        auto J = SE2d::left_jacobian(twist);
        auto J_inv = SE2d::left_jacobian_inverse(twist);

        // Compute J * J_inv manually
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
        CHECK(approx_equal(I(0, 1), 0.0, 1e-9));
        CHECK(approx_equal(I(0, 2), 0.0, 1e-9));
        CHECK(approx_equal(I(1, 0), 0.0, 1e-9));
        CHECK(approx_equal(I(1, 1), 1.0, 1e-9));
        CHECK(approx_equal(I(1, 2), 0.0, 1e-9));
        CHECK(approx_equal(I(2, 0), 0.0, 1e-9));
        CHECK(approx_equal(I(2, 1), 0.0, 1e-9));
        CHECK(approx_equal(I(2, 2), 1.0, 1e-9));
    }
}

// ===== INTERPOLATION TESTS =====

TEST_CASE("SE2 interpolation") {
    SUBCASE("interpolate(a, b, 0) = a") {
        SE2d a(0.2, 1.0, 2.0);
        SE2d b(0.8, 3.0, 4.0);
        auto result = interpolate(a, b, 0.0);
        CHECK(result.is_approx(a, 1e-10));
    }

    SUBCASE("interpolate(a, b, 1) = b") {
        SE2d a(0.2, 1.0, 2.0);
        SE2d b(0.8, 3.0, 4.0);
        auto result = interpolate(a, b, 1.0);
        CHECK(result.is_approx(b, 1e-10));
    }

    SUBCASE("interpolate(a, b, 0.5) is midpoint") {
        SE2d a(0.0, 0.0, 0.0);
        SE2d b(0.0, 2.0, 4.0); // Pure translation
        auto mid = interpolate(a, b, 0.5);
        CHECK(approx_equal(mid.x(), 1.0, 1e-10));
        CHECK(approx_equal(mid.y(), 2.0, 1e-10));
    }
}

// ===== ACCESSORS AND MUTATORS =====

TEST_CASE("SE2 accessors") {
    SE2d T(0.5, 1.0, 2.0);

    CHECK(approx_equal(T.angle(), 0.5, 1e-10));
    CHECK(approx_equal(T.x(), 1.0));
    CHECK(approx_equal(T.y(), 2.0));

    auto params = T.params();
    CHECK(approx_equal(params[0], std::cos(0.5), 1e-10));
    CHECK(approx_equal(params[1], std::sin(0.5), 1e-10));
    CHECK(approx_equal(params[2], 1.0));
    CHECK(approx_equal(params[3], 2.0));
}

TEST_CASE("SE2 mutators") {
    SE2d T;

    T.set_angle(0.5);
    CHECK(approx_equal(T.angle(), 0.5, 1e-10));

    T.set_translation(1.0, 2.0);
    CHECK(approx_equal(T.x(), 1.0));
    CHECK(approx_equal(T.y(), 2.0));
}

// ===== TYPE CONVERSION =====

TEST_CASE("SE2 cast to different scalar type") {
    SE2d Td(0.5, 1.0, 2.0);
    SE2f Tf = Td.cast<float>();

    CHECK(approx_equal(static_cast<double>(Tf.angle()), 0.5, 1e-5));
    CHECK(approx_equal(static_cast<double>(Tf.x()), 1.0, 1e-5));
    CHECK(approx_equal(static_cast<double>(Tf.y()), 2.0, 1e-5));
}

// ===== EDGE CASES =====

TEST_CASE("SE2 edge cases") {
    SUBCASE("very small rotations") {
        simd::Vector<double, 3> twist{1.0, 2.0, 1e-15};
        auto T = SE2d::exp(twist);
        auto twist2 = T.log();
        CHECK(vec_approx_equal(twist, twist2, 1e-10));
    }

    SUBCASE("rotation near pi") {
        SE2d T(std::numbers::pi - 1e-10, 1.0, 2.0);
        auto twist = T.log();
        auto T2 = SE2d::exp(twist);
        CHECK(T.is_approx(T2, 1e-8));
    }
}

// ===== RANDOM SAMPLING =====

TEST_CASE("SE2 sample_uniform produces valid transforms") {
    std::mt19937 rng(42);

    for (int i = 0; i < 100; ++i) {
        auto T = SE2d::sample_uniform(rng, 10.0);

        // Check rotation is valid
        const auto &R = T.so2();
        const double norm = std::sqrt(R.real() * R.real() + R.imag() * R.imag());
        CHECK(approx_equal(norm, 1.0, 1e-10));

        // Translation should be bounded
        CHECK(std::abs(T.x()) <= 10.0);
        CHECK(std::abs(T.y()) <= 10.0);
    }
}
