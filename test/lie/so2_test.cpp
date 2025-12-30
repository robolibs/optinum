#include <doctest/doctest.h>
#include <optinum/lie/lie.hpp>

#include <cmath>
#include <numbers>
#include <random>
#include <vector>

using namespace optinum::lie;
using namespace optinum;

namespace dp = ::datapod;

// ===== HELPER FUNCTIONS =====

template <typename T> bool approx_equal(T a, T b, T tol = T(1e-10)) { return std::abs(a - b) < tol; }

// ===== CONSTRUCTION TESTS =====

TEST_CASE("SO2 default construction is identity") {
    SO2d R;

    CHECK(approx_equal(R.real(), 1.0));
    CHECK(approx_equal(R.imag(), 0.0));
    CHECK(R.is_identity());
}

TEST_CASE("SO2 construction from angle") {
    SUBCASE("zero angle is identity") {
        SO2d R(0.0);
        CHECK(R.is_identity());
    }

    SUBCASE("90 degrees") {
        SO2d R(std::numbers::pi / 2);
        CHECK(approx_equal(R.real(), 0.0, 1e-10));
        CHECK(approx_equal(R.imag(), 1.0, 1e-10));
    }

    SUBCASE("180 degrees") {
        SO2d R(std::numbers::pi);
        CHECK(approx_equal(R.real(), -1.0, 1e-10));
        CHECK(approx_equal(R.imag(), 0.0, 1e-10));
    }

    SUBCASE("45 degrees") {
        SO2d R(std::numbers::pi / 4);
        const double sqrt2_2 = std::sqrt(2.0) / 2.0;
        CHECK(approx_equal(R.real(), sqrt2_2, 1e-10));
        CHECK(approx_equal(R.imag(), sqrt2_2, 1e-10));
    }
}

TEST_CASE("SO2 construction from complex normalizes") {
    SUBCASE("unit complex stays unit") {
        SO2d R(1.0, 0.0);
        CHECK(approx_equal(R.real(), 1.0));
        CHECK(approx_equal(R.imag(), 0.0));
    }

    SUBCASE("non-unit complex is normalized") {
        SO2d R(3.0, 4.0);                          // norm = 5
        CHECK(approx_equal(R.real(), 0.6, 1e-10)); // 3/5
        CHECK(approx_equal(R.imag(), 0.8, 1e-10)); // 4/5

        // Verify it's actually unit
        const double norm = std::sqrt(R.real() * R.real() + R.imag() * R.imag());
        CHECK(approx_equal(norm, 1.0, 1e-10));
    }
}

TEST_CASE("SO2 construction from rotation matrix") {
    const double theta = 0.5;
    dp::mat::matrix<double, 2, 2> R_mat;
    R_mat(0, 0) = std::cos(theta);
    R_mat(0, 1) = -std::sin(theta);
    R_mat(1, 0) = std::sin(theta);
    R_mat(1, 1) = std::cos(theta);

    SO2d R(R_mat);
    CHECK(approx_equal(R.angle(), theta, 1e-10));
}

// ===== IDENTITY AND FACTORY TESTS =====

TEST_CASE("SO2::identity") {
    auto R = SO2d::identity();
    CHECK(R.is_identity());
    CHECK(approx_equal(R.angle(), 0.0, 1e-10));
}

TEST_CASE("SO2::sample_uniform produces valid rotations") {
    std::mt19937 rng(42);

    for (int i = 0; i < 100; ++i) {
        auto R = SO2d::sample_uniform(rng);

        // Check it's a valid unit complex
        const double norm = std::sqrt(R.real() * R.real() + R.imag() * R.imag());
        CHECK(approx_equal(norm, 1.0, 1e-10));

        // Angle should be in [0, 2*pi)
        const double angle = R.angle();
        CHECK(angle >= -std::numbers::pi);
        CHECK(angle <= std::numbers::pi);
    }
}

// ===== EXP AND LOG TESTS =====

TEST_CASE("SO2 exp and log are inverses") {
    SUBCASE("exp(log(R)) = R") {
        std::mt19937 rng(42);
        for (int i = 0; i < 100; ++i) {
            auto R = SO2d::sample_uniform(rng);
            const double theta = R.log();
            auto R2 = SO2d::exp(theta);
            CHECK(R.is_approx(R2, 1e-10));
        }
    }

    SUBCASE("log(exp(theta)) = theta for theta in (-pi, pi]") {
        for (double theta = -3.0; theta <= 3.0; theta += 0.1) {
            auto R = SO2d::exp(theta);
            const double theta2 = R.log();
            CHECK(approx_equal(theta, theta2, 1e-10));
        }
    }
}

TEST_CASE("SO2 exp of specific angles") {
    SUBCASE("exp(0) = identity") {
        auto R = SO2d::exp(0.0);
        CHECK(R.is_identity());
    }

    SUBCASE("exp(pi/2) = 90 degree rotation") {
        auto R = SO2d::exp(std::numbers::pi / 2);
        CHECK(approx_equal(R.real(), 0.0, 1e-10));
        CHECK(approx_equal(R.imag(), 1.0, 1e-10));
    }

    SUBCASE("exp(pi) = 180 degree rotation") {
        auto R = SO2d::exp(std::numbers::pi);
        CHECK(approx_equal(R.real(), -1.0, 1e-10));
        CHECK(approx_equal(R.imag(), 0.0, 1e-10));
    }
}

// ===== INVERSE TESTS =====

TEST_CASE("SO2 inverse") {
    SUBCASE("R * R^-1 = identity") {
        std::mt19937 rng(42);
        for (int i = 0; i < 100; ++i) {
            auto R = SO2d::sample_uniform(rng);
            auto R_inv = R.inverse();
            auto I = R * R_inv;
            CHECK(I.is_identity(1e-10));
        }
    }

    SUBCASE("(R^-1)^-1 = R") {
        SO2d R(0.7);
        auto R_inv_inv = R.inverse().inverse();
        CHECK(R.is_approx(R_inv_inv, 1e-10));
    }

    SUBCASE("inverse of identity is identity") {
        auto I = SO2d::identity();
        CHECK(I.inverse().is_identity());
    }
}

// ===== COMPOSITION TESTS =====

TEST_CASE("SO2 composition") {
    SUBCASE("identity * R = R") {
        SO2d R(0.5);
        auto result = SO2d::identity() * R;
        CHECK(result.is_approx(R, 1e-10));
    }

    SUBCASE("R * identity = R") {
        SO2d R(0.5);
        auto result = R * SO2d::identity();
        CHECK(result.is_approx(R, 1e-10));
    }

    SUBCASE("composition adds angles") {
        const double a = 0.3;
        const double b = 0.5;
        SO2d Ra(a);
        SO2d Rb(b);
        auto Rab = Ra * Rb;
        CHECK(approx_equal(Rab.angle(), a + b, 1e-10));
    }

    SUBCASE("associativity: (A*B)*C = A*(B*C)") {
        SO2d A(0.3);
        SO2d B(0.5);
        SO2d C(0.7);

        auto AB_C = (A * B) * C;
        auto A_BC = A * (B * C);
        CHECK(AB_C.is_approx(A_BC, 1e-10));
    }
}

// ===== POINT ROTATION TESTS =====

TEST_CASE("SO2 rotates points correctly") {
    SUBCASE("identity doesn't change point") {
        dp::mat::vector<double, 2> p{{1.0, 2.0}};
        auto p2 = SO2d::identity() * p;
        CHECK(approx_equal(p2[0], 1.0));
        CHECK(approx_equal(p2[1], 2.0));
    }

    SUBCASE("90 degree rotation") {
        dp::mat::vector<double, 2> p{{1.0, 0.0}};
        SO2d R(std::numbers::pi / 2);
        auto p2 = R * p;
        CHECK(approx_equal(p2[0], 0.0, 1e-10));
        CHECK(approx_equal(p2[1], 1.0, 1e-10));
    }

    SUBCASE("180 degree rotation") {
        dp::mat::vector<double, 2> p{{1.0, 0.0}};
        SO2d R(std::numbers::pi);
        auto p2 = R * p;
        CHECK(approx_equal(p2[0], -1.0, 1e-10));
        CHECK(approx_equal(p2[1], 0.0, 1e-10));
    }

    SUBCASE("rotation preserves norm") {
        dp::mat::vector<double, 2> p{{3.0, 4.0}}; // norm = 5
        SO2d R(0.7);
        auto p2 = R * p;
        const double norm_before = std::sqrt(p[0] * p[0] + p[1] * p[1]);
        const double norm_after = std::sqrt(p2[0] * p2[0] + p2[1] * p2[1]);
        CHECK(approx_equal(norm_before, norm_after, 1e-10));
    }
}

// ===== ROTATION MATRIX TESTS =====

TEST_CASE("SO2 rotation matrix") {
    SUBCASE("matrix is orthogonal") {
        SO2d R(0.7);
        auto M = R.matrix();

        // R^T * R = I
        const double a = M(0, 0), b = M(0, 1), c = M(1, 0), d = M(1, 1);
        CHECK(approx_equal(a * a + c * c, 1.0, 1e-10)); // col 1 unit
        CHECK(approx_equal(b * b + d * d, 1.0, 1e-10)); // col 2 unit
        CHECK(approx_equal(a * b + c * d, 0.0, 1e-10)); // cols orthogonal
    }

    SUBCASE("matrix determinant is 1") {
        SO2d R(0.7);
        auto M = R.matrix();
        const double det = M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0);
        CHECK(approx_equal(det, 1.0, 1e-10));
    }

    SUBCASE("matrix has correct structure") {
        const double theta = 0.5;
        SO2d R(theta);
        auto M = R.matrix();

        CHECK(approx_equal(M(0, 0), std::cos(theta), 1e-10));
        CHECK(approx_equal(M(0, 1), -std::sin(theta), 1e-10));
        CHECK(approx_equal(M(1, 0), std::sin(theta), 1e-10));
        CHECK(approx_equal(M(1, 1), std::cos(theta), 1e-10));
    }
}

// ===== LIE ALGEBRA TESTS =====

TEST_CASE("SO2 hat and vee") {
    SUBCASE("vee(hat(theta)) = theta") {
        for (double theta = -3.0; theta <= 3.0; theta += 0.1) {
            auto Omega = SO2d::hat(theta);
            double theta2 = SO2d::vee(Omega);
            CHECK(approx_equal(theta, theta2, 1e-10));
        }
    }

    SUBCASE("hat produces skew-symmetric matrix") {
        const double theta = 0.5;
        auto Omega = SO2d::hat(theta);

        // Skew-symmetric: Omega^T = -Omega
        CHECK(approx_equal(Omega(0, 0), 0.0, 1e-10));
        CHECK(approx_equal(Omega(1, 1), 0.0, 1e-10));
        CHECK(approx_equal(Omega(0, 1), -Omega(1, 0), 1e-10));
    }
}

TEST_CASE("SO2 Adjoint is 1") {
    SO2d R(0.7);
    CHECK(approx_equal(R.Adj(), 1.0));
}

TEST_CASE("SO2 Lie bracket is 0 (commutative)") { CHECK(approx_equal(SO2d::lie_bracket(0.3, 0.5), 0.0)); }

// ===== DERIVATIVE TESTS =====

TEST_CASE("SO2 derivatives") {
    SUBCASE("Dx_exp_x at 0") {
        auto J = SO2d::Dx_exp_x_at_0();
        CHECK(approx_equal(J[0], 0.0, 1e-10)); // d(cos)/d(theta) at 0
        CHECK(approx_equal(J[1], 1.0, 1e-10)); // d(sin)/d(theta) at 0
    }

    SUBCASE("Dx_exp_x matches numerical derivative") {
        const double theta = 0.5;
        const double h = 1e-8;

        // Numerical derivative
        SO2d R_plus = SO2d::exp(theta + h);
        SO2d R_minus = SO2d::exp(theta - h);
        const double d_cos = (R_plus.real() - R_minus.real()) / (2 * h);
        const double d_sin = (R_plus.imag() - R_minus.imag()) / (2 * h);

        // Analytical derivative
        auto J = SO2d::Dx_exp_x(theta);

        CHECK(approx_equal(J[0], d_cos, 1e-5));
        CHECK(approx_equal(J[1], d_sin, 1e-5));
    }
}

// ===== INTERPOLATION TESTS =====

TEST_CASE("SO2 interpolation") {
    SUBCASE("interpolate(a, b, 0) = a") {
        SO2d a(0.2);
        SO2d b(0.8);
        auto result = interpolate(a, b, 0.0);
        CHECK(result.is_approx(a, 1e-10));
    }

    SUBCASE("interpolate(a, b, 1) = b") {
        SO2d a(0.2);
        SO2d b(0.8);
        auto result = interpolate(a, b, 1.0);
        CHECK(result.is_approx(b, 1e-10));
    }

    SUBCASE("interpolate(a, b, 0.5) is midpoint") {
        SO2d a(0.0);
        SO2d b(1.0);
        auto mid = interpolate(a, b, 0.5);
        CHECK(approx_equal(mid.angle(), 0.5, 1e-10));
    }
}

// ===== AVERAGE TESTS =====

TEST_CASE("SO2 average") {
    SUBCASE("average of single element") {
        std::vector<SO2d> rotations = {SO2d(0.5)};
        auto avg = average(rotations.begin(), rotations.end());
        CHECK(avg.is_approx(rotations[0], 1e-10));
    }

    SUBCASE("average of symmetric rotations around identity") {
        std::vector<SO2d> rotations = {SO2d(0.3), SO2d(-0.3)};
        auto avg = average(rotations.begin(), rotations.end());
        CHECK(avg.is_identity(1e-10));
    }

    SUBCASE("average of two rotations") {
        std::vector<SO2d> rotations = {SO2d(0.2), SO2d(0.6)};
        auto avg = average(rotations.begin(), rotations.end());
        // Average angle should be approximately 0.4
        CHECK(approx_equal(avg.angle(), 0.4, 1e-2)); // Looser tolerance for averaging
    }
}

// ===== TYPE CONVERSION TESTS =====

TEST_CASE("SO2 cast to different scalar type") {
    SO2d Rd(0.5);
    SO2f Rf = Rd.cast<float>();

    CHECK(approx_equal(static_cast<double>(Rf.real()), Rd.real(), 1e-6));
    CHECK(approx_equal(static_cast<double>(Rf.imag()), Rd.imag(), 1e-6));
}

// ===== MUTATOR TESTS =====

TEST_CASE("SO2 mutators") {
    SUBCASE("set_angle") {
        SO2d R;
        R.set_angle(0.5);
        CHECK(approx_equal(R.angle(), 0.5, 1e-10));
    }

    SUBCASE("set_complex normalizes") {
        SO2d R;
        R.set_complex(3.0, 4.0);
        CHECK(approx_equal(R.real(), 0.6, 1e-10));
        CHECK(approx_equal(R.imag(), 0.8, 1e-10));
    }
}

// ===== EDGE CASES =====

TEST_CASE("SO2 edge cases") {
    SUBCASE("very small angles") {
        const double tiny = 1e-15;
        SO2d R(tiny);
        CHECK(approx_equal(R.angle(), tiny, 1e-14));
    }

    SUBCASE("angles near pi") {
        const double near_pi = std::numbers::pi - 1e-10;
        SO2d R(near_pi);
        CHECK(approx_equal(R.angle(), near_pi, 1e-9));
    }

    SUBCASE("angles slightly greater than pi wrap correctly") {
        // atan2 returns values in (-pi, pi], so angles > pi wrap to negative
        const double theta = std::numbers::pi + 0.1;
        SO2d R(theta);
        const double recovered = R.log();
        // Should be equivalent to theta - 2*pi
        const double expected = theta - 2 * std::numbers::pi;
        CHECK(approx_equal(recovered, expected, 1e-10));
    }
}
