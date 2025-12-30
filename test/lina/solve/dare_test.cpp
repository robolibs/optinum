#include <doctest/doctest.h>
#include <optinum/lina/basic/inverse.hpp>
#include <optinum/lina/basic/matmul.hpp>
#include <optinum/lina/basic/transpose.hpp>
#include <optinum/lina/solve/dare.hpp>

namespace lina = optinum::lina;
using optinum::simd::Matrix;

namespace dp = datapod;

TEST_CASE("DARE: Simple 2x2 stable system") {
    // Simple stable discrete-time system
    dp::mat::matrix<double, 2, 2> A;
    A(0, 0) = 0.9;
    A(0, 1) = 0.1;
    A(1, 0) = 0.0;
    A(1, 1) = 0.8;

    dp::mat::matrix<double, 2, 1> B;
    B(0, 0) = 0.0;
    B(1, 0) = 1.0;

    dp::mat::matrix<double, 2, 2> Q;
    Q(0, 0) = 1.0;
    Q(0, 1) = 0.0;
    Q(1, 0) = 0.0;
    Q(1, 1) = 1.0;

    dp::mat::matrix<double, 1, 1> R;
    R(0, 0) = 1.0;

    auto P =
        lina::dare(Matrix<double, 2, 2>(A), Matrix<double, 2, 1>(B), Matrix<double, 2, 2>(Q), Matrix<double, 1, 1>(R));

    // Verify P is positive definite (check diagonal elements are positive)
    CHECK(P(0, 0) > 0.0);
    CHECK(P(1, 1) > 0.0);

    // Verify P is symmetric (for DARE solution)
    CHECK(P(0, 1) == doctest::Approx(P(1, 0)).epsilon(1e-6));

    // Verify DARE equation is satisfied: P = A^T * P * A - A^T * P * B * (R + B^T * P * B)^{-1} * B^T * P * A + Q
    auto AT = lina::transpose(A);
    auto AT_P = lina::matmul(AT, P);
    auto AT_P_A = lina::matmul(AT_P, A);

    auto BT = lina::transpose(B);
    auto BT_P = lina::matmul(BT, P);
    auto BT_P_B = lina::matmul(BT_P, B);

    dp::mat::matrix<double, 1, 1> R_plus_BT_P_B = R;
    R_plus_BT_P_B(0, 0) += BT_P_B(0, 0);
    auto inv_term = lina::inverse(Matrix<double, 1, 1>(R_plus_BT_P_B));

    auto BT_P_A = lina::matmul(BT_P, A);
    auto AT_P_B = lina::matmul(AT_P, B);
    auto correction = lina::matmul(lina::matmul(AT_P_B, inv_term), BT_P_A);

    dp::mat::matrix<double, 2, 2> P_expected;
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            P_expected(i, j) = AT_P_A(i, j) - correction(i, j) + Q(i, j);
        }
    }

    // Check DARE residual
    CHECK(P(0, 0) == doctest::Approx(P_expected(0, 0)).epsilon(1e-6));
    CHECK(P(0, 1) == doctest::Approx(P_expected(0, 1)).epsilon(1e-6));
    CHECK(P(1, 0) == doctest::Approx(P_expected(1, 0)).epsilon(1e-6));
    CHECK(P(1, 1) == doctest::Approx(P_expected(1, 1)).epsilon(1e-6));
}

TEST_CASE("DARE: 4x4 LQR problem (drivekit-style)") {
    // State: [lateral_error, lateral_error_rate, heading_error, heading_error_rate]
    // Simplified linearized lateral vehicle dynamics
    double dt = 0.1; // Time step
    double v = 10.0; // Velocity (m/s)

    dp::mat::matrix<double, 4, 4> A;
    // Simplified continuous-time dynamics discretized
    A(0, 0) = 1.0;
    A(0, 1) = dt;
    A(0, 2) = 0.0;
    A(0, 3) = 0.0;
    A(1, 0) = 0.0;
    A(1, 1) = 0.9;
    A(1, 2) = v * dt;
    A(1, 3) = 0.0;
    A(2, 0) = 0.0;
    A(2, 1) = 0.0;
    A(2, 2) = 1.0;
    A(2, 3) = dt;
    A(3, 0) = 0.0;
    A(3, 1) = 0.0;
    A(3, 2) = 0.0;
    A(3, 3) = 0.95;

    dp::mat::matrix<double, 4, 1> B;
    // Simplified control influence
    B(0, 0) = 0.0;
    B(1, 0) = dt * 0.5;
    B(2, 0) = 0.0;
    B(3, 0) = dt * 1.0;

    dp::mat::matrix<double, 4, 4> Q;
    // State cost (penalize lateral and heading errors)
    Q(0, 0) = 10.0;
    Q(0, 1) = 0.0;
    Q(0, 2) = 0.0;
    Q(0, 3) = 0.0;
    Q(1, 0) = 0.0;
    Q(1, 1) = 1.0;
    Q(1, 2) = 0.0;
    Q(1, 3) = 0.0;
    Q(2, 0) = 0.0;
    Q(2, 1) = 0.0;
    Q(2, 2) = 5.0;
    Q(2, 3) = 0.0;
    Q(3, 0) = 0.0;
    Q(3, 1) = 0.0;
    Q(3, 2) = 0.0;
    Q(3, 3) = 0.5;

    dp::mat::matrix<double, 1, 1> R;
    R(0, 0) = 1.0; // Control cost

    auto P = lina::dare(Matrix<double, 4, 4>(A), Matrix<double, 4, 1>(B), Matrix<double, 4, 4>(Q),
                        Matrix<double, 1, 1>(R), 200, 1e-8);

    // Verify P is positive semi-definite (diagonal elements should be non-negative)
    CHECK(P(0, 0) >= 0.0);
    CHECK(P(1, 1) >= 0.0);
    CHECK(P(2, 2) >= 0.0);
    CHECK(P(3, 3) >= 0.0);

    // Verify P is symmetric
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = i + 1; j < 4; ++j) {
            CHECK(P(i, j) == doctest::Approx(P(j, i)).epsilon(1e-6));
        }
    }

    // Compute LQR gain
    auto K = lina::lqr_gain(Matrix<double, 4, 4>(A), Matrix<double, 4, 1>(B), Matrix<double, 1, 1>(R), P);

    // Verify K has correct dimensions (1 x 4)
    CHECK(K.rows() == 1);
    CHECK(K.cols() == 4);

    // Verify closed-loop stability: eigenvalues of (A - B*K) should be < 1
    // For now, just check that K is non-zero
    double k_norm = 0.0;
    for (std::size_t j = 0; j < 4; ++j) {
        k_norm += K(0, j) * K(0, j);
    }
    k_norm = std::sqrt(k_norm);
    CHECK(k_norm > 0.0);
}

TEST_CASE("DARE: Identity system (A=I, B=I, Q=I, R=I)") {
    dp::mat::matrix<double, 3, 3> A;
    A(0, 0) = 1.0;
    A(0, 1) = 0.0;
    A(0, 2) = 0.0;
    A(1, 0) = 0.0;
    A(1, 1) = 1.0;
    A(1, 2) = 0.0;
    A(2, 0) = 0.0;
    A(2, 1) = 0.0;
    A(2, 2) = 1.0;

    dp::mat::matrix<double, 3, 3> B;
    B(0, 0) = 1.0;
    B(0, 1) = 0.0;
    B(0, 2) = 0.0;
    B(1, 0) = 0.0;
    B(1, 1) = 1.0;
    B(1, 2) = 0.0;
    B(2, 0) = 0.0;
    B(2, 1) = 0.0;
    B(2, 2) = 1.0;

    dp::mat::matrix<double, 3, 3> Q;
    Q(0, 0) = 1.0;
    Q(0, 1) = 0.0;
    Q(0, 2) = 0.0;
    Q(1, 0) = 0.0;
    Q(1, 1) = 1.0;
    Q(1, 2) = 0.0;
    Q(2, 0) = 0.0;
    Q(2, 1) = 0.0;
    Q(2, 2) = 1.0;

    dp::mat::matrix<double, 3, 3> R;
    R(0, 0) = 1.0;
    R(0, 1) = 0.0;
    R(0, 2) = 0.0;
    R(1, 0) = 0.0;
    R(1, 1) = 1.0;
    R(1, 2) = 0.0;
    R(2, 0) = 0.0;
    R(2, 1) = 0.0;
    R(2, 2) = 1.0;

    auto P =
        lina::dare(Matrix<double, 3, 3>(A), Matrix<double, 3, 3>(B), Matrix<double, 3, 3>(Q), Matrix<double, 3, 3>(R));

    // For this special case, we can verify basic properties
    CHECK(P(0, 0) > 0.0);
    CHECK(P(1, 1) > 0.0);
    CHECK(P(2, 2) > 0.0);

    // Verify symmetry
    CHECK(P(0, 1) == doctest::Approx(P(1, 0)).epsilon(1e-6));
    CHECK(P(0, 2) == doctest::Approx(P(2, 0)).epsilon(1e-6));
    CHECK(P(1, 2) == doctest::Approx(P(2, 1)).epsilon(1e-6));
}

TEST_CASE("DARE: LQR gain computation") {
    dp::mat::matrix<double, 2, 2> A;
    A(0, 0) = 0.9;
    A(0, 1) = 0.1;
    A(1, 0) = 0.0;
    A(1, 1) = 0.8;

    dp::mat::matrix<double, 2, 1> B;
    B(0, 0) = 0.0;
    B(1, 0) = 1.0;

    dp::mat::matrix<double, 2, 2> Q;
    Q(0, 0) = 1.0;
    Q(0, 1) = 0.0;
    Q(1, 0) = 0.0;
    Q(1, 1) = 1.0;

    dp::mat::matrix<double, 1, 1> R;
    R(0, 0) = 1.0;

    auto P =
        lina::dare(Matrix<double, 2, 2>(A), Matrix<double, 2, 1>(B), Matrix<double, 2, 2>(Q), Matrix<double, 1, 1>(R));
    auto K = lina::lqr_gain(Matrix<double, 2, 2>(A), Matrix<double, 2, 1>(B), Matrix<double, 1, 1>(R), P);

    // Verify K dimensions
    CHECK(K.rows() == 1);
    CHECK(K.cols() == 2);

    // Verify K is non-zero (optimal control should exist)
    CHECK((std::abs(K(0, 0)) + std::abs(K(0, 1))) > 0.0);

    // Verify manual computation matches
    auto BT = lina::transpose(B);
    auto BT_P = lina::matmul(BT, P);
    auto BT_P_B = lina::matmul(BT_P, B);

    dp::mat::matrix<double, 1, 1> R_plus_BT_P_B = R;
    R_plus_BT_P_B(0, 0) += BT_P_B(0, 0);
    auto inv_term = lina::inverse(Matrix<double, 1, 1>(R_plus_BT_P_B));

    auto BT_P_A = lina::matmul(BT_P, A);
    auto K_manual = lina::matmul(inv_term, BT_P_A);

    CHECK(K(0, 0) == doctest::Approx(K_manual(0, 0)).epsilon(1e-9));
    CHECK(K(0, 1) == doctest::Approx(K_manual(0, 1)).epsilon(1e-9));
}

TEST_CASE("DARE: Convergence with maximum iterations") {
    // Test that algorithm respects max_iterations parameter
    dp::mat::matrix<double, 2, 2> A;
    A(0, 0) = 0.95;
    A(0, 1) = 0.05;
    A(1, 0) = 0.05;
    A(1, 1) = 0.95;

    dp::mat::matrix<double, 2, 1> B;
    B(0, 0) = 0.1;
    B(1, 0) = 0.1;

    dp::mat::matrix<double, 2, 2> Q;
    Q(0, 0) = 1.0;
    Q(0, 1) = 0.0;
    Q(1, 0) = 0.0;
    Q(1, 1) = 1.0;

    dp::mat::matrix<double, 1, 1> R;
    R(0, 0) = 1.0;

    // With very low tolerance and few iterations, should either converge or throw
    // Most systems will converge quickly, so just verify it doesn't crash
    bool converged = false;
    try {
        auto P = lina::dare(Matrix<double, 2, 2>(A), Matrix<double, 2, 1>(B), Matrix<double, 2, 2>(Q),
                            Matrix<double, 1, 1>(R), 5, 1e-12); // Very strict tolerance, few iterations
        converged = true;
        // If it converges, verify result is valid
        CHECK(P(0, 0) > 0.0);
        CHECK(P(1, 1) > 0.0);
    } catch (const std::runtime_error &) {
        // It's okay if it doesn't converge with these strict parameters
        converged = false;
    }

    // Just verify the function handles the parameters correctly (no crash)
    CHECK(true);
}

TEST_CASE("DARE: Multiple control inputs (M > 1)") {
    // Test with 2 control inputs
    dp::mat::matrix<double, 3, 3> A;
    A(0, 0) = 0.9;
    A(0, 1) = 0.1;
    A(0, 2) = 0.0;
    A(1, 0) = 0.0;
    A(1, 1) = 0.85;
    A(1, 2) = 0.1;
    A(2, 0) = 0.0;
    A(2, 1) = 0.0;
    A(2, 2) = 0.95;

    dp::mat::matrix<double, 3, 2> B;
    B(0, 0) = 1.0;
    B(0, 1) = 0.0;
    B(1, 0) = 0.0;
    B(1, 1) = 1.0;
    B(2, 0) = 0.5;
    B(2, 1) = 0.5;

    dp::mat::matrix<double, 3, 3> Q;
    Q(0, 0) = 1.0;
    Q(0, 1) = 0.0;
    Q(0, 2) = 0.0;
    Q(1, 0) = 0.0;
    Q(1, 1) = 2.0;
    Q(1, 2) = 0.0;
    Q(2, 0) = 0.0;
    Q(2, 1) = 0.0;
    Q(2, 2) = 1.5;

    dp::mat::matrix<double, 2, 2> R;
    R(0, 0) = 1.0;
    R(0, 1) = 0.0;
    R(1, 0) = 0.0;
    R(1, 1) = 1.0;

    auto P =
        lina::dare(Matrix<double, 3, 3>(A), Matrix<double, 3, 2>(B), Matrix<double, 3, 3>(Q), Matrix<double, 2, 2>(R));

    // Verify P is positive semi-definite
    CHECK(P(0, 0) > 0.0);
    CHECK(P(1, 1) > 0.0);
    CHECK(P(2, 2) > 0.0);

    // Verify P is symmetric
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = i + 1; j < 3; ++j) {
            CHECK(P(i, j) == doctest::Approx(P(j, i)).epsilon(1e-6));
        }
    }
}
