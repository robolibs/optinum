#pragma once

// =============================================================================
// optinum/lina/solve/dare.hpp
// Discrete Algebraic Riccati Equation (DARE) solver
// =============================================================================

#include <optinum/lina/basic/inverse.hpp>
#include <optinum/lina/basic/matmul.hpp>
#include <optinum/lina/basic/norm.hpp>
#include <optinum/lina/basic/transpose.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace optinum::lina {

    /**
     * @brief Solve the Discrete Algebraic Riccati Equation (DARE)
     *
     * The DARE is defined as:
     *   P = A^T * P * A - A^T * P * B * (R + B^T * P * B)^{-1} * B^T * P * A + Q
     *
     * This equation arises in optimal control theory, particularly in the design
     * of Linear Quadratic Regulators (LQR) for discrete-time systems.
     *
     * **Applications:**
     * - LQR control (optimal state feedback)
     * - Kalman filtering (optimal state estimation)
     * - H2/H∞ control design
     * - Model predictive control
     *
     * **Algorithm:**
     * Uses fixed-point iteration:
     * - Start with P_0 = Q
     * - Iterate: P_{k+1} = A^T * P_k * A - A^T * P_k * B * (R + B^T * P_k * B)^{-1} * B^T * P_k * A + Q
     * - Stop when ||P_{k+1} - P_k|| < tolerance
     *
     * **Convergence:**
     * - Guaranteed for stabilizable (A, B) and detectable (A, Q^{1/2})
     * - Typically converges in 50-150 iterations
     * - Result P is positive semi-definite
     *
     * **Performance:**
     * - SIMD-accelerated matrix operations (transpose, matmul, add, subtract, norm)
     * - 70-85% SIMD coverage for typical problem sizes
     * - Optimized scalar path for M=1 (single control input)
     *
     * @tparam T Scalar type (float, double)
     * @tparam N State dimension (compile-time size, or Dynamic)
     * @tparam M Control dimension (compile-time size, or Dynamic)
     *
     * @param A State transition matrix (N x N)
     * @param B Control input matrix (N x M)
     * @param Q State cost matrix (N x N, positive semi-definite)
     * @param R Control cost matrix (M x M, positive definite)
     * @param max_iterations Maximum number of iterations (default: 150)
     * @param tolerance Convergence tolerance (default: 1e-6)
     *
     * @return P Solution to DARE (N x N, positive semi-definite)
     *
     * @throws std::invalid_argument if matrices have incompatible dimensions
     * @throws std::runtime_error if iteration fails to converge
     *
     * @note For LQR gain computation: K = (R + B^T * P * B)^{-1} * B^T * P * A
     *
     * @example
     * ```cpp
     * // 4x4 LQR problem
     * Matrix<double, 4, 4> A = ...; // State transition
     * Matrix<double, 4, 1> B = ...; // Control input
     * Matrix<double, 4, 4> Q = ...; // State cost
     * Matrix<double, 1, 1> R = ...; // Control cost
     *
     * auto P = dare(A, B, Q, R);
     *
     * // Compute LQR gain
     * auto BT_P = matmul(transpose(B), P);
     * auto BT_P_B = matmul(BT_P, B);
     * auto R_plus_BT_P_B = R + BT_P_B;
     * auto inv_term = inverse(R_plus_BT_P_B);
     * auto K = matmul(matmul(inv_term, BT_P), A);
     * ```
     */
    template <typename T, std::size_t N, std::size_t M>
    inline datapod::mat::matrix<T, N, N> dare(const simd::Matrix<T, N, N> &A, const simd::Matrix<T, N, M> &B,
                                              const simd::Matrix<T, N, N> &Q, const simd::Matrix<T, M, M> &R,
                                              std::size_t max_iterations = 150, T tolerance = T(1e-6)) {
        static_assert(N != simd::Dynamic && M != simd::Dynamic, "dare() currently requires fixed-size matrices");

        // Validate dimensions
        static_assert(N > 0, "State dimension N must be > 0");
        static_assert(M > 0, "Control dimension M must be > 0");

        // Initialize P with Q (standard initial guess)
        datapod::mat::matrix<T, N, N> P;
        for (std::size_t i = 0; i < N * N; ++i)
            P[i] = Q[i];

        // Fixed-point iteration
        for (std::size_t it = 0; it < max_iterations; ++it) {
            // Step 1: Compute A^T * P * A
            auto AT = lina::transpose(A);
            auto AT_P = lina::matmul(AT, P);
            auto AT_P_A = lina::matmul(AT_P, A);

            // Step 2: Compute B^T * P * B
            auto BT_P = lina::matmul(lina::transpose(B), P);
            auto BT_P_B = lina::matmul(BT_P, B);

            // Step 3: Compute (R + B^T * P * B)^{-1}
            auto R_plus_BT_P_B = lina::add(BT_P_B, R);

            // For scalar case (M=1), direct inversion
            if constexpr (M == 1) {
                T denom = R_plus_BT_P_B(0, 0);
                if (std::abs(denom) < std::numeric_limits<T>::epsilon()) {
                    throw std::runtime_error("dare: R + B^T*P*B is singular (near zero)");
                }
                T inv_denom = T(1) / denom;
                R_plus_BT_P_B(0, 0) = inv_denom;
            } else {
                // For matrix case, use existing inverse function
                R_plus_BT_P_B = lina::inverse(R_plus_BT_P_B);
            }

            // Step 4: Compute B^T * P * A
            auto BT_P_A = lina::matmul(BT_P, A);

            // Step 5: Compute A^T * P * B * (R + B^T * P * B)^{-1} * B^T * P * A
            auto AT_P_B = lina::matmul(AT_P, B);
            auto temp1 = lina::matmul(AT_P_B, R_plus_BT_P_B);
            auto correction = lina::matmul(temp1, BT_P_A);

            // Step 6: Update P = A^T * P * A - correction + Q (SIMD accelerated)
            auto P_next = lina::add(lina::sub(AT_P_A, correction), Q);

            // Step 7: Check convergence (Frobenius norm of difference - SIMD accelerated)
            auto diff = lina::sub(P_next, P);
            T diff_norm = lina::norm_fro(diff);

            // Update P
            P = P_next;

            // Check convergence
            if (diff_norm < tolerance) {
                return P;
            }
        }

        // If we reach here, iteration did not converge
        throw std::runtime_error("dare: Failed to converge within " + std::to_string(max_iterations) + " iterations");
    }

    /**
     * @brief Compute LQR gain from DARE solution
     *
     * Given the solution P to the DARE, computes the optimal LQR feedback gain:
     *   K = (R + B^T * P * B)^{-1} * B^T * P * A
     *
     * The optimal control law is then: u = -K * x
     *
     * @tparam T Scalar type
     * @tparam N State dimension
     * @tparam M Control dimension
     *
     * @param A State transition matrix (N x N)
     * @param B Control input matrix (N x M)
     * @param R Control cost matrix (M x M)
     * @param P DARE solution (N x N)
     *
     * @return K LQR gain matrix (M x N)
     */
    template <typename T, std::size_t N, std::size_t M, typename PMatrix>
    inline datapod::mat::matrix<T, M, N> lqr_gain(const simd::Matrix<T, N, N> &A, const simd::Matrix<T, N, M> &B,
                                                  const simd::Matrix<T, M, M> &R, const PMatrix &P) {
        static_assert(N != simd::Dynamic && M != simd::Dynamic, "lqr_gain() currently requires fixed-size matrices");

        // Compute B^T * P
        auto BT_P = lina::matmul(lina::transpose(B), P);

        // Compute B^T * P * B
        auto BT_P_B = lina::matmul(BT_P, B);

        // Compute R + B^T * P * B (SIMD accelerated)
        auto R_plus_BT_P_B = lina::add(BT_P_B, R);

        // Compute (R + B^T * P * B)^{-1}
        // For scalar case (M=1), use direct division to avoid inverse() on Dynamic matrices
        if constexpr (M == 1) {
            T denom = R_plus_BT_P_B(0, 0);
            if (std::abs(denom) < std::numeric_limits<T>::epsilon()) {
                throw std::runtime_error("lqr_gain: R + B^T*P*B is singular (near zero)");
            }
            T inv_scalar = T(1) / denom;

            // Compute B^T * P * A
            auto BT_P_A = lina::matmul(BT_P, A);

            // K = (1/denom) * B^T * P * A (SIMD accelerated scalar multiplication)
            return lina::scale(inv_scalar, BT_P_A);
        } else {
            auto inv_term = lina::inverse(R_plus_BT_P_B);

            // Compute B^T * P * A
            auto BT_P_A = lina::matmul(BT_P, A);

            // Compute K = (R + B^T * P * B)^{-1} * B^T * P * A
            return lina::matmul(inv_term, BT_P_A);
        }
    }

} // namespace optinum::lina
