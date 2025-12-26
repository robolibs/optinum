#pragma once

// =============================================================================
// optinum/lina/algebra/kron.hpp
// Kronecker product
// =============================================================================

#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/matrix.hpp>

#include <cstddef>

namespace optinum::lina {

    /**
     * Compute Kronecker product A ⊗ B
     *
     * For A (m x n) and B (p x q), the result is (mp x nq) where:
     *   (A ⊗ B)[i*p + k, j*q + l] = A[i,j] * B[k,l]
     *
     * Each element of A multiplies the entire matrix B.
     *
     * @param a First matrix (M x N)
     * @param b Second matrix (P x Q)
     * @return Kronecker product (MP x NQ)
     */
    template <typename T, std::size_t M, std::size_t N, std::size_t P, std::size_t Q>
    [[nodiscard]] simd::Matrix<T, M * P, N * Q> kron(const simd::Matrix<T, M, N> &a,
                                                     const simd::Matrix<T, P, Q> &b) noexcept {
        simd::Matrix<T, M * P, N * Q> result{};

        // For each element of A
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                T a_ij = a(i, j);

                // Multiply entire B by a_ij and place in result
                for (std::size_t k = 0; k < P; ++k) {
                    for (std::size_t l = 0; l < Q; ++l) {
                        result(i * P + k, j * Q + l) = a_ij * b(k, l);
                    }
                }
            }
        }

        return result;
    }

} // namespace optinum::lina
