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
     * Compute Kronecker product A x B
     *
     * For A (m x n) and B (p x q), the result is (mp x nq) where:
     *   (A x B)[i*p + k, j*q + l] = A[i,j] * B[k,l]
     *
     * Each element of A multiplies the entire matrix B.
     *
     * SIMD optimization: For each a_ij, we scale the entire B matrix using SIMD
     * and copy it to the appropriate block in the result.
     *
     * @param a First matrix (M x N)
     * @param b Second matrix (P x Q)
     * @return Kronecker product (MP x NQ)
     */
    template <typename T, std::size_t M, std::size_t N, std::size_t P, std::size_t Q>
    [[nodiscard]] datapod::mat::matrix<T, M * P, N * Q> kron(const simd::Matrix<T, M, N> &a,
                                                             const simd::Matrix<T, P, Q> &b) noexcept {
        datapod::mat::matrix<T, M * P, N * Q> result;
        result.fill(T{});

        // Temporary buffer for scaled B (a_ij * B)
        datapod::mat::matrix<T, P, Q> scaled_b_pod;
        simd::Matrix<T, P, Q> scaled_b(scaled_b_pod);

        // For each element of A
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                T a_ij = a(i, j);

                // SIMD: Scale entire B by a_ij
                simd::backend::mul_scalar<T, P * Q>(scaled_b.data(), b.data(), a_ij);

                // Copy scaled_b to the appropriate block in result
                // Result block starts at row i*P, column j*Q
                // For column-major storage, we copy column by column
                for (std::size_t l = 0; l < Q; ++l) {
                    // Column l of scaled_b goes to column (j*Q + l) of result
                    // Starting at row i*P
                    const std::size_t result_col = j * Q + l;
                    const std::size_t result_row_start = i * P;

                    // In column-major: result column starts at result.data() + result_col * (M*P)
                    // We need to copy P elements starting at row result_row_start
                    T *dst = result.data() + result_col * (M * P) + result_row_start;
                    const T *src = scaled_b.data() + l * P; // Column l of scaled_b

                    // Copy P elements (could use SIMD copy for large P)
                    if constexpr (P >= 4) {
                        simd::backend::copy_runtime<T>(dst, src, P);
                    } else {
                        for (std::size_t k = 0; k < P; ++k) {
                            dst[k] = src[k];
                        }
                    }
                }
            }
        }

        return result;
    }

} // namespace optinum::lina
