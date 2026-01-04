#pragma once

// =============================================================================
// optinum/lina/basic/pinv.hpp
// Moore-Penrose pseudo-inverse via SVD
// =============================================================================

#include <datapod/datapod.hpp>
#include <optinum/lina/decompose/svd.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/backend/matmul.hpp>
#include <optinum/simd/matrix.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace optinum::lina {

    /**
     * Compute Moore-Penrose pseudo-inverse via SVD
     *
     * For A = U Σ V^T, the pseudo-inverse is:
     *   A^+ = V Σ^+ U^T
     * where Σ^+ has reciprocals of non-zero singular values.
     *
     * @param a Input matrix (M x N)
     * @param tol Tolerance for singular value (default: ε * max(M,N) * σ_max)
     * @return Pseudo-inverse (N x M)
     */
    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] datapod::mat::Matrix<T, N, M> pinv(const simd::Matrix<T, M, N> &a, T tol = T{-1}) noexcept {
        static_assert(std::is_floating_point_v<T>, "pinv() requires floating-point type");

        // Compute SVD: A = U Σ V^T
        auto svd_result = svd(a);

        // Determine tolerance if not provided
        if (tol < T{0}) {
            constexpr std::size_t max_dim = (M > N) ? M : N;
            T sigma_max = svd_result.s[0];
            tol = std::numeric_limits<T>::epsilon() * static_cast<T>(max_dim) * sigma_max;
        }

        // Compute Σ^+ (reciprocal of non-zero singular values)
        constexpr std::size_t min_dim = (M < N) ? M : N;
        datapod::mat::Vector<T, min_dim> sigma_inv;
        for (std::size_t i = 0; i < min_dim; ++i) {
            if (svd_result.s[i] > tol) {
                sigma_inv[i] = T{1} / svd_result.s[i];
            } else {
                sigma_inv[i] = T{0};
            }
        }

        // Compute A^+ = V Σ^+ U^T
        // Since SVD gives us V^T, we have: A^+ = (V^T)^T Σ^+ U^T = V Σ^+ U^T
        //
        // First: (V^T)^T Σ^+ = V Σ^+ (N x min_dim)
        // V = (V^T)^T, so V(i,j) = V^T(j,i)
        // V Σ^+ means scaling each column j of V by sigma_inv[j]
        datapod::mat::Matrix<T, N, min_dim> v_sigma_inv;
        for (std::size_t j = 0; j < min_dim; ++j) {
            // Column j of V is row j of V^T (contiguous in column-major V^T)
            // V(i,j) = V^T(j,i), and we want V(i,j) * sigma_inv[j]
            // In column-major V^T: row j starts at vt.data() + j (stride = min_dim)
            // In column-major v_sigma_inv: column j is contiguous at v_sigma_inv.data() + j*N
            T scale = sigma_inv[j];
            for (std::size_t i = 0; i < N; ++i) {
                v_sigma_inv(i, j) = svd_result.vt(j, i) * scale;
            }
        }

        // Then: (V Σ^+) U^T = result (N x M)
        // result(i,j) = sum_k v_sigma_inv(i,k) * U(j,k)
        // For column-major U: column k is contiguous at u.data() + k*M
        // For column-major v_sigma_inv: column k is contiguous at v_sigma_inv.data() + k*N
        datapod::mat::Matrix<T, N, M> result;
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < M; ++j) {
                // Dot product of row i of v_sigma_inv with row j of U
                // Row i of v_sigma_inv: v_sigma_inv(i,k) for k=0..min_dim-1 (not contiguous)
                // Row j of U: U(j,k) for k=0..min_dim-1 (not contiguous)
                // Extract to contiguous buffers for SIMD if min_dim is large enough
                if constexpr (min_dim >= 8) {
                    alignas(32) T row_v[min_dim];
                    alignas(32) T row_u[min_dim];
                    for (std::size_t k = 0; k < min_dim; ++k) {
                        row_v[k] = v_sigma_inv(i, k);
                        row_u[k] = svd_result.u(j, k);
                    }
                    result(i, j) = simd::backend::dot_runtime<T>(row_v, row_u, min_dim);
                } else {
                    T sum = T{0};
                    for (std::size_t k = 0; k < min_dim; ++k) {
                        sum += v_sigma_inv(i, k) * svd_result.u(j, k);
                    }
                    result(i, j) = sum;
                }
            }
        }

        return result;
    }

    // Overload for dp::mat::matrix (owning type)
    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] datapod::mat::Matrix<T, N, M> pinv(const datapod::mat::Matrix<T, M, N> &a, T tol = T{-1}) noexcept {
        simd::Matrix<T, M, N> view(const_cast<datapod::mat::Matrix<T, M, N> &>(a));
        return pinv(view, tol);
    }

} // namespace optinum::lina
