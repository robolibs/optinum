#pragma once

// =============================================================================
// optinum/lina/basic/pinv.hpp
// Moore-Penrose pseudo-inverse via SVD
// =============================================================================

#include <optinum/lina/decompose/svd.hpp>
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
    [[nodiscard]] simd::Matrix<T, N, M> pinv(const simd::Matrix<T, M, N> &a, T tol = T{-1}) noexcept {
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
        simd::Vector<T, min_dim> sigma_inv{};
        for (std::size_t i = 0; i < min_dim; ++i) {
            if (svd_result.s[i] > tol) {
                sigma_inv[i] = T{1} / svd_result.s[i];
            } else {
                sigma_inv[i] = T{0};
            }
        }

        // Compute A^+ = V Σ^+ U^T
        // Since SVD gives us V^T, we have: A^+ = (V^T)^T Σ^+ U^T = V Σ^+ U^T
        // First: (V^T)^T Σ^+ = V Σ^+ (N x min_dim)
        simd::Matrix<T, N, min_dim> v_sigma_inv{};
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < min_dim; ++j) {
                v_sigma_inv(i, j) = svd_result.vt(j, i) * sigma_inv[j]; // Transpose vt to get v
            }
        }

        // Then: (V Σ^+) U^T = V Σ^+ U^T (N x M)
        simd::Matrix<T, N, M> result{};
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < M; ++j) {
                T sum = T{0};
                for (std::size_t k = 0; k < min_dim; ++k) {
                    sum += v_sigma_inv(i, k) * svd_result.u(j, k); // U^T means we transpose
                }
                result(i, j) = sum;
            }
        }

        return result;
    }

} // namespace optinum::lina
