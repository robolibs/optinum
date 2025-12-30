#pragma once

// =============================================================================
// optinum/lina/basic/null.hpp
// Null space basis via SVD
// =============================================================================

#include <datapod/matrix/matrix.hpp>
#include <optinum/lina/decompose/svd.hpp>
#include <optinum/simd/matrix.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace optinum::lina {

    /**
     * Compute null space basis via SVD
     *
     * Returns the right singular vectors corresponding to near-zero singular values.
     * These form an orthonormal basis for the null space of A.
     *
     * For A of size M x N with rank r, the null space has dimension N - r.
     *
     * @param a Input matrix (M x N)
     * @param tol Tolerance for singular value (default: ε * max(M,N) * σ_max)
     * @return Matrix whose columns span the null space (N x (N-r))
     *         Returns empty matrix if A has full rank
     */
    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] datapod::mat::matrix<T, N, N> null(const simd::Matrix<T, M, N> &a, T tol = T{-1}) noexcept {
        static_assert(std::is_floating_point_v<T>, "null() requires floating-point type");

        // Compute SVD
        auto svd_result = svd(a);

        // Determine tolerance if not provided
        if (tol < T{0}) {
            constexpr std::size_t max_dim = (M > N) ? M : N;
            T sigma_max = svd_result.s[0];
            tol = std::numeric_limits<T>::epsilon() * static_cast<T>(max_dim) * sigma_max;
        }

        // Find rank (number of non-zero singular values)
        constexpr std::size_t min_dim = (M < N) ? M : N;
        std::size_t r = 0;
        for (std::size_t i = 0; i < min_dim; ++i) {
            if (svd_result.s[i] > tol) {
                ++r;
            }
        }

        // Null space dimension = N - r
        std::size_t null_dim = N - r;

        // Extract right singular vectors corresponding to zero singular values
        // These are columns r to N-1 of V
        // Since SVD gives us V^T, rows r to N-1 of V^T are the null space vectors
        datapod::mat::matrix<T, N, N> null_basis{};

        if (null_dim > 0) {
            for (std::size_t i = 0; i < N; ++i) {
                for (std::size_t j = 0; j < null_dim; ++j) {
                    null_basis(i, j) = svd_result.vt(r + j, i); // Transpose vt to get v
                }
            }
        }

        return null_basis;
    }

} // namespace optinum::lina
