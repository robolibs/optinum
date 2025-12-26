#pragma once

// =============================================================================
// optinum/lina/basic/orth.hpp
// Orthonormal basis via QR decomposition
// =============================================================================

#include <optinum/lina/basic/rank.hpp>
#include <optinum/lina/decompose/qr.hpp>
#include <optinum/simd/matrix.hpp>

#include <cstddef>
#include <type_traits>

namespace optinum::lina {

    /**
     * Compute orthonormal basis for range of A via QR
     *
     * Returns the first r columns of Q from QR decomposition,
     * where r = rank(A). These form an orthonormal basis for
     * the column space (range) of A.
     *
     * @param a Input matrix (M x N)
     * @param tol Tolerance for rank determination
     * @return Matrix whose columns are orthonormal basis (M x r)
     */
    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] simd::Matrix<T, M, M> orth(const simd::Matrix<T, M, N> &a, T tol = T{-1}) noexcept {
        static_assert(std::is_floating_point_v<T>, "orth() requires floating-point type");

        // Compute QR decomposition
        auto qr_result = qr(a);

        // Determine rank
        std::size_t r = rank(a, tol);

        // Extract first r columns of Q
        simd::Matrix<T, M, M> basis{};
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < r; ++j) {
                basis(i, j) = qr_result.q(i, j); // lowercase q
            }
        }

        return basis;
    }

} // namespace optinum::lina
