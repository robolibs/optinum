#pragma once

// =============================================================================
// optinum/lina/basic/determinant.hpp
// Determinant computation with specialized kernels for small matrices
// =============================================================================

#include <optinum/lina/decompose/lu.hpp>
#include <optinum/simd/backend/det_small.hpp>
#include <optinum/simd/matrix.hpp>

namespace optinum::lina {

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr T determinant(const simd::Matrix<T, N, N> &a) noexcept {
        // Use specialized kernels for small matrices (much faster)
        if constexpr (N == 2) {
            return simd::backend::det_2x2(a.data());
        } else if constexpr (N == 3) {
            return simd::backend::det_3x3(a.data());
        } else if constexpr (N == 4) {
            return simd::backend::det_4x4(a.data());
        } else {
            // Fall back to LU decomposition for larger matrices
            const auto f = lu<T, N>(a);
            if (f.singular) {
                return T{};
            }
            T det = static_cast<T>(f.sign);
            for (std::size_t i = 0; i < N; ++i) {
                det *= f.u(i, i);
            }
            return det;
        }
    }

} // namespace optinum::lina
