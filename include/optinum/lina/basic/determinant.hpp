#pragma once

// =============================================================================
// optinum/lina/basic/determinant.hpp
// =============================================================================

#include <optinum/lina/decompose/lu.hpp>
#include <optinum/simd/matrix.hpp>

namespace optinum::lina {

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr T determinant(const simd::Matrix<T, N, N> &a) noexcept {
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

} // namespace optinum::lina
