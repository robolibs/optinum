#pragma once

// =============================================================================
// optinum/lina/basic/transpose.hpp
// =============================================================================

#include <optinum/simd/matrix.hpp>

namespace optinum::lina {

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr simd::Matrix<T, C, R> transpose(const simd::Matrix<T, R, C> &a) noexcept {
        return simd::transpose(a);
    }

} // namespace optinum::lina
