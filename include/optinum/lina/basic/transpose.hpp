#pragma once

// =============================================================================
// optinum/lina/basic/transpose.hpp
// =============================================================================

#include <optinum/simd/matrix.hpp>

namespace optinum::lina {

    // Transpose returning owning type
    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr datapod::mat::Matrix<T, C, R> transpose(const simd::Matrix<T, R, C> &a) noexcept {
        datapod::mat::Matrix<T, C, R> result;
        simd::transpose_to(result.data(), a);
        return result;
    }

    // Transpose for owning dp::mat::matrix
    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr datapod::mat::Matrix<T, C, R> transpose(const datapod::mat::Matrix<T, R, C> &a) noexcept {
        datapod::mat::Matrix<T, C, R> result;
        simd::Matrix<T, R, C> view(a);
        simd::transpose_to(result.data(), view);
        return result;
    }

} // namespace optinum::lina
