#pragma once

// =============================================================================
// optinum/lina/basic/matmul.hpp
// =============================================================================

#include <optinum/simd/matrix.hpp>
#include <optinum/simd/tensor.hpp>

namespace optinum::lina {

    template <typename T, std::size_t R, std::size_t K, std::size_t C>
    [[nodiscard]] constexpr simd::Matrix<T, R, C> matmul(const simd::Matrix<T, R, K> &a,
                                                         const simd::Matrix<T, K, C> &b) noexcept {
        return a * b;
    }

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr simd::Tensor<T, R> matmul(const simd::Matrix<T, R, C> &a,
                                                      const simd::Tensor<T, C> &x) noexcept {
        return a * x;
    }

} // namespace optinum::lina
