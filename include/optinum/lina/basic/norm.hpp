#pragma once

// =============================================================================
// optinum/lina/basic/norm.hpp
// =============================================================================

#include <optinum/simd/matrix.hpp>
#include <optinum/simd/tensor.hpp>

#include <cmath>

namespace optinum::lina {

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr T dot(const simd::Tensor<T, N> &a, const simd::Tensor<T, N> &b) noexcept {
        return simd::dot(a, b);
    }

    template <typename T, std::size_t N> [[nodiscard]] T norm(const simd::Tensor<T, N> &x) noexcept {
        return simd::norm(x);
    }

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] T norm_fro(const simd::Matrix<T, R, C> &a) noexcept {
        return simd::frobenius_norm(a);
    }

    template <typename T>
    [[nodiscard]] constexpr simd::Tensor<T, 3> cross(const simd::Tensor<T, 3> &a,
                                                     const simd::Tensor<T, 3> &b) noexcept {
        simd::Tensor<T, 3> r;
        r[0] = a[1] * b[2] - a[2] * b[1];
        r[1] = a[2] * b[0] - a[0] * b[2];
        r[2] = a[0] * b[1] - a[1] * b[0];
        return r;
    }

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr simd::Matrix<T, R, C> scale(const simd::Matrix<T, R, C> &a, T s) noexcept {
        return a * s;
    }

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr simd::Tensor<T, N> scale(const simd::Tensor<T, N> &x, T s) noexcept {
        return x * s;
    }

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr simd::Matrix<T, R, C> axpy(T alpha, const simd::Matrix<T, R, C> &x,
                                                       const simd::Matrix<T, R, C> &y) noexcept {
        // alpha*x + y
        return (x * alpha) + y;
    }

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr simd::Tensor<T, N> axpy(T alpha, const simd::Tensor<T, N> &x,
                                                    const simd::Tensor<T, N> &y) noexcept {
        return (x * alpha) + y;
    }

} // namespace optinum::lina
