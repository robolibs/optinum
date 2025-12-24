#pragma once

// =============================================================================
// optinum/lina/algebra/contraction.hpp
// Tensor algebra helpers (fixed-size rank-1/2)
// =============================================================================

#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::lina {

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr T inner(const simd::Vector<T, N> &a, const simd::Vector<T, N> &b) noexcept {
        return simd::dot(a, b);
    }

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr T inner(const simd::Matrix<T, R, C> &a, const simd::Matrix<T, R, C> &b) noexcept {
        // Frobenius inner product: sum_ij a_ij * b_ij
        T sum{};
        for (std::size_t i = 0; i < R * C; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr simd::Matrix<T, R, C> hadamard(const simd::Matrix<T, R, C> &a,
                                                           const simd::Matrix<T, R, C> &b) noexcept {
        simd::Matrix<T, R, C> out;
        for (std::size_t i = 0; i < R * C; ++i) {
            out[i] = a[i] * b[i];
        }
        return out;
    }

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr simd::Matrix<T, R, C> contraction(const simd::Matrix<T, R, C> &a,
                                                              const simd::Matrix<T, R, C> &b) noexcept {
        // Rank-2 contraction over both indices: same as Frobenius inner, but returning elementwise product here is
        // sometimes expected; provide hadamard under this name for rank-2.
        return hadamard(a, b);
    }

    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] constexpr simd::Matrix<T, M, N> outer(const simd::Vector<T, M> &a,
                                                        const simd::Vector<T, N> &b) noexcept {
        simd::Matrix<T, M, N> out;
        for (std::size_t j = 0; j < N; ++j) {
            for (std::size_t i = 0; i < M; ++i) {
                out(i, j) = a[i] * b[j];
            }
        }
        return out;
    }

} // namespace optinum::lina
