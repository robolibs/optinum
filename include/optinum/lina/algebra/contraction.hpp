#pragma once

// =============================================================================
// optinum/lina/algebra/contraction.hpp
// Tensor algebra helpers (fixed-size rank-1/2)
// Uses SIMD backend for accelerated operations.
// =============================================================================

#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::lina {

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr T inner(const simd::Vector<T, N> &a, const simd::Vector<T, N> &b) noexcept {
        return simd::dot(a, b);
    }

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] inline T inner(const simd::Matrix<T, R, C> &a, const simd::Matrix<T, R, C> &b) noexcept {
        // Frobenius inner product: sum_ij a_ij * b_ij
        // Uses SIMD dot product over flattened matrix data
        return simd::backend::dot<T, R * C>(a.data(), b.data());
    }

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] inline simd::Matrix<T, R, C> hadamard(const simd::Matrix<T, R, C> &a,
                                                        const simd::Matrix<T, R, C> &b) noexcept {
        // Element-wise (Hadamard) product using SIMD backend
        simd::Matrix<T, R, C> out;
        simd::backend::mul<T, R * C>(out.data(), a.data(), b.data());
        return out;
    }

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] inline simd::Matrix<T, R, C> contraction(const simd::Matrix<T, R, C> &a,
                                                           const simd::Matrix<T, R, C> &b) noexcept {
        // Rank-2 contraction over both indices: same as Frobenius inner, but returning elementwise product here is
        // sometimes expected; provide hadamard under this name for rank-2.
        return hadamard(a, b);
    }

    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] inline simd::Matrix<T, M, N> outer(const simd::Vector<T, M> &a,
                                                     const simd::Vector<T, N> &b) noexcept {
        // Outer product: out(i,j) = a[i] * b[j]
        // Column-major: each column j is a[0..M-1] * b[j]
        // Use SIMD mul_scalar for each column
        simd::Matrix<T, M, N> out;
        for (std::size_t j = 0; j < N; ++j) {
            simd::backend::mul_scalar<T, M>(&out(0, j), a.data(), b[j]);
        }
        return out;
    }

} // namespace optinum::lina
