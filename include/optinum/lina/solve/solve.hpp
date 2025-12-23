#pragma once

// =============================================================================
// optinum/lina/solve/solve.hpp
// Linear system solvers (fixed-size)
// =============================================================================

#include <datapod/adapters/error.hpp>
#include <datapod/adapters/result.hpp>
#include <optinum/lina/decompose/lu.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/tensor.hpp>

namespace optinum::lina {

    namespace dp = ::datapod;

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr dp::Result<simd::Tensor<T, N>, dp::Error> try_solve(const simd::Matrix<T, N, N> &a,
                                                                                const simd::Tensor<T, N> &b) noexcept {
        const auto f = lu<T, N>(a);
        if (f.singular) {
            return dp::Result<simd::Tensor<T, N>, dp::Error>::err(dp::Error::invalid_argument("matrix is singular"));
        }
        return dp::Result<simd::Tensor<T, N>, dp::Error>::ok(lu_solve(f, b));
    }

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr simd::Tensor<T, N> solve(const simd::Matrix<T, N, N> &a,
                                                     const simd::Tensor<T, N> &b) noexcept {
        auto r = try_solve<T, N>(a, b);
        if (r.is_ok()) {
            return r.value();
        }
        simd::Tensor<T, N> zero;
        zero.fill(T{});
        return zero;
    }

    template <typename T, std::size_t N, std::size_t M>
    [[nodiscard]] constexpr dp::Result<simd::Matrix<T, N, M>, dp::Error>
    try_solve(const simd::Matrix<T, N, N> &a, const simd::Matrix<T, N, M> &b) noexcept {
        const auto f = lu<T, N>(a);
        if (f.singular) {
            return dp::Result<simd::Matrix<T, N, M>, dp::Error>::err(dp::Error::invalid_argument("matrix is singular"));
        }

        simd::Matrix<T, N, M> x;
        for (std::size_t col = 0; col < M; ++col) {
            simd::Tensor<T, N> rhs;
            for (std::size_t i = 0; i < N; ++i) {
                rhs[i] = b(i, col);
            }
            const auto sol = lu_solve(f, rhs);
            for (std::size_t i = 0; i < N; ++i) {
                x(i, col) = sol[i];
            }
        }

        return dp::Result<simd::Matrix<T, N, M>, dp::Error>::ok(x);
    }

    template <typename T, std::size_t N, std::size_t M>
    [[nodiscard]] constexpr simd::Matrix<T, N, M> solve(const simd::Matrix<T, N, N> &a,
                                                        const simd::Matrix<T, N, M> &b) noexcept {
        auto r = try_solve<T, N, M>(a, b);
        if (r.is_ok()) {
            return r.value();
        }
        simd::Matrix<T, N, M> zero;
        zero.fill(T{});
        return zero;
    }

} // namespace optinum::lina
