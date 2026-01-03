#pragma once

// =============================================================================
// optinum/lina/solve/solve.hpp
// Linear system solvers (fixed-size)
// =============================================================================

#include <datapod/adapters/error.hpp>
#include <datapod/adapters/result.hpp>
#include <datapod/matrix.hpp>
#include <optinum/lina/decompose/lu.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::lina {

    namespace dp = ::datapod;

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr dp::Result<dp::mat::Vector<T, N>, dp::Error>
    try_solve(const simd::Matrix<T, N, N> &a, const simd::Vector<T, N> &b) noexcept {
        const auto f = lu<T, N>(a);
        if (f.singular) {
            return dp::Result<dp::mat::Vector<T, N>, dp::Error>::err(dp::Error::invalid_argument("matrix is singular"));
        }
        return dp::Result<dp::mat::Vector<T, N>, dp::Error>::ok(lu_solve(f, b));
    }

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr dp::mat::Vector<T, N> solve(const simd::Matrix<T, N, N> &a,
                                                        const simd::Vector<T, N> &b) noexcept {
        auto r = try_solve<T, N>(a, b);
        if (r.is_ok()) {
            return r.value();
        }
        dp::mat::Vector<T, N> zero;
        for (std::size_t i = 0; i < N; ++i) {
            zero[i] = T{};
        }
        return zero;
    }

    template <typename T, std::size_t N, std::size_t M>
    [[nodiscard]] constexpr dp::Result<dp::mat::Matrix<T, N, M>, dp::Error>
    try_solve(const simd::Matrix<T, N, N> &a, const simd::Matrix<T, N, M> &b) noexcept {
        const auto f = lu<T, N>(a);
        if (f.singular) {
            return dp::Result<dp::mat::Matrix<T, N, M>, dp::Error>::err(
                dp::Error::invalid_argument("matrix is singular"));
        }

        dp::mat::Matrix<T, N, M> x;
        for (std::size_t col = 0; col < M; ++col) {
            // Extract column from b (columns are contiguous in column-major)
            dp::mat::Vector<T, N> rhs_storage;
            simd::Vector<T, N> rhs(rhs_storage);
            const T *b_col = b.data() + col * N;
            for (std::size_t i = 0; i < N; ++i) {
                rhs[i] = b_col[i];
            }

            const auto sol = lu_solve(f, rhs);

            // Store solution column (contiguous in column-major)
            T *x_col = x.data() + col * N;
            for (std::size_t i = 0; i < N; ++i) {
                x_col[i] = sol[i];
            }
        }

        return dp::Result<dp::mat::Matrix<T, N, M>, dp::Error>::ok(x);
    }

    template <typename T, std::size_t N, std::size_t M>
    [[nodiscard]] constexpr dp::mat::Matrix<T, N, M> solve(const simd::Matrix<T, N, N> &a,
                                                           const simd::Matrix<T, N, M> &b) noexcept {
        auto r = try_solve<T, N, M>(a, b);
        if (r.is_ok()) {
            return r.value();
        }
        dp::mat::Matrix<T, N, M> zero;
        for (std::size_t i = 0; i < N * M; ++i) {
            zero[i] = T{};
        }
        return zero;
    }

} // namespace optinum::lina
