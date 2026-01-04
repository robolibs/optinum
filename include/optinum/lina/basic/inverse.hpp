#pragma once

// =============================================================================
// optinum/lina/basic/inverse.hpp
// Matrix inverse with specialized kernels for small matrices
// =============================================================================

#include <datapod/datapod.hpp>
#include <optinum/lina/decompose/lu.hpp>
#include <optinum/simd/backend/inverse_small.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::lina {

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr dp::Result<dp::mat::Matrix<T, N, N>, dp::Error>
    try_inverse(const simd::Matrix<T, N, N> &a) noexcept {
        // Use specialized kernels for small matrices (much faster)
        if constexpr (N == 2 || N == 3 || N == 4) {
            dp::mat::Matrix<T, N, N> inv;
            bool success = false;

            if constexpr (N == 2) {
                success = simd::backend::inverse_2x2(a.data(), inv.data());
            } else if constexpr (N == 3) {
                success = simd::backend::inverse_3x3(a.data(), inv.data());
            } else if constexpr (N == 4) {
                success = simd::backend::inverse_4x4(a.data(), inv.data());
            }

            if (success) {
                return dp::Result<dp::mat::Matrix<T, N, N>, dp::Error>::ok(inv);
            } else {
                return dp::Result<dp::mat::Matrix<T, N, N>, dp::Error>::err(
                    dp::Error::invalid_argument("matrix is singular"));
            }
        } else {
            // Fall back to LU decomposition for larger matrices
            const auto f = lu<T, N>(a);
            if (f.singular) {
                return dp::Result<dp::mat::Matrix<T, N, N>, dp::Error>::err(
                    dp::Error::invalid_argument("matrix is singular"));
            }

            dp::mat::Matrix<T, N, N> inv;
            for (std::size_t col = 0; col < N; ++col) {
                dp::mat::Vector<T, N> e;
                e.fill(T{});
                e[col] = T{1};
                simd::Vector<T, N> e_view(e);
                const auto x = lu_solve(f, e_view);
                for (std::size_t row = 0; row < N; ++row) {
                    inv(row, col) = x[row];
                }
            }

            return dp::Result<dp::mat::Matrix<T, N, N>, dp::Error>::ok(inv);
        }
    }

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr dp::mat::Matrix<T, N, N> inverse(const simd::Matrix<T, N, N> &a) noexcept {
        auto r = try_inverse<T, N>(a);
        if (r.is_ok()) {
            return r.value();
        }
        dp::mat::Matrix<T, N, N> zero;
        zero.fill(T{});
        return zero;
    }

    // Overload for dp::mat::matrix (owning type)
    template <typename T, std::size_t N>
    [[nodiscard]] constexpr dp::mat::Matrix<T, N, N> inverse(const dp::mat::Matrix<T, N, N> &a) noexcept {
        // Create a view and call the simd::Matrix version
        simd::Matrix<T, N, N> view(const_cast<dp::mat::Matrix<T, N, N> &>(a));
        return inverse(view);
    }

} // namespace optinum::lina
