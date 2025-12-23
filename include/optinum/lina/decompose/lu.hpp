#pragma once

// =============================================================================
// optinum/lina/decompose/lu.hpp
// LU decomposition with partial pivoting (fixed-size)
// =============================================================================

#include <datapod/sequential/array.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/tensor.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace optinum::lina {

    namespace dp = ::datapod;

    template <typename T, std::size_t N> struct LU {
        simd::Matrix<T, N, N> l{};
        simd::Matrix<T, N, N> u{};
        dp::Array<std::size_t, N> p{}; // row permutation (P*A = L*U)
        int sign = 1;                  // sign of the permutation
        bool singular = false;
    };

    namespace lu_detail {

        template <typename T> [[nodiscard]] OPTINUM_INLINE T abs_val(T x) noexcept {
            if constexpr (std::is_unsigned_v<T>) {
                return x;
            } else {
                return static_cast<T>(std::abs(x));
            }
        }

        template <typename T, std::size_t N>
        OPTINUM_INLINE void swap_rows(simd::Matrix<T, N, N> &a, std::size_t r1, std::size_t r2) noexcept {
            if (r1 == r2)
                return;
            for (std::size_t col = 0; col < N; ++col) {
                const T tmp = a(r1, col);
                a(r1, col) = a(r2, col);
                a(r2, col) = tmp;
            }
        }

        template <std::size_t N>
        OPTINUM_INLINE void swap_elems(dp::Array<std::size_t, N> &a, std::size_t i, std::size_t j) noexcept {
            const std::size_t tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }

    } // namespace lu_detail

    template <typename T, std::size_t N> [[nodiscard]] constexpr LU<T, N> lu(const simd::Matrix<T, N, N> &a) noexcept {
        LU<T, N> out;

        simd::Matrix<T, N, N> lu_mat = a; // in-place LU
        for (std::size_t i = 0; i < N; ++i)
            out.p[i] = i;

        for (std::size_t k = 0; k < N; ++k) {
            // Pivot
            std::size_t piv = k;
            T max_val = lu_detail::abs_val(lu_mat(k, k));
            for (std::size_t i = k + 1; i < N; ++i) {
                const T v = lu_detail::abs_val(lu_mat(i, k));
                if (v > max_val) {
                    max_val = v;
                    piv = i;
                }
            }

            if (max_val == T{}) {
                out.singular = true;
                break;
            }

            if (piv != k) {
                lu_detail::swap_rows(lu_mat, piv, k);
                lu_detail::swap_elems(out.p, piv, k);
                out.sign = -out.sign;
            }

            const T pivval = lu_mat(k, k);
            for (std::size_t i = k + 1; i < N; ++i) {
                lu_mat(i, k) /= pivval;
                const T lik = lu_mat(i, k);
                for (std::size_t j = k + 1; j < N; ++j) {
                    lu_mat(i, j) -= lik * lu_mat(k, j);
                }
            }
        }

        // Split into L and U (even if singular, fill what we can)
        out.l.fill(T{});
        out.u.fill(T{});
        for (std::size_t i = 0; i < N; ++i) {
            out.l(i, i) = T{1};
            for (std::size_t j = 0; j < N; ++j) {
                if (i > j)
                    out.l(i, j) = lu_mat(i, j);
                else
                    out.u(i, j) = lu_mat(i, j);
            }
        }

        return out;
    }

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr simd::Tensor<T, N> lu_solve(const LU<T, N> &f, const simd::Tensor<T, N> &b) noexcept {
        simd::Tensor<T, N> x;
        simd::Tensor<T, N> y;

        // Apply permutation: Pb
        for (std::size_t i = 0; i < N; ++i) {
            y[i] = b[f.p[i]];
        }

        // Forward substitution: L z = Pb (use y as z)
        for (std::size_t i = 0; i < N; ++i) {
            T sum = y[i];
            for (std::size_t j = 0; j < i; ++j) {
                sum -= f.l(i, j) * y[j];
            }
            y[i] = sum; // diag of L is 1
        }

        // Back substitution: U x = z
        for (std::size_t ii = 0; ii < N; ++ii) {
            const std::size_t i = N - 1 - ii;
            T sum = y[i];
            for (std::size_t j = i + 1; j < N; ++j) {
                sum -= f.u(i, j) * x[j];
            }
            x[i] = sum / f.u(i, i);
        }

        return x;
    }

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr simd::Matrix<T, N, N> permutation_matrix(const dp::Array<std::size_t, N> &p) noexcept {
        simd::Matrix<T, N, N> P;
        P.fill(T{});
        for (std::size_t i = 0; i < N; ++i) {
            P(i, p[i]) = T{1};
        }
        return P;
    }

} // namespace optinum::lina
