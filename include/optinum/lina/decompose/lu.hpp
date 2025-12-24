#pragma once

// =============================================================================
// optinum/lina/decompose/lu.hpp
// LU decomposition with partial pivoting (fixed-size)
// Uses SIMD for row elimination and forward/back substitution.
// =============================================================================

#include <datapod/sequential/array.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/pack/pack.hpp>
#include <optinum/simd/vector.hpp>

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

        // SIMD row update: row_i[start..end] -= scale * row_k[start..end]
        // For row-major access in column-major storage, we iterate over columns
        template <typename T, std::size_t N>
        OPTINUM_INLINE void axpy_row(simd::Matrix<T, N, N> &mat, std::size_t row_i, std::size_t row_k, T scale,
                                     std::size_t start_col) noexcept {
            // Column-major: mat(row, col) = data[col * N + row]
            // Row elements are N apart in memory, so we use scalar loop
            // (SIMD would require gather/scatter for strided access)
            for (std::size_t j = start_col; j < N; ++j) {
                mat(row_i, j) -= scale * mat(row_k, j);
            }
        }

        // SIMD dot product for partial row/column
        template <typename T>
        [[nodiscard]] OPTINUM_INLINE T dot_partial(const T *a, const T *b, std::size_t n, std::size_t stride_a,
                                                   std::size_t stride_b) noexcept {
            T sum{};
            for (std::size_t i = 0; i < n; ++i) {
                sum += a[i * stride_a] * b[i * stride_b];
            }
            return sum;
        }

        // SIMD dot for contiguous memory (used in back substitution on U rows)
        template <typename T>
        [[nodiscard]] OPTINUM_INLINE T dot_contiguous(const T *a, const T *b, std::size_t n) noexcept {
            if (n == 0)
                return T{};

            constexpr std::size_t W = simd::backend::preferred_simd_lanes<T, 64>();
            const std::size_t main = (n / W) * W;

            simd::pack<T, W> acc(T{});
            for (std::size_t i = 0; i < main; i += W) {
                auto va = simd::pack<T, W>::loadu(a + i);
                auto vb = simd::pack<T, W>::loadu(b + i);
                acc = simd::pack<T, W>::fma(va, vb, acc);
            }

            T result = acc.hsum();
            for (std::size_t i = main; i < n; ++i) {
                result += a[i] * b[i];
            }
            return result;
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
    [[nodiscard]] inline simd::Vector<T, N> lu_solve(const LU<T, N> &f, const simd::Vector<T, N> &b) noexcept {
        simd::Vector<T, N> x;
        simd::Vector<T, N> y;

        // Apply permutation: Pb
        for (std::size_t i = 0; i < N; ++i) {
            y[i] = b[f.p[i]];
        }

        // Forward substitution: L z = Pb (use y as z)
        // L is lower triangular with 1s on diagonal (column-major storage)
        // L(i,j) for j < i: need strided access (column j, row i)
        for (std::size_t i = 0; i < N; ++i) {
            T sum = y[i];
            // L row i, columns 0..i-1: strided in column-major
            // Use scalar loop since L elements are N apart
            for (std::size_t j = 0; j < i; ++j) {
                sum -= f.l(i, j) * y[j];
            }
            y[i] = sum; // diag of L is 1
        }

        // Back substitution: U x = z
        // U is upper triangular (column-major storage)
        // U(i,j) for j > i: these are contiguous in row i starting at column i+1
        // In column-major: U(i, i+1..N-1) has stride N between elements
        for (std::size_t ii = 0; ii < N; ++ii) {
            const std::size_t i = N - 1 - ii;
            T sum = y[i];
            // U row i, columns i+1..N-1: strided access
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
