#pragma once

// =============================================================================
// optinum/lina/decompose/lu_dynamic.hpp
// LU decomposition with partial pivoting (dynamic/runtime-size)
// =============================================================================

#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace optinum::lina {

    /**
     * @brief Result of dynamic LU decomposition with partial pivoting
     *
     * Stores L (lower triangular with unit diagonal), U (upper triangular),
     * and permutation vector P such that P*A = L*U.
     */
    template <typename T> struct LUDynamic {
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> l;
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> u;
        simd::Vector<std::size_t, simd::Dynamic> p;
        int sign = 1;
        bool singular = false;

        LUDynamic() = default;

        explicit LUDynamic(std::size_t n) : l(n, n), u(n, n), p(n) {
            l.fill(T{});
            u.fill(T{});
            for (std::size_t i = 0; i < n; ++i) {
                p[i] = i;
            }
        }
    };

    namespace lu_dynamic_detail {

        template <typename T> [[nodiscard]] inline T abs_val(T x) noexcept {
            if constexpr (std::is_unsigned_v<T>) {
                return x;
            } else {
                return static_cast<T>(std::abs(x));
            }
        }

        template <typename T>
        inline void swap_rows(simd::Matrix<T, simd::Dynamic, simd::Dynamic> &mat, std::size_t r1,
                              std::size_t r2) noexcept {
            if (r1 == r2)
                return;
            const std::size_t n = mat.cols();
            for (std::size_t col = 0; col < n; ++col) {
                T tmp = mat(r1, col);
                mat(r1, col) = mat(r2, col);
                mat(r2, col) = tmp;
            }
        }

    } // namespace lu_dynamic_detail

    /**
     * @brief Compute LU decomposition with partial pivoting for dynamic-size matrix
     *
     * @tparam T Scalar type (float, double)
     * @param a Input matrix (n x n)
     * @return LUDynamic<T> containing L, U, and permutation
     */
    template <typename T>
    [[nodiscard]] inline LUDynamic<T> lu_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a) {
        const std::size_t n = a.rows();
        LUDynamic<T> out(n);

        // Copy input to working matrix
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> lu_mat(n, n);
        for (std::size_t i = 0; i < n * n; ++i) {
            lu_mat[i] = a[i];
        }

        for (std::size_t k = 0; k < n; ++k) {
            // Find pivot
            std::size_t piv = k;
            T max_val = lu_dynamic_detail::abs_val(lu_mat(k, k));
            for (std::size_t i = k + 1; i < n; ++i) {
                const T v = lu_dynamic_detail::abs_val(lu_mat(i, k));
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
                lu_dynamic_detail::swap_rows(lu_mat, piv, k);
                std::size_t tmp = out.p[piv];
                out.p[piv] = out.p[k];
                out.p[k] = tmp;
                out.sign = -out.sign;
            }

            const T pivval = lu_mat(k, k);
            for (std::size_t i = k + 1; i < n; ++i) {
                lu_mat(i, k) /= pivval;
                const T lik = lu_mat(i, k);
                for (std::size_t j = k + 1; j < n; ++j) {
                    lu_mat(i, j) -= lik * lu_mat(k, j);
                }
            }
        }

        // Split into L and U
        for (std::size_t i = 0; i < n; ++i) {
            out.l(i, i) = T{1};
            for (std::size_t j = 0; j < n; ++j) {
                if (i > j) {
                    out.l(i, j) = lu_mat(i, j);
                } else {
                    out.u(i, j) = lu_mat(i, j);
                }
            }
        }

        return out;
    }

    /**
     * @brief Solve Ax = b using precomputed LU decomposition
     *
     * @tparam T Scalar type
     * @param f LU decomposition result
     * @param b Right-hand side vector
     * @return Solution vector x
     */
    template <typename T>
    [[nodiscard]] inline simd::Vector<T, simd::Dynamic> lu_solve_dynamic(const LUDynamic<T> &f,
                                                                         const simd::Vector<T, simd::Dynamic> &b) {
        const std::size_t n = b.size();
        simd::Vector<T, simd::Dynamic> x(n);
        simd::Vector<T, simd::Dynamic> y(n);

        // Apply permutation: y = Pb
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = b[f.p[i]];
        }

        // Forward substitution: L z = Pb (store in y)
        for (std::size_t i = 0; i < n; ++i) {
            T sum = y[i];
            for (std::size_t j = 0; j < i; ++j) {
                sum -= f.l(i, j) * y[j];
            }
            y[i] = sum; // L has unit diagonal
        }

        // Back substitution: U x = z
        for (std::size_t ii = 0; ii < n; ++ii) {
            const std::size_t i = n - 1 - ii;
            T sum = y[i];
            for (std::size_t j = i + 1; j < n; ++j) {
                sum -= f.u(i, j) * x[j];
            }
            x[i] = sum / f.u(i, i);
        }

        return x;
    }

} // namespace optinum::lina
