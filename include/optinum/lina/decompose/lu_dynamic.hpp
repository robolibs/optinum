#pragma once

// =============================================================================
// optinum/lina/decompose/lu_dynamic.hpp
// LU decomposition with partial pivoting (dynamic/runtime-size)
// Uses SIMD for column operations (column-major layout).
// =============================================================================

#include <datapod/matrix.hpp>
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

    /**
     * @brief Result of dynamic LU decomposition with partial pivoting
     *
     * Stores L (lower triangular with unit diagonal), U (upper triangular),
     * and permutation vector P such that P*A = L*U.
     * Uses owning storage types (dp::mat::matrix/vector) with simd views.
     */
    template <typename T> struct LUDynamic {
        // Owning storage
        dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> l_storage;
        dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> u_storage;
        dp::mat::vector<std::size_t, dp::mat::Dynamic> p_storage;

        // Views for SIMD operations
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> l;
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> u;
        simd::Vector<std::size_t, simd::Dynamic> p;

        int sign = 1;
        bool singular = false;

        LUDynamic() = default;

        explicit LUDynamic(std::size_t n)
            : l_storage(n, n), u_storage(n, n), p_storage(n), l(l_storage), u(u_storage), p(p_storage) {
            l.fill(T{});
            u.fill(T{});
            for (std::size_t i = 0; i < n; ++i) {
                p[i] = i;
            }
        }

        // Copy constructor - must re-establish views after copying storage
        LUDynamic(const LUDynamic &other)
            : l_storage(other.l_storage), u_storage(other.u_storage), p_storage(other.p_storage), l(l_storage),
              u(u_storage), p(p_storage), sign(other.sign), singular(other.singular) {}

        // Move constructor - must re-establish views after moving storage
        LUDynamic(LUDynamic &&other) noexcept
            : l_storage(std::move(other.l_storage)), u_storage(std::move(other.u_storage)),
              p_storage(std::move(other.p_storage)), l(l_storage), u(u_storage), p(p_storage), sign(other.sign),
              singular(other.singular) {}

        // Copy assignment
        LUDynamic &operator=(const LUDynamic &other) {
            if (this != &other) {
                l_storage = other.l_storage;
                u_storage = other.u_storage;
                p_storage = other.p_storage;
                l = simd::Matrix<T, simd::Dynamic, simd::Dynamic>(l_storage);
                u = simd::Matrix<T, simd::Dynamic, simd::Dynamic>(u_storage);
                p = simd::Vector<std::size_t, simd::Dynamic>(p_storage);
                sign = other.sign;
                singular = other.singular;
            }
            return *this;
        }

        // Move assignment
        LUDynamic &operator=(LUDynamic &&other) noexcept {
            if (this != &other) {
                l_storage = std::move(other.l_storage);
                u_storage = std::move(other.u_storage);
                p_storage = std::move(other.p_storage);
                l = simd::Matrix<T, simd::Dynamic, simd::Dynamic>(l_storage);
                u = simd::Matrix<T, simd::Dynamic, simd::Dynamic>(u_storage);
                p = simd::Vector<std::size_t, simd::Dynamic>(p_storage);
                sign = other.sign;
                singular = other.singular;
            }
            return *this;
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

        // SIMD column swap for column-major matrix
        // In column-major: column j is at data[j * n], contiguous
        // Swapping rows means swapping elements at positions [col * n + r1] and [col * n + r2] for each column
        template <typename T>
        inline void swap_rows_colmajor(T *data, std::size_t n, std::size_t ncols, std::size_t r1,
                                       std::size_t r2) noexcept {
            if (r1 == r2)
                return;

            // For each column, swap elements at row r1 and r2
            for (std::size_t col = 0; col < ncols; ++col) {
                T tmp = data[col * n + r1];
                data[col * n + r1] = data[col * n + r2];
                data[col * n + r2] = tmp;
            }
        }

        // SIMD axpy for column update in column-major layout
        // dst_col[start:n] -= scale * src_col[start:n]
        // In column-major, column k starts at data[k * n], elements are contiguous
        template <typename T>
        inline void axpy_col_simd(T *dst_col, const T *src_col, T scale, std::size_t start, std::size_t n) noexcept {
            constexpr std::size_t W = simd::backend::default_pack_width<T>();
            const std::size_t len = n - start;
            const std::size_t main = (len / W) * W;

            for (std::size_t i = 0; i < main; i += W) {
                auto vd = simd::pack<T, W>::loadu(dst_col + start + i);
                auto vs = simd::pack<T, W>::loadu(src_col + start + i);
                (simd::pack<T, W>::fma(simd::pack<T, W>(-scale), vs, vd)).storeu(dst_col + start + i);
            }

            for (std::size_t i = main; i < len; ++i) {
                dst_col[start + i] -= scale * src_col[start + i];
            }
        }

        // SIMD scale column: dst_col[start:n] /= scalar
        template <typename T> inline void scale_col_simd(T *col, T scalar, std::size_t start, std::size_t n) noexcept {
            constexpr std::size_t W = simd::backend::default_pack_width<T>();
            const std::size_t len = n - start;
            const std::size_t main = (len / W) * W;

            const simd::pack<T, W> vs(scalar);

            for (std::size_t i = 0; i < main; i += W) {
                auto vd = simd::pack<T, W>::loadu(col + start + i);
                (vd / vs).storeu(col + start + i);
            }

            for (std::size_t i = main; i < len; ++i) {
                col[start + i] /= scalar;
            }
        }

    } // namespace lu_dynamic_detail

    /**
     * @brief Compute LU decomposition with partial pivoting for dynamic-size matrix
     *
     * Uses SIMD for column operations (column-major layout).
     * Performs in-place decomposition where L is stored below diagonal and U on/above.
     *
     * @tparam T Scalar type (float, double)
     * @param a Input matrix (n x n)
     * @return LUDynamic<T> containing L, U, and permutation
     */
    template <typename T>
    [[nodiscard]] inline LUDynamic<T> lu_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a) {
        const std::size_t n = a.rows();
        LUDynamic<T> out(n);

        // Copy input to working matrix (owning storage + view)
        dp::mat::matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> lu_storage(n, n);
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> lu_mat(lu_storage);
        for (std::size_t i = 0; i < n * n; ++i) {
            lu_mat[i] = a[i];
        }

        T *lu_data = lu_mat.data();

        for (std::size_t k = 0; k < n; ++k) {
            // Find pivot in column k, rows k to n-1
            // In column-major: column k is at lu_data[k * n], element (i, k) is at lu_data[k * n + i]
            std::size_t piv = k;
            T max_val = lu_dynamic_detail::abs_val(lu_data[k * n + k]);
            for (std::size_t i = k + 1; i < n; ++i) {
                const T v = lu_dynamic_detail::abs_val(lu_data[k * n + i]);
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
                // Swap rows piv and k across all columns
                lu_dynamic_detail::swap_rows_colmajor(lu_data, n, n, piv, k);
                std::size_t tmp = out.p[piv];
                out.p[piv] = out.p[k];
                out.p[k] = tmp;
                out.sign = -out.sign;
            }

            const T pivval = lu_data[k * n + k]; // A(k, k) in column-major

            // Scale column k below diagonal: L[i,k] = A[i,k] / A[k,k] for i > k
            // In column-major, these are contiguous: lu_data[k * n + (k+1)] to lu_data[k * n + (n-1)]
            for (std::size_t i = k + 1; i < n; ++i) {
                lu_data[k * n + i] /= pivval;
            }

            // Update submatrix: A[i,j] -= L[i,k] * A[k,j] for i > k, j > k
            // For each column j > k, update rows i > k
            for (std::size_t j = k + 1; j < n; ++j) {
                const T ukj = lu_data[j * n + k]; // A(k, j) = U[k,j]
                // Column j: subtract ukj * column k (below diagonal part)
                // A[i,j] -= L[i,k] * U[k,j] for i > k
                // L[i,k] is at lu_data[k * n + i]
                // A[i,j] is at lu_data[j * n + i]
                T *col_j = lu_data + j * n;
                const T *col_k = lu_data + k * n;
                lu_dynamic_detail::axpy_col_simd(col_j, col_k, ukj, k + 1, n);
            }
        }

        // Split into L and U
        // L: lower triangular with unit diagonal
        // U: upper triangular (including diagonal)
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                if (i == j) {
                    out.l(i, j) = T{1};
                    out.u(i, j) = lu_data[j * n + i];
                } else if (i > j) {
                    // Below diagonal: L
                    out.l(i, j) = lu_data[j * n + i];
                } else {
                    // Above diagonal: U
                    out.u(i, j) = lu_data[j * n + i];
                }
            }
        }

        return out;
    }

    /**
     * @brief Solve Ax = b using precomputed LU decomposition
     *
     * Uses SIMD for dot products in forward/back substitution.
     *
     * @tparam T Scalar type
     * @param f LU decomposition result
     * @param b Right-hand side vector
     * @return Solution vector x
     */
    template <typename T>
    [[nodiscard]] inline dp::mat::vector<T, dp::mat::Dynamic>
    lu_solve_dynamic(const LUDynamic<T> &f, const simd::Vector<T, simd::Dynamic> &b) {
        const std::size_t n = b.size();

        // Owning storage for result vectors
        dp::mat::vector<T, dp::mat::Dynamic> x_storage(n);
        dp::mat::vector<T, dp::mat::Dynamic> y_storage(n);
        simd::Vector<T, simd::Dynamic> x(x_storage);
        simd::Vector<T, simd::Dynamic> y(y_storage);

        // Apply permutation: y = Pb
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = b[f.p[i]];
        }

        // Forward substitution: L z = Pb (store in y)
        // y[i] = y[i] - sum_{j<i} L[i,j] * y[j]
        for (std::size_t i = 0; i < n; ++i) {
            T sum = T{};
            for (std::size_t j = 0; j < i; ++j) {
                sum += f.l(i, j) * y[j];
            }
            y[i] -= sum;
        }

        // Back substitution: U x = z
        // x[i] = (y[i] - sum_{j>i} U[i,j] * x[j]) / U[i,i]
        for (std::size_t ii = 0; ii < n; ++ii) {
            const std::size_t i = n - 1 - ii;
            T sum = y[i];
            for (std::size_t j = i + 1; j < n; ++j) {
                sum -= f.u(i, j) * x[j];
            }
            x[i] = sum / f.u(i, i);
        }

        return x_storage;
    }

} // namespace optinum::lina
