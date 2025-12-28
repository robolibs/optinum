#pragma once

// =============================================================================
// optinum/lina/decompose/qr_dynamic.hpp
// QR decomposition via Householder reflections (dynamic/runtime-size)
// Uses SIMD for column operations where memory is contiguous.
// =============================================================================

#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/pack/pack.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace optinum::lina {

    /**
     * @brief Result of dynamic QR decomposition
     *
     * Stores Q (orthogonal, m x m) and R (upper triangular, m x n)
     * such that A = Q * R.
     */
    template <typename T> struct QRDynamic {
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> q;
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> r;

        QRDynamic() = default;

        QRDynamic(std::size_t m, std::size_t n) : q(m, m), r(m, n) {
            // Initialize Q to identity
            q.fill(T{});
            r.fill(T{});
            for (std::size_t i = 0; i < m; ++i) {
                q(i, i) = T{1};
            }
        }
    };

    namespace qr_dynamic_detail {

        // SIMD dot product for contiguous memory
        template <typename T> [[nodiscard]] inline T dot_simd(const T *a, const T *b, std::size_t n) noexcept {
            return simd::backend::dot_runtime<T>(a, b, n);
        }

        // SIMD axpy: a[0:n] -= s * b[0:n]
        template <typename T> inline void axpy_simd(T *a, const T *b, T s, std::size_t n) noexcept {
            constexpr std::size_t W = std::is_same_v<T, double> ? 4 : 8;
            const std::size_t main = (n / W) * W;

            const simd::pack<T, W> vs(s);

            for (std::size_t i = 0; i < main; i += W) {
                auto va = simd::pack<T, W>::loadu(a + i);
                auto vb = simd::pack<T, W>::loadu(b + i);
                // va = va - s * vb
                (simd::pack<T, W>::fma(simd::pack<T, W>(-s), vb, va)).storeu(a + i);
            }

            for (std::size_t i = main; i < n; ++i) {
                a[i] -= s * b[i];
            }
        }

        // Apply Householder reflection from the left to matrix R (column-major)
        // H = I - beta * w * w^T
        // R = H * R
        // For column-major: R[:,j] is contiguous at R.data() + j*m
        template <typename T>
        inline void apply_householder_left_simd(simd::Matrix<T, simd::Dynamic, simd::Dynamic> &r,
                                                const simd::Vector<T, simd::Dynamic> &w, T beta,
                                                std::size_t k) noexcept {
            const std::size_t m = r.rows();
            const std::size_t n = r.cols();
            const std::size_t len = m - k;

            // For each column j >= k
            for (std::size_t j = k; j < n; ++j) {
                // Column j starts at r.data() + j*m (column-major)
                // We need R[k:m-1, j] which is contiguous at r.data() + j*m + k
                T *col_ptr = r.data() + j * m + k;
                const T *w_ptr = w.data() + k;

                // SIMD dot: w[k:m-1] . R[k:m-1, j]
                T dot = dot_simd(w_ptr, col_ptr, len);
                const T s = beta * dot;

                // SIMD axpy: R[k:m-1, j] -= s * w[k:m-1]
                axpy_simd(col_ptr, w_ptr, s, len);
            }
        }

        // Apply Householder reflection from the right to matrix Q (column-major)
        // Q = Q * H = Q * (I - beta * w * w^T)
        // For column-major Q: Q[:,j] is contiguous
        // Q * H means: for each row i, Q[i,:] = Q[i,:] - beta * (Q[i,:] . w) * w^T
        // But row access is strided in column-major...
        // Alternative: Q * H column by column
        // (Q * H)[:,j] = Q[:,j] - beta * w[j] * (Q * w)
        // Let v = Q * w, then (Q * H)[:,j] = Q[:,j] - beta * w[j] * v
        template <typename T>
        inline void apply_householder_right_simd(simd::Matrix<T, simd::Dynamic, simd::Dynamic> &q,
                                                 const simd::Vector<T, simd::Dynamic> &w, T beta,
                                                 std::size_t k) noexcept {
            const std::size_t m = q.rows();

            // Compute v = Q * w (only w[k:m-1] is non-zero)
            // v[i] = sum_{j=k}^{m-1} Q[i,j] * w[j]
            simd::Vector<T, simd::Dynamic> v(m);
            v.fill(T{});

            // For each column j >= k, add w[j] * Q[:,j] to v
            constexpr std::size_t W = std::is_same_v<T, double> ? 4 : 8;
            for (std::size_t j = k; j < m; ++j) {
                const T wj = w[j];
                if (wj == T{})
                    continue;

                T *v_ptr = v.data();
                const T *q_col = q.data() + j * m;

                // SIMD: v += wj * Q[:,j]
                const std::size_t main = (m / W) * W;
                const simd::pack<T, W> vwj(wj);

                for (std::size_t i = 0; i < main; i += W) {
                    auto vv = simd::pack<T, W>::loadu(v_ptr + i);
                    auto vq = simd::pack<T, W>::loadu(q_col + i);
                    (simd::pack<T, W>::fma(vwj, vq, vv)).storeu(v_ptr + i);
                }
                for (std::size_t i = main; i < m; ++i) {
                    v_ptr[i] += wj * q_col[i];
                }
            }

            // Now update Q: Q[:,j] -= beta * w[j] * v for j >= k
            for (std::size_t j = k; j < m; ++j) {
                const T scale = beta * w[j];
                if (scale == T{})
                    continue;

                T *q_col = q.data() + j * m;
                const T *v_ptr = v.data();

                // SIMD: Q[:,j] -= scale * v
                axpy_simd(q_col, v_ptr, scale, m);
            }
        }

    } // namespace qr_dynamic_detail

    /**
     * @brief Compute QR decomposition for dynamic-size matrix
     *
     * Uses Householder reflections with SIMD-accelerated column operations.
     * A = Q * R where:
     * - Q is orthogonal (m x m)
     * - R is upper triangular (m x n)
     *
     * @tparam T Scalar type (float, double)
     * @param a Input matrix (m x n)
     * @return QRDynamic<T> containing Q and R
     */
    template <typename T>
    [[nodiscard]] inline QRDynamic<T> qr_dynamic(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &a) {
        static_assert(std::is_floating_point_v<T>, "qr_dynamic() requires floating-point type");

        const std::size_t m = a.rows();
        const std::size_t n = a.cols();

        QRDynamic<T> out(m, n);

        // Copy input to R (column-major, so columns are contiguous)
        for (std::size_t i = 0; i < m * n; ++i) {
            out.r[i] = a[i];
        }

        const std::size_t k_max = (m < n) ? m : n;
        simd::Vector<T, simd::Dynamic> w(m);

        for (std::size_t k = 0; k < k_max; ++k) {
            // Column k of R starts at out.r.data() + k*m
            // R[k:m-1, k] is contiguous at out.r.data() + k*m + k
            const T *col_k = out.r.data() + k * m + k;
            const std::size_t len = m - k;

            // SIMD norm: ||R[k:m-1, k]||
            T norm_x = qr_dynamic_detail::dot_simd(col_k, col_k, len);
            norm_x = std::sqrt(norm_x);

            if (norm_x == T{}) {
                continue;
            }

            // Build Householder vector w
            w.fill(T{});

            const T x0 = out.r(k, k);
            const T alpha = (x0 >= T{}) ? -norm_x : norm_x;
            w[k] = x0 - alpha;
            for (std::size_t i = k + 1; i < m; ++i) {
                w[i] = out.r(i, k);
            }

            // SIMD: beta = 2 / (w^T w)
            T wtw = qr_dynamic_detail::dot_simd(w.data() + k, w.data() + k, len);
            if (wtw == T{}) {
                continue;
            }
            const T beta = T{2} / wtw;

            // Apply Householder to R and Q with SIMD
            qr_dynamic_detail::apply_householder_left_simd(out.r, w, beta, k);
            qr_dynamic_detail::apply_householder_right_simd(out.q, w, beta, k);

            // Set R(k,k) = alpha and below-diagonal to 0
            out.r(k, k) = alpha;
            for (std::size_t i = k + 1; i < m; ++i) {
                out.r(i, k) = T{};
            }
        }

        return out;
    }

} // namespace optinum::lina
