#pragma once

// =============================================================================
// optinum/lina/decompose/qr_dynamic.hpp
// QR decomposition via Householder reflections (dynamic/runtime-size)
// =============================================================================

#include <optinum/simd/matrix.hpp>
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

        template <typename T>
        [[nodiscard]] inline T dot_partial(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &mat, std::size_t col,
                                           std::size_t start_row, std::size_t end_row) noexcept {
            T sum{};
            for (std::size_t i = start_row; i < end_row; ++i) {
                const T val = mat(i, col);
                sum += val * val;
            }
            return sum;
        }

        // Apply Householder reflection from the left to matrix R
        // H = I - beta * w * w^T
        // R = H * R
        template <typename T>
        inline void apply_householder_left(simd::Matrix<T, simd::Dynamic, simd::Dynamic> &r,
                                           const simd::Vector<T, simd::Dynamic> &w, T beta, std::size_t k) noexcept {
            const std::size_t m = r.rows();
            const std::size_t n = r.cols();

            for (std::size_t j = k; j < n; ++j) {
                T dot{};
                for (std::size_t i = k; i < m; ++i) {
                    dot += w[i] * r(i, j);
                }
                const T s = beta * dot;
                for (std::size_t i = k; i < m; ++i) {
                    r(i, j) -= s * w[i];
                }
            }
        }

        // Apply Householder reflection from the right to matrix Q
        // Q = Q * H = Q * (I - beta * w * w^T)
        template <typename T>
        inline void apply_householder_right(simd::Matrix<T, simd::Dynamic, simd::Dynamic> &q,
                                            const simd::Vector<T, simd::Dynamic> &w, T beta, std::size_t k) noexcept {
            const std::size_t m = q.rows();

            for (std::size_t i = 0; i < m; ++i) {
                T dot{};
                for (std::size_t j = k; j < m; ++j) {
                    dot += q(i, j) * w[j];
                }
                const T s = beta * dot;
                for (std::size_t j = k; j < m; ++j) {
                    q(i, j) -= s * w[j];
                }
            }
        }

    } // namespace qr_dynamic_detail

    /**
     * @brief Compute QR decomposition for dynamic-size matrix
     *
     * Uses Householder reflections to compute A = Q * R where:
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

        // Copy input to R
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                out.r(i, j) = a(i, j);
            }
        }

        const std::size_t k_max = (m < n) ? m : n;
        simd::Vector<T, simd::Dynamic> w(m);

        for (std::size_t k = 0; k < k_max; ++k) {
            // Compute norm of x = R[k:m-1, k]
            T norm_x = qr_dynamic_detail::dot_partial(out.r, k, k, m);
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

            // Compute beta = 2 / (w^T w)
            T wtw{};
            for (std::size_t i = k; i < m; ++i) {
                wtw += w[i] * w[i];
            }
            if (wtw == T{}) {
                continue;
            }
            const T beta = T{2} / wtw;

            // Apply Householder to R and Q
            qr_dynamic_detail::apply_householder_left(out.r, w, beta, k);
            qr_dynamic_detail::apply_householder_right(out.q, w, beta, k);

            // Set R(k,k) = alpha and below-diagonal to 0
            out.r(k, k) = alpha;
            for (std::size_t i = k + 1; i < m; ++i) {
                out.r(i, k) = T{};
            }
        }

        return out;
    }

} // namespace optinum::lina
