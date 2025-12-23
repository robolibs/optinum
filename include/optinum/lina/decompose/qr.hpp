#pragma once

// =============================================================================
// optinum/lina/decompose/qr.hpp
// QR decomposition via Householder reflections (fixed-size)
// =============================================================================

#include <optinum/simd/matrix.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace optinum::lina {

    template <typename T, std::size_t M, std::size_t N> struct QR {
        simd::Matrix<T, M, M> q{};
        simd::Matrix<T, M, N> r{};
    };

    namespace qr_detail {

        template <typename T, std::size_t M, std::size_t N>
        OPTINUM_INLINE void apply_householder_left(simd::Matrix<T, M, N> &a, const T *w, T beta,
                                                   std::size_t k) noexcept {
            // a = (I - beta w w^T) a; w has non-zero only for indices >= k
            for (std::size_t col = k; col < N; ++col) {
                T dot{};
                for (std::size_t i = k; i < M; ++i) {
                    dot += w[i] * a(i, col);
                }
                const T s = beta * dot;
                for (std::size_t i = k; i < M; ++i) {
                    a(i, col) -= s * w[i];
                }
            }
        }

        template <typename T, std::size_t M>
        OPTINUM_INLINE void apply_householder_right(simd::Matrix<T, M, M> &q, const T *w, T beta,
                                                    std::size_t k) noexcept {
            // q = q (I - beta w w^T)
            for (std::size_t row = 0; row < M; ++row) {
                T dot{};
                for (std::size_t i = k; i < M; ++i) {
                    dot += q(row, i) * w[i];
                }
                const T s = beta * dot;
                for (std::size_t i = k; i < M; ++i) {
                    q(row, i) -= s * w[i];
                }
            }
        }

    } // namespace qr_detail

    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] QR<T, M, N> qr(const simd::Matrix<T, M, N> &a) noexcept {
        static_assert(std::is_floating_point_v<T>, "qr() currently requires floating-point T");

        QR<T, M, N> out;
        out.r = a;
        out.q = simd::identity<T, M>();

        constexpr std::size_t K = (M < N) ? M : N;
        T w[M]; // Householder vector (full length, but zeros before k)

        for (std::size_t k = 0; k < K; ++k) {
            // Compute norm of x = R[k:M-1, k]
            T norm_x{};
            for (std::size_t i = k; i < M; ++i) {
                const T v = out.r(i, k);
                norm_x += v * v;
            }
            norm_x = std::sqrt(norm_x);
            if (norm_x == T{}) {
                continue;
            }

            // Build Householder vector w
            for (std::size_t i = 0; i < M; ++i)
                w[i] = T{};

            const T x0 = out.r(k, k);
            const T alpha = (x0 >= T{}) ? -norm_x : norm_x;
            w[k] = x0 - alpha;
            for (std::size_t i = k + 1; i < M; ++i) {
                w[i] = out.r(i, k);
            }

            // beta = 2 / (w^T w)
            T wtw{};
            for (std::size_t i = k; i < M; ++i) {
                wtw += w[i] * w[i];
            }
            if (wtw == T{}) {
                continue;
            }
            const T beta = T{2} / wtw;

            // Apply to R and Q
            qr_detail::apply_householder_left(out.r, w, beta, k);
            qr_detail::apply_householder_right(out.q, w, beta, k);

            // Set R(k,k) = alpha and below diagonal to 0 for cleanliness
            out.r(k, k) = alpha;
            for (std::size_t i = k + 1; i < M; ++i) {
                out.r(i, k) = T{};
            }
        }

        return out;
    }

} // namespace optinum::lina
