#pragma once

// =============================================================================
// optinum/lina/decompose/qr.hpp
// QR decomposition via Householder reflections (fixed-size)
// Uses SIMD for column operations where memory is contiguous.
// =============================================================================

#include <optinum/lina/basic/identity.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/pack/pack.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace optinum::lina {

    template <typename T, std::size_t M, std::size_t N> struct QR {
        datapod::mat::matrix<T, M, M> q{};
        datapod::mat::matrix<T, M, N> r{};
    };

    namespace qr_detail {

        // SIMD dot product for partial column (contiguous in column-major)
        template <typename T>
        [[nodiscard]] OPTINUM_INLINE T dot_partial(const T *a, const T *b, std::size_t n) noexcept {
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

        // SIMD axpy for partial column: a[0:n] -= s * b[0:n]
        template <typename T> OPTINUM_INLINE void axpy_partial(T *a, const T *b, T s, std::size_t n) noexcept {
            if (n == 0)
                return;

            constexpr std::size_t W = simd::backend::preferred_simd_lanes<T, 64>();
            const std::size_t main = (n / W) * W;

            const simd::pack<T, W> vs(s);
            for (std::size_t i = 0; i < main; i += W) {
                auto va = simd::pack<T, W>::loadu(a + i);
                auto vb = simd::pack<T, W>::loadu(b + i);
                // va = va - s * vb = va + (-s) * vb
                (simd::pack<T, W>::fma(simd::pack<T, W>(-s), vb, va)).storeu(a + i);
            }

            for (std::size_t i = main; i < n; ++i) {
                a[i] -= s * b[i];
            }
        }

        template <typename T, std::size_t M, std::size_t N>
        OPTINUM_INLINE void apply_householder_left(simd::Matrix<T, M, N> &a, const T *w, T beta,
                                                   std::size_t k) noexcept {
            // a = (I - beta w w^T) a; w has non-zero only for indices >= k
            // Column-major: a(k:M-1, col) is contiguous at &a(k, col)
            const std::size_t len = M - k;
            for (std::size_t col = k; col < N; ++col) {
                // dot = w[k:M-1] . a[k:M-1, col]
                T dot = dot_partial(&a(k, col), w + k, len);
                const T s = beta * dot;
                // a[k:M-1, col] -= s * w[k:M-1]
                axpy_partial(&a(k, col), w + k, s, len);
            }
        }

        template <typename T, std::size_t M>
        OPTINUM_INLINE void apply_householder_right(simd::Matrix<T, M, M> &q, const T *w, T beta,
                                                    std::size_t k) noexcept {
            // q = q (I - beta w w^T)
            // For each row of q: q[row, k:M-1] -= beta * (q[row, k:M-1] . w[k:M-1]) * w[k:M-1]
            // Row access is strided (M apart in column-major), so use scalar loop
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
    [[nodiscard]] inline QR<T, M, N> qr(const simd::Matrix<T, M, N> &a) noexcept {
        static_assert(std::is_floating_point_v<T>, "qr() currently requires floating-point T");

        QR<T, M, N> out;
        // Copy input to out.r
        for (std::size_t i = 0; i < M * N; ++i)
            out.r[i] = a[i];
        out.q = identity<T, M>();

        // Create views for in-place operations
        simd::Matrix<T, M, N> r_view(out.r);
        simd::Matrix<T, M, M> q_view(out.q);

        constexpr std::size_t K = (M < N) ? M : N;
        T w[M]; // Householder vector (full length, but zeros before k)

        for (std::size_t k = 0; k < K; ++k) {
            // Compute norm of x = R[k:M-1, k] using SIMD
            // Column k is contiguous at &out.r(k, k)
            const std::size_t len = M - k;
            T norm_x = qr_detail::dot_partial(&r_view(k, k), &r_view(k, k), len);
            norm_x = std::sqrt(norm_x);
            if (norm_x == T{}) {
                continue;
            }

            // Build Householder vector w
            for (std::size_t i = 0; i < M; ++i)
                w[i] = T{};

            const T x0 = r_view(k, k);
            const T alpha = (x0 >= T{}) ? -norm_x : norm_x;
            w[k] = x0 - alpha;
            for (std::size_t i = k + 1; i < M; ++i) {
                w[i] = r_view(i, k);
            }

            // beta = 2 / (w^T w) using SIMD
            T wtw = qr_detail::dot_partial(w + k, w + k, len);
            if (wtw == T{}) {
                continue;
            }
            const T beta = T{2} / wtw;

            // Apply to R and Q
            qr_detail::apply_householder_left(r_view, w, beta, k);
            qr_detail::apply_householder_right(q_view, w, beta, k);

            // Set R(k,k) = alpha and below diagonal to 0 for cleanliness
            r_view(k, k) = alpha;
            for (std::size_t i = k + 1; i < M; ++i) {
                r_view(i, k) = T{};
            }
        }

        return out;
    }

} // namespace optinum::lina
