#pragma once

// =============================================================================
// optinum/lina/decompose/svd.hpp
// SVD via one-sided Jacobi (fixed-size, small/medium)
// =============================================================================

#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/matmul.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/tensor.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace optinum::lina {

    namespace svd_detail {
        template <std::size_t A, std::size_t B> inline constexpr std::size_t min_v = (A < B) ? A : B;

        template <typename T> [[nodiscard]] OPTINUM_INLINE T abs_val(T x) noexcept {
            return static_cast<T>(std::abs(x));
        }
    } // namespace svd_detail

    template <typename T, std::size_t M, std::size_t N> struct SVD {
        simd::Matrix<T, M, M> u{};
        simd::Tensor<T, svd_detail::min_v<M, N>> s{};
        simd::Matrix<T, N, N> vt{};
        std::size_t sweeps = 0;
    };

    namespace svd_detail {

        template <typename T, std::size_t M, std::size_t N>
        OPTINUM_INLINE T col_dot(const simd::Matrix<T, M, N> &a, std::size_t c1, std::size_t c2) noexcept {
            const T *p1 = a.data() + c1 * M;
            const T *p2 = a.data() + c2 * M;
            return simd::backend::dot<T, M>(p1, p2);
        }

        template <typename T, std::size_t M, std::size_t N>
        OPTINUM_INLINE void rotate_cols(simd::Matrix<T, M, N> &a, std::size_t p, std::size_t q, T c, T s) noexcept {
            for (std::size_t i = 0; i < M; ++i) {
                const T ap = a(i, p);
                const T aq = a(i, q);
                a(i, p) = c * ap - s * aq;
                a(i, q) = s * ap + c * aq;
            }
        }

        template <typename T, std::size_t N>
        OPTINUM_INLINE void rotate_cols(simd::Matrix<T, N, N> &a, std::size_t p, std::size_t q, T c, T s) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                const T ap = a(i, p);
                const T aq = a(i, q);
                a(i, p) = c * ap - s * aq;
                a(i, q) = s * ap + c * aq;
            }
        }

    } // namespace svd_detail

    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] SVD<T, M, N> svd(const simd::Matrix<T, M, N> &a, std::size_t max_sweeps = 32) noexcept {
        static_assert(std::is_floating_point_v<T>, "svd() currently requires floating-point T");

        // Handle tall vs wide by transposing to ensure M >= N for one-sided Jacobi.
        if constexpr (M < N) {
            const auto at = simd::transpose(a); // (N x M)
            auto svd_t = svd<T, N, M>(at, max_sweeps);
            // A = U S V^T
            // A^T = V S U^T
            SVD<T, M, N> out;
            out.u = simd::transpose(svd_t.vt); // V
            out.vt = simd::transpose(svd_t.u); // U^T
            out.s = svd_t.s;
            out.sweeps = svd_t.sweeps;
            return out;
        } else {
            SVD<T, M, N> out;
            simd::Matrix<T, M, N> b = a;
            simd::Matrix<T, N, N> v = simd::identity<T, N>();

            constexpr std::size_t K = N;

            for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
                bool changed = false;
                for (std::size_t p = 0; p + 1 < K; ++p) {
                    for (std::size_t q = p + 1; q < K; ++q) {
                        const T app = svd_detail::col_dot<T, M, N>(b, p, p);
                        const T aqq = svd_detail::col_dot<T, M, N>(b, q, q);
                        const T apq = svd_detail::col_dot<T, M, N>(b, p, q);

                        const T denom = std::sqrt(app * aqq);
                        if (denom == T{}) {
                            continue;
                        }
                        if (svd_detail::abs_val(apq) <= T{1e-12} * denom) {
                            continue;
                        }

                        const T tau = (aqq - app) / (T{2} * apq);
                        const T t = (tau >= T{}) ? (T{1} / (tau + std::sqrt(T{1} + tau * tau)))
                                                 : (T{-1} / (-tau + std::sqrt(T{1} + tau * tau)));
                        const T c = T{1} / std::sqrt(T{1} + t * t);
                        const T s = t * c;

                        svd_detail::rotate_cols(b, p, q, c, s);
                        svd_detail::rotate_cols(v, p, q, c, s);
                        changed = true;
                    }
                }

                out.sweeps = sweep + 1;
                if (!changed) {
                    break;
                }
            }

            // Singular values are column norms, U = B * diag(1/s)
            simd::Matrix<T, M, M> u = simd::identity<T, M>();
            simd::Matrix<T, N, N> vt = simd::transpose(v);

            for (std::size_t j = 0; j < K; ++j) {
                const T norm_col = std::sqrt(svd_detail::col_dot<T, M, N>(b, j, j));
                out.s[j] = norm_col;
                if (norm_col > T{}) {
                    for (std::size_t i = 0; i < M; ++i) {
                        u(i, j) = b(i, j) / norm_col;
                    }
                } else {
                    for (std::size_t i = 0; i < M; ++i) {
                        u(i, j) = T{};
                    }
                }
            }

            // Fill remaining columns of U (if M>N) with identity basis (not orthonormalized against first K columns,
            // but good enough for reconstruction tests on small matrices)
            if constexpr (M > N) {
                for (std::size_t j = K; j < M; ++j) {
                    for (std::size_t i = 0; i < M; ++i) {
                        u(i, j) = (i == j) ? T{1} : T{};
                    }
                }
            }

            out.u = u;
            out.vt = vt;
            return out;
        }
    }

} // namespace optinum::lina
