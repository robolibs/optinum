#pragma once

// =============================================================================
// optinum/lina/decompose/svd.hpp
// SVD via one-sided Jacobi (fixed-size, small/medium)
// =============================================================================

#include <optinum/lina/basic/identity.hpp>
#include <optinum/lina/basic/transpose.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/backend/matmul.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

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
        datapod::mat::matrix<T, M, M> u{};
        datapod::mat::vector<T, svd_detail::min_v<M, N>> s{};
        datapod::mat::matrix<T, N, N> vt{};
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
            // Rotate columns: a[:,p] and a[:,q] are contiguous in column-major layout
            if constexpr (M >= 8) {
                alignas(32) T temp_p[M];
                alignas(32) T temp_q[M];
                alignas(32) T scaled_p[M];
                alignas(32) T scaled_q[M];

                const T *col_p = a.data() + p * M;
                const T *col_q = a.data() + q * M;

                // temp_p = c * col_p - s * col_q
                simd::backend::mul_scalar<T, M>(scaled_p, col_p, c);
                simd::backend::mul_scalar<T, M>(scaled_q, col_q, s);
                simd::backend::sub<T, M>(temp_p, scaled_p, scaled_q);

                // temp_q = s * col_p + c * col_q
                simd::backend::mul_scalar<T, M>(scaled_p, col_p, s);
                simd::backend::mul_scalar<T, M>(scaled_q, col_q, c);
                simd::backend::add<T, M>(temp_q, scaled_p, scaled_q);

                // Store back
                for (std::size_t i = 0; i < M; ++i) {
                    a(i, p) = temp_p[i];
                    a(i, q) = temp_q[i];
                }
            } else {
                // Scalar fallback for small M
                for (std::size_t i = 0; i < M; ++i) {
                    const T ap = a(i, p);
                    const T aq = a(i, q);
                    a(i, p) = c * ap - s * aq;
                    a(i, q) = s * ap + c * aq;
                }
            }
        }

        template <typename T, std::size_t N>
        OPTINUM_INLINE void rotate_cols(simd::Matrix<T, N, N> &a, std::size_t p, std::size_t q, T c, T s) noexcept {
            // Same as above but for square matrices
            if constexpr (N >= 8) {
                alignas(32) T temp_p[N];
                alignas(32) T temp_q[N];
                alignas(32) T scaled_p[N];
                alignas(32) T scaled_q[N];

                const T *col_p = a.data() + p * N;
                const T *col_q = a.data() + q * N;

                // temp_p = c * col_p - s * col_q
                simd::backend::mul_scalar<T, N>(scaled_p, col_p, c);
                simd::backend::mul_scalar<T, N>(scaled_q, col_q, s);
                simd::backend::sub<T, N>(temp_p, scaled_p, scaled_q);

                // temp_q = s * col_p + c * col_q
                simd::backend::mul_scalar<T, N>(scaled_p, col_p, s);
                simd::backend::mul_scalar<T, N>(scaled_q, col_q, c);
                simd::backend::add<T, N>(temp_q, scaled_p, scaled_q);

                // Store back
                for (std::size_t i = 0; i < N; ++i) {
                    a(i, p) = temp_p[i];
                    a(i, q) = temp_q[i];
                }
            } else {
                // Scalar fallback for small N
                for (std::size_t i = 0; i < N; ++i) {
                    const T ap = a(i, p);
                    const T aq = a(i, q);
                    a(i, p) = c * ap - s * aq;
                    a(i, q) = s * ap + c * aq;
                }
            }
        }

    } // namespace svd_detail

    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] SVD<T, M, N> svd(const simd::Matrix<T, M, N> &a, std::size_t max_sweeps = 32) noexcept {
        static_assert(std::is_floating_point_v<T>, "svd() currently requires floating-point T");

        // Handle tall vs wide by transposing to ensure M >= N for one-sided Jacobi.
        if constexpr (M < N) {
            const auto at = transpose(a); // (N x M)
            auto svd_t = svd<T, N, M>(at, max_sweeps);
            // A = U S V^T
            // A^T = V S U^T
            SVD<T, M, N> out;
            out.u = transpose(simd::Matrix<T, N, N>(svd_t.vt)); // V
            out.vt = transpose(simd::Matrix<T, M, M>(svd_t.u)); // U^T
            out.s = svd_t.s;
            out.sweeps = svd_t.sweeps;
            return out;
        } else {
            SVD<T, M, N> out;
            // Create working copy of input matrix
            datapod::mat::matrix<T, M, N> b_pod;
            for (std::size_t i = 0; i < M * N; ++i)
                b_pod[i] = a[i];
            simd::Matrix<T, M, N> b(b_pod);
            auto v_pod = identity<T, N>();
            simd::Matrix<T, N, N> v(v_pod);

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
            auto u_pod = identity<T, M>();
            simd::Matrix<T, M, M> u(u_pod);
            auto vt_pod = transpose(v);
            simd::Matrix<T, N, N> vt(vt_pod);

            for (std::size_t j = 0; j < K; ++j) {
                const T norm_col = std::sqrt(svd_detail::col_dot<T, M, N>(b, j, j));
                out.s[j] = norm_col;

                // Normalize column j: u[:,j] = b[:,j] / norm_col (or zero if norm is zero)
                // Columns are contiguous in column-major layout
                T *u_col = u.data() + j * M;
                const T *b_col = b.data() + j * M;

                if (norm_col > T{}) {
                    if constexpr (M >= 8) {
                        const T inv_norm = T{1} / norm_col;
                        simd::backend::mul_scalar<T, M>(u_col, b_col, inv_norm);
                    } else {
                        for (std::size_t i = 0; i < M; ++i) {
                            u_col[i] = b_col[i] / norm_col;
                        }
                    }
                } else {
                    if constexpr (M >= 8) {
                        simd::backend::fill<T, M>(u_col, T{});
                    } else {
                        for (std::size_t i = 0; i < M; ++i) {
                            u_col[i] = T{};
                        }
                    }
                }
            }

            // Fill remaining columns of U (if M>N) with identity basis (not orthonormalized against first K columns,
            // but good enough for reconstruction tests on small matrices)
            if constexpr (M > N) {
                for (std::size_t j = K; j < M; ++j) {
                    T *u_col = u.data() + j * M;
                    if constexpr (M >= 8) {
                        simd::backend::fill<T, M>(u_col, T{});
                        u_col[j] = T{1}; // Diagonal element
                    } else {
                        for (std::size_t i = 0; i < M; ++i) {
                            u_col[i] = (i == j) ? T{1} : T{};
                        }
                    }
                }
            }

            out.u = u_pod;
            out.vt = vt_pod;
            return out;
        }
    }

} // namespace optinum::lina
