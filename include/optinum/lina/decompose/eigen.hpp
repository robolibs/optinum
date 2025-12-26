#pragma once

// =============================================================================
// optinum/lina/decompose/eigen.hpp
// Symmetric eigen decomposition via Jacobi rotations (fixed-size)
// =============================================================================

#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace optinum::lina {

    template <typename T, std::size_t N> struct EigenSym {
        simd::Vector<T, N> values{};
        simd::Matrix<T, N, N> vectors{}; // columns are eigenvectors
        std::size_t iterations = 0;
    };

    namespace eigen_detail {

        template <typename T> [[nodiscard]] OPTINUM_INLINE T abs_val(T x) noexcept {
            return static_cast<T>(std::abs(x));
        }

        template <typename T, std::size_t N>
        OPTINUM_INLINE void rotate(simd::Matrix<T, N, N> &a, simd::Matrix<T, N, N> &v, std::size_t p, std::size_t q,
                                   T c, T s) noexcept {
            // Update A = J^T A J for symmetric A, and V = V J
            for (std::size_t i = 0; i < N; ++i) {
                if (i != p && i != q) {
                    const T aip = a(i, p);
                    const T aiq = a(i, q);
                    a(i, p) = c * aip - s * aiq;
                    a(p, i) = a(i, p);
                    a(i, q) = s * aip + c * aiq;
                    a(q, i) = a(i, q);
                }
            }

            const T app = a(p, p);
            const T aqq = a(q, q);
            const T apq = a(p, q);

            a(p, p) = c * c * app - T{2} * s * c * apq + s * s * aqq;
            a(q, q) = s * s * app + T{2} * s * c * apq + c * c * aqq;
            a(p, q) = T{};
            a(q, p) = T{};

            // Update eigenvector columns: V[:,p] and V[:,q]
            // Columns are contiguous in column-major layout, use SIMD
            if constexpr (N >= 8) {
                alignas(32) T temp_p[N];
                alignas(32) T temp_q[N];
                alignas(32) T scaled_p[N];
                alignas(32) T scaled_q[N];

                // Load columns
                const T *col_p = v.data() + p * N;
                const T *col_q = v.data() + q * N;

                // temp_p = c * col_p - s * col_q
                simd::backend::mul_scalar<T, N>(scaled_p, col_p, c);
                simd::backend::mul_scalar<T, N>(scaled_q, col_q, s);
                simd::backend::sub<T, N>(temp_p, scaled_p, scaled_q);

                // temp_q = s * col_p + c * col_q
                simd::backend::mul_scalar<T, N>(scaled_p, col_p, s);
                simd::backend::mul_scalar<T, N>(scaled_q, col_q, c);
                simd::backend::add<T, N>(temp_q, scaled_p, scaled_q);

                // Store back (direct pointer copy)
                for (std::size_t i = 0; i < N; ++i) {
                    v(i, p) = temp_p[i];
                    v(i, q) = temp_q[i];
                }
            } else {
                // Scalar fallback for small N
                for (std::size_t i = 0; i < N; ++i) {
                    const T vip = v(i, p);
                    const T viq = v(i, q);
                    v(i, p) = c * vip - s * viq;
                    v(i, q) = s * vip + c * viq;
                }
            }
        }

    } // namespace eigen_detail

    template <typename T, std::size_t N>
    [[nodiscard]] EigenSym<T, N> eigen_sym(const simd::Matrix<T, N, N> &a, std::size_t max_sweeps = 64) noexcept {
        static_assert(std::is_floating_point_v<T>, "eigen_sym() currently requires floating-point T");

        EigenSym<T, N> out;
        simd::Matrix<T, N, N> A = a;
        out.vectors = simd::identity<T, N>();

        for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
            // Find largest off-diagonal element
            std::size_t p = 0, q = 1;
            T max_off = eigen_detail::abs_val(A(0, 1));
            for (std::size_t i = 0; i < N; ++i) {
                for (std::size_t j = i + 1; j < N; ++j) {
                    const T v = eigen_detail::abs_val(A(i, j));
                    if (v > max_off) {
                        max_off = v;
                        p = i;
                        q = j;
                    }
                }
            }

            if (max_off == T{}) {
                out.iterations = sweep;
                break;
            }

            const T app = A(p, p);
            const T aqq = A(q, q);
            const T apq = A(p, q);

            // Compute rotation
            const T tau = (aqq - app) / (T{2} * apq);
            const T t = (tau >= T{}) ? (T{1} / (tau + std::sqrt(T{1} + tau * tau)))
                                     : (T{-1} / (-tau + std::sqrt(T{1} + tau * tau)));
            const T c = T{1} / std::sqrt(T{1} + t * t);
            const T s = t * c;

            eigen_detail::rotate(A, out.vectors, p, q, c, s);
            out.iterations = sweep + 1;
        }

        for (std::size_t i = 0; i < N; ++i) {
            out.values[i] = A(i, i);
        }

        return out;
    }

} // namespace optinum::lina
