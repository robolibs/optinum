#pragma once

// =============================================================================
// optinum/lina/solve/lstsq.hpp
// Least squares (fixed-size) via QR decomposition
// =============================================================================

#include <datapod/adapters/error.hpp>
#include <datapod/adapters/result.hpp>
#include <datapod/matrix/vector.hpp>
#include <optinum/lina/decompose/qr.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::lina {

    namespace dp = ::datapod;

    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] dp::Result<dp::mat::Vector<T, N>, dp::Error> try_lstsq(const simd::Matrix<T, M, N> &a,
                                                                         const simd::Vector<T, M> &b) noexcept {
        static_assert(std::is_floating_point_v<T>, "lstsq() currently requires floating-point T");
        static_assert(M >= N, "lstsq(A,b) expects M >= N (over/fully determined)");

        const auto f = qr<T, M, N>(a);

        // y = Q^T b
        // Each y[i] = dot(Q[:,i], b), where Q[:,i] is the i-th column of Q (contiguous in column-major layout)
        // Use SIMD backend for dot products
        dp::mat::Vector<T, M> y;
        for (std::size_t i = 0; i < M; ++i) {
            // Q[:,i] starts at f.q.data() + i*M (column-major layout)
            y[i] = simd::backend::dot<T, M>(f.q.data() + i * M, b.data());
        }

        // Solve R(0:N-1,0:N-1) x = y(0:N-1) (upper triangular)
        dp::mat::Vector<T, N> x;
        for (std::size_t ii = 0; ii < N; ++ii) {
            const std::size_t i = N - 1 - ii;
            T sum = y[i];
            for (std::size_t j = i + 1; j < N; ++j) {
                sum -= f.r(i, j) * x[j];
            }
            const T diag = f.r(i, i);
            if (diag == T{}) {
                return dp::Result<dp::mat::Vector<T, N>, dp::Error>::err(
                    dp::Error::invalid_argument("rank deficient R"));
            }
            x[i] = sum / diag;
        }

        return dp::Result<dp::mat::Vector<T, N>, dp::Error>::ok(x);
    }

    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] dp::mat::Vector<T, N> lstsq(const simd::Matrix<T, M, N> &a, const simd::Vector<T, M> &b) noexcept {
        auto r = try_lstsq<T, M, N>(a, b);
        if (r.is_ok()) {
            return r.value();
        }
        dp::mat::Vector<T, N> zero;
        for (std::size_t i = 0; i < N; ++i) {
            zero[i] = T{};
        }
        return zero;
    }

    // Overloads for dp::mat types
    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] dp::Result<dp::mat::Vector<T, N>, dp::Error> try_lstsq(const dp::mat::Matrix<T, M, N> &a,
                                                                         const dp::mat::Vector<T, M> &b) noexcept {
        simd::Matrix<T, M, N> a_view(const_cast<dp::mat::Matrix<T, M, N> &>(a));
        simd::Vector<T, M> b_view(const_cast<dp::mat::Vector<T, M> &>(b));
        return try_lstsq(a_view, b_view);
    }

    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] dp::mat::Vector<T, N> lstsq(const dp::mat::Matrix<T, M, N> &a,
                                              const dp::mat::Vector<T, M> &b) noexcept {
        simd::Matrix<T, M, N> a_view(const_cast<dp::mat::Matrix<T, M, N> &>(a));
        simd::Vector<T, M> b_view(const_cast<dp::mat::Vector<T, M> &>(b));
        return lstsq(a_view, b_view);
    }

} // namespace optinum::lina
