#pragma once

// =============================================================================
// optinum/lina/solve/lstsq.hpp
// Least squares (fixed-size) via QR decomposition
// =============================================================================

#include <datapod/adapters/error.hpp>
#include <datapod/adapters/result.hpp>
#include <optinum/lina/decompose/qr.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/tensor.hpp>

namespace optinum::lina {

    namespace dp = ::datapod;

    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] dp::Result<simd::Tensor<T, N>, dp::Error> try_lstsq(const simd::Matrix<T, M, N> &a,
                                                                      const simd::Tensor<T, M> &b) noexcept {
        static_assert(std::is_floating_point_v<T>, "lstsq() currently requires floating-point T");
        static_assert(M >= N, "lstsq(A,b) expects M >= N (over/fully determined)");

        const auto f = qr<T, M, N>(a);

        // y = Q^T b
        simd::Tensor<T, M> y;
        for (std::size_t i = 0; i < M; ++i) {
            T sum{};
            for (std::size_t j = 0; j < M; ++j) {
                sum += f.q(j, i) * b[j];
            }
            y[i] = sum;
        }

        // Solve R(0:N-1,0:N-1) x = y(0:N-1) (upper triangular)
        simd::Tensor<T, N> x;
        for (std::size_t ii = 0; ii < N; ++ii) {
            const std::size_t i = N - 1 - ii;
            T sum = y[i];
            for (std::size_t j = i + 1; j < N; ++j) {
                sum -= f.r(i, j) * x[j];
            }
            const T diag = f.r(i, i);
            if (diag == T{}) {
                return dp::Result<simd::Tensor<T, N>, dp::Error>::err(dp::Error::invalid_argument("rank deficient R"));
            }
            x[i] = sum / diag;
        }

        return dp::Result<simd::Tensor<T, N>, dp::Error>::ok(x);
    }

    template <typename T, std::size_t M, std::size_t N>
    [[nodiscard]] simd::Tensor<T, N> lstsq(const simd::Matrix<T, M, N> &a, const simd::Tensor<T, M> &b) noexcept {
        auto r = try_lstsq<T, M, N>(a, b);
        if (r.is_ok()) {
            return r.value();
        }
        simd::Tensor<T, N> zero;
        zero.fill(T{});
        return zero;
    }

} // namespace optinum::lina
