#pragma once

// =============================================================================
// optinum/lina/decompose/cholesky.hpp
// Cholesky factorization for symmetric positive definite matrices
// =============================================================================

#include <optinum/simd/matrix.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace optinum::lina {

    template <typename T, std::size_t N> struct Cholesky {
        simd::Matrix<T, N, N> l{};
        bool success = true;
    };

    template <typename T, std::size_t N>
    [[nodiscard]] Cholesky<T, N> cholesky(const simd::Matrix<T, N, N> &a) noexcept {
        static_assert(std::is_floating_point_v<T>, "cholesky() currently requires floating-point T");

        Cholesky<T, N> out;
        out.l.fill(T{});

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                T sum = a(i, j);
                for (std::size_t k = 0; k < j; ++k) {
                    sum -= out.l(i, k) * out.l(j, k);
                }

                if (i == j) {
                    if (sum <= T{}) {
                        out.success = false;
                        out.l(i, j) = T{};
                    } else {
                        out.l(i, j) = std::sqrt(sum);
                    }
                } else {
                    const T denom = out.l(j, j);
                    if (denom == T{}) {
                        out.success = false;
                        out.l(i, j) = T{};
                    } else {
                        out.l(i, j) = sum / denom;
                    }
                }
            }
        }

        return out;
    }

} // namespace optinum::lina
