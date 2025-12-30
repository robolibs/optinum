#pragma once

// =============================================================================
// optinum/lina/decompose/cholesky.hpp
// Cholesky factorization for symmetric positive definite matrices
// =============================================================================

#include <datapod/matrix/matrix.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/matrix.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace optinum::lina {

    namespace dp = ::datapod;

    template <typename T, std::size_t N> struct Cholesky {
        dp::mat::matrix<T, N, N> l{}; // Owning storage for L
        bool success = true;
    };

    template <typename T, std::size_t N>
    [[nodiscard]] Cholesky<T, N> cholesky(const simd::Matrix<T, N, N> &a) noexcept {
        static_assert(std::is_floating_point_v<T>, "cholesky() currently requires floating-point T");

        Cholesky<T, N> out;
        // Initialize L to zero
        for (std::size_t i = 0; i < N * N; ++i) {
            out.l.data()[i] = T{};
        }

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                T sum = a(i, j);

                // Inner product: sum -= L[i, :j] · L[j, :j]
                // Note: Rows are strided in column-major layout (stride = N)
                // SIMD optimization requires contiguous data, so we extract to temp arrays
                // This is only beneficial for larger j (overhead of copy vs SIMD speedup)
                if (j >= 8) {
                    // For longer rows, use SIMD via temporary contiguous arrays
                    alignas(32) T row_i[N];
                    alignas(32) T row_j[N];

                    // Extract partial rows (columns 0..j-1)
                    for (std::size_t k = 0; k < j; ++k) {
                        row_i[k] = out.l(i, k);
                        row_j[k] = out.l(j, k);
                    }
                    // Pad remaining elements with zeros for SIMD safety
                    for (std::size_t k = j; k < N; ++k) {
                        row_i[k] = T{};
                        row_j[k] = T{};
                    }

                    // Use SIMD dot product on full arrays (zeros don't affect result)
                    sum -= simd::backend::dot<T, N>(row_i, row_j);
                } else {
                    // For short rows, scalar loop is more efficient
                    for (std::size_t k = 0; k < j; ++k) {
                        sum -= out.l(i, k) * out.l(j, k);
                    }
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

    // Overload for dp::mat::matrix input
    template <typename T, std::size_t N>
    [[nodiscard]] Cholesky<T, N> cholesky(const dp::mat::matrix<T, N, N> &a) noexcept {
        simd::Matrix<T, N, N> view(const_cast<dp::mat::matrix<T, N, N> &>(a));
        return cholesky(view);
    }

} // namespace optinum::lina
