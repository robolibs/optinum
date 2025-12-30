#pragma once

// =============================================================================
// optinum/lina/basic/norm.hpp
// =============================================================================

#include <datapod/matrix/matrix.hpp>
#include <datapod/matrix/vector.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>

namespace optinum::lina {

    namespace dp = ::datapod;

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr T dot(const simd::Vector<T, N> &a, const simd::Vector<T, N> &b) noexcept {
        return simd::dot(a, b);
    }

    template <typename T, std::size_t N> [[nodiscard]] T norm(const simd::Vector<T, N> &x) noexcept {
        return simd::norm(x);
    }

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] T norm_fro(const simd::Matrix<T, R, C> &a) noexcept {
        return simd::frobenius_norm(a);
    }

    // Overload for dp::mat::matrix (owning type)
    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] T norm_fro(const dp::mat::matrix<T, R, C> &a) noexcept {
        simd::Matrix<T, R, C> view(const_cast<dp::mat::matrix<T, R, C> &>(a));
        return simd::frobenius_norm(view);
    }

    // Cross product - returns owning type
    template <typename T>
    [[nodiscard]] constexpr dp::mat::vector<T, 3> cross(const simd::Vector<T, 3> &a,
                                                        const simd::Vector<T, 3> &b) noexcept {
        dp::mat::vector<T, 3> r;
        r[0] = a[1] * b[2] - a[2] * b[1];
        r[1] = a[2] * b[0] - a[0] * b[2];
        r[2] = a[0] * b[1] - a[1] * b[0];
        return r;
    }

    // Cross product for owning input types
    template <typename T>
    [[nodiscard]] constexpr dp::mat::vector<T, 3> cross(const dp::mat::vector<T, 3> &a,
                                                        const dp::mat::vector<T, 3> &b) noexcept {
        dp::mat::vector<T, 3> r;
        r[0] = a[1] * b[2] - a[2] * b[1];
        r[1] = a[2] * b[0] - a[0] * b[2];
        r[2] = a[0] * b[1] - a[1] * b[0];
        return r;
    }

    // Scale - returns owning type
    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr dp::mat::matrix<T, R, C> scale(const simd::Matrix<T, R, C> &a, T s) noexcept {
        dp::mat::matrix<T, R, C> result;
        for (std::size_t i = 0; i < R * C; ++i) {
            result[i] = a[i] * s;
        }
        return result;
    }

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr dp::mat::vector<T, N> scale(const simd::Vector<T, N> &x, T s) noexcept {
        dp::mat::vector<T, N> result;
        for (std::size_t i = 0; i < N; ++i) {
            result[i] = x[i] * s;
        }
        return result;
    }

    // axpy: alpha*x + y - returns owning type
    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr dp::mat::matrix<T, R, C> axpy(T alpha, const simd::Matrix<T, R, C> &x,
                                                          const simd::Matrix<T, R, C> &y) noexcept {
        dp::mat::matrix<T, R, C> result;
        for (std::size_t i = 0; i < R * C; ++i) {
            result[i] = alpha * x[i] + y[i];
        }
        return result;
    }

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr dp::mat::vector<T, N> axpy(T alpha, const simd::Vector<T, N> &x,
                                                       const simd::Vector<T, N> &y) noexcept {
        dp::mat::vector<T, N> result;
        for (std::size_t i = 0; i < N; ++i) {
            result[i] = alpha * x[i] + y[i];
        }
        return result;
    }

} // namespace optinum::lina
