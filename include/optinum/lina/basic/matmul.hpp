#pragma once

// =============================================================================
// optinum/lina/basic/matmul.hpp
// =============================================================================

#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/backend/matmul.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::lina {

    // Matrix-matrix multiplication: returns owning type dp::mat::matrix
    template <typename T, std::size_t R, std::size_t K, std::size_t C>
    [[nodiscard]] constexpr datapod::mat::matrix<T, R, C> matmul(const simd::Matrix<T, R, K> &a,
                                                                 const simd::Matrix<T, K, C> &b) noexcept {
        datapod::mat::matrix<T, R, C> result;
        simd::matmul_to(result.data(), a, b);
        return result;
    }

    // Mixed: simd::Matrix * dp::mat::matrix
    template <typename T, std::size_t R, std::size_t K, std::size_t C>
    [[nodiscard]] constexpr datapod::mat::matrix<T, R, C> matmul(const simd::Matrix<T, R, K> &a,
                                                                 const datapod::mat::matrix<T, K, C> &b) noexcept {
        datapod::mat::matrix<T, R, C> result;
        simd::Matrix<T, K, C> b_view(b);
        simd::matmul_to(result.data(), a, b_view);
        return result;
    }

    // Mixed: dp::mat::matrix * simd::Matrix
    template <typename T, std::size_t R, std::size_t K, std::size_t C>
    [[nodiscard]] constexpr datapod::mat::matrix<T, R, C> matmul(const datapod::mat::matrix<T, R, K> &a,
                                                                 const simd::Matrix<T, K, C> &b) noexcept {
        datapod::mat::matrix<T, R, C> result;
        simd::Matrix<T, R, K> a_view(a);
        simd::matmul_to(result.data(), a_view, b);
        return result;
    }

    // Owning: dp::mat::matrix * dp::mat::matrix
    template <typename T, std::size_t R, std::size_t K, std::size_t C>
    [[nodiscard]] constexpr datapod::mat::matrix<T, R, C> matmul(const datapod::mat::matrix<T, R, K> &a,
                                                                 const datapod::mat::matrix<T, K, C> &b) noexcept {
        datapod::mat::matrix<T, R, C> result;
        simd::backend::matmul<T, R, K, C>(result.data(), a.data(), b.data());
        return result;
    }

    // Matrix-vector multiplication
    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr datapod::mat::vector<T, R> matmul(const simd::Matrix<T, R, C> &a,
                                                              const simd::Vector<T, C> &x) noexcept {
        datapod::mat::vector<T, R> result;
        simd::matvec_to(result.data(), a, x);
        return result;
    }

    // Matrix-vector multiplication with owning vector
    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr datapod::mat::vector<T, R> matmul(const simd::Matrix<T, R, C> &a,
                                                              const datapod::mat::vector<T, C> &x) noexcept {
        datapod::mat::vector<T, R> result;
        simd::Vector<T, C> x_view(x);
        simd::matvec_to(result.data(), a, x_view);
        return result;
    }

    // =============================================================================
    // Matrix addition/subtraction for dp::mat::matrix types
    // (ADL doesn't find simd:: operators for datapod:: types)
    // =============================================================================

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr datapod::mat::matrix<T, R, C> add(const datapod::mat::matrix<T, R, C> &a,
                                                              const datapod::mat::matrix<T, R, C> &b) noexcept {
        datapod::mat::matrix<T, R, C> result;
        simd::backend::add<T, R * C>(result.data(), a.data(), b.data());
        return result;
    }

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr datapod::mat::matrix<T, R, C> add(const datapod::mat::matrix<T, R, C> &a,
                                                              const simd::Matrix<T, R, C> &b) noexcept {
        datapod::mat::matrix<T, R, C> result;
        simd::backend::add<T, R * C>(result.data(), a.data(), b.data());
        return result;
    }

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr datapod::mat::matrix<T, R, C> sub(const datapod::mat::matrix<T, R, C> &a,
                                                              const datapod::mat::matrix<T, R, C> &b) noexcept {
        datapod::mat::matrix<T, R, C> result;
        simd::backend::sub<T, R * C>(result.data(), a.data(), b.data());
        return result;
    }

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr datapod::mat::matrix<T, R, C> sub(const datapod::mat::matrix<T, R, C> &a,
                                                              const simd::Matrix<T, R, C> &b) noexcept {
        datapod::mat::matrix<T, R, C> result;
        simd::backend::sub<T, R * C>(result.data(), a.data(), b.data());
        return result;
    }

    // =============================================================================
    // Scalar multiplication for dp::mat::matrix types
    // =============================================================================

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr datapod::mat::matrix<T, R, C> scale(T scalar,
                                                                const datapod::mat::matrix<T, R, C> &a) noexcept {
        datapod::mat::matrix<T, R, C> result;
        simd::backend::mul_scalar<T, R * C>(result.data(), a.data(), scalar);
        return result;
    }

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr datapod::mat::matrix<T, R, C> scale(T scalar, const simd::Matrix<T, R, C> &a) noexcept {
        datapod::mat::matrix<T, R, C> result;
        simd::backend::mul_scalar<T, R * C>(result.data(), a.data(), scalar);
        return result;
    }

} // namespace optinum::lina
