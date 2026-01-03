#pragma once

// =============================================================================
// optinum/lina/basic/adjoint.hpp
// Adjoint (adjugate) matrix computation
// =============================================================================
//
// The adjoint (or adjugate) of a square matrix A is the transpose of its cofactor matrix:
//   adj(A) = C^T
// where C is the cofactor matrix.
//
// Properties:
//   - A * adj(A) = adj(A) * A = det(A) * I
//   - If det(A) ≠ 0: A^(-1) = adj(A) / det(A)
//   - adj(A^T) = adj(A)^T
//   - det(adj(A)) = det(A)^(n-1)
//
// For small matrices (2x2, 3x3, 4x4), we use direct formulas.
// For larger matrices, we compute cofactor then transpose.
//
// =============================================================================

#include <datapod/matrix/matrix.hpp>
#include <optinum/lina/basic/cofactor.hpp>
#include <optinum/lina/basic/transpose.hpp>
#include <optinum/simd/matrix.hpp>

namespace optinum::lina {

    // =========================================================================
    // Adjoint Matrix - Specialized for 2x2
    // =========================================================================
    // For 2x2: adj(A) = [[a11, -a01], [-a10, a00]]
    template <typename T>
    [[nodiscard]] constexpr datapod::mat::Matrix<T, 2, 2> adjoint(const simd::Matrix<T, 2, 2> &a) noexcept {
        datapod::mat::Matrix<T, 2, 2> result;

        result(0, 0) = a(1, 1);  // adj_00 = a11
        result(0, 1) = -a(0, 1); // adj_01 = -a01
        result(1, 0) = -a(1, 0); // adj_10 = -a10
        result(1, 1) = a(0, 0);  // adj_11 = a00

        return result;
    }

    // =========================================================================
    // Adjoint Matrix - Specialized for 3x3
    // =========================================================================
    // For 3x3: adj(A) = cofactor(A)^T
    // Directly compute to avoid intermediate matrix
    template <typename T>
    [[nodiscard]] constexpr datapod::mat::Matrix<T, 3, 3> adjoint(const simd::Matrix<T, 3, 3> &a) noexcept {
        datapod::mat::Matrix<T, 3, 3> result;

        // Compute cofactor matrix elements and transpose simultaneously
        // adj(A) = C^T, so adj_ij = C_ji

        // Column 0 (= cofactor row 0)
        result(0, 0) = +(a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)); // C_00
        result(1, 0) = -(a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)); // C_01
        result(2, 0) = +(a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)); // C_02

        // Column 1 (= cofactor row 1)
        result(0, 1) = -(a(0, 1) * a(2, 2) - a(0, 2) * a(2, 1)); // C_10
        result(1, 1) = +(a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)); // C_11
        result(2, 1) = -(a(0, 0) * a(2, 1) - a(0, 1) * a(2, 0)); // C_12

        // Column 2 (= cofactor row 2)
        result(0, 2) = +(a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)); // C_20
        result(1, 2) = -(a(0, 0) * a(1, 2) - a(0, 2) * a(1, 0)); // C_21
        result(2, 2) = +(a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)); // C_22

        return result;
    }

    // =========================================================================
    // Adjoint Matrix - General case (all sizes including 4x4 and larger)
    // =========================================================================
    // adj(A) = cofactor(A)^T
    template <typename T, std::size_t N>
    [[nodiscard]] constexpr datapod::mat::Matrix<T, N, N> adjoint(const simd::Matrix<T, N, N> &a) noexcept
    requires(N > 3)
    {
        // Compute cofactor matrix then transpose
        const auto cof = cofactor(a);
        return lina::transpose(cof);
    }

    // =========================================================================
    // Overloads for dp::mat::matrix (owning type)
    // =========================================================================
    template <typename T>
    [[nodiscard]] constexpr datapod::mat::Matrix<T, 2, 2> adjoint(const datapod::mat::Matrix<T, 2, 2> &a) noexcept {
        simd::Matrix<T, 2, 2> view(const_cast<datapod::mat::Matrix<T, 2, 2> &>(a));
        return adjoint(view);
    }

    template <typename T>
    [[nodiscard]] constexpr datapod::mat::Matrix<T, 3, 3> adjoint(const datapod::mat::Matrix<T, 3, 3> &a) noexcept {
        simd::Matrix<T, 3, 3> view(const_cast<datapod::mat::Matrix<T, 3, 3> &>(a));
        return adjoint(view);
    }

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr datapod::mat::Matrix<T, N, N> adjoint(const datapod::mat::Matrix<T, N, N> &a) noexcept
    requires(N > 3)
    {
        simd::Matrix<T, N, N> view(const_cast<datapod::mat::Matrix<T, N, N> &>(a));
        return adjoint(view);
    }

    // =========================================================================
    // Convenience alias - adjugate is the standard mathematical term
    // =========================================================================
    template <typename T, std::size_t N>
    [[nodiscard]] constexpr datapod::mat::Matrix<T, N, N> adjugate(const simd::Matrix<T, N, N> &a) noexcept {
        return adjoint(a);
    }

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr datapod::mat::Matrix<T, N, N> adjugate(const datapod::mat::Matrix<T, N, N> &a) noexcept {
        return adjoint(a);
    }

} // namespace optinum::lina
