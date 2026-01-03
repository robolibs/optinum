#pragma once

// =============================================================================
// optinum/simd/layout.hpp
// Layout conversion utilities (column-major ↔ row-major)
// =============================================================================

#include <datapod/matrix.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/tensor.hpp>

namespace optinum::simd {

    namespace dp = ::datapod;

    /**
     * @brief Convert matrix from column-major to row-major layout
     *
     * Optinum uses column-major (Fortran-style) by default.
     * This creates a copy in row-major (C-style) layout.
     *
     * Column-major: data[col * rows + row]
     * Row-major:    data[row * cols + col]
     */
    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr dp::mat::Matrix<T, C, R> torowmajor(const Matrix<T, R, C> &m) noexcept {
        // Transpose to convert layout
        // Column-major (R x C) → Row-major = Transpose
        dp::mat::Matrix<T, C, R> result{};
        for (std::size_t i = 0; i < R; ++i) {
            for (std::size_t j = 0; j < C; ++j) {
                result(j, i) = m(i, j);
            }
        }
        return result;
    }

    /**
     * @brief Convert matrix from row-major to column-major layout
     *
     * This assumes input is in row-major and converts to column-major.
     * Returns transpose to convert layout.
     */
    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr dp::mat::Matrix<T, C, R> tocolumnmajor(const Matrix<T, R, C> &m) noexcept {
        // Same as torowmajor - transpose converts between layouts
        dp::mat::Matrix<T, C, R> result{};
        for (std::size_t i = 0; i < R; ++i) {
            for (std::size_t j = 0; j < C; ++j) {
                result(j, i) = m(i, j);
            }
        }
        return result;
    }

    /**
     * @brief Copy matrix data to row-major array
     *
     * @param m Matrix in column-major layout
     * @param out Output array (must have R*C elements)
     */
    template <typename T, std::size_t R, std::size_t C>
    constexpr void copy_to_rowmajor(const Matrix<T, R, C> &m, T *out) noexcept {
        for (std::size_t i = 0; i < R; ++i) {
            for (std::size_t j = 0; j < C; ++j) {
                out[i * C + j] = m(i, j);
            }
        }
    }

    /**
     * @brief Copy matrix data from row-major array
     *
     * @param m Matrix in column-major layout (output)
     * @param in Input array in row-major layout (must have R*C elements)
     */
    template <typename T, std::size_t R, std::size_t C>
    constexpr void copy_from_rowmajor(Matrix<T, R, C> &m, const T *in) noexcept {
        for (std::size_t i = 0; i < R; ++i) {
            for (std::size_t j = 0; j < C; ++j) {
                m(i, j) = in[i * C + j];
            }
        }
    }

    /**
     * @brief Copy tensor data to row-major (C-style) array
     *
     * Converts from column-major to row-major layout.
     * For 3D tensor [D0, D1, D2]:
     *   Column-major: data[d2 * D0*D1 + d1 * D0 + d0]
     *   Row-major:    data[d0 * D1*D2 + d1 * D2 + d2]
     */
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2>
    constexpr void copy_to_rowmajor(const Tensor<T, D0, D1, D2> &t, T *out) noexcept {
        for (std::size_t i = 0; i < D0; ++i) {
            for (std::size_t j = 0; j < D1; ++j) {
                for (std::size_t k = 0; k < D2; ++k) {
                    out[i * D1 * D2 + j * D2 + k] = t(i, j, k);
                }
            }
        }
    }

    /**
     * @brief Copy tensor data from row-major (C-style) array
     *
     * Converts from row-major to column-major layout.
     */
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2>
    constexpr void copy_from_rowmajor(Tensor<T, D0, D1, D2> &t, const T *in) noexcept {
        for (std::size_t i = 0; i < D0; ++i) {
            for (std::size_t j = 0; j < D1; ++j) {
                for (std::size_t k = 0; k < D2; ++k) {
                    t(i, j, k) = in[i * D1 * D2 + j * D2 + k];
                }
            }
        }
    }

    /**
     * @brief Copy tensor data to column-major (Fortran-style) array
     *
     * This is essentially a straight copy since optinum uses column-major.
     */
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2>
    constexpr void copy_to_columnmajor(const Tensor<T, D0, D1, D2> &t, T *out) noexcept {
        const T *data = t.data();
        for (std::size_t i = 0; i < D0 * D1 * D2; ++i) {
            out[i] = data[i];
        }
    }

    /**
     * @brief Copy tensor data from column-major (Fortran-style) array
     *
     * This is essentially a straight copy since optinum uses column-major.
     */
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2>
    constexpr void copy_from_columnmajor(Tensor<T, D0, D1, D2> &t, const T *in) noexcept {
        T *data = t.data();
        for (std::size_t i = 0; i < D0 * D1 * D2; ++i) {
            data[i] = in[i];
        }
    }

} // namespace optinum::simd
