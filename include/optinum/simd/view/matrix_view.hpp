#pragma once

// =============================================================================
// optinum/simd/view/matrix_view.hpp
// matrix_view<T,W> - Non-owning view over a 2D array (column-major)
// =============================================================================

#include <optinum/simd/kernel.hpp>
#include <optinum/simd/view/vector_view.hpp>

namespace optinum::simd {

    // =============================================================================
    // matrix_view<T, W> - Rank-2 view (2D array, column-major layout)
    //
    // Layout: column-major (Fortran-style), matching datapod::mat::matrix
    //   - stride(0) = 1 (row stride)
    //   - stride(1) = rows (column stride)
    //
    // Provides:
    //   - rows(), cols(): dimensions
    //   - operator()(r,c): element access
    //   - row(r), col(c): vector views
    //   - Linear pack iteration over the entire matrix
    // =============================================================================

    template <typename T, std::size_t W> struct matrix_view {
        using value_type = std::remove_const_t<T>;
        using kernel_type = Kernel<T, W, 2>;
        static constexpr std::size_t width = W;
        static constexpr std::size_t rank = 2;

        kernel_type kernel_;

        // ==========================================================================
        // Constructors
        // ==========================================================================

        OPTINUM_INLINE constexpr matrix_view() noexcept = default;

        // Column-major: stride[0]=1, stride[1]=rows
        OPTINUM_INLINE constexpr matrix_view(T *ptr, std::size_t rows, std::size_t cols) noexcept
            : kernel_(ptr, {rows, cols}, {1, rows}) {}

        OPTINUM_INLINE constexpr explicit matrix_view(const kernel_type &k) noexcept : kernel_(k) {}

        // ==========================================================================
        // Size queries
        // ==========================================================================

        OPTINUM_INLINE constexpr std::size_t rows() const noexcept { return kernel_.extent(0); }

        OPTINUM_INLINE constexpr std::size_t cols() const noexcept { return kernel_.extent(1); }

        OPTINUM_INLINE constexpr std::size_t size() const noexcept { return rows() * cols(); }

        OPTINUM_INLINE constexpr std::size_t num_packs() const noexcept { return kernel_.num_packs(); }

        OPTINUM_INLINE constexpr std::size_t tail_size() const noexcept { return kernel_.tail_size(); }

        OPTINUM_INLINE constexpr bool is_contiguous() const noexcept { return kernel_.is_contiguous(); }

        // ==========================================================================
        // Element access (2D indexing)
        // ==========================================================================

        OPTINUM_INLINE value_type &operator()(std::size_t r, std::size_t c) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot get non-const reference from const view");
            return kernel_.at(r, c);
        }

        OPTINUM_INLINE const value_type &at(std::size_t r, std::size_t c) const noexcept {
            return kernel_.at_const(r, c);
        }

        // ==========================================================================
        // Row/Column views
        // ==========================================================================

        // Get a view of row r (non-contiguous, stride = rows)
        OPTINUM_INLINE vector_view<T, W> row(std::size_t r) const noexcept {
            Kernel<T, W, 1> row_kernel(kernel_.data() + r, {cols()}, {rows()});
            return vector_view<T, W>(row_kernel);
        }

        // Get a view of column c (contiguous, stride = 1)
        OPTINUM_INLINE vector_view<T, W> col(std::size_t c) const noexcept {
            Kernel<T, W, 1> col_kernel(kernel_.data() + c * rows(), {rows()}, {1});
            return vector_view<T, W>(col_kernel);
        }

        // ==========================================================================
        // Linear pack access (treats matrix as flat array)
        // ==========================================================================

        OPTINUM_INLINE pack<value_type, W> load_pack(std::size_t pack_idx) const noexcept {
            return kernel_.load_pack(pack_idx);
        }

        OPTINUM_INLINE void store_pack(std::size_t pack_idx, const pack<value_type, W> &v) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot store to const view");
            kernel_.store_pack(pack_idx, v);
        }

        OPTINUM_INLINE pack<value_type, W> load_pack_tail(std::size_t pack_idx) const noexcept {
            const std::size_t valid = (pack_idx == num_packs() - 1) ? tail_size() : W;
            return kernel_.load_pack_tail(pack_idx, valid);
        }

        OPTINUM_INLINE void store_pack_tail(std::size_t pack_idx, const pack<value_type, W> &v) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot store to const view");
            const std::size_t valid = (pack_idx == num_packs() - 1) ? tail_size() : W;
            kernel_.store_pack_tail(pack_idx, v, valid);
        }

        // ==========================================================================
        // Data access
        // ==========================================================================

        OPTINUM_INLINE T *data() const noexcept { return kernel_.data(); }

        OPTINUM_INLINE const value_type *data_const() const noexcept { return kernel_.data_const(); }

        // ==========================================================================
        // Subview (block)
        // ==========================================================================

        OPTINUM_INLINE matrix_view subview(std::size_t row_offset, std::size_t col_offset, std::size_t nrows,
                                           std::size_t ncols) const noexcept {
            T *block_ptr = kernel_.data() + row_offset + col_offset * rows();
            Kernel<T, W, 2> block_kernel(block_ptr, {nrows, ncols}, {1, rows()});
            return matrix_view(block_kernel);
        }
    };

} // namespace optinum::simd
