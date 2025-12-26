#pragma once

// =============================================================================
// optinum/simd/view/matrix_view.hpp
// matrix_view<T,W> - Non-owning view over a 2D array (column-major)
// =============================================================================

#include <optinum/simd/kernel.hpp>
#include <optinum/simd/view/slice.hpp>
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

        // ==========================================================================
        // Slicing - 2D slicing with seq/fseq/all/fix
        // ==========================================================================

        // Two-dimensional slicing: m.slice(row_slice, col_slice)
        // Returns a matrix_view of the sliced region
        //
        // Examples:
        //   m.slice(seq(0, 3), all)        // rows 0..2, all columns
        //   m.slice(all, fix<2>)           // all rows, column 2
        //   m.slice(seq(1,5), seq(2,8))    // rows 1..4, cols 2..7
        //   m.slice(fseq<0,4>(), all)      // compile-time rows 0..3, all cols
        template <typename SliceR, typename SliceC>
        OPTINUM_INLINE auto slice(const SliceR &row_slice, const SliceC &col_slice) const noexcept {
            static_assert(is_slice_v<SliceR> || is_fixed_index_v<SliceR>, "Invalid row slice type");
            static_assert(is_slice_v<SliceC> || is_fixed_index_v<SliceC>, "Invalid col slice type");

            // Resolve both slices to concrete indices
            seq row_seq = resolve_slice(row_slice, rows());
            seq col_seq = resolve_slice(col_slice, cols());

            // Handle special case: both are single indices -> return scalar view
            // (For now, just handle the matrix case)

            // Calculate new pointer offset
            // Column-major layout: element(r,c) is at data[r + c*rows]
            T *slice_ptr = kernel_.data() + row_seq.start + col_seq.start * rows();

            // Calculate new dimensions
            std::size_t new_rows = row_seq.size();
            std::size_t new_cols = col_seq.size();

            // Calculate new strides
            // row_stride remains 1 in column-major (unless we have row stepping)
            // col_stride is rows() in original matrix (unless we have col stepping)
            std::size_t new_row_stride = row_seq.step;
            std::size_t new_col_stride = rows() * col_seq.step;

            // Create new kernel with adjusted pointer, dimensions, and strides
            Kernel<T, W, 2> slice_kernel(slice_ptr, {new_rows, new_cols}, {new_row_stride, new_col_stride});

            // Handle special cases for dimensionality reduction
            if constexpr (is_fixed_index_v<SliceR> && is_fixed_index_v<SliceC>) {
                // Both dimensions fixed -> return scalar_view (not implemented yet)
                // For now, just return a 1x1 matrix
                return matrix_view(slice_kernel);
            } else if constexpr (is_fixed_index_v<SliceR>) {
                // Row is fixed -> return vector_view (column slice of single row)
                Kernel<T, W, 1> vec_kernel(slice_ptr, {new_cols}, {new_col_stride});
                return vector_view<T, W>(vec_kernel);
            } else if constexpr (is_fixed_index_v<SliceC>) {
                // Column is fixed -> return vector_view (row slice of single column)
                Kernel<T, W, 1> vec_kernel(slice_ptr, {new_rows}, {new_row_stride});
                return vector_view<T, W>(vec_kernel);
            } else {
                // Both are ranges -> return matrix_view
                return matrix_view(slice_kernel);
            }
        }
    };

} // namespace optinum::simd
