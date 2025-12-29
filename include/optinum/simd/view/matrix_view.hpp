#pragma once

// =============================================================================
// optinum/simd/view/matrix_view.hpp
// matrix_view<T,W> - Non-owning view over a 2D array (column-major)
// =============================================================================

#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/backend/matmul.hpp>
#include <optinum/simd/backend/reduce.hpp>
#include <optinum/simd/backend/transpose.hpp>
#include <optinum/simd/kernel.hpp>
#include <optinum/simd/view/slice.hpp>
#include <optinum/simd/view/vector_view.hpp>

#include <cmath>

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
    //   - Arithmetic operations: +, -, *, +=, -=, *=, /=
    //   - Matrix multiplication, transpose, trace
    //   - fill(), set_identity()
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

        // 1D linear indexing (column-major order)
        OPTINUM_INLINE value_type &operator[](std::size_t i) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot get non-const reference from const view");
            return kernel_.at_linear(i);
        }

        OPTINUM_INLINE const value_type &at_linear(std::size_t i) const noexcept { return kernel_.at_linear_const(i); }

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
        // Fill operations (in-place)
        // ==========================================================================

        // Fill all elements with a constant value
        OPTINUM_INLINE matrix_view &fill(const value_type &value) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot fill const view");
            backend::fill_runtime<value_type>(data(), size(), value);
            return *this;
        }

        // Set to identity matrix (square matrices only, in-place)
        OPTINUM_INLINE matrix_view &set_identity() noexcept {
            static_assert(!std::is_const_v<T>, "Cannot set_identity on const view");
            fill(value_type{0});
            const std::size_t n = (rows() < cols()) ? rows() : cols();
            for (std::size_t i = 0; i < n; ++i) {
                (*this)(i, i) = value_type{1};
            }
            return *this;
        }

        // ==========================================================================
        // Static factory-like functions (write to output pointer)
        // These are static functions that fill an external buffer
        // ==========================================================================

        // Fill output with zeros
        static OPTINUM_INLINE void zeros(T *output, std::size_t rows, std::size_t cols) noexcept {
            backend::fill_runtime<value_type>(output, rows * cols, value_type{0});
        }

        // Fill output with ones
        static OPTINUM_INLINE void ones(T *output, std::size_t rows, std::size_t cols) noexcept {
            backend::fill_runtime<value_type>(output, rows * cols, value_type{1});
        }

        // Fill output with identity matrix
        static OPTINUM_INLINE void identity(T *output, std::size_t rows, std::size_t cols) noexcept {
            zeros(output, rows, cols);
            const std::size_t n = (rows < cols) ? rows : cols;
            for (std::size_t i = 0; i < n; ++i) {
                output[i + i * rows] = value_type{1}; // Column-major: element(i,i) = data[i + i*rows]
            }
        }

        // ==========================================================================
        // Compound assignment operators (in-place, element-wise)
        // ==========================================================================

        // Add another matrix view element-wise (accepts any const-qualified view)
        template <typename U> OPTINUM_INLINE matrix_view &operator+=(const matrix_view<U, W> &rhs) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            static_assert(std::is_same_v<std::remove_const_t<U>, value_type>, "Type mismatch");
            backend::add_runtime<value_type>(data(), data_const(), rhs.data_const(), size());
            return *this;
        }

        // Subtract another matrix view element-wise (accepts any const-qualified view)
        template <typename U> OPTINUM_INLINE matrix_view &operator-=(const matrix_view<U, W> &rhs) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            static_assert(std::is_same_v<std::remove_const_t<U>, value_type>, "Type mismatch");
            backend::sub_runtime<value_type>(data(), data_const(), rhs.data_const(), size());
            return *this;
        }

        // Multiply by scalar
        OPTINUM_INLINE matrix_view &operator*=(value_type scalar) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            backend::mul_scalar_runtime<value_type>(data(), data_const(), scalar, size());
            return *this;
        }

        // Divide by scalar
        OPTINUM_INLINE matrix_view &operator/=(value_type scalar) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            backend::div_scalar_runtime<value_type>(data(), data_const(), scalar, size());
            return *this;
        }

        // Element-wise multiplication (Hadamard product)
        template <typename U> OPTINUM_INLINE matrix_view &hadamard_inplace(const matrix_view<U, W> &rhs) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            static_assert(std::is_same_v<std::remove_const_t<U>, value_type>, "Type mismatch");
            backend::mul_runtime<value_type>(data(), data_const(), rhs.data_const(), size());
            return *this;
        }

        // Element-wise division
        template <typename U> OPTINUM_INLINE matrix_view &div_inplace(const matrix_view<U, W> &rhs) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            static_assert(std::is_same_v<std::remove_const_t<U>, value_type>, "Type mismatch");
            backend::div_runtime<value_type>(data(), data_const(), rhs.data_const(), size());
            return *this;
        }

        // ==========================================================================
        // Reductions
        // ==========================================================================

        // Trace (sum of diagonal elements)
        [[nodiscard]] OPTINUM_INLINE value_type trace() const noexcept {
            const std::size_t n = (rows() < cols()) ? rows() : cols();
            value_type result{0};
            for (std::size_t i = 0; i < n; ++i) {
                result += at(i, i);
            }
            return result;
        }

        // Sum of all elements
        [[nodiscard]] OPTINUM_INLINE value_type sum() const noexcept {
            return backend::reduce_sum_runtime<value_type>(data_const(), size());
        }

        // Frobenius norm
        [[nodiscard]] OPTINUM_INLINE value_type frobenius_norm() const noexcept {
            return std::sqrt(backend::dot_runtime<value_type>(data_const(), data_const(), size()));
        }

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

    // =============================================================================
    // Free functions for matrix_view operations
    // =============================================================================

    // Element-wise addition: C = A + B
    // Writes result to output buffer
    template <typename T, std::size_t W>
    OPTINUM_INLINE void add(T *output, const matrix_view<const T, W> &a, const matrix_view<const T, W> &b) noexcept {
        backend::add_runtime<T>(output, a.data_const(), b.data_const(), a.size());
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void add(T *output, const matrix_view<T, W> &a, const matrix_view<T, W> &b) noexcept {
        backend::add_runtime<T>(output, a.data_const(), b.data_const(), a.size());
    }

    // Element-wise subtraction: C = A - B
    template <typename T, std::size_t W>
    OPTINUM_INLINE void sub(T *output, const matrix_view<const T, W> &a, const matrix_view<const T, W> &b) noexcept {
        backend::sub_runtime<T>(output, a.data_const(), b.data_const(), a.size());
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void sub(T *output, const matrix_view<T, W> &a, const matrix_view<T, W> &b) noexcept {
        backend::sub_runtime<T>(output, a.data_const(), b.data_const(), a.size());
    }

    // Element-wise multiplication (Hadamard product): C = A * B (element-wise)
    template <typename T, std::size_t W>
    OPTINUM_INLINE void hadamard(T *output, const matrix_view<const T, W> &a,
                                 const matrix_view<const T, W> &b) noexcept {
        backend::mul_runtime<T>(output, a.data_const(), b.data_const(), a.size());
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void hadamard(T *output, const matrix_view<T, W> &a, const matrix_view<T, W> &b) noexcept {
        backend::mul_runtime<T>(output, a.data_const(), b.data_const(), a.size());
    }

    // Element-wise division: C = A / B (element-wise)
    template <typename T, std::size_t W>
    OPTINUM_INLINE void div(T *output, const matrix_view<const T, W> &a, const matrix_view<const T, W> &b) noexcept {
        backend::div_runtime<T>(output, a.data_const(), b.data_const(), a.size());
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void div(T *output, const matrix_view<T, W> &a, const matrix_view<T, W> &b) noexcept {
        backend::div_runtime<T>(output, a.data_const(), b.data_const(), a.size());
    }

    // Scalar multiplication: C = A * scalar
    template <typename T, std::size_t W>
    OPTINUM_INLINE void mul_scalar(T *output, const matrix_view<const T, W> &a, T scalar) noexcept {
        backend::mul_scalar_runtime<T>(output, a.data_const(), scalar, a.size());
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void mul_scalar(T *output, const matrix_view<T, W> &a, T scalar) noexcept {
        backend::mul_scalar_runtime<T>(output, a.data_const(), scalar, a.size());
    }

    // Scalar division: C = A / scalar
    template <typename T, std::size_t W>
    OPTINUM_INLINE void div_scalar(T *output, const matrix_view<const T, W> &a, T scalar) noexcept {
        backend::div_scalar_runtime<T>(output, a.data_const(), scalar, a.size());
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void div_scalar(T *output, const matrix_view<T, W> &a, T scalar) noexcept {
        backend::div_scalar_runtime<T>(output, a.data_const(), scalar, a.size());
    }

    // =============================================================================
    // Matrix multiplication (write to output buffer)
    // C = A * B where A is (R x K), B is (K x C), output is (R x C)
    // =============================================================================

    // Runtime matrix multiplication
    template <typename T, std::size_t W>
    OPTINUM_INLINE void matmul(T *output, const matrix_view<const T, W> &a, const matrix_view<const T, W> &b) noexcept {
        const std::size_t R = a.rows();
        const std::size_t K = a.cols();
        const std::size_t C = b.cols();

        // Column-major GEMM: C = A * B
        // A: (R x K), B: (K x C), C: (R x C)
        for (std::size_t j = 0; j < C; ++j) {
            const T *bcol = b.data_const() + j * K; // B column j
            T *outcol = output + j * R;             // C column j

            for (std::size_t i = 0; i < R; ++i) {
                T acc{0};
                for (std::size_t k = 0; k < K; ++k) {
                    acc += a.at(i, k) * bcol[k];
                }
                outcol[i] = acc;
            }
        }
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void matmul(T *output, const matrix_view<T, W> &a, const matrix_view<T, W> &b) noexcept {
        matmul(output, matrix_view<const T, W>(a.data(), a.rows(), a.cols()),
               matrix_view<const T, W>(b.data(), b.rows(), b.cols()));
    }

    // Matrix-vector multiplication: y = A * x
    // A: (R x C), x: (C), y: (R)
    template <typename T, std::size_t W>
    OPTINUM_INLINE void matvec(T *output, const matrix_view<const T, W> &a, const vector_view<const T, W> &x) noexcept {
        const std::size_t R = a.rows();
        const std::size_t C = a.cols();

        for (std::size_t i = 0; i < R; ++i) {
            T acc{0};
            for (std::size_t j = 0; j < C; ++j) {
                acc += a.at(i, j) * x.at(j);
            }
            output[i] = acc;
        }
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void matvec(T *output, const matrix_view<T, W> &a, const vector_view<T, W> &x) noexcept {
        matvec(output, matrix_view<const T, W>(a.data(), a.rows(), a.cols()),
               vector_view<const T, W>(x.data(), x.size()));
    }

    // =============================================================================
    // Transpose (write to output buffer)
    // Output: (cols x rows) from input (rows x cols)
    // =============================================================================

    template <typename T, std::size_t W>
    OPTINUM_INLINE void transpose(T *output, const matrix_view<const T, W> &a) noexcept {
        const std::size_t R = a.rows();
        const std::size_t C = a.cols();

        // in(row, col)  = in[col * R + row]
        // out(row',col') = out[col' * C + row'] with row'=col, col'=row
        for (std::size_t col = 0; col < C; ++col) {
            for (std::size_t row = 0; row < R; ++row) {
                output[row * C + col] = a.at(row, col);
            }
        }
    }

    template <typename T, std::size_t W> OPTINUM_INLINE void transpose(T *output, const matrix_view<T, W> &a) noexcept {
        transpose(output, matrix_view<const T, W>(a.data(), a.rows(), a.cols()));
    }

    // =============================================================================
    // Trace (sum of diagonal elements)
    // =============================================================================

    template <typename T, std::size_t W>
    [[nodiscard]] OPTINUM_INLINE T trace(const matrix_view<const T, W> &a) noexcept {
        return a.trace();
    }

    template <typename T, std::size_t W> [[nodiscard]] OPTINUM_INLINE T trace(const matrix_view<T, W> &a) noexcept {
        return a.trace();
    }

    // =============================================================================
    // Frobenius norm
    // =============================================================================

    template <typename T, std::size_t W>
    [[nodiscard]] OPTINUM_INLINE T frobenius_norm(const matrix_view<const T, W> &a) noexcept {
        return a.frobenius_norm();
    }

    template <typename T, std::size_t W>
    [[nodiscard]] OPTINUM_INLINE T frobenius_norm(const matrix_view<T, W> &a) noexcept {
        return a.frobenius_norm();
    }

    // =============================================================================
    // Comparison (element-wise equality)
    // =============================================================================

    template <typename T, std::size_t W>
    [[nodiscard]] OPTINUM_INLINE bool operator==(const matrix_view<T, W> &lhs, const matrix_view<T, W> &rhs) noexcept {
        if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
            return false;
        }
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            if (lhs.at_linear(i) != rhs.at_linear(i)) {
                return false;
            }
        }
        return true;
    }

    template <typename T, std::size_t W>
    [[nodiscard]] OPTINUM_INLINE bool operator!=(const matrix_view<T, W> &lhs, const matrix_view<T, W> &rhs) noexcept {
        return !(lhs == rhs);
    }

    // =============================================================================
    // Copy between views
    // =============================================================================

    // Copy from source view to destination view
    template <typename T, std::size_t W>
    OPTINUM_INLINE void copy(matrix_view<T, W> &dst, const matrix_view<const T, W> &src) noexcept {
        const std::size_t n = dst.size();
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = src.at_linear(i);
        }
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void copy(matrix_view<T, W> &dst, const matrix_view<T, W> &src) noexcept {
        const std::size_t n = dst.size();
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = src.at_linear(i);
        }
    }

    // =============================================================================
    // Negate (unary minus) - write to output buffer
    // =============================================================================

    template <typename T, std::size_t W>
    OPTINUM_INLINE void negate(T *output, const matrix_view<const T, W> &a) noexcept {
        const std::size_t n = a.size();
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = -a.at_linear(i);
        }
    }

    template <typename T, std::size_t W> OPTINUM_INLINE void negate(T *output, const matrix_view<T, W> &a) noexcept {
        const std::size_t n = a.size();
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = -a.at_linear(i);
        }
    }

} // namespace optinum::simd
