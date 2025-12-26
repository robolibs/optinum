#pragma once

// =============================================================================
// optinum/simd/view/diagonal_view.hpp
// diagonal_view<T,W> - Non-owning view over a matrix diagonal
// =============================================================================

#include <optinum/simd/kernel.hpp>

namespace optinum::simd {

    // Forward declaration
    template <typename T, std::size_t W> struct matrix_view;

    // =============================================================================
    // diagonal_view<T, W> - View over a matrix diagonal
    //
    // Provides access to matrix diagonals without copying data:
    //   - Main diagonal (k=0): elements m(0,0), m(1,1), m(2,2), ...
    //   - Upper diagonals (k>0): elements m(0,k), m(1,k+1), m(2,k+2), ...
    //   - Lower diagonals (k<0): elements m(-k,0), m(-k+1,1), m(-k+2,2), ...
    //
    // For column-major matrices, the stride is:
    //   - Main diagonal: rows + 1
    //   - All diagonals: rows + 1
    //
    // Provides:
    //   - size(): number of elements in the diagonal
    //   - operator[]: scalar element access
    //   - load_pack(i): load the i-th pack (strided access)
    //   - store_pack(i, v): store to the i-th pack (strided access)
    //   - Compatible with algo:: functions for SIMD operations
    // =============================================================================

    template <typename T, std::size_t W> struct diagonal_view {
        using value_type = std::remove_const_t<T>;
        using kernel_type = Kernel<T, W, 1>;
        static constexpr std::size_t width = W;
        static constexpr std::size_t rank = 1;

        kernel_type kernel_;
        std::size_t rows_;
        std::size_t cols_;
        std::ptrdiff_t k_; // Diagonal offset

        // ==========================================================================
        // Constructors
        // ==========================================================================

        OPTINUM_INLINE constexpr diagonal_view() noexcept = default;

        // Constructor for main diagonal (k=0)
        OPTINUM_INLINE constexpr diagonal_view(T *ptr, std::size_t rows, std::size_t cols) noexcept
            : rows_(rows), cols_(cols), k_(0) {
            // Main diagonal: starts at (0,0), stride = rows + 1
            std::size_t diag_size = std::min(rows, cols);
            std::size_t stride = rows + 1;
            kernel_ = kernel_type(ptr, {diag_size}, {stride});
        }

        // Constructor for arbitrary diagonal (k != 0)
        OPTINUM_INLINE constexpr diagonal_view(T *ptr, std::size_t rows, std::size_t cols, std::ptrdiff_t k) noexcept
            : rows_(rows), cols_(cols), k_(k) {
            // Calculate starting position and size
            std::size_t diag_size;
            T *start_ptr;

            if (k >= 0) {
                // Upper diagonal: starts at (0, k)
                std::size_t k_u = static_cast<std::size_t>(k);
                if (k_u >= cols) {
                    // Out of bounds diagonal
                    diag_size = 0;
                    start_ptr = ptr;
                } else {
                    diag_size = std::min(rows, cols - k_u);
                    start_ptr = ptr + k_u * rows; // Column offset in column-major
                }
            } else {
                // Lower diagonal: starts at (-k, 0)
                std::size_t k_l = static_cast<std::size_t>(-k);
                if (k_l >= rows) {
                    // Out of bounds diagonal
                    diag_size = 0;
                    start_ptr = ptr;
                } else {
                    diag_size = std::min(rows - k_l, cols);
                    start_ptr = ptr + k_l; // Row offset
                }
            }

            // For all diagonals, stride is rows + 1 in column-major layout
            std::size_t stride = rows + 1;
            kernel_ = kernel_type(start_ptr, {diag_size}, {stride});
        }

        OPTINUM_INLINE constexpr explicit diagonal_view(const kernel_type &k, std::size_t rows, std::size_t cols,
                                                        std::ptrdiff_t diag_k) noexcept
            : kernel_(k), rows_(rows), cols_(cols), k_(diag_k) {}

        // ==========================================================================
        // Size queries
        // ==========================================================================

        OPTINUM_INLINE constexpr std::size_t size() const noexcept { return kernel_.extent(0); }

        OPTINUM_INLINE constexpr std::size_t num_packs() const noexcept { return kernel_.num_packs(); }

        OPTINUM_INLINE constexpr std::size_t tail_size() const noexcept { return kernel_.tail_size(); }

        OPTINUM_INLINE constexpr bool is_contiguous() const noexcept {
            return false; // Diagonal views are never contiguous (stride != 1)
        }

        OPTINUM_INLINE constexpr std::ptrdiff_t diagonal_offset() const noexcept { return k_; }

        // ==========================================================================
        // Element access (scalar)
        // ==========================================================================

        OPTINUM_INLINE value_type &operator[](std::size_t i) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot get non-const reference from const view");
            return kernel_.at_linear(i);
        }

        OPTINUM_INLINE const value_type &at(std::size_t i) const noexcept { return kernel_.at_linear_const(i); }

        // ==========================================================================
        // Pack access (SIMD) - Strided load/store
        // ==========================================================================

        // Note: Diagonal views are strided, so we need to load/store elements individually
        OPTINUM_INLINE pack<value_type, W> load_pack(std::size_t pack_idx) const noexcept {
            const std::size_t base = pack_idx * W;
            const std::size_t stride = rows_ + 1;

            alignas(W * sizeof(value_type)) value_type temp[W];
            for (std::size_t i = 0; i < W; ++i) {
                if (base + i < size()) {
                    temp[i] = kernel_.at_linear(base + i);
                } else {
                    temp[i] = value_type{}; // Padding for partial packs
                }
            }
            return pack<value_type, W>::load(temp);
        }

        OPTINUM_INLINE void store_pack(std::size_t pack_idx, const pack<value_type, W> &v) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot store to const view");
            const std::size_t base = pack_idx * W;

            alignas(W * sizeof(value_type)) value_type temp[W];
            v.store(temp);

            for (std::size_t i = 0; i < W; ++i) {
                if (base + i < size()) {
                    kernel_.at_linear(base + i) = temp[i];
                }
            }
        }

        // Tail-safe pack access
        OPTINUM_INLINE pack<value_type, W> load_pack_tail(std::size_t pack_idx) const noexcept {
            const std::size_t base = pack_idx * W;
            const std::size_t valid = (pack_idx == num_packs() - 1) ? tail_size() : W;

            alignas(W * sizeof(value_type)) value_type temp[W];
            for (std::size_t i = 0; i < valid; ++i) {
                temp[i] = kernel_.at_linear(base + i);
            }
            for (std::size_t i = valid; i < W; ++i) {
                temp[i] = value_type{}; // Padding
            }
            return pack<value_type, W>::load(temp);
        }

        OPTINUM_INLINE void store_pack_tail(std::size_t pack_idx, const pack<value_type, W> &v) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot store to const view");
            const std::size_t base = pack_idx * W;
            const std::size_t valid = (pack_idx == num_packs() - 1) ? tail_size() : W;

            alignas(W * sizeof(value_type)) value_type temp[W];
            v.store(temp);

            for (std::size_t i = 0; i < valid; ++i) {
                kernel_.at_linear(base + i) = temp[i];
            }
        }

        // ==========================================================================
        // Data access
        // ==========================================================================

        OPTINUM_INLINE T *data() const noexcept { return kernel_.data(); }

        OPTINUM_INLINE const value_type *data_const() const noexcept { return kernel_.data_const(); }
    };

    // =============================================================================
    // Helper function to create diagonal views
    // =============================================================================

    // Create a view of the main diagonal (k=0)
    template <typename T, std::size_t W>
    OPTINUM_INLINE diagonal_view<T, W> diagonal(T *ptr, std::size_t rows, std::size_t cols) noexcept {
        return diagonal_view<T, W>(ptr, rows, cols);
    }

    // Create a view of the k-th diagonal
    // k=0: main diagonal
    // k>0: k-th upper diagonal
    // k<0: k-th lower diagonal
    template <typename T, std::size_t W>
    OPTINUM_INLINE diagonal_view<T, W> diagonal(T *ptr, std::size_t rows, std::size_t cols, std::ptrdiff_t k) noexcept {
        return diagonal_view<T, W>(ptr, rows, cols, k);
    }

    // Create diagonal view from matrix_view
    template <typename T, std::size_t W>
    OPTINUM_INLINE diagonal_view<T, W> diagonal(const matrix_view<T, W> &m, std::ptrdiff_t k = 0) noexcept {
        return diagonal_view<T, W>(m.data(), m.rows(), m.cols(), k);
    }

} // namespace optinum::simd
