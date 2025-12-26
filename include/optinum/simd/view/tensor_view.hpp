#pragma once

// =============================================================================
// optinum/simd/view/tensor_view.hpp
// tensor_view<T,W,Rank> - Non-owning view over an N-dimensional array
// =============================================================================

#include <optinum/simd/kernel.hpp>
#include <optinum/simd/view/slice.hpp>
#include <tuple>

namespace optinum::simd {

    // Forward declarations for dimensionality reduction
    template <typename T, std::size_t W> struct vector_view;
    template <typename T, std::size_t W> struct matrix_view;
    template <typename T, std::size_t W> struct vector_view;
    template <typename T, std::size_t W> struct matrix_view;
} // namespace optinum::simd

namespace optinum::simd {

    // =============================================================================
    // tensor_view<T, W, Rank> - Rank-N view (N-dimensional array)
    //
    // General-purpose multi-dimensional view for Rank >= 3
    // Provides:
    //   - extent(dim): size along dimension
    //   - operator()(...): multi-dimensional indexing
    //   - Linear pack iteration over the entire tensor
    // =============================================================================

    template <typename T, std::size_t W, std::size_t Rank> struct tensor_view {
        static_assert(Rank >= 3, "tensor_view requires Rank >= 3 (use scalar/vector/matrix_view for lower ranks)");

        using value_type = std::remove_const_t<T>;
        using kernel_type = Kernel<T, W, Rank>;
        static constexpr std::size_t width = W;
        static constexpr std::size_t rank = Rank;

        kernel_type kernel_;

        // ==========================================================================
        // Constructors
        // ==========================================================================

        OPTINUM_INLINE constexpr tensor_view() noexcept = default;

        OPTINUM_INLINE constexpr explicit tensor_view(const kernel_type &k) noexcept : kernel_(k) {}

        OPTINUM_INLINE constexpr tensor_view(T *ptr, const std::array<std::size_t, Rank> &extents,
                                             const std::array<std::size_t, Rank> &strides) noexcept
            : kernel_(ptr, extents, strides) {}

        // ==========================================================================
        // Size queries
        // ==========================================================================

        OPTINUM_INLINE constexpr std::size_t extent(std::size_t dim) const noexcept { return kernel_.extent(dim); }

        OPTINUM_INLINE constexpr std::size_t stride(std::size_t dim) const noexcept { return kernel_.stride(dim); }

        OPTINUM_INLINE constexpr std::size_t size() const noexcept { return kernel_.linear_size(); }

        OPTINUM_INLINE constexpr std::size_t num_packs() const noexcept { return kernel_.num_packs(); }

        OPTINUM_INLINE constexpr std::size_t tail_size() const noexcept { return kernel_.tail_size(); }

        OPTINUM_INLINE constexpr bool is_contiguous() const noexcept { return kernel_.is_contiguous(); }

        // ==========================================================================
        // Element access (multi-dimensional indexing)
        // ==========================================================================

        template <typename... Indices> OPTINUM_INLINE value_type &operator()(Indices... indices) const noexcept {
            static_assert(sizeof...(Indices) == Rank, "Number of indices must match Rank");
            static_assert(!std::is_const_v<T>, "Cannot get non-const reference from const view");
            return kernel_.at(indices...);
        }

        template <typename... Indices> OPTINUM_INLINE const value_type &at(Indices... indices) const noexcept {
            static_assert(sizeof...(Indices) == Rank, "Number of indices must match Rank");
            return kernel_.at_const(indices...);
        }

        // ==========================================================================
        // Linear pack access (treats tensor as flat array)
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
        // Slicing - N-dimensional slicing with seq/fseq/all/fix
        // ==========================================================================

        // Multi-dimensional slicing: t.slice(slice0, slice1, ..., sliceN)
        // Returns a tensor_view, matrix_view, vector_view, or scalar_view based on dimensionality reduction
        //
        // Examples (for a 3D tensor):
        //   t.slice(seq(0, 3), all, all)       // 3D tensor (no fix<>)
        //   t.slice(all, fix<2>(), all)        // 2D matrix (1 fix<> -> rank-1)
        //   t.slice(fix<0>(), fix<2>(), all)   // 1D vector (2 fix<> -> rank-2)
        //   t.slice(fix<0>(), fix<1>(), fix<2>()) // scalar (3 fix<> -> rank-3)
        template <typename... Slices> OPTINUM_INLINE auto slice(const Slices &...slices) const noexcept {
            static_assert(sizeof...(Slices) == Rank, "Number of slices must match tensor rank");

            // Count how many fix<> indices we have (compile-time)
            constexpr std::size_t num_fixed = (is_fixed_index_v<Slices> + ... + 0);
            constexpr std::size_t result_rank = Rank - num_fixed;

            // Resolve all slices to concrete seq objects
            std::array<seq, Rank> resolved_slices =
                resolve_slices_helper(std::index_sequence_for<Slices...>{}, slices...);

            // Dispatch based on resulting rank
            if constexpr (result_rank == 0) {
                // All dimensions fixed -> return scalar_view (1x1x...x1 tensor for now)
                return slice_impl_same_rank(resolved_slices, std::make_index_sequence<Rank>{});
            } else if constexpr (result_rank == 1) {
                // Rank-1 -> return vector_view
                return slice_impl_to_vector(resolved_slices, std::make_index_sequence<Rank>{}, slices...);
            } else if constexpr (result_rank == 2) {
                // Rank-2 -> return matrix_view
                return slice_impl_to_matrix(resolved_slices, std::make_index_sequence<Rank>{}, slices...);
            } else {
                // Rank >= 3 -> return tensor_view of lower rank
                return slice_impl_same_rank(resolved_slices, std::make_index_sequence<Rank>{});
            }
        }

      private:
        // Helper to resolve all slices into an array
        template <std::size_t... Is, typename... Slices>
        OPTINUM_INLINE std::array<seq, Rank> resolve_slices_helper(std::index_sequence<Is...>,
                                                                   const Slices &...slices) const noexcept {
            // Use tuple to access each argument by index
            auto tuple = std::make_tuple(slices...);
            return {resolve_slice(std::get<Is>(tuple), kernel_.extent(Is))...};
        }

        // Helper: slice to same-rank tensor (no dimensionality reduction)
        template <std::size_t... Is>
        OPTINUM_INLINE tensor_view slice_impl_same_rank(const std::array<seq, Rank> &slices,
                                                        std::index_sequence<Is...>) const noexcept {
            // Calculate pointer offset
            std::size_t offset = 0;
            ((offset += slices[Is].start * kernel_.stride(Is)), ...);

            T *slice_ptr = kernel_.data() + offset;

            // Calculate new extents and strides
            std::array<std::size_t, Rank> new_extents;
            std::array<std::size_t, Rank> new_strides;

            ((new_extents[Is] = slices[Is].size(), new_strides[Is] = kernel_.stride(Is) * slices[Is].step), ...);

            Kernel<T, W, Rank> slice_kernel(slice_ptr, new_extents, new_strides);
            return tensor_view(slice_kernel);
        }

        // Helper: slice to vector_view (rank reduced to 1)
        template <std::size_t... Is, typename... Slices>
        OPTINUM_INLINE vector_view<T, W> slice_impl_to_vector(const std::array<seq, Rank> &slices,
                                                              std::index_sequence<Is...>,
                                                              const Slices &...slice_args) const noexcept {
            // Calculate pointer offset
            std::size_t offset = 0;
            ((offset += slices[Is].start * kernel_.stride(Is)), ...);

            T *slice_ptr = kernel_.data() + offset;

            // Find the one non-fixed dimension
            std::size_t extent = 0;
            std::size_t stride = 0;

            // Use fold expression with early return effect
            std::size_t dummy[] = {0, (void(is_fixed_index_v<Slices> ? 0
                                                                     : (extent = slices[Is].size(),
                                                                        stride = kernel_.stride(Is) * slices[Is].step)),
                                       0)...};
            (void)dummy;

            Kernel<T, W, 1> vec_kernel(slice_ptr, {extent}, {stride});
            return vector_view<T, W>(vec_kernel);
        }

        // Helper: slice to matrix_view (rank reduced to 2)
        template <std::size_t... Is, typename... Slices>
        OPTINUM_INLINE matrix_view<T, W> slice_impl_to_matrix(const std::array<seq, Rank> &slices,
                                                              std::index_sequence<Is...>,
                                                              const Slices &...slice_args) const noexcept {
            // Calculate pointer offset
            std::size_t offset = 0;
            ((offset += slices[Is].start * kernel_.stride(Is)), ...);

            T *slice_ptr = kernel_.data() + offset;

            // Find the two non-fixed dimensions
            std::array<std::size_t, 2> extents = {0, 0};
            std::array<std::size_t, 2> strides = {0, 0};
            std::size_t idx = 0;

            // Collect non-fixed dimensions
            std::size_t dummy[] = {
                0, (void(is_fixed_index_v<Slices> ? 0
                                                  : (extents[idx] = slices[Is].size(),
                                                     strides[idx] = kernel_.stride(Is) * slices[Is].step, ++idx)),
                    0)...};
            (void)dummy;

            Kernel<T, W, 2> mat_kernel(slice_ptr, extents, strides);
            return matrix_view<T, W>(mat_kernel);
        }
    };

} // namespace optinum::simd

// Include view implementations for dimensionality reduction
// (must be after tensor_view definition to avoid circular dependencies)
#include <optinum/simd/view/matrix_view.hpp>
#include <optinum/simd/view/vector_view.hpp>
