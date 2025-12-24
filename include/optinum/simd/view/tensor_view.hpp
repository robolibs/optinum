#pragma once

// =============================================================================
// optinum/simd/view/tensor_view.hpp
// tensor_view<T,W,Rank> - Non-owning view over an N-dimensional array
// =============================================================================

#include <optinum/simd/kernel.hpp>

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
    };

} // namespace optinum::simd
