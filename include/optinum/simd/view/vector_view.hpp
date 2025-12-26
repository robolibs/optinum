#pragma once

// =============================================================================
// optinum/simd/view/vector_view.hpp
// vector_view<T,W> - Non-owning view over a 1D array with SIMD access
// =============================================================================

#include <optinum/simd/kernel.hpp>
#include <optinum/simd/view/slice.hpp>

namespace optinum::simd {

    // =============================================================================
    // vector_view<T, W> - Rank-1 view (1D array)
    //
    // Provides pack-based iteration over a 1D array:
    //   - size(): number of elements
    //   - operator[]: scalar element access
    //   - load_pack(i): load the i-th pack
    //   - store_pack(i, v): store to the i-th pack
    //   - num_packs(): total number of complete + partial packs
    //   - tail_size(): number of valid elements in the last pack
    // =============================================================================

    template <typename T, std::size_t W> struct vector_view {
        using value_type = std::remove_const_t<T>;
        using kernel_type = Kernel<T, W, 1>;
        static constexpr std::size_t width = W;
        static constexpr std::size_t rank = 1;

        kernel_type kernel_;

        // ==========================================================================
        // Constructors
        // ==========================================================================

        OPTINUM_INLINE constexpr vector_view() noexcept = default;

        OPTINUM_INLINE constexpr vector_view(T *ptr, std::size_t n) noexcept : kernel_(ptr, {n}, {1}) {}

        OPTINUM_INLINE constexpr explicit vector_view(const kernel_type &k) noexcept : kernel_(k) {}

        // ==========================================================================
        // Size queries
        // ==========================================================================

        OPTINUM_INLINE constexpr std::size_t size() const noexcept { return kernel_.extent(0); }

        OPTINUM_INLINE constexpr std::size_t num_packs() const noexcept { return kernel_.num_packs(); }

        OPTINUM_INLINE constexpr std::size_t tail_size() const noexcept { return kernel_.tail_size(); }

        OPTINUM_INLINE constexpr bool is_contiguous() const noexcept { return kernel_.is_contiguous(); }

        // ==========================================================================
        // Element access (scalar) - integer index
        // ==========================================================================

        OPTINUM_INLINE value_type &operator[](std::size_t i) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot get non-const reference from const view");
            return kernel_.at_linear(i);
        }

        OPTINUM_INLINE const value_type &at(std::size_t i) const noexcept { return kernel_.at_linear_const(i); }

        // ==========================================================================
        // Pack access (SIMD)
        // ==========================================================================

        OPTINUM_INLINE pack<value_type, W> load_pack(std::size_t pack_idx) const noexcept {
            return kernel_.load_pack(pack_idx);
        }

        OPTINUM_INLINE void store_pack(std::size_t pack_idx, const pack<value_type, W> &v) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot store to const view");
            kernel_.store_pack(pack_idx, v);
        }

        // Tail-safe pack access (loads/stores only valid elements)
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
        // Subview (slice)
        // ==========================================================================

        OPTINUM_INLINE vector_view subview(std::size_t offset, std::size_t count) const noexcept {
            return vector_view(kernel_.data() + offset, count);
        }

        // Slicing with seq/fseq/all - use slice() method
        template <typename Slice> OPTINUM_INLINE vector_view slice(const Slice &s) const noexcept {
            static_assert(is_slice_v<Slice> || is_fixed_index_v<Slice>, "Invalid slice type");

            // Resolve slice to concrete indices
            seq sl = resolve_slice(s, size());

            // If step == 1, we can create a contiguous subview
            if (sl.step == 1) {
                return vector_view(kernel_.data() + sl.start, sl.size());
            }

            // For strided slicing, create a view with custom stride
            // Note: This requires the Kernel to support non-unit stride
            // For now, just return a contiguous copy of the indices (simplified)
            return vector_view(kernel_.data() + sl.start, sl.size());
        }
    };

} // namespace optinum::simd
