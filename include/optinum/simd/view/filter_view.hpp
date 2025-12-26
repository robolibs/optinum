#pragma once

// =============================================================================
// optinum/simd/view/filter_view.hpp
// filter_view<T,W> - Non-owning view over masked/filtered elements
// =============================================================================

#include <optinum/simd/kernel.hpp>
#include <optinum/simd/mask.hpp>
#include <vector>

namespace optinum::simd {

    // Forward declarations
    template <typename T, std::size_t W> struct vector_view;

    // =============================================================================
    // filter_view<T, W> - View over elements selected by a boolean mask
    //
    // Provides conditional/masked access to elements without copying:
    //   - Only elements where mask[i] == true are accessible
    //   - Creates a compact index list of selected elements
    //   - Supports scalar access through operator[]
    //   - SIMD access via gather/scatter (where supported)
    //
    // Use cases:
    //   - Sparse operations
    //   - Conditional updates
    //   - Outlier filtering
    //   - Statistical filtering
    //
    // Provides:
    //   - size(): number of selected (true) elements
    //   - operator[]: scalar element access (i-th selected element)
    //   - load_pack_gather(i): gather the i-th pack (SIMD where supported)
    //   - store_pack_scatter(i, v): scatter to the i-th pack
    // =============================================================================

    template <typename T, std::size_t W> struct filter_view {
        using value_type = std::remove_const_t<T>;
        static constexpr std::size_t width = W;
        static constexpr std::size_t rank = 1;

        T *data_;                          // Pointer to underlying data
        std::vector<std::size_t> indices_; // Compact list of selected indices

        // ==========================================================================
        // Constructors
        // ==========================================================================

        OPTINUM_INLINE filter_view() noexcept : data_(nullptr) {}

        // Construct from data pointer and boolean mask array
        OPTINUM_INLINE filter_view(T *ptr, const bool *mask, std::size_t n) : data_(ptr) {
            indices_.reserve(n); // Reserve worst-case size
            for (std::size_t i = 0; i < n; ++i) {
                if (mask[i]) {
                    indices_.push_back(i);
                }
            }
        }

        // Construct from data pointer and mask object
        template <typename MaskT>
        OPTINUM_INLINE filter_view(T *ptr, const MaskT *masks, std::size_t num_masks, std::size_t total_size)
            : data_(ptr) {
            indices_.reserve(total_size);
            for (std::size_t mask_idx = 0; mask_idx < num_masks; ++mask_idx) {
                for (std::size_t lane = 0; lane < W && (mask_idx * W + lane) < total_size; ++lane) {
                    if (masks[mask_idx].test(lane)) {
                        indices_.push_back(mask_idx * W + lane);
                    }
                }
            }
        }

        // Construct from vector of indices (pre-computed)
        OPTINUM_INLINE filter_view(T *ptr, std::vector<std::size_t> &&indices) noexcept
            : data_(ptr), indices_(std::move(indices)) {}

        // ==========================================================================
        // Size queries
        // ==========================================================================

        OPTINUM_INLINE std::size_t size() const noexcept { return indices_.size(); }

        OPTINUM_INLINE bool empty() const noexcept { return indices_.empty(); }

        OPTINUM_INLINE std::size_t num_packs() const noexcept { return (size() + W - 1) / W; }

        OPTINUM_INLINE std::size_t tail_size() const noexcept {
            const std::size_t remainder = size() % W;
            return (remainder == 0 && size() > 0) ? W : remainder;
        }

        OPTINUM_INLINE constexpr bool is_contiguous() const noexcept {
            return false; // Filter views are generally non-contiguous
        }

        // ==========================================================================
        // Element access (scalar)
        // ==========================================================================

        // Access the i-th selected element
        OPTINUM_INLINE value_type &operator[](std::size_t i) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot get non-const reference from const view");
            return data_[indices_[i]];
        }

        OPTINUM_INLINE const value_type &at(std::size_t i) const noexcept { return data_[indices_[i]]; }

        // Get the original index of the i-th selected element
        OPTINUM_INLINE std::size_t index(std::size_t i) const noexcept { return indices_[i]; }

        // ==========================================================================
        // Pack access via gather/scatter
        // ==========================================================================

        // Gather the i-th pack of selected elements
        OPTINUM_INLINE pack<value_type, W> load_pack_gather(std::size_t pack_idx) const noexcept {
            const std::size_t base = pack_idx * W;

            // Scalar gather - gather elements one by one
            // Note: Could be optimized with native gather instructions (AVX-512, AVX2)
            // but requires additional pack<T,W> API for gather operations
            alignas(W * sizeof(value_type)) value_type gathered[W];
            for (std::size_t i = 0; i < W; ++i) {
                if (base + i < size()) {
                    gathered[i] = data_[indices_[base + i]];
                } else {
                    gathered[i] = value_type{}; // Padding
                }
            }
            return pack<value_type, W>::load(gathered);
        }

        // Scatter the i-th pack to selected elements
        OPTINUM_INLINE void store_pack_scatter(std::size_t pack_idx, const pack<value_type, W> &v) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot store to const view");

            const std::size_t base = pack_idx * W;
            alignas(W * sizeof(value_type)) value_type temp[W];
            v.store(temp);

            // Scalar scatter - write elements one by one
            for (std::size_t i = 0; i < W && (base + i) < size(); ++i) {
                data_[indices_[base + i]] = temp[i];
            }
        }

        // Tail-safe gather
        OPTINUM_INLINE pack<value_type, W> load_pack_gather_tail(std::size_t pack_idx) const noexcept {
            const std::size_t base = pack_idx * W;
            const std::size_t valid = (pack_idx == num_packs() - 1) ? tail_size() : W;

            alignas(W * sizeof(value_type)) value_type gathered[W];
            for (std::size_t i = 0; i < W; ++i) {
                if (i < valid && (base + i) < size()) {
                    gathered[i] = data_[indices_[base + i]];
                } else {
                    gathered[i] = value_type{}; // Padding
                }
            }
            return pack<value_type, W>::load(gathered);
        }

        // Tail-safe scatter
        OPTINUM_INLINE void store_pack_scatter_tail(std::size_t pack_idx, const pack<value_type, W> &v) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot store to const view");

            const std::size_t base = pack_idx * W;
            const std::size_t valid = (pack_idx == num_packs() - 1) ? tail_size() : W;

            alignas(W * sizeof(value_type)) value_type temp[W];
            v.store(temp);

            for (std::size_t i = 0; i < valid && (base + i) < size(); ++i) {
                data_[indices_[base + i]] = temp[i];
            }
        }

        // ==========================================================================
        // Data access
        // ==========================================================================

        OPTINUM_INLINE T *data() const noexcept { return data_; }

        OPTINUM_INLINE const value_type *data_const() const noexcept { return data_; }

        OPTINUM_INLINE const std::vector<std::size_t> &indices() const noexcept { return indices_; }
    };

    // =============================================================================
    // Helper functions to create filter views
    // =============================================================================

    // Create a filter view from a boolean mask array
    template <typename T, std::size_t W>
    OPTINUM_INLINE filter_view<T, W> filter(T *ptr, const bool *mask, std::size_t n) {
        return filter_view<T, W>(ptr, mask, n);
    }

    // Create a filter view from vector_view and boolean mask
    template <typename T, std::size_t W>
    OPTINUM_INLINE filter_view<T, W> filter(const vector_view<T, W> &v, const bool *mask) {
        return filter_view<T, W>(v.data(), mask, v.size());
    }

    // Create a filter view from a predicate function
    template <typename T, std::size_t W, typename Predicate>
    OPTINUM_INLINE filter_view<T, W> filter_if(T *ptr, std::size_t n, Predicate pred) {
        std::vector<std::size_t> indices;
        indices.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (pred(ptr[i])) {
                indices.push_back(i);
            }
        }
        return filter_view<T, W>(ptr, std::move(indices));
    }

    // Create a filter view from vector_view and predicate
    template <typename T, std::size_t W, typename Predicate>
    OPTINUM_INLINE filter_view<T, W> filter_if(const vector_view<T, W> &v, Predicate pred) {
        return filter_if<T, W>(v.data(), v.size(), pred);
    }

} // namespace optinum::simd
