#pragma once

// =============================================================================
// optinum/simd/kernel.hpp
// Kernel<T,W,Rank> - Memory layout descriptor for SIMD views
// =============================================================================

#include <optinum/simd/mask.hpp>
#include <optinum/simd/pack/pack.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace optinum::simd {

    // =============================================================================
    // Kernel<T, W, Rank> - Low-level memory layout descriptor
    //
    // This is a non-owning view that describes how to access memory in SIMD packs.
    // It stores:
    //   - ptr: pointer to data (can be const T*)
    //   - extents: size in each dimension
    //   - strides: element stride in each dimension
    //
    // The Kernel provides pack-based access:
    //   - load_pack(i): load the i-th pack
    //   - store_pack(i, v): store to the i-th pack
    //   - load_pack_tail(i, n): load partial pack with n valid elements
    //   - store_pack_tail(i, v, n): store partial pack with n valid elements
    // =============================================================================

    template <typename T, std::size_t W, std::size_t Rank> struct Kernel {
        static_assert(W > 0, "Kernel<T,W,Rank> requires W > 0");
        static_assert(Rank >= 0, "Kernel<T,W,Rank> requires Rank >= 0");
        static_assert(std::is_arithmetic_v<T> || std::is_same_v<T, const float> || std::is_same_v<T, const double> ||
                          std::is_same_v<T, const int32_t> || std::is_same_v<T, const int64_t>,
                      "Kernel<T,W,Rank> requires arithmetic T or const arithmetic T");

        using value_type = std::remove_const_t<T>;
        using pointer_type = T *;
        using const_pointer_type = const std::remove_const_t<T> *;
        static constexpr std::size_t width = W;
        static constexpr std::size_t rank = Rank;

        // Storage
        pointer_type ptr_ = nullptr;
        std::array<std::size_t, Rank> extents_{};
        std::array<std::size_t, Rank> strides_{};

        // ==========================================================================
        // Constructors
        // ==========================================================================

        OPTINUM_INLINE constexpr Kernel() noexcept = default;

        OPTINUM_INLINE constexpr Kernel(pointer_type ptr, const std::array<std::size_t, Rank> &extents,
                                        const std::array<std::size_t, Rank> &strides) noexcept
            : ptr_(ptr), extents_(extents), strides_(strides) {}

        // ==========================================================================
        // Metadata queries
        // ==========================================================================

        OPTINUM_INLINE constexpr std::size_t extent(std::size_t dim) const noexcept { return extents_[dim]; }

        OPTINUM_INLINE constexpr std::size_t stride(std::size_t dim) const noexcept { return strides_[dim]; }

        OPTINUM_INLINE constexpr std::size_t linear_size() const noexcept {
            if constexpr (Rank == 0) {
                return 1;
            } else {
                std::size_t size = 1;
                for (std::size_t i = 0; i < Rank; ++i)
                    size *= extents_[i];
                return size;
            }
        }

        OPTINUM_INLINE constexpr std::size_t num_packs() const noexcept { return (linear_size() + W - 1) / W; }

        OPTINUM_INLINE constexpr std::size_t tail_size() const noexcept {
            const std::size_t rem = linear_size() % W;
            return (rem == 0) ? W : rem;
        }

        OPTINUM_INLINE constexpr bool is_contiguous() const noexcept {
            if constexpr (Rank == 0) {
                return true;
            } else if constexpr (Rank == 1) {
                return strides_[0] == 1;
            } else {
                // Check if stride[i] == product(extents[0..i-1])
                std::size_t expected_stride = 1;
                for (std::size_t i = 0; i < Rank; ++i) {
                    if (strides_[i] != expected_stride)
                        return false;
                    expected_stride *= extents_[i];
                }
                return true;
            }
        }

        // ==========================================================================
        // Linear (contiguous) pack access
        // These assume contiguous memory layout
        // ==========================================================================

        OPTINUM_INLINE pack<value_type, W> load_pack(std::size_t pack_idx) const noexcept {
            return pack<value_type, W>::loadu(ptr_ + pack_idx * W);
        }

        OPTINUM_INLINE void store_pack(std::size_t pack_idx, const pack<value_type, W> &v) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot store to const Kernel");
            v.storeu(ptr_ + pack_idx * W);
        }

        // Tail access: load/store only the first 'valid' elements
        OPTINUM_INLINE pack<value_type, W> load_pack_tail(std::size_t pack_idx, std::size_t valid) const noexcept {
            auto m = mask<value_type, W>::first_n(valid);
            return maskload(ptr_ + pack_idx * W, m);
        }

        OPTINUM_INLINE void store_pack_tail(std::size_t pack_idx, const pack<value_type, W> &v,
                                            std::size_t valid) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot store to const Kernel");
            auto m = mask<value_type, W>::first_n(valid);
            maskstore(ptr_ + pack_idx * W, v, m);
        }

        // ==========================================================================
        // Scalar access (linear indexing)
        // ==========================================================================

        OPTINUM_INLINE value_type &at_linear(std::size_t i) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot get non-const reference from const Kernel");
            if constexpr (Rank == 1) {
                return ptr_[i * strides_[0]];
            } else {
                return ptr_[i];
            }
        }

        OPTINUM_INLINE const value_type &at_linear_const(std::size_t i) const noexcept {
            if constexpr (Rank == 1) {
                return ptr_[i * strides_[0]];
            } else {
                return ptr_[i];
            }
        }

        // ==========================================================================
        // Multi-dimensional indexing (for Rank > 1)
        // ==========================================================================

        template <typename... Indices> OPTINUM_INLINE std::size_t compute_offset(Indices... indices) const noexcept {
            static_assert(sizeof...(Indices) == Rank, "Number of indices must match Rank");
            std::array<std::size_t, Rank> idx_array = {static_cast<std::size_t>(indices)...};
            std::size_t offset = 0;
            for (std::size_t i = 0; i < Rank; ++i) {
                offset += idx_array[i] * strides_[i];
            }
            return offset;
        }

        template <typename... Indices> OPTINUM_INLINE value_type &at(Indices... indices) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot get non-const reference from const Kernel");
            return ptr_[compute_offset(indices...)];
        }

        template <typename... Indices> OPTINUM_INLINE const value_type &at_const(Indices... indices) const noexcept {
            return ptr_[compute_offset(indices...)];
        }

        // ==========================================================================
        // Pointer access
        // ==========================================================================

        OPTINUM_INLINE pointer_type data() const noexcept { return ptr_; }

        OPTINUM_INLINE const_pointer_type data_const() const noexcept { return ptr_; }
    };

    // =============================================================================
    // Kernel specialization for Rank=0 (scalar)
    // =============================================================================

    template <typename T, std::size_t W> struct Kernel<T, W, 0> {
        using value_type = std::remove_const_t<T>;
        using pointer_type = T *;
        using const_pointer_type = const std::remove_const_t<T> *;
        static constexpr std::size_t width = W;
        static constexpr std::size_t rank = 0;

        pointer_type ptr_ = nullptr;
        std::array<std::size_t, 0> extents_{};
        std::array<std::size_t, 0> strides_{};

        OPTINUM_INLINE constexpr Kernel() noexcept = default;

        OPTINUM_INLINE constexpr Kernel(pointer_type ptr) noexcept : ptr_(ptr) {}

        OPTINUM_INLINE constexpr std::size_t linear_size() const noexcept { return 1; }
        OPTINUM_INLINE constexpr std::size_t num_packs() const noexcept { return 1; }
        OPTINUM_INLINE constexpr std::size_t tail_size() const noexcept { return 1; }
        OPTINUM_INLINE constexpr bool is_contiguous() const noexcept { return true; }

        OPTINUM_INLINE value_type &get() const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot get non-const reference from const Kernel");
            return *ptr_;
        }

        OPTINUM_INLINE const value_type &get_const() const noexcept { return *ptr_; }

        OPTINUM_INLINE pointer_type data() const noexcept { return ptr_; }
        OPTINUM_INLINE const_pointer_type data_const() const noexcept { return ptr_; }
    };

} // namespace optinum::simd
