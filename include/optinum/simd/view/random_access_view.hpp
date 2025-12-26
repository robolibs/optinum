#pragma once

// =============================================================================
// optinum/simd/view/random_access_view.hpp
// Non-contiguous element access using index arrays (gather/scatter)
// =============================================================================

#include <optinum/simd/kernel.hpp>
#include <optinum/simd/pack/pack.hpp>

#include <cstddef>
#include <type_traits>

namespace optinum::simd {

    /**
     * @brief Random access view - access elements via index array
     *
     * Allows non-contiguous access patterns like:
     *   data[indices[0]], data[indices[1]], ...
     *
     * Useful for:
     * - Sparse operations
     * - Indirect indexing
     * - Permutations/shuffles
     * - Graph algorithms
     */
    template <typename T, std::size_t W, std::size_t N> class random_access_view {
        static_assert(std::is_arithmetic_v<T>, "random_access_view requires arithmetic type");
        static_assert(W > 0 && N > 0, "Width and size must be > 0");

      public:
        using value_type = T;
        using pack_type = pack<T, W>;
        using index_type = std::size_t;

        static constexpr std::size_t width = W;
        static constexpr std::size_t size = N;
        static constexpr std::size_t rank = 1;

      private:
        T *data_;                   // Pointer to data array
        const index_type *indices_; // Pointer to index array (size N)

      public:
        // Constructors
        constexpr random_access_view() noexcept : data_(nullptr), indices_(nullptr) {}

        constexpr random_access_view(T *data, const index_type *indices) noexcept : data_(data), indices_(indices) {}

        // Element access (scalar)
        [[nodiscard]] constexpr T &operator[](std::size_t i) noexcept { return data_[indices_[i]]; }

        [[nodiscard]] constexpr const T &operator[](std::size_t i) const noexcept { return data_[indices_[i]]; }

        // Pack access (gather/scatter)
        // Note: Gather can be slow if indices are scattered in memory
        [[nodiscard]] pack_type load_pack(std::size_t i) const noexcept {
            alignas(32) T temp[W];
            for (std::size_t j = 0; j < W && (i + j) < N; ++j) {
                temp[j] = data_[indices_[i + j]];
            }
            // Pad with zeros if needed
            for (std::size_t j = i + W; j < W; ++j) {
                if ((i + j) >= N)
                    temp[j] = T{};
            }
            return pack_type::loadu(temp);
        }

        // Store pack (scatter)
        void store_pack(std::size_t i, const pack_type &p) noexcept {
            alignas(32) T temp[W];
            p.storeu(temp);
            for (std::size_t j = 0; j < W && (i + j) < N; ++j) {
                data_[indices_[i + j]] = temp[j];
            }
        }

        // Raw access
        [[nodiscard]] constexpr T *data() noexcept { return data_; }
        [[nodiscard]] constexpr const T *data() const noexcept { return data_; }
        [[nodiscard]] constexpr const index_type *indices() const noexcept { return indices_; }

        // Iteration helpers
        [[nodiscard]] constexpr std::size_t num_packs() const noexcept { return (N + W - 1) / W; }

        // Fill operation
        void fill(T value) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                data_[indices_[i]] = value;
            }
        }

        // Copy from contiguous source
        void copy_from(const T *src) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                data_[indices_[i]] = src[i];
            }
        }

        // Copy to contiguous destination
        void copy_to(T *dst) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                dst[i] = data_[indices_[i]];
            }
        }

        // Elementwise operations
        random_access_view &operator+=(const random_access_view &other) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                data_[indices_[i]] += other.data_[other.indices_[i]];
            }
            return *this;
        }

        random_access_view &operator-=(const random_access_view &other) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                data_[indices_[i]] -= other.data_[other.indices_[i]];
            }
            return *this;
        }

        random_access_view &operator*=(const random_access_view &other) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                data_[indices_[i]] *= other.data_[other.indices_[i]];
            }
            return *this;
        }

        random_access_view &operator*=(T scalar) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                data_[indices_[i]] *= scalar;
            }
            return *this;
        }
    };

    // Factory function
    template <typename T, std::size_t W, std::size_t N>
    [[nodiscard]] constexpr random_access_view<T, W, N> make_random_access_view(T *data,
                                                                                const std::size_t *indices) noexcept {
        return random_access_view<T, W, N>(data, indices);
    }

} // namespace optinum::simd
