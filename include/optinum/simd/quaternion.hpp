#pragma once

// =============================================================================
// optinum/simd/quaternion.hpp
// Non-owning view over quaternion arrays with transparent SIMD operations
// =============================================================================
//
// This file provides a non-owning view over quaternion arrays.
// For SIMD pack operations, use pack/quaternion.hpp
// For lower-level SIMD views, use view/quaternion_view.hpp
//
// Usage:
//   dp::mat::vector<dp::mat::quaternion<double>, 8> storage;
//   Quaternion<double, 8> rotations(storage);
//   rotations[0] = dp::mat::quaternion<double>::from_axis_angle(0, 0, 1, M_PI/4);
//   rotations.normalize_inplace();  // SIMD under the hood
//
// Or from raw pointer:
//   dp::mat::quaternion<double>* ptr = ...;
//   Quaternion<double, 8> view(ptr, 8);
// =============================================================================

#include <datapod/matrix/math/quaternion.hpp>
#include <datapod/matrix/vector.hpp>
#include <optinum/simd/bridge.hpp>

#include <cstddef>
#include <iostream>
#include <type_traits>

namespace optinum::simd {

    namespace dp = ::datapod;

    /**
     * @brief Non-owning view over quaternion arrays with transparent SIMD operations
     *
     * Provides a view over N quaternions. Operations use SIMD internally
     * via quaternion_view. User works with dp::mat::quaternion<T> directly.
     *
     * Use cases:
     *   - Batch rotation operations
     *   - Lie group optimization on SO(3)
     *   - Parallel quaternion interpolation
     *   - SLAM, robotics, computer graphics
     */
    template <typename T, std::size_t N> class Quaternion {
        static_assert(N > 0, "Quaternion array size must be > 0");
        static_assert(std::is_floating_point_v<T>, "Quaternion<T, N> requires floating-point type");

      public:
        using value_type = dp::mat::quaternion<T>;
        using pod_type = dp::mat::vector<value_type, N>;
        using real_type = T;
        using view_type = quaternion_view<T, detail::default_width<T>()>;
        using size_type = std::size_t;
        using pointer = value_type *;
        using const_pointer = const value_type *;

        static constexpr std::size_t extent = N;
        static constexpr std::size_t rank = 1;

      private:
        pointer ptr_ = nullptr;
        size_type size_ = 0;

      public:
        // ===== CONSTRUCTORS =====

        // Default constructor (null view)
        constexpr Quaternion() noexcept = default;

        // Constructor from raw pointer (non-owning view)
        constexpr Quaternion(pointer ptr, size_type n) noexcept : ptr_(ptr), size_(n) {}

        // Constructor from const pointer (non-owning view)
        constexpr Quaternion(const_pointer ptr, size_type n) noexcept : ptr_(const_cast<pointer>(ptr)), size_(n) {}

        // Constructor from pod_type reference (non-owning view)
        constexpr Quaternion(pod_type &pod) noexcept : ptr_(pod.data()), size_(N) {}
        constexpr Quaternion(const pod_type &pod) noexcept : ptr_(const_cast<pointer>(pod.data())), size_(N) {}

        // Copy/move (shallow - copies pointer, not data)
        constexpr Quaternion(const Quaternion &) noexcept = default;
        constexpr Quaternion(Quaternion &&) noexcept = default;
        constexpr Quaternion &operator=(const Quaternion &) noexcept = default;
        constexpr Quaternion &operator=(Quaternion &&) noexcept = default;

        // ===== VALIDITY CHECK =====

        [[nodiscard]] constexpr bool valid() const noexcept { return ptr_ != nullptr; }
        constexpr explicit operator bool() const noexcept { return valid(); }

        // ===== ELEMENT ACCESS =====

        [[nodiscard]] constexpr value_type &operator[](size_type i) noexcept { return ptr_[i]; }
        [[nodiscard]] constexpr const value_type &operator[](size_type i) const noexcept { return ptr_[i]; }

        // ===== RAW DATA ACCESS =====

        [[nodiscard]] constexpr pointer data() noexcept { return ptr_; }
        [[nodiscard]] constexpr const_pointer data() const noexcept { return ptr_; }

        // Get SIMD view (for advanced operations)
        [[nodiscard]] view_type as_view() noexcept { return view_type(ptr_, size_); }
        [[nodiscard]] view_type as_view() const noexcept { return view_type(ptr_, size_); }

        // ===== SIZE =====

        [[nodiscard]] constexpr size_type size() const noexcept { return size_; }
        [[nodiscard]] static constexpr bool empty() noexcept { return false; }

        // ===== FILL OPERATIONS =====

        constexpr void fill(const value_type &val) noexcept {
            for (size_type i = 0; i < size_; ++i) {
                ptr_[i] = val;
            }
        }

        // ===== IN-PLACE OPERATIONS (SIMD accelerated) =====

        void conjugate_inplace() noexcept { as_view().conjugate_inplace(); }

        void normalize_inplace() noexcept { as_view().normalize_inplace(); }

        void inverse_inplace() noexcept { as_view().inverse_inplace(); }

        // ===== OPERATIONS WRITING TO OUTPUT =====

        void conjugate_to(pointer out) const noexcept { (void)as_view().conjugate_to(out); }

        void normalized_to(pointer out) const noexcept { (void)as_view().normalized_to(out); }

        void inverse_to(pointer out) const noexcept { (void)as_view().inverse_to(out); }

        // ===== BINARY OPERATIONS =====

        void multiply_to(const Quaternion &other, pointer out) const noexcept {
            (void)as_view().multiply_to(other.as_view(), out);
        }

        void multiply_inplace(const Quaternion &other) noexcept {
            // Need temporary storage for in-place multiply
            alignas(32) value_type temp[N];
            (void)as_view().multiply_to(other.as_view(), temp);
            for (size_type i = 0; i < size_; ++i) {
                ptr_[i] = temp[i];
            }
        }

        // ===== INTERPOLATION =====

        void slerp_to(const Quaternion &other, T t, pointer out) const noexcept {
            (void)as_view().slerp_to(other.as_view(), t, out);
        }

        void nlerp_to(const Quaternion &other, T t, pointer out) const noexcept {
            (void)as_view().nlerp_to(other.as_view(), t, out);
        }

        // ===== REDUCTION OPERATIONS =====

        void norms_to(T *out) const noexcept { as_view().norms_to(out); }

        void dot_to(const Quaternion &other, T *out) const noexcept { as_view().dot_to(other.as_view(), out); }

        // ===== ROTATION OPERATIONS =====

        void rotate_vectors(T *vx, T *vy, T *vz) const noexcept { as_view().rotate_vectors(vx, vy, vz); }

        // ===== CONVERSION OPERATIONS =====

        void to_euler(T *roll, T *pitch, T *yaw) const noexcept { as_view().to_euler(roll, pitch, yaw); }

        static void from_euler(const T *roll, const T *pitch, const T *yaw, pointer out) noexcept {
            view_type::from_euler(roll, pitch, yaw, out, N);
        }

        void to_axis_angle(T *ax, T *ay, T *az, T *angle) const noexcept { as_view().to_axis_angle(ax, ay, az, angle); }

        // ===== STREAM OUTPUT =====

        friend std::ostream &operator<<(std::ostream &os, const Quaternion &q) {
            os << "[";
            for (size_type i = 0; i < q.size_; ++i) {
                os << "(" << q[i].w << "+" << q[i].x << "i+" << q[i].y << "j+" << q[i].z << "k)";
                if (i + 1 < q.size_)
                    os << ", ";
            }
            os << "]";
            return os;
        }

        // ===== ITERATORS =====

        pointer begin() noexcept { return ptr_; }
        pointer end() noexcept { return ptr_ + size_; }
        const_pointer begin() const noexcept { return ptr_; }
        const_pointer end() const noexcept { return ptr_ + size_; }
    };

    // ===== TYPE ALIASES =====

    template <std::size_t N> using Quaternionf = Quaternion<float, N>;
    template <std::size_t N> using Quaterniond = Quaternion<double, N>;

} // namespace optinum::simd
