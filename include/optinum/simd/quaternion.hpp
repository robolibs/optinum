#pragma once

// =============================================================================
// optinum/simd/quaternion.hpp
// High-level quaternion array container with transparent SIMD
// =============================================================================
//
// This file provides a simple owning container for quaternion arrays.
// For non-owning views with SIMD operations, use quaternion_view.hpp
// For low-level SIMD pack operations, use pack/quaternion.hpp
//
// Usage:
//   Quaternion<double, 8> rotations;
//   rotations[0] = dp::mat::quaternion<double>::from_axis_angle(0, 0, 1, M_PI/4);
//   rotations.normalize_inplace();  // SIMD under the hood
//
// Or use the view() bridge for existing arrays:
//   dp::mat::quaternion<double> quats[8];
//   auto qv = view(quats);  // auto-detect SIMD width
//   qv.normalize_inplace();
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
     * @brief Quaternion array container with transparent SIMD operations
     *
     * High-level container for N quaternions. Operations use SIMD internally
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

        static constexpr std::size_t extent = N;
        static constexpr std::size_t rank = 1;

      private:
        pod_type pod_;

      public:
        // ===== CONSTRUCTORS =====

        constexpr Quaternion() noexcept : pod_() {}
        constexpr Quaternion(const Quaternion &) = default;
        constexpr Quaternion(Quaternion &&) noexcept = default;
        constexpr Quaternion &operator=(const Quaternion &) = default;
        constexpr Quaternion &operator=(Quaternion &&) noexcept = default;

        // Direct initialization from pod
        constexpr explicit Quaternion(const pod_type &p) noexcept : pod_(p) {}

        // ===== ELEMENT ACCESS =====

        [[nodiscard]] constexpr value_type &operator[](std::size_t i) noexcept { return pod_[i]; }
        [[nodiscard]] constexpr const value_type &operator[](std::size_t i) const noexcept { return pod_[i]; }

        // ===== RAW DATA ACCESS =====

        [[nodiscard]] constexpr value_type *data() noexcept { return pod_.data(); }
        [[nodiscard]] constexpr const value_type *data() const noexcept { return pod_.data(); }

        // Get underlying pod
        [[nodiscard]] constexpr pod_type &pod() noexcept { return pod_; }
        [[nodiscard]] constexpr const pod_type &pod() const noexcept { return pod_; }

        // Get SIMD view (for advanced operations)
        [[nodiscard]] view_type as_view() noexcept { return view(pod_); }
        [[nodiscard]] view_type as_view() const noexcept { return view(pod_); }

        // ===== SIZE =====

        [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }

        // ===== FILL OPERATIONS =====

        constexpr void fill(const value_type &val) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                pod_[i] = val;
            }
        }

        // ===== FACTORY FUNCTIONS =====

        [[nodiscard]] static constexpr Quaternion identity() noexcept {
            Quaternion q;
            q.fill(value_type::identity());
            return q;
        }

        [[nodiscard]] static constexpr Quaternion zeros() noexcept {
            Quaternion q;
            q.fill(value_type{T{}, T{}, T{}, T{}});
            return q;
        }

        // ===== IN-PLACE OPERATIONS (SIMD accelerated) =====

        void conjugate_inplace() noexcept { as_view().conjugate_inplace(); }

        void normalize_inplace() noexcept { as_view().normalize_inplace(); }

        void inverse_inplace() noexcept { as_view().inverse_inplace(); }

        // ===== OPERATIONS RETURNING NEW ARRAY =====

        [[nodiscard]] Quaternion conjugate() const noexcept {
            Quaternion result;
            as_view().conjugate_to(result.data());
            return result;
        }

        [[nodiscard]] Quaternion normalized() const noexcept {
            Quaternion result;
            as_view().normalized_to(result.data());
            return result;
        }

        [[nodiscard]] Quaternion inverse() const noexcept {
            Quaternion result;
            as_view().inverse_to(result.data());
            return result;
        }

        // ===== BINARY OPERATIONS =====

        [[nodiscard]] Quaternion operator*(const Quaternion &other) const noexcept {
            Quaternion result;
            as_view().multiply_to(other.as_view(), result.data());
            return result;
        }

        Quaternion &operator*=(const Quaternion &other) noexcept {
            Quaternion result;
            as_view().multiply_to(other.as_view(), result.data());
            pod_ = result.pod_;
            return *this;
        }

        // ===== INTERPOLATION =====

        [[nodiscard]] Quaternion slerp(const Quaternion &other, T t) const noexcept {
            Quaternion result;
            as_view().slerp_to(other.as_view(), t, result.data());
            return result;
        }

        [[nodiscard]] Quaternion nlerp(const Quaternion &other, T t) const noexcept {
            Quaternion result;
            as_view().nlerp_to(other.as_view(), t, result.data());
            return result;
        }

        // ===== REDUCTION OPERATIONS =====

        void norms(T *out) const noexcept { as_view().norms_to(out); }

        void dot(const Quaternion &other, T *out) const noexcept { as_view().dot_to(other.as_view(), out); }

        // ===== ROTATION OPERATIONS =====

        void rotate_vectors(T *vx, T *vy, T *vz) const noexcept { as_view().rotate_vectors(vx, vy, vz); }

        // ===== CONVERSION OPERATIONS =====

        void to_euler(T *roll, T *pitch, T *yaw) const noexcept { as_view().to_euler(roll, pitch, yaw); }

        static Quaternion from_euler(const T *roll, const T *pitch, const T *yaw) noexcept {
            Quaternion result;
            view_type::from_euler(roll, pitch, yaw, result.data(), N);
            return result;
        }

        void to_axis_angle(T *ax, T *ay, T *az, T *angle) const noexcept { as_view().to_axis_angle(ax, ay, az, angle); }

        // ===== STREAM OUTPUT =====

        friend std::ostream &operator<<(std::ostream &os, const Quaternion &q) {
            os << "[";
            for (std::size_t i = 0; i < N; ++i) {
                os << "(" << q[i].w << "+" << q[i].x << "i+" << q[i].y << "j+" << q[i].z << "k)";
                if (i + 1 < N)
                    os << ", ";
            }
            os << "]";
            return os;
        }

        // ===== ITERATORS =====

        value_type *begin() noexcept { return pod_.data(); }
        value_type *end() noexcept { return pod_.data() + N; }
        const value_type *begin() const noexcept { return pod_.data(); }
        const value_type *end() const noexcept { return pod_.data() + N; }
    };

    // ===== TYPE ALIASES =====

    template <std::size_t N> using Quaternionf = Quaternion<float, N>;
    template <std::size_t N> using Quaterniond = Quaternion<double, N>;

} // namespace optinum::simd
