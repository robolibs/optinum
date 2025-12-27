#pragma once

#include <datapod/matrix/math/quaternion.hpp>
#include <datapod/matrix/vector.hpp>
#include <optinum/simd/debug.hpp>

#include <cstddef>
#include <iostream>
#include <type_traits>

namespace optinum::simd {

    namespace dp = ::datapod;

    /**
     * @brief Quaternion array wrapper (owns data via datapod)
     *
     * High-level container for N quaternions, designed for SIMD operations.
     * Wraps datapod::mat::vector<quaternion<T>, N> and provides quaternion-specific
     * operations useful for Lie group optimization (SO(3)).
     *
     * Use cases:
     *   - Batch rotation operations
     *   - Lie group optimization on SO(3)
     *   - Parallel quaternion interpolation
     *   - SLAM, robotics, computer graphics
     *
     * Example:
     *   Quaternion<double, 8> rotations;
     *   rotations[0] = dp::mat::quaternion<double>::from_axis_angle(0, 0, 1, M_PI/4);
     *   auto normalized = rotations.normalized();
     */
    template <typename T, std::size_t N> class Quaternion {
        static_assert(N > 0, "Quaternion array size must be > 0");
        static_assert(std::is_floating_point_v<T>, "Quaternion<T, N> requires floating-point type");

      public:
        using value_type = dp::mat::quaternion<T>;
        using pod_type = dp::mat::vector<value_type, N>;
        using real_type = T;

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

        // ===== FILL OPERATIONS =====

        constexpr void fill(const value_type &val) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                pod_[i] = val;
            }
        }

        // ===== FACTORY FUNCTIONS =====

        // All identity quaternions (1, 0, 0, 0)
        [[nodiscard]] static constexpr Quaternion identity() noexcept {
            Quaternion q;
            q.fill(value_type::identity());
            return q;
        }

        // All zero quaternions (0, 0, 0, 0) - not valid rotations, but useful for accumulators
        [[nodiscard]] static constexpr Quaternion zeros() noexcept {
            Quaternion q;
            q.fill(value_type{T{}, T{}, T{}, T{}});
            return q;
        }

        // ===== COMPONENT EXTRACTION =====

        // Extract w (scalar) components
        constexpr void w_parts(T *out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = pod_[i].w;
            }
        }

        // Extract x components
        constexpr void x_parts(T *out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = pod_[i].x;
            }
        }

        // Extract y components
        constexpr void y_parts(T *out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = pod_[i].y;
            }
        }

        // Extract z components
        constexpr void z_parts(T *out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = pod_[i].z;
            }
        }

        // ===== QUATERNION OPERATIONS =====

        // Conjugate all quaternions
        [[nodiscard]] constexpr Quaternion conjugate() const noexcept {
            Quaternion result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = pod_[i].conjugate();
            }
            return result;
        }

        // Inverse all quaternions
        [[nodiscard]] Quaternion inverse() const noexcept {
            Quaternion result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = pod_[i].inverse();
            }
            return result;
        }

        // Normalize all quaternions to unit quaternions
        [[nodiscard]] Quaternion normalized() const noexcept {
            Quaternion result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = pod_[i].normalized();
            }
            return result;
        }

        // Get norms of all quaternions
        void norms(T *out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = pod_[i].norm();
            }
        }

        // Get squared norms of all quaternions
        constexpr void norms_squared(T *out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = pod_[i].norm_squared();
            }
        }

        // ===== ARITHMETIC OPERATORS =====

        // Component-wise addition
        [[nodiscard]] constexpr Quaternion operator+(const Quaternion &other) const noexcept {
            Quaternion result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = pod_[i] + other[i];
            }
            return result;
        }

        // Component-wise subtraction
        [[nodiscard]] constexpr Quaternion operator-(const Quaternion &other) const noexcept {
            Quaternion result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = pod_[i] - other[i];
            }
            return result;
        }

        // Hamilton product (element-wise quaternion multiplication)
        [[nodiscard]] constexpr Quaternion operator*(const Quaternion &other) const noexcept {
            Quaternion result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = pod_[i] * other[i];
            }
            return result;
        }

        // Quaternion division (element-wise)
        [[nodiscard]] Quaternion operator/(const Quaternion &other) const noexcept {
            Quaternion result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = pod_[i] / other[i];
            }
            return result;
        }

        // Scalar multiplication
        [[nodiscard]] constexpr Quaternion operator*(T scalar) const noexcept {
            Quaternion result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = pod_[i] * scalar;
            }
            return result;
        }

        [[nodiscard]] friend constexpr Quaternion operator*(T scalar, const Quaternion &q) noexcept {
            return q * scalar;
        }

        // Scalar division
        [[nodiscard]] constexpr Quaternion operator/(T scalar) const noexcept {
            Quaternion result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = pod_[i] / scalar;
            }
            return result;
        }

        // ===== COMPOUND ASSIGNMENT =====

        constexpr Quaternion &operator+=(const Quaternion &other) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                pod_[i] += other[i];
            }
            return *this;
        }

        constexpr Quaternion &operator-=(const Quaternion &other) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                pod_[i] -= other[i];
            }
            return *this;
        }

        constexpr Quaternion &operator*=(const Quaternion &other) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                pod_[i] *= other[i];
            }
            return *this;
        }

        Quaternion &operator/=(const Quaternion &other) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                pod_[i] /= other[i];
            }
            return *this;
        }

        constexpr Quaternion &operator*=(T scalar) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                pod_[i] *= scalar;
            }
            return *this;
        }

        // ===== LIE GROUP OPERATIONS =====
        // Essential for SO(3) optimization

        // Dot product (element-wise, returns array of scalars)
        void dot(const Quaternion &other, T *out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = dp::mat::dot(pod_[i], other[i]);
            }
        }

        // Normalized linear interpolation (element-wise)
        [[nodiscard]] Quaternion nlerp(const Quaternion &other, T t) const noexcept {
            Quaternion result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = dp::mat::nlerp(pod_[i], other[i], t);
            }
            return result;
        }

        // Spherical linear interpolation (element-wise)
        [[nodiscard]] Quaternion slerp(const Quaternion &other, T t) const noexcept {
            Quaternion result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = dp::mat::slerp(pod_[i], other[i], t);
            }
            return result;
        }

        // Exponential map (from Lie algebra so(3) to SO(3))
        // Input: array of axis-angle vectors (pure quaternions with w=0)
        [[nodiscard]] static Quaternion exp(const Quaternion &lie_algebra) noexcept {
            Quaternion result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = dp::mat::exp(lie_algebra[i]);
            }
            return result;
        }

        // Logarithm map (from SO(3) to Lie algebra so(3))
        // Returns pure quaternions (w ≈ 0 for unit quaternions)
        [[nodiscard]] Quaternion log() const noexcept {
            Quaternion result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = dp::mat::log(pod_[i]);
            }
            return result;
        }

        // ===== ROTATION OPERATIONS =====

        // Rotate vectors by these quaternions
        // vx, vy, vz are arrays of N vectors
        void rotate_vectors(T *vx, T *vy, T *vz) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                pod_[i].rotate_vector(vx[i], vy[i], vz[i]);
            }
        }

        // ===== EULER ANGLE CONVERSION =====

        // Convert to Euler angles (roll, pitch, yaw in radians)
        void to_euler(T *roll, T *pitch, T *yaw) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                pod_[i].to_euler(roll[i], pitch[i], yaw[i]);
            }
        }

        // Create from Euler angles
        static Quaternion from_euler(const T *roll, const T *pitch, const T *yaw) noexcept {
            Quaternion result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = value_type::from_euler(roll[i], pitch[i], yaw[i]);
            }
            return result;
        }

        // ===== AXIS-ANGLE CONVERSION =====

        // Create from axis-angle representation
        static Quaternion from_axis_angle(const T *ax, const T *ay, const T *az, const T *angle) noexcept {
            Quaternion result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = value_type::from_axis_angle(ax[i], ay[i], az[i], angle[i]);
            }
            return result;
        }

        // Extract axis-angle representation
        void to_axis_angle(T *ax, T *ay, T *az, T *angle) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                pod_[i].to_axis_angle(ax[i], ay[i], az[i], angle[i]);
            }
        }

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
    };

    // ===== TYPE ALIASES =====

    template <std::size_t N> using Quaternionf = Quaternion<float, N>;
    template <std::size_t N> using Quaterniond = Quaternion<double, N>;

} // namespace optinum::simd
