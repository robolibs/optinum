#pragma once

#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/so3.hpp>

#include <datapod/matrix/matrix.hpp>
#include <datapod/matrix/vector.hpp>

#include <cmath>
#include <type_traits>

namespace optinum::lie {

    namespace dp = ::datapod;

    // ===== EULER ANGLE CONVENTIONS =====
    //
    // EulerAnglesZYX (yaw-pitch-roll): Most common in robotics
    //   - First rotation: yaw (Z axis)
    //   - Second rotation: pitch (Y' axis)
    //   - Third rotation: roll (X'' axis)
    //   - Storage: [yaw, pitch, roll]
    //   - Gimbal lock at pitch = ±π/2
    //
    // EulerAnglesXYZ (roll-pitch-yaw): Aerospace convention
    //   - First rotation: roll (X axis)
    //   - Second rotation: pitch (Y' axis)
    //   - Third rotation: yaw (Z'' axis)
    //   - Storage: [roll, pitch, yaw]
    //   - Gimbal lock at pitch = ±π/2
    //
    // Jacobians:
    //   - euler_rates_to_angular_velocity: Maps Euler angle rates to body angular velocity
    //   - angular_velocity_to_euler_rates: Maps body angular velocity to Euler angle rates (inverse)

    /// EulerAnglesZYX: Euler angles in Z-Y'-X'' (yaw-pitch-roll) convention.
    ///
    /// This is the most common convention in robotics. The rotation is applied as:
    /// R = Rz(yaw) * Ry(pitch) * Rx(roll)
    ///
    /// Storage order: [yaw, pitch, roll]
    template <typename T = double> class EulerAnglesZYX {
        static_assert(std::is_floating_point_v<T>, "EulerAnglesZYX requires floating-point scalar type");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Vector3 = dp::mat::vector<T, 3>;    // Owning storage type
        using Matrix3 = dp::mat::matrix<T, 3, 3>; // Owning storage type

        // ===== CONSTRUCTORS =====

        /// Default constructor: identity (all angles zero)
        constexpr EulerAnglesZYX() noexcept : yaw_(T(0)), pitch_(T(0)), roll_(T(0)) {}

        /// Construct from yaw, pitch, roll angles (in radians)
        constexpr EulerAnglesZYX(T yaw, T pitch, T roll) noexcept : yaw_(yaw), pitch_(pitch), roll_(roll) {}

        /// Construct from Vector3 [yaw, pitch, roll]
        explicit constexpr EulerAnglesZYX(const Vector3 &v) noexcept : yaw_(v[0]), pitch_(v[1]), roll_(v[2]) {}

        /// Construct from SO3 rotation
        explicit EulerAnglesZYX(const SO3<T> &rotation) noexcept { set_from_rotation(rotation); }

        // ===== STATIC FACTORY METHODS =====

        /// Identity (zero rotation)
        [[nodiscard]] static constexpr EulerAnglesZYX identity() noexcept { return EulerAnglesZYX(); }

        /// Create from SO3 rotation
        [[nodiscard]] static EulerAnglesZYX from_rotation(const SO3<T> &rotation) noexcept {
            return EulerAnglesZYX(rotation);
        }

        /// Create from rotation matrix
        [[nodiscard]] static EulerAnglesZYX from_rotation_matrix(const Matrix3 &R) noexcept {
            return EulerAnglesZYX(SO3<T>(R));
        }

        // ===== ACCESSORS =====

        [[nodiscard]] constexpr T yaw() const noexcept { return yaw_; }
        [[nodiscard]] constexpr T pitch() const noexcept { return pitch_; }
        [[nodiscard]] constexpr T roll() const noexcept { return roll_; }

        /// Alias: z() = yaw()
        [[nodiscard]] constexpr T z() const noexcept { return yaw_; }
        /// Alias: y() = pitch()
        [[nodiscard]] constexpr T y() const noexcept { return pitch_; }
        /// Alias: x() = roll()
        [[nodiscard]] constexpr T x() const noexcept { return roll_; }

        /// Get as vector [yaw, pitch, roll]
        [[nodiscard]] Vector3 vector() const noexcept { return Vector3{{yaw_, pitch_, roll_}}; }

        // ===== MUTATORS =====

        void set_yaw(T yaw) noexcept { yaw_ = yaw; }
        void set_pitch(T pitch) noexcept { pitch_ = pitch; }
        void set_roll(T roll) noexcept { roll_ = roll; }

        void set_z(T z) noexcept { yaw_ = z; }
        void set_y(T y) noexcept { pitch_ = y; }
        void set_x(T x) noexcept { roll_ = x; }

        /// Set to identity
        EulerAnglesZYX &set_identity() noexcept {
            yaw_ = pitch_ = roll_ = T(0);
            return *this;
        }

        // ===== CONVERSION TO/FROM SO3 =====

        /// Convert to SO3 rotation
        [[nodiscard]] SO3<T> to_rotation() const noexcept {
            // R = Rz(yaw) * Ry(pitch) * Rx(roll)
            return SO3<T>::rot_z(yaw_) * SO3<T>::rot_y(pitch_) * SO3<T>::rot_x(roll_);
        }

        /// Convert to rotation matrix
        [[nodiscard]] Matrix3 to_rotation_matrix() const noexcept { return to_rotation().matrix(); }

        /// Set from SO3 rotation
        void set_from_rotation(const SO3<T> &rotation) noexcept {
            // Extract Euler angles from rotation matrix using ZYX convention
            const auto R = rotation.matrix();

            // pitch = -asin(R(2,0))
            // Handle gimbal lock
            const T sinp = -R(2, 0);

            if (std::abs(sinp) >= T(1) - epsilon<T>) {
                // Gimbal lock: pitch = ±π/2
                pitch_ = std::copysign(half_pi<T>, sinp);
                // In gimbal lock, yaw and roll are coupled
                // Convention: set roll = 0, compute yaw
                roll_ = T(0);
                yaw_ = std::atan2(-R(0, 1), R(1, 1));
            } else {
                pitch_ = std::asin(sinp);
                roll_ = std::atan2(R(2, 1), R(2, 2));
                yaw_ = std::atan2(R(1, 0), R(0, 0));
            }
        }

        // ===== GIMBAL LOCK DETECTION =====

        /// Check if near gimbal lock (pitch ≈ ±π/2)
        [[nodiscard]] bool is_near_gimbal_lock(T tolerance = T(0.01)) const noexcept {
            return std::abs(std::abs(pitch_) - half_pi<T>) < tolerance;
        }

        /// Get distance to gimbal lock (0 = at gimbal lock, π/2 = far from gimbal lock)
        [[nodiscard]] T gimbal_lock_distance() const noexcept { return std::abs(std::abs(pitch_) - half_pi<T>); }

        // ===== JACOBIANS =====

        /// Jacobian mapping Euler angle rates to body-frame angular velocity.
        ///
        /// ω_body = J * [yaw_dot, pitch_dot, roll_dot]^T
        ///
        /// This is the "E" matrix in many robotics texts.
        /// Note: This Jacobian becomes singular at gimbal lock (pitch = ±π/2).
        [[nodiscard]] Matrix3 euler_rates_to_angular_velocity() const noexcept {
            const T cp = std::cos(pitch_);
            const T sp = std::sin(pitch_);
            const T cr = std::cos(roll_);
            const T sr = std::sin(roll_);

            Matrix3 J;
            // Column 0: contribution of yaw_dot
            J(0, 0) = -sp;
            J(1, 0) = cp * sr;
            J(2, 0) = cp * cr;

            // Column 1: contribution of pitch_dot
            J(0, 1) = T(0);
            J(1, 1) = cr;
            J(2, 1) = -sr;

            // Column 2: contribution of roll_dot
            J(0, 2) = T(1);
            J(1, 2) = T(0);
            J(2, 2) = T(0);

            return J;
        }

        /// Jacobian mapping body-frame angular velocity to Euler angle rates.
        ///
        /// [yaw_dot, pitch_dot, roll_dot]^T = J_inv * ω_body
        ///
        /// This is the inverse of euler_rates_to_angular_velocity().
        /// Note: This Jacobian is undefined at gimbal lock (pitch = ±π/2).
        ///
        /// @return The Jacobian matrix, or identity if at gimbal lock
        [[nodiscard]] Matrix3 angular_velocity_to_euler_rates() const noexcept {
            const T cp = std::cos(pitch_);

            // Check for gimbal lock
            if (std::abs(cp) < epsilon<T>) {
                // At gimbal lock, return identity as a fallback
                // Caller should check is_near_gimbal_lock() before using this
                Matrix3 I;
                I(0, 0) = T(1);
                I(0, 1) = T(0);
                I(0, 2) = T(0);
                I(1, 0) = T(0);
                I(1, 1) = T(1);
                I(1, 2) = T(0);
                I(2, 0) = T(0);
                I(2, 1) = T(0);
                I(2, 2) = T(1);
                return I;
            }

            const T sp = std::sin(pitch_);
            const T cr = std::cos(roll_);
            const T sr = std::sin(roll_);
            const T inv_cp = T(1) / cp;

            Matrix3 J_inv;
            // Row 0: yaw_dot
            J_inv(0, 0) = T(0);
            J_inv(0, 1) = sr * inv_cp;
            J_inv(0, 2) = cr * inv_cp;

            // Row 1: pitch_dot
            J_inv(1, 0) = T(0);
            J_inv(1, 1) = cr;
            J_inv(1, 2) = -sr;

            // Row 2: roll_dot
            J_inv(2, 0) = T(1);
            J_inv(2, 1) = sr * sp * inv_cp;
            J_inv(2, 2) = cr * sp * inv_cp;

            return J_inv;
        }

        // ===== NORMALIZATION =====

        /// Wrap angles to canonical range: yaw,roll in [-π,π), pitch in [-π/2,π/2)
        [[nodiscard]] EulerAnglesZYX get_unique() const noexcept {
            // Wrap all angles to [-π, π)
            auto wrap = [](T angle) -> T {
                angle = std::fmod(angle + pi<T>, two_pi<T>);
                if (angle < T(0))
                    angle += two_pi<T>;
                return angle - pi<T>;
            };

            T y = wrap(yaw_);
            T p = wrap(pitch_);
            T r = wrap(roll_);

            // If pitch is outside [-π/2, π/2), flip it
            if (p < -half_pi<T>) {
                y = wrap(y + pi<T>);
                p = -(p + pi<T>);
                r = wrap(r + pi<T>);
            } else if (p > half_pi<T>) {
                y = wrap(y + pi<T>);
                p = -(p - pi<T>);
                r = wrap(r + pi<T>);
            }

            return EulerAnglesZYX(y, p, r);
        }

        /// Normalize angles in place
        EulerAnglesZYX &set_unique() noexcept {
            *this = get_unique();
            return *this;
        }

        // ===== COMPARISON =====

        [[nodiscard]] bool operator==(const EulerAnglesZYX &other) const noexcept {
            return std::abs(yaw_ - other.yaw_) < epsilon<T> && std::abs(pitch_ - other.pitch_) < epsilon<T> &&
                   std::abs(roll_ - other.roll_) < epsilon<T>;
        }

        [[nodiscard]] bool operator!=(const EulerAnglesZYX &other) const noexcept { return !(*this == other); }

        [[nodiscard]] bool is_approx(const EulerAnglesZYX &other, T tol = epsilon<T>) const noexcept {
            return std::abs(yaw_ - other.yaw_) < tol && std::abs(pitch_ - other.pitch_) < tol &&
                   std::abs(roll_ - other.roll_) < tol;
        }

        [[nodiscard]] bool is_identity(T tol = epsilon<T>) const noexcept {
            return std::abs(yaw_) < tol && std::abs(pitch_) < tol && std::abs(roll_) < tol;
        }

        // ===== TYPE CONVERSION =====

        template <typename NewScalar> [[nodiscard]] EulerAnglesZYX<NewScalar> cast() const noexcept {
            return EulerAnglesZYX<NewScalar>(static_cast<NewScalar>(yaw_), static_cast<NewScalar>(pitch_),
                                             static_cast<NewScalar>(roll_));
        }

      private:
        T yaw_;   // Z rotation (first)
        T pitch_; // Y' rotation (second)
        T roll_;  // X'' rotation (third)
    };

    // ===== TYPE ALIASES =====

    using EulerAnglesZYXf = EulerAnglesZYX<float>;
    using EulerAnglesZYXd = EulerAnglesZYX<double>;

    // Yaw-Pitch-Roll is the same as ZYX
    template <typename T = double> using EulerAnglesYPR = EulerAnglesZYX<T>;
    using EulerAnglesYPRf = EulerAnglesZYX<float>;
    using EulerAnglesYPRd = EulerAnglesZYX<double>;

    // ===== EulerAnglesXYZ =====

    /// EulerAnglesXYZ: Euler angles in X-Y'-Z'' (roll-pitch-yaw) convention.
    ///
    /// This is the aerospace convention. The rotation is applied as:
    /// R = Rx(roll) * Ry(pitch) * Rz(yaw)
    ///
    /// Storage order: [roll, pitch, yaw]
    template <typename T = double> class EulerAnglesXYZ {
        static_assert(std::is_floating_point_v<T>, "EulerAnglesXYZ requires floating-point scalar type");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Vector3 = dp::mat::vector<T, 3>;    // Owning storage type
        using Matrix3 = dp::mat::matrix<T, 3, 3>; // Owning storage type

        // ===== CONSTRUCTORS =====

        /// Default constructor: identity (all angles zero)
        constexpr EulerAnglesXYZ() noexcept : roll_(T(0)), pitch_(T(0)), yaw_(T(0)) {}

        /// Construct from roll, pitch, yaw angles (in radians)
        constexpr EulerAnglesXYZ(T roll, T pitch, T yaw) noexcept : roll_(roll), pitch_(pitch), yaw_(yaw) {}

        /// Construct from Vector3 [roll, pitch, yaw]
        explicit constexpr EulerAnglesXYZ(const Vector3 &v) noexcept : roll_(v[0]), pitch_(v[1]), yaw_(v[2]) {}

        /// Construct from SO3 rotation
        explicit EulerAnglesXYZ(const SO3<T> &rotation) noexcept { set_from_rotation(rotation); }

        // ===== STATIC FACTORY METHODS =====

        /// Identity (zero rotation)
        [[nodiscard]] static constexpr EulerAnglesXYZ identity() noexcept { return EulerAnglesXYZ(); }

        /// Create from SO3 rotation
        [[nodiscard]] static EulerAnglesXYZ from_rotation(const SO3<T> &rotation) noexcept {
            return EulerAnglesXYZ(rotation);
        }

        /// Create from rotation matrix
        [[nodiscard]] static EulerAnglesXYZ from_rotation_matrix(const Matrix3 &R) noexcept {
            return EulerAnglesXYZ(SO3<T>(R));
        }

        // ===== ACCESSORS =====

        [[nodiscard]] constexpr T roll() const noexcept { return roll_; }
        [[nodiscard]] constexpr T pitch() const noexcept { return pitch_; }
        [[nodiscard]] constexpr T yaw() const noexcept { return yaw_; }

        /// Alias: x() = roll()
        [[nodiscard]] constexpr T x() const noexcept { return roll_; }
        /// Alias: y() = pitch()
        [[nodiscard]] constexpr T y() const noexcept { return pitch_; }
        /// Alias: z() = yaw()
        [[nodiscard]] constexpr T z() const noexcept { return yaw_; }

        /// Get as vector [roll, pitch, yaw]
        [[nodiscard]] Vector3 vector() const noexcept { return Vector3{{roll_, pitch_, yaw_}}; }

        // ===== MUTATORS =====

        void set_roll(T roll) noexcept { roll_ = roll; }
        void set_pitch(T pitch) noexcept { pitch_ = pitch; }
        void set_yaw(T yaw) noexcept { yaw_ = yaw; }

        void set_x(T x) noexcept { roll_ = x; }
        void set_y(T y) noexcept { pitch_ = y; }
        void set_z(T z) noexcept { yaw_ = z; }

        /// Set to identity
        EulerAnglesXYZ &set_identity() noexcept {
            roll_ = pitch_ = yaw_ = T(0);
            return *this;
        }

        // ===== CONVERSION TO/FROM SO3 =====

        /// Convert to SO3 rotation
        [[nodiscard]] SO3<T> to_rotation() const noexcept {
            // R = Rx(roll) * Ry(pitch) * Rz(yaw)
            return SO3<T>::rot_x(roll_) * SO3<T>::rot_y(pitch_) * SO3<T>::rot_z(yaw_);
        }

        /// Convert to rotation matrix
        [[nodiscard]] Matrix3 to_rotation_matrix() const noexcept { return to_rotation().matrix(); }

        /// Set from SO3 rotation
        void set_from_rotation(const SO3<T> &rotation) noexcept {
            // Extract Euler angles from rotation matrix using XYZ convention
            const auto R = rotation.matrix();

            // pitch = asin(R(0,2))
            const T sinp = R(0, 2);

            if (std::abs(sinp) >= T(1) - epsilon<T>) {
                // Gimbal lock: pitch = ±π/2
                pitch_ = std::copysign(half_pi<T>, sinp);
                // In gimbal lock, roll and yaw are coupled
                // Convention: set yaw = 0, compute roll
                yaw_ = T(0);
                roll_ = std::atan2(-R(1, 0), R(1, 1));
            } else {
                pitch_ = std::asin(sinp);
                roll_ = std::atan2(-R(1, 2), R(2, 2));
                yaw_ = std::atan2(-R(0, 1), R(0, 0));
            }
        }

        // ===== GIMBAL LOCK DETECTION =====

        /// Check if near gimbal lock (pitch ≈ ±π/2)
        [[nodiscard]] bool is_near_gimbal_lock(T tolerance = T(0.01)) const noexcept {
            return std::abs(std::abs(pitch_) - half_pi<T>) < tolerance;
        }

        /// Get distance to gimbal lock (0 = at gimbal lock, π/2 = far from gimbal lock)
        [[nodiscard]] T gimbal_lock_distance() const noexcept { return std::abs(std::abs(pitch_) - half_pi<T>); }

        // ===== JACOBIANS =====

        /// Jacobian mapping Euler angle rates to body-frame angular velocity.
        ///
        /// ω_body = J * [roll_dot, pitch_dot, yaw_dot]^T
        ///
        /// Note: This Jacobian becomes singular at gimbal lock (pitch = ±π/2).
        [[nodiscard]] Matrix3 euler_rates_to_angular_velocity() const noexcept {
            const T cp = std::cos(pitch_);
            const T sp = std::sin(pitch_);
            const T cy = std::cos(yaw_);
            const T sy = std::sin(yaw_);

            Matrix3 J;
            // Column 0: contribution of roll_dot
            J(0, 0) = cp * cy;
            J(1, 0) = cp * sy;
            J(2, 0) = -sp;

            // Column 1: contribution of pitch_dot
            J(0, 1) = -sy;
            J(1, 1) = cy;
            J(2, 1) = T(0);

            // Column 2: contribution of yaw_dot
            J(0, 2) = T(0);
            J(1, 2) = T(0);
            J(2, 2) = T(1);

            return J;
        }

        /// Jacobian mapping body-frame angular velocity to Euler angle rates.
        ///
        /// [roll_dot, pitch_dot, yaw_dot]^T = J_inv * ω_body
        ///
        /// Note: This Jacobian is undefined at gimbal lock (pitch = ±π/2).
        [[nodiscard]] Matrix3 angular_velocity_to_euler_rates() const noexcept {
            const T cp = std::cos(pitch_);

            // Check for gimbal lock
            if (std::abs(cp) < epsilon<T>) {
                // At gimbal lock, return identity as a fallback
                Matrix3 I;
                I(0, 0) = T(1);
                I(0, 1) = T(0);
                I(0, 2) = T(0);
                I(1, 0) = T(0);
                I(1, 1) = T(1);
                I(1, 2) = T(0);
                I(2, 0) = T(0);
                I(2, 1) = T(0);
                I(2, 2) = T(1);
                return I;
            }

            const T sp = std::sin(pitch_);
            const T cy = std::cos(yaw_);
            const T sy = std::sin(yaw_);
            const T inv_cp = T(1) / cp;

            Matrix3 J_inv;
            // Row 0: roll_dot
            J_inv(0, 0) = cy * inv_cp;
            J_inv(0, 1) = sy * inv_cp;
            J_inv(0, 2) = T(0);

            // Row 1: pitch_dot
            J_inv(1, 0) = -sy;
            J_inv(1, 1) = cy;
            J_inv(1, 2) = T(0);

            // Row 2: yaw_dot
            J_inv(2, 0) = cy * sp * inv_cp;
            J_inv(2, 1) = sy * sp * inv_cp;
            J_inv(2, 2) = T(1);

            return J_inv;
        }

        // ===== NORMALIZATION =====

        /// Wrap angles to canonical range
        [[nodiscard]] EulerAnglesXYZ get_unique() const noexcept {
            auto wrap = [](T angle) -> T {
                angle = std::fmod(angle + pi<T>, two_pi<T>);
                if (angle < T(0))
                    angle += two_pi<T>;
                return angle - pi<T>;
            };

            T r = wrap(roll_);
            T p = wrap(pitch_);
            T y = wrap(yaw_);

            // If pitch is outside [-π/2, π/2), flip it
            if (p < -half_pi<T>) {
                r = wrap(r + pi<T>);
                p = -(p + pi<T>);
                y = wrap(y + pi<T>);
            } else if (p > half_pi<T>) {
                r = wrap(r + pi<T>);
                p = -(p - pi<T>);
                y = wrap(y + pi<T>);
            }

            return EulerAnglesXYZ(r, p, y);
        }

        /// Normalize angles in place
        EulerAnglesXYZ &set_unique() noexcept {
            *this = get_unique();
            return *this;
        }

        // ===== COMPARISON =====

        [[nodiscard]] bool operator==(const EulerAnglesXYZ &other) const noexcept {
            return std::abs(roll_ - other.roll_) < epsilon<T> && std::abs(pitch_ - other.pitch_) < epsilon<T> &&
                   std::abs(yaw_ - other.yaw_) < epsilon<T>;
        }

        [[nodiscard]] bool operator!=(const EulerAnglesXYZ &other) const noexcept { return !(*this == other); }

        [[nodiscard]] bool is_approx(const EulerAnglesXYZ &other, T tol = epsilon<T>) const noexcept {
            return std::abs(roll_ - other.roll_) < tol && std::abs(pitch_ - other.pitch_) < tol &&
                   std::abs(yaw_ - other.yaw_) < tol;
        }

        [[nodiscard]] bool is_identity(T tol = epsilon<T>) const noexcept {
            return std::abs(roll_) < tol && std::abs(pitch_) < tol && std::abs(yaw_) < tol;
        }

        // ===== TYPE CONVERSION =====

        template <typename NewScalar> [[nodiscard]] EulerAnglesXYZ<NewScalar> cast() const noexcept {
            return EulerAnglesXYZ<NewScalar>(static_cast<NewScalar>(roll_), static_cast<NewScalar>(pitch_),
                                             static_cast<NewScalar>(yaw_));
        }

      private:
        T roll_;  // X rotation (first)
        T pitch_; // Y' rotation (second)
        T yaw_;   // Z'' rotation (third)
    };

    // ===== TYPE ALIASES =====

    using EulerAnglesXYZf = EulerAnglesXYZ<float>;
    using EulerAnglesXYZd = EulerAnglesXYZ<double>;

    // Roll-Pitch-Yaw is the same as XYZ
    template <typename T = double> using EulerAnglesRPY = EulerAnglesXYZ<T>;
    using EulerAnglesRPYf = EulerAnglesXYZ<float>;
    using EulerAnglesRPYd = EulerAnglesXYZ<double>;

    // ===== CONVERSION BETWEEN CONVENTIONS =====

    /// Convert EulerAnglesZYX to EulerAnglesXYZ (via SO3)
    template <typename T> [[nodiscard]] EulerAnglesXYZ<T> to_xyz(const EulerAnglesZYX<T> &zyx) noexcept {
        return EulerAnglesXYZ<T>(zyx.to_rotation());
    }

    /// Convert EulerAnglesXYZ to EulerAnglesZYX (via SO3)
    template <typename T> [[nodiscard]] EulerAnglesZYX<T> to_zyx(const EulerAnglesXYZ<T> &xyz) noexcept {
        return EulerAnglesZYX<T>(xyz.to_rotation());
    }

} // namespace optinum::lie
