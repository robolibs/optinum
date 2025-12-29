#pragma once

#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/angular_velocity.hpp>
#include <optinum/lie/groups/se3.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>
#include <type_traits>

namespace optinum::lie {

    // ===== TWIST CLASSES =====
    //
    // A twist represents the velocity of a rigid body in 3D space.
    // It combines linear velocity (v) and angular velocity (ω).
    //
    // Twist = [v; ω] ∈ R^6 where:
    //   v = [vx, vy, vz] - linear velocity
    //   ω = [ωx, ωy, ωz] - angular velocity
    //
    // LocalTwist (body-frame):
    //   - Linear and angular velocities expressed in body-fixed frame
    //   - This is what IMU/sensors typically measure
    //   - Integration: T_new = T_old * exp(twist * dt)
    //
    // GlobalTwist (world-frame):
    //   - Linear and angular velocities expressed in world frame
    //   - Integration: T_new = exp(twist * dt) * T_old
    //
    // Frame conversion uses the Adjoint of SE3:
    //   twist_global = Ad_T * twist_local
    //   twist_local = Ad_T^{-1} * twist_global

    // Forward declarations
    template <typename T> class LocalTwist;
    template <typename T> class GlobalTwist;

    /// LocalTwist: Twist (linear + angular velocity) expressed in body-fixed frame.
    ///
    /// This represents the velocity of a rigid body as measured in its own frame.
    /// The linear velocity is the velocity of the body origin expressed in body coordinates.
    /// The angular velocity is the rotation rate around body axes.
    template <typename T = double> class LocalTwist {
        static_assert(std::is_floating_point_v<T>, "LocalTwist requires floating-point scalar type");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Vector3 = simd::Vector<T, 3>;
        using Vector6 = simd::Vector<T, 6>;

        // ===== CONSTRUCTORS =====

        /// Default constructor: zero twist
        constexpr LocalTwist() noexcept : linear_{T(0), T(0), T(0)}, angular_{T(0), T(0), T(0)} {}

        /// Construct from linear and angular velocity vectors
        constexpr LocalTwist(const Vector3 &linear, const Vector3 &angular) noexcept
            : linear_(linear), angular_(angular) {}

        /// Construct from 6 components [vx, vy, vz, ωx, ωy, ωz]
        constexpr LocalTwist(T vx, T vy, T vz, T wx, T wy, T wz) noexcept : linear_{vx, vy, vz}, angular_{wx, wy, wz} {}

        /// Construct from Vector6 [v; ω]
        explicit constexpr LocalTwist(const Vector6 &vec) noexcept
            : linear_{vec[0], vec[1], vec[2]}, angular_{vec[3], vec[4], vec[5]} {}

        /// Construct from LocalAngularVelocity (zero linear velocity)
        explicit constexpr LocalTwist(const LocalAngularVelocity<T> &angular) noexcept
            : linear_{T(0), T(0), T(0)}, angular_(angular.vector()) {}

        // ===== STATIC FACTORY METHODS =====

        /// Zero twist
        [[nodiscard]] static constexpr LocalTwist zero() noexcept { return LocalTwist(); }

        /// Create from Vector6 [v; ω]
        [[nodiscard]] static constexpr LocalTwist from_vector(const Vector6 &vec) noexcept { return LocalTwist(vec); }

        /// Create from global twist using SE3 transform T (body-to-world)
        [[nodiscard]] static LocalTwist from_global(const GlobalTwist<T> &global, const SE3<T> &T_body_world) noexcept;

        // ===== ACCESSORS =====

        /// Get linear velocity
        [[nodiscard]] constexpr const Vector3 &linear() const noexcept { return linear_; }
        [[nodiscard]] constexpr Vector3 &linear() noexcept { return linear_; }

        /// Get angular velocity
        [[nodiscard]] constexpr const Vector3 &angular() const noexcept { return angular_; }
        [[nodiscard]] constexpr Vector3 &angular() noexcept { return angular_; }

        /// Get angular velocity as LocalAngularVelocity
        [[nodiscard]] LocalAngularVelocity<T> angular_velocity() const noexcept {
            return LocalAngularVelocity<T>(angular_);
        }

        /// Get as Vector6 [v; ω]
        [[nodiscard]] Vector6 vector() const noexcept {
            return Vector6{linear_[0], linear_[1], linear_[2], angular_[0], angular_[1], angular_[2]};
        }

        /// Get individual components
        [[nodiscard]] constexpr T vx() const noexcept { return linear_[0]; }
        [[nodiscard]] constexpr T vy() const noexcept { return linear_[1]; }
        [[nodiscard]] constexpr T vz() const noexcept { return linear_[2]; }
        [[nodiscard]] constexpr T wx() const noexcept { return angular_[0]; }
        [[nodiscard]] constexpr T wy() const noexcept { return angular_[1]; }
        [[nodiscard]] constexpr T wz() const noexcept { return angular_[2]; }

        // ===== MUTATORS =====

        void set_linear(const Vector3 &v) noexcept { linear_ = v; }
        void set_angular(const Vector3 &w) noexcept { angular_ = w; }

        void set_linear(T vx, T vy, T vz) noexcept {
            linear_[0] = vx;
            linear_[1] = vy;
            linear_[2] = vz;
        }

        void set_angular(T wx, T wy, T wz) noexcept {
            angular_[0] = wx;
            angular_[1] = wy;
            angular_[2] = wz;
        }

        /// Set from Vector6
        void set_vector(const Vector6 &vec) noexcept {
            linear_[0] = vec[0];
            linear_[1] = vec[1];
            linear_[2] = vec[2];
            angular_[0] = vec[3];
            angular_[1] = vec[4];
            angular_[2] = vec[5];
        }

        /// Set to zero
        LocalTwist &set_zero() noexcept {
            linear_[0] = linear_[1] = linear_[2] = T(0);
            angular_[0] = angular_[1] = angular_[2] = T(0);
            return *this;
        }

        // ===== FRAME CONVERSION =====

        /// Convert to global twist using SE3 transform T (body-to-world)
        /// twist_global = Ad_T * twist_local
        [[nodiscard]] GlobalTwist<T> to_global(const SE3<T> &T_body_world) const noexcept;

        // ===== INTEGRATION =====

        /// Integrate twist over time dt to get SE3 increment.
        /// Returns the transform dT such that: T_new = T_old * dT
        /// This is the exponential map: exp(twist * dt)
        [[nodiscard]] SE3<T> integrate(T dt) const noexcept {
            Vector6 twist_dt{linear_[0] * dt,  linear_[1] * dt,  linear_[2] * dt,
                             angular_[0] * dt, angular_[1] * dt, angular_[2] * dt};
            return SE3<T>::exp(twist_dt);
        }

        /// Apply twist to a pose over time dt.
        /// T_new = T_old * exp(twist_local * dt)
        [[nodiscard]] SE3<T> apply_to(const SE3<T> &pose, T dt) const noexcept { return pose * integrate(dt); }

        // ===== ADJOINT TRANSFORMATION =====

        /// Apply adjoint transformation: Ad_T * this
        /// Transforms this local twist to a different frame
        [[nodiscard]] LocalTwist adjoint(const SE3<T> &transform) const noexcept {
            auto Ad = transform.Adj();
            auto v = vector();

            // Ad * v (6x6 matrix * 6-vector)
            Vector6 result;
            for (int i = 0; i < 6; ++i) {
                Scalar s = Scalar(0);
                for (int j = 0; j < 6; ++j) {
                    s += Ad(i, j) * v[j];
                }
                result[i] = s;
            }

            return LocalTwist(result);
        }

        // ===== ARITHMETIC OPERATORS =====

        [[nodiscard]] LocalTwist operator+(const LocalTwist &other) const noexcept {
            return LocalTwist(
                Vector3{linear_[0] + other.linear_[0], linear_[1] + other.linear_[1], linear_[2] + other.linear_[2]},
                Vector3{angular_[0] + other.angular_[0], angular_[1] + other.angular_[1],
                        angular_[2] + other.angular_[2]});
        }

        [[nodiscard]] LocalTwist operator-(const LocalTwist &other) const noexcept {
            return LocalTwist(
                Vector3{linear_[0] - other.linear_[0], linear_[1] - other.linear_[1], linear_[2] - other.linear_[2]},
                Vector3{angular_[0] - other.angular_[0], angular_[1] - other.angular_[1],
                        angular_[2] - other.angular_[2]});
        }

        [[nodiscard]] LocalTwist operator*(T scalar) const noexcept {
            return LocalTwist(Vector3{linear_[0] * scalar, linear_[1] * scalar, linear_[2] * scalar},
                              Vector3{angular_[0] * scalar, angular_[1] * scalar, angular_[2] * scalar});
        }

        [[nodiscard]] LocalTwist operator/(T scalar) const noexcept {
            const T inv = T(1) / scalar;
            return LocalTwist(Vector3{linear_[0] * inv, linear_[1] * inv, linear_[2] * inv},
                              Vector3{angular_[0] * inv, angular_[1] * inv, angular_[2] * inv});
        }

        [[nodiscard]] LocalTwist operator-() const noexcept {
            return LocalTwist(Vector3{-linear_[0], -linear_[1], -linear_[2]},
                              Vector3{-angular_[0], -angular_[1], -angular_[2]});
        }

        LocalTwist &operator+=(const LocalTwist &other) noexcept {
            linear_[0] += other.linear_[0];
            linear_[1] += other.linear_[1];
            linear_[2] += other.linear_[2];
            angular_[0] += other.angular_[0];
            angular_[1] += other.angular_[1];
            angular_[2] += other.angular_[2];
            return *this;
        }

        LocalTwist &operator-=(const LocalTwist &other) noexcept {
            linear_[0] -= other.linear_[0];
            linear_[1] -= other.linear_[1];
            linear_[2] -= other.linear_[2];
            angular_[0] -= other.angular_[0];
            angular_[1] -= other.angular_[1];
            angular_[2] -= other.angular_[2];
            return *this;
        }

        LocalTwist &operator*=(T scalar) noexcept {
            linear_[0] *= scalar;
            linear_[1] *= scalar;
            linear_[2] *= scalar;
            angular_[0] *= scalar;
            angular_[1] *= scalar;
            angular_[2] *= scalar;
            return *this;
        }

        // ===== COMPARISON =====

        [[nodiscard]] bool operator==(const LocalTwist &other) const noexcept {
            return std::abs(linear_[0] - other.linear_[0]) < epsilon<T> &&
                   std::abs(linear_[1] - other.linear_[1]) < epsilon<T> &&
                   std::abs(linear_[2] - other.linear_[2]) < epsilon<T> &&
                   std::abs(angular_[0] - other.angular_[0]) < epsilon<T> &&
                   std::abs(angular_[1] - other.angular_[1]) < epsilon<T> &&
                   std::abs(angular_[2] - other.angular_[2]) < epsilon<T>;
        }

        [[nodiscard]] bool operator!=(const LocalTwist &other) const noexcept { return !(*this == other); }

        [[nodiscard]] bool is_approx(const LocalTwist &other, T tol = epsilon<T>) const noexcept {
            return std::abs(linear_[0] - other.linear_[0]) < tol && std::abs(linear_[1] - other.linear_[1]) < tol &&
                   std::abs(linear_[2] - other.linear_[2]) < tol && std::abs(angular_[0] - other.angular_[0]) < tol &&
                   std::abs(angular_[1] - other.angular_[1]) < tol && std::abs(angular_[2] - other.angular_[2]) < tol;
        }

        [[nodiscard]] bool is_zero(T tol = epsilon<T>) const noexcept {
            return std::abs(linear_[0]) < tol && std::abs(linear_[1]) < tol && std::abs(linear_[2]) < tol &&
                   std::abs(angular_[0]) < tol && std::abs(angular_[1]) < tol && std::abs(angular_[2]) < tol;
        }

        // ===== TYPE CONVERSION =====

        template <typename NewScalar> [[nodiscard]] LocalTwist<NewScalar> cast() const noexcept {
            return LocalTwist<NewScalar>(
                dp::mat::vector<NewScalar, 3>{static_cast<NewScalar>(linear_[0]), static_cast<NewScalar>(linear_[1]),
                                              static_cast<NewScalar>(linear_[2])},
                dp::mat::vector<NewScalar, 3>{static_cast<NewScalar>(angular_[0]), static_cast<NewScalar>(angular_[1]),
                                              static_cast<NewScalar>(angular_[2])});
        }

      private:
        Vector3 linear_;  // [vx, vy, vz] linear velocity
        Vector3 angular_; // [ωx, ωy, ωz] angular velocity
    };

    /// GlobalTwist: Twist (linear + angular velocity) expressed in world/inertial frame.
    ///
    /// This represents the velocity of a rigid body in the fixed world frame.
    template <typename T = double> class GlobalTwist {
        static_assert(std::is_floating_point_v<T>, "GlobalTwist requires floating-point scalar type");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Vector3 = simd::Vector<T, 3>;
        using Vector6 = simd::Vector<T, 6>;

        // ===== CONSTRUCTORS =====

        /// Default constructor: zero twist
        constexpr GlobalTwist() noexcept : linear_{T(0), T(0), T(0)}, angular_{T(0), T(0), T(0)} {}

        /// Construct from linear and angular velocity vectors
        constexpr GlobalTwist(const Vector3 &linear, const Vector3 &angular) noexcept
            : linear_(linear), angular_(angular) {}

        /// Construct from 6 components [vx, vy, vz, ωx, ωy, ωz]
        constexpr GlobalTwist(T vx, T vy, T vz, T wx, T wy, T wz) noexcept
            : linear_{vx, vy, vz}, angular_{wx, wy, wz} {}

        /// Construct from Vector6 [v; ω]
        explicit constexpr GlobalTwist(const Vector6 &vec) noexcept
            : linear_{vec[0], vec[1], vec[2]}, angular_{vec[3], vec[4], vec[5]} {}

        /// Construct from GlobalAngularVelocity (zero linear velocity)
        explicit constexpr GlobalTwist(const GlobalAngularVelocity<T> &angular) noexcept
            : linear_{T(0), T(0), T(0)}, angular_(angular.vector()) {}

        // ===== STATIC FACTORY METHODS =====

        /// Zero twist
        [[nodiscard]] static constexpr GlobalTwist zero() noexcept { return GlobalTwist(); }

        /// Create from Vector6 [v; ω]
        [[nodiscard]] static constexpr GlobalTwist from_vector(const Vector6 &vec) noexcept { return GlobalTwist(vec); }

        /// Create from local twist using SE3 transform T (body-to-world)
        [[nodiscard]] static GlobalTwist from_local(const LocalTwist<T> &local, const SE3<T> &T_body_world) noexcept {
            // twist_global = Ad_T * twist_local
            auto Ad = T_body_world.Adj();
            auto v = local.vector();

            // Ad * v (6x6 matrix * 6-vector)
            Vector6 result;
            for (int i = 0; i < 6; ++i) {
                T sum = T(0);
                for (int j = 0; j < 6; ++j) {
                    sum += Ad(i, j) * v[j];
                }
                result[i] = sum;
            }

            return GlobalTwist(result);
        }

        // ===== ACCESSORS =====

        /// Get linear velocity
        [[nodiscard]] constexpr const Vector3 &linear() const noexcept { return linear_; }
        [[nodiscard]] constexpr Vector3 &linear() noexcept { return linear_; }

        /// Get angular velocity
        [[nodiscard]] constexpr const Vector3 &angular() const noexcept { return angular_; }
        [[nodiscard]] constexpr Vector3 &angular() noexcept { return angular_; }

        /// Get angular velocity as GlobalAngularVelocity
        [[nodiscard]] GlobalAngularVelocity<T> angular_velocity() const noexcept {
            return GlobalAngularVelocity<T>(angular_);
        }

        /// Get as Vector6 [v; ω]
        [[nodiscard]] Vector6 vector() const noexcept {
            return Vector6{linear_[0], linear_[1], linear_[2], angular_[0], angular_[1], angular_[2]};
        }

        /// Get individual components
        [[nodiscard]] constexpr T vx() const noexcept { return linear_[0]; }
        [[nodiscard]] constexpr T vy() const noexcept { return linear_[1]; }
        [[nodiscard]] constexpr T vz() const noexcept { return linear_[2]; }
        [[nodiscard]] constexpr T wx() const noexcept { return angular_[0]; }
        [[nodiscard]] constexpr T wy() const noexcept { return angular_[1]; }
        [[nodiscard]] constexpr T wz() const noexcept { return angular_[2]; }

        // ===== MUTATORS =====

        void set_linear(const Vector3 &v) noexcept { linear_ = v; }
        void set_angular(const Vector3 &w) noexcept { angular_ = w; }

        void set_linear(T vx, T vy, T vz) noexcept {
            linear_[0] = vx;
            linear_[1] = vy;
            linear_[2] = vz;
        }

        void set_angular(T wx, T wy, T wz) noexcept {
            angular_[0] = wx;
            angular_[1] = wy;
            angular_[2] = wz;
        }

        /// Set from Vector6
        void set_vector(const Vector6 &vec) noexcept {
            linear_[0] = vec[0];
            linear_[1] = vec[1];
            linear_[2] = vec[2];
            angular_[0] = vec[3];
            angular_[1] = vec[4];
            angular_[2] = vec[5];
        }

        /// Set to zero
        GlobalTwist &set_zero() noexcept {
            linear_[0] = linear_[1] = linear_[2] = T(0);
            angular_[0] = angular_[1] = angular_[2] = T(0);
            return *this;
        }

        // ===== FRAME CONVERSION =====

        /// Convert to local twist using SE3 transform T (body-to-world)
        /// twist_local = Ad_T^{-1} * twist_global
        [[nodiscard]] LocalTwist<T> to_local(const SE3<T> &T_body_world) const noexcept {
            // Ad_T^{-1} = Ad_{T^{-1}}
            auto T_inv = T_body_world.inverse();
            auto Ad_inv = T_inv.Adj();
            auto v = vector();

            // Ad_inv * v (6x6 matrix * 6-vector)
            typename LocalTwist<T>::Vector6 result;
            for (int i = 0; i < 6; ++i) {
                T sum = T(0);
                for (int j = 0; j < 6; ++j) {
                    sum += Ad_inv(i, j) * v[j];
                }
                result[i] = sum;
            }

            return LocalTwist<T>(result);
        }

        // ===== INTEGRATION =====

        /// Integrate twist over time dt to get SE3 increment.
        /// Returns the transform dT such that: T_new = dT * T_old
        /// This is the exponential map: exp(twist * dt)
        [[nodiscard]] SE3<T> integrate(T dt) const noexcept {
            Vector6 twist_dt{linear_[0] * dt,  linear_[1] * dt,  linear_[2] * dt,
                             angular_[0] * dt, angular_[1] * dt, angular_[2] * dt};
            return SE3<T>::exp(twist_dt);
        }

        /// Apply twist to a pose over time dt.
        /// T_new = exp(twist_global * dt) * T_old
        [[nodiscard]] SE3<T> apply_to(const SE3<T> &pose, T dt) const noexcept { return integrate(dt) * pose; }

        // ===== ARITHMETIC OPERATORS =====

        [[nodiscard]] GlobalTwist operator+(const GlobalTwist &other) const noexcept {
            return GlobalTwist(
                Vector3{linear_[0] + other.linear_[0], linear_[1] + other.linear_[1], linear_[2] + other.linear_[2]},
                Vector3{angular_[0] + other.angular_[0], angular_[1] + other.angular_[1],
                        angular_[2] + other.angular_[2]});
        }

        [[nodiscard]] GlobalTwist operator-(const GlobalTwist &other) const noexcept {
            return GlobalTwist(
                Vector3{linear_[0] - other.linear_[0], linear_[1] - other.linear_[1], linear_[2] - other.linear_[2]},
                Vector3{angular_[0] - other.angular_[0], angular_[1] - other.angular_[1],
                        angular_[2] - other.angular_[2]});
        }

        [[nodiscard]] GlobalTwist operator*(T scalar) const noexcept {
            return GlobalTwist(Vector3{linear_[0] * scalar, linear_[1] * scalar, linear_[2] * scalar},
                               Vector3{angular_[0] * scalar, angular_[1] * scalar, angular_[2] * scalar});
        }

        [[nodiscard]] GlobalTwist operator/(T scalar) const noexcept {
            const T inv = T(1) / scalar;
            return GlobalTwist(Vector3{linear_[0] * inv, linear_[1] * inv, linear_[2] * inv},
                               Vector3{angular_[0] * inv, angular_[1] * inv, angular_[2] * inv});
        }

        [[nodiscard]] GlobalTwist operator-() const noexcept {
            return GlobalTwist(Vector3{-linear_[0], -linear_[1], -linear_[2]},
                               Vector3{-angular_[0], -angular_[1], -angular_[2]});
        }

        GlobalTwist &operator+=(const GlobalTwist &other) noexcept {
            linear_[0] += other.linear_[0];
            linear_[1] += other.linear_[1];
            linear_[2] += other.linear_[2];
            angular_[0] += other.angular_[0];
            angular_[1] += other.angular_[1];
            angular_[2] += other.angular_[2];
            return *this;
        }

        GlobalTwist &operator-=(const GlobalTwist &other) noexcept {
            linear_[0] -= other.linear_[0];
            linear_[1] -= other.linear_[1];
            linear_[2] -= other.linear_[2];
            angular_[0] -= other.angular_[0];
            angular_[1] -= other.angular_[1];
            angular_[2] -= other.angular_[2];
            return *this;
        }

        GlobalTwist &operator*=(T scalar) noexcept {
            linear_[0] *= scalar;
            linear_[1] *= scalar;
            linear_[2] *= scalar;
            angular_[0] *= scalar;
            angular_[1] *= scalar;
            angular_[2] *= scalar;
            return *this;
        }

        // ===== COMPARISON =====

        [[nodiscard]] bool operator==(const GlobalTwist &other) const noexcept {
            return std::abs(linear_[0] - other.linear_[0]) < epsilon<T> &&
                   std::abs(linear_[1] - other.linear_[1]) < epsilon<T> &&
                   std::abs(linear_[2] - other.linear_[2]) < epsilon<T> &&
                   std::abs(angular_[0] - other.angular_[0]) < epsilon<T> &&
                   std::abs(angular_[1] - other.angular_[1]) < epsilon<T> &&
                   std::abs(angular_[2] - other.angular_[2]) < epsilon<T>;
        }

        [[nodiscard]] bool operator!=(const GlobalTwist &other) const noexcept { return !(*this == other); }

        [[nodiscard]] bool is_approx(const GlobalTwist &other, T tol = epsilon<T>) const noexcept {
            return std::abs(linear_[0] - other.linear_[0]) < tol && std::abs(linear_[1] - other.linear_[1]) < tol &&
                   std::abs(linear_[2] - other.linear_[2]) < tol && std::abs(angular_[0] - other.angular_[0]) < tol &&
                   std::abs(angular_[1] - other.angular_[1]) < tol && std::abs(angular_[2] - other.angular_[2]) < tol;
        }

        [[nodiscard]] bool is_zero(T tol = epsilon<T>) const noexcept {
            return std::abs(linear_[0]) < tol && std::abs(linear_[1]) < tol && std::abs(linear_[2]) < tol &&
                   std::abs(angular_[0]) < tol && std::abs(angular_[1]) < tol && std::abs(angular_[2]) < tol;
        }

        // ===== TYPE CONVERSION =====

        template <typename NewScalar> [[nodiscard]] GlobalTwist<NewScalar> cast() const noexcept {
            return GlobalTwist<NewScalar>(
                dp::mat::vector<NewScalar, 3>{static_cast<NewScalar>(linear_[0]), static_cast<NewScalar>(linear_[1]),
                                              static_cast<NewScalar>(linear_[2])},
                dp::mat::vector<NewScalar, 3>{static_cast<NewScalar>(angular_[0]), static_cast<NewScalar>(angular_[1]),
                                              static_cast<NewScalar>(angular_[2])});
        }

      private:
        Vector3 linear_;  // [vx, vy, vz] linear velocity
        Vector3 angular_; // [ωx, ωy, ωz] angular velocity
    };

    // ===== DEFERRED IMPLEMENTATIONS =====

    template <typename T> GlobalTwist<T> LocalTwist<T>::to_global(const SE3<T> &T_body_world) const noexcept {
        return GlobalTwist<T>::from_local(*this, T_body_world);
    }

    template <typename T>
    LocalTwist<T> LocalTwist<T>::from_global(const GlobalTwist<T> &global, const SE3<T> &T_body_world) noexcept {
        return global.to_local(T_body_world);
    }

    // ===== SCALAR * TWIST =====

    template <typename T> [[nodiscard]] LocalTwist<T> operator*(T scalar, const LocalTwist<T> &twist) noexcept {
        return twist * scalar;
    }

    template <typename T> [[nodiscard]] GlobalTwist<T> operator*(T scalar, const GlobalTwist<T> &twist) noexcept {
        return twist * scalar;
    }

    // ===== TYPE ALIASES =====

    using LocalTwistf = LocalTwist<float>;
    using LocalTwistd = LocalTwist<double>;

    using GlobalTwistf = GlobalTwist<float>;
    using GlobalTwistd = GlobalTwist<double>;

    // Short aliases (matching kindr naming)
    template <typename T = double> using BodyTwist = LocalTwist<T>;
    template <typename T = double> using SpatialTwist = GlobalTwist<T>;

} // namespace optinum::lie
