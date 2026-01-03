#pragma once

#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/so3.hpp>

#include <datapod/matrix/vector.hpp>

#include <cmath>
#include <type_traits>

namespace optinum::lie {

    namespace dp = ::datapod;

    // ===== ANGULAR VELOCITY CLASSES =====
    //
    // Angular velocity can be expressed in two frames:
    //
    // LocalAngularVelocity (body-frame):
    //   - ω expressed in the body-fixed frame
    //   - This is what a gyroscope measures
    //   - Also called "body angular velocity" or ω_B
    //
    // GlobalAngularVelocity (world-frame):
    //   - ω expressed in the world/inertial frame
    //   - Also called "spatial angular velocity" or ω_W
    //
    // Conversion:
    //   ω_global = R * ω_local    (R rotates body to world)
    //   ω_local = R^T * ω_global  (R^T rotates world to body)
    //
    // Integration:
    //   R(t + dt) = R(t) * exp(ω_local * dt)   for local angular velocity
    //   R(t + dt) = exp(ω_global * dt) * R(t)  for global angular velocity

    // Forward declarations
    template <typename T> class LocalAngularVelocity;
    template <typename T> class GlobalAngularVelocity;

    /// LocalAngularVelocity: Angular velocity expressed in body-fixed frame.
    ///
    /// This represents the angular velocity as measured by a gyroscope attached to the body.
    /// The components [ωx, ωy, ωz] are the rotation rates around the body's x, y, z axes.
    template <typename T = double> class LocalAngularVelocity {
        static_assert(std::is_floating_point_v<T>, "LocalAngularVelocity requires floating-point scalar type");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Vector3 = dp::mat::Vector<T, 3>; // Owning storage type

        // ===== CONSTRUCTORS =====

        /// Default constructor: zero angular velocity
        constexpr LocalAngularVelocity() noexcept : omega_{{T(0), T(0), T(0)}} {}

        /// Construct from components (ωx, ωy, ωz) in rad/s
        constexpr LocalAngularVelocity(T wx, T wy, T wz) noexcept : omega_{{wx, wy, wz}} {}

        /// Construct from Vector3
        explicit constexpr LocalAngularVelocity(const Vector3 &omega) noexcept : omega_(omega) {}

        // ===== STATIC FACTORY METHODS =====

        /// Zero angular velocity
        [[nodiscard]] static constexpr LocalAngularVelocity zero() noexcept { return LocalAngularVelocity(); }

        /// Create from global angular velocity using rotation R (body-to-world)
        [[nodiscard]] static LocalAngularVelocity from_global(const GlobalAngularVelocity<T> &global,
                                                              const SO3<T> &R) noexcept;

        // ===== ACCESSORS =====

        [[nodiscard]] constexpr T x() const noexcept { return omega_[0]; }
        [[nodiscard]] constexpr T y() const noexcept { return omega_[1]; }
        [[nodiscard]] constexpr T z() const noexcept { return omega_[2]; }

        [[nodiscard]] constexpr T &x() noexcept { return omega_[0]; }
        [[nodiscard]] constexpr T &y() noexcept { return omega_[1]; }
        [[nodiscard]] constexpr T &z() noexcept { return omega_[2]; }

        /// Get as Vector3
        [[nodiscard]] constexpr const Vector3 &vector() const noexcept { return omega_; }
        [[nodiscard]] constexpr Vector3 &vector() noexcept { return omega_; }

        /// Get magnitude (rotation rate in rad/s)
        [[nodiscard]] T norm() const noexcept {
            return std::sqrt(omega_[0] * omega_[0] + omega_[1] * omega_[1] + omega_[2] * omega_[2]);
        }

        /// Get squared magnitude
        [[nodiscard]] T norm_squared() const noexcept {
            return omega_[0] * omega_[0] + omega_[1] * omega_[1] + omega_[2] * omega_[2];
        }

        // ===== MUTATORS =====

        void set_x(T wx) noexcept { omega_[0] = wx; }
        void set_y(T wy) noexcept { omega_[1] = wy; }
        void set_z(T wz) noexcept { omega_[2] = wz; }

        /// Set to zero
        LocalAngularVelocity &set_zero() noexcept {
            omega_[0] = omega_[1] = omega_[2] = T(0);
            return *this;
        }

        // ===== FRAME CONVERSION =====

        /// Convert to global angular velocity using rotation R (body-to-world)
        /// ω_global = R * ω_local
        [[nodiscard]] GlobalAngularVelocity<T> to_global(const SO3<T> &R) const noexcept;

        // ===== INTEGRATION =====

        /// Integrate angular velocity over time dt to get rotation increment.
        /// Returns the rotation R such that: R_new = R_old * R
        /// This is the exponential map: exp(ω * dt)
        [[nodiscard]] SO3<T> integrate(T dt) const noexcept {
            Vector3 omega_dt{{omega_[0] * dt, omega_[1] * dt, omega_[2] * dt}};
            return SO3<T>::exp(omega_dt);
        }

        /// Apply angular velocity to a rotation over time dt.
        /// R_new = R_old * exp(ω_local * dt)
        [[nodiscard]] SO3<T> apply_to(const SO3<T> &R, T dt) const noexcept { return R * integrate(dt); }

        // ===== ARITHMETIC OPERATORS =====

        [[nodiscard]] LocalAngularVelocity operator+(const LocalAngularVelocity &other) const noexcept {
            return LocalAngularVelocity(omega_[0] + other.omega_[0], omega_[1] + other.omega_[1],
                                        omega_[2] + other.omega_[2]);
        }

        [[nodiscard]] LocalAngularVelocity operator-(const LocalAngularVelocity &other) const noexcept {
            return LocalAngularVelocity(omega_[0] - other.omega_[0], omega_[1] - other.omega_[1],
                                        omega_[2] - other.omega_[2]);
        }

        [[nodiscard]] LocalAngularVelocity operator*(T scalar) const noexcept {
            return LocalAngularVelocity(omega_[0] * scalar, omega_[1] * scalar, omega_[2] * scalar);
        }

        [[nodiscard]] LocalAngularVelocity operator/(T scalar) const noexcept {
            const T inv = T(1) / scalar;
            return LocalAngularVelocity(omega_[0] * inv, omega_[1] * inv, omega_[2] * inv);
        }

        [[nodiscard]] LocalAngularVelocity operator-() const noexcept {
            return LocalAngularVelocity(-omega_[0], -omega_[1], -omega_[2]);
        }

        LocalAngularVelocity &operator+=(const LocalAngularVelocity &other) noexcept {
            omega_[0] += other.omega_[0];
            omega_[1] += other.omega_[1];
            omega_[2] += other.omega_[2];
            return *this;
        }

        LocalAngularVelocity &operator-=(const LocalAngularVelocity &other) noexcept {
            omega_[0] -= other.omega_[0];
            omega_[1] -= other.omega_[1];
            omega_[2] -= other.omega_[2];
            return *this;
        }

        LocalAngularVelocity &operator*=(T scalar) noexcept {
            omega_[0] *= scalar;
            omega_[1] *= scalar;
            omega_[2] *= scalar;
            return *this;
        }

        LocalAngularVelocity &operator/=(T scalar) noexcept {
            const T inv = T(1) / scalar;
            omega_[0] *= inv;
            omega_[1] *= inv;
            omega_[2] *= inv;
            return *this;
        }

        // ===== COMPARISON =====

        [[nodiscard]] bool operator==(const LocalAngularVelocity &other) const noexcept {
            return std::abs(omega_[0] - other.omega_[0]) < epsilon<T> &&
                   std::abs(omega_[1] - other.omega_[1]) < epsilon<T> &&
                   std::abs(omega_[2] - other.omega_[2]) < epsilon<T>;
        }

        [[nodiscard]] bool operator!=(const LocalAngularVelocity &other) const noexcept { return !(*this == other); }

        [[nodiscard]] bool is_approx(const LocalAngularVelocity &other, T tol = epsilon<T>) const noexcept {
            return std::abs(omega_[0] - other.omega_[0]) < tol && std::abs(omega_[1] - other.omega_[1]) < tol &&
                   std::abs(omega_[2] - other.omega_[2]) < tol;
        }

        [[nodiscard]] bool is_zero(T tol = epsilon<T>) const noexcept { return norm_squared() < tol * tol; }

        // ===== TYPE CONVERSION =====

        template <typename NewScalar> [[nodiscard]] LocalAngularVelocity<NewScalar> cast() const noexcept {
            return LocalAngularVelocity<NewScalar>(static_cast<NewScalar>(omega_[0]), static_cast<NewScalar>(omega_[1]),
                                                   static_cast<NewScalar>(omega_[2]));
        }

      private:
        Vector3 omega_; // [ωx, ωy, ωz] in rad/s
    };

    /// GlobalAngularVelocity: Angular velocity expressed in world/inertial frame.
    ///
    /// This represents the angular velocity in the fixed world frame.
    /// The components [ωx, ωy, ωz] are the rotation rates around the world's x, y, z axes.
    template <typename T = double> class GlobalAngularVelocity {
        static_assert(std::is_floating_point_v<T>, "GlobalAngularVelocity requires floating-point scalar type");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Vector3 = dp::mat::Vector<T, 3>; // Owning storage type

        // ===== CONSTRUCTORS =====

        /// Default constructor: zero angular velocity
        constexpr GlobalAngularVelocity() noexcept : omega_{{T(0), T(0), T(0)}} {}

        /// Construct from components (ωx, ωy, ωz) in rad/s
        constexpr GlobalAngularVelocity(T wx, T wy, T wz) noexcept : omega_{{wx, wy, wz}} {}

        /// Construct from Vector3
        explicit constexpr GlobalAngularVelocity(const Vector3 &omega) noexcept : omega_(omega) {}

        // ===== STATIC FACTORY METHODS =====

        /// Zero angular velocity
        [[nodiscard]] static constexpr GlobalAngularVelocity zero() noexcept { return GlobalAngularVelocity(); }

        /// Create from local angular velocity using rotation R (body-to-world)
        [[nodiscard]] static GlobalAngularVelocity from_local(const LocalAngularVelocity<T> &local,
                                                              const SO3<T> &R) noexcept {
            // ω_global = R * ω_local
            auto omega_global = R * local.vector();
            return GlobalAngularVelocity(omega_global);
        }

        // ===== ACCESSORS =====

        [[nodiscard]] constexpr T x() const noexcept { return omega_[0]; }
        [[nodiscard]] constexpr T y() const noexcept { return omega_[1]; }
        [[nodiscard]] constexpr T z() const noexcept { return omega_[2]; }

        [[nodiscard]] constexpr T &x() noexcept { return omega_[0]; }
        [[nodiscard]] constexpr T &y() noexcept { return omega_[1]; }
        [[nodiscard]] constexpr T &z() noexcept { return omega_[2]; }

        /// Get as Vector3
        [[nodiscard]] constexpr const Vector3 &vector() const noexcept { return omega_; }
        [[nodiscard]] constexpr Vector3 &vector() noexcept { return omega_; }

        /// Get magnitude (rotation rate in rad/s)
        [[nodiscard]] T norm() const noexcept {
            return std::sqrt(omega_[0] * omega_[0] + omega_[1] * omega_[1] + omega_[2] * omega_[2]);
        }

        /// Get squared magnitude
        [[nodiscard]] T norm_squared() const noexcept {
            return omega_[0] * omega_[0] + omega_[1] * omega_[1] + omega_[2] * omega_[2];
        }

        // ===== MUTATORS =====

        void set_x(T wx) noexcept { omega_[0] = wx; }
        void set_y(T wy) noexcept { omega_[1] = wy; }
        void set_z(T wz) noexcept { omega_[2] = wz; }

        /// Set to zero
        GlobalAngularVelocity &set_zero() noexcept {
            omega_[0] = omega_[1] = omega_[2] = T(0);
            return *this;
        }

        // ===== FRAME CONVERSION =====

        /// Convert to local angular velocity using rotation R (body-to-world)
        /// ω_local = R^T * ω_global
        [[nodiscard]] LocalAngularVelocity<T> to_local(const SO3<T> &R) const noexcept {
            // R^T * ω_global = R.inverse() * ω_global
            auto omega_local = R.inverse() * omega_;
            return LocalAngularVelocity<T>(omega_local);
        }

        // ===== INTEGRATION =====

        /// Integrate angular velocity over time dt to get rotation increment.
        /// Returns the rotation R such that: R_new = R * R_old
        /// This is the exponential map: exp(ω * dt)
        [[nodiscard]] SO3<T> integrate(T dt) const noexcept {
            Vector3 omega_dt{{omega_[0] * dt, omega_[1] * dt, omega_[2] * dt}};
            return SO3<T>::exp(omega_dt);
        }

        /// Apply angular velocity to a rotation over time dt.
        /// R_new = exp(ω_global * dt) * R_old
        [[nodiscard]] SO3<T> apply_to(const SO3<T> &R, T dt) const noexcept { return integrate(dt) * R; }

        // ===== ARITHMETIC OPERATORS =====

        [[nodiscard]] GlobalAngularVelocity operator+(const GlobalAngularVelocity &other) const noexcept {
            return GlobalAngularVelocity(omega_[0] + other.omega_[0], omega_[1] + other.omega_[1],
                                         omega_[2] + other.omega_[2]);
        }

        [[nodiscard]] GlobalAngularVelocity operator-(const GlobalAngularVelocity &other) const noexcept {
            return GlobalAngularVelocity(omega_[0] - other.omega_[0], omega_[1] - other.omega_[1],
                                         omega_[2] - other.omega_[2]);
        }

        [[nodiscard]] GlobalAngularVelocity operator*(T scalar) const noexcept {
            return GlobalAngularVelocity(omega_[0] * scalar, omega_[1] * scalar, omega_[2] * scalar);
        }

        [[nodiscard]] GlobalAngularVelocity operator/(T scalar) const noexcept {
            const T inv = T(1) / scalar;
            return GlobalAngularVelocity(omega_[0] * inv, omega_[1] * inv, omega_[2] * inv);
        }

        [[nodiscard]] GlobalAngularVelocity operator-() const noexcept {
            return GlobalAngularVelocity(-omega_[0], -omega_[1], -omega_[2]);
        }

        GlobalAngularVelocity &operator+=(const GlobalAngularVelocity &other) noexcept {
            omega_[0] += other.omega_[0];
            omega_[1] += other.omega_[1];
            omega_[2] += other.omega_[2];
            return *this;
        }

        GlobalAngularVelocity &operator-=(const GlobalAngularVelocity &other) noexcept {
            omega_[0] -= other.omega_[0];
            omega_[1] -= other.omega_[1];
            omega_[2] -= other.omega_[2];
            return *this;
        }

        GlobalAngularVelocity &operator*=(T scalar) noexcept {
            omega_[0] *= scalar;
            omega_[1] *= scalar;
            omega_[2] *= scalar;
            return *this;
        }

        GlobalAngularVelocity &operator/=(T scalar) noexcept {
            const T inv = T(1) / scalar;
            omega_[0] *= inv;
            omega_[1] *= inv;
            omega_[2] *= inv;
            return *this;
        }

        // ===== COMPARISON =====

        [[nodiscard]] bool operator==(const GlobalAngularVelocity &other) const noexcept {
            return std::abs(omega_[0] - other.omega_[0]) < epsilon<T> &&
                   std::abs(omega_[1] - other.omega_[1]) < epsilon<T> &&
                   std::abs(omega_[2] - other.omega_[2]) < epsilon<T>;
        }

        [[nodiscard]] bool operator!=(const GlobalAngularVelocity &other) const noexcept { return !(*this == other); }

        [[nodiscard]] bool is_approx(const GlobalAngularVelocity &other, T tol = epsilon<T>) const noexcept {
            return std::abs(omega_[0] - other.omega_[0]) < tol && std::abs(omega_[1] - other.omega_[1]) < tol &&
                   std::abs(omega_[2] - other.omega_[2]) < tol;
        }

        [[nodiscard]] bool is_zero(T tol = epsilon<T>) const noexcept { return norm_squared() < tol * tol; }

        // ===== TYPE CONVERSION =====

        template <typename NewScalar> [[nodiscard]] GlobalAngularVelocity<NewScalar> cast() const noexcept {
            return GlobalAngularVelocity<NewScalar>(static_cast<NewScalar>(omega_[0]),
                                                    static_cast<NewScalar>(omega_[1]),
                                                    static_cast<NewScalar>(omega_[2]));
        }

      private:
        Vector3 omega_; // [ωx, ωy, ωz] in rad/s
    };

    // ===== DEFERRED IMPLEMENTATIONS =====

    template <typename T> GlobalAngularVelocity<T> LocalAngularVelocity<T>::to_global(const SO3<T> &R) const noexcept {
        // ω_global = R * ω_local
        auto omega_global = R * omega_;
        return GlobalAngularVelocity<T>(omega_global);
    }

    template <typename T>
    LocalAngularVelocity<T> LocalAngularVelocity<T>::from_global(const GlobalAngularVelocity<T> &global,
                                                                 const SO3<T> &R) noexcept {
        return global.to_local(R);
    }

    // ===== SCALAR * ANGULAR_VELOCITY =====

    template <typename T>
    [[nodiscard]] LocalAngularVelocity<T> operator*(T scalar, const LocalAngularVelocity<T> &omega) noexcept {
        return omega * scalar;
    }

    template <typename T>
    [[nodiscard]] GlobalAngularVelocity<T> operator*(T scalar, const GlobalAngularVelocity<T> &omega) noexcept {
        return omega * scalar;
    }

    // ===== TYPE ALIASES =====

    using LocalAngularVelocityf = LocalAngularVelocity<float>;
    using LocalAngularVelocityd = LocalAngularVelocity<double>;

    using GlobalAngularVelocityf = GlobalAngularVelocity<float>;
    using GlobalAngularVelocityd = GlobalAngularVelocity<double>;

    // Short aliases
    template <typename T = double> using BodyAngularVelocity = LocalAngularVelocity<T>;
    template <typename T = double> using SpatialAngularVelocity = GlobalAngularVelocity<T>;

} // namespace optinum::lie
