#pragma once

#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/se3.hpp>
#include <optinum/lie/groups/twist.hpp>

#include <datapod/datapod.hpp>

#include <cmath>
#include <type_traits>

namespace optinum::lie {

    // ===== WRENCH CLASS =====
    //
    // A wrench represents forces and torques acting on a rigid body.
    // It combines force (f) and torque (τ).
    //
    // Wrench = [f; τ] ∈ R^6 where:
    //   f = [fx, fy, fz] - force vector
    //   τ = [τx, τy, τz] - torque vector
    //
    // Wrench is the dual of Twist in spatial algebra:
    //   Power = Twist · Wrench = v·f + ω·τ
    //
    // Frame transformation uses the co-adjoint (adjoint transpose):
    //   wrench_new = Ad_T^{-T} * wrench_old
    //
    // This is different from twist transformation which uses Ad_T directly.
    // The co-adjoint preserves the power pairing:
    //   twist_new · wrench_new = twist_old · wrench_old

    /// Wrench: Force and torque acting on a rigid body.
    ///
    /// The wrench is expressed in a specific frame. Use transform() to
    /// change the frame of reference.
    template <typename T = double> class Wrench {
        static_assert(std::is_floating_point_v<T>, "Wrench requires floating-point scalar type");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Vector3 = dp::mat::Vector<T, 3>; // Owning storage type
        using Vector6 = dp::mat::Vector<T, 6>; // Owning storage type

        // ===== CONSTRUCTORS =====

        /// Default constructor: zero wrench
        constexpr Wrench() noexcept : force_{{T(0), T(0), T(0)}}, torque_{{T(0), T(0), T(0)}} {}

        /// Construct from force and torque vectors
        constexpr Wrench(const Vector3 &force, const Vector3 &torque) noexcept : force_(force), torque_(torque) {}

        /// Construct from 6 components [fx, fy, fz, τx, τy, τz]
        constexpr Wrench(T fx, T fy, T fz, T tx, T ty, T tz) noexcept : force_{{fx, fy, fz}}, torque_{{tx, ty, tz}} {}

        /// Construct from Vector6 [f; τ]
        explicit constexpr Wrench(const Vector6 &vec) noexcept
            : force_{{vec[0], vec[1], vec[2]}}, torque_{{vec[3], vec[4], vec[5]}} {}

        // ===== STATIC FACTORY METHODS =====

        /// Zero wrench
        [[nodiscard]] static constexpr Wrench zero() noexcept { return Wrench(); }

        /// Create from Vector6 [f; τ]
        [[nodiscard]] static constexpr Wrench from_vector(const Vector6 &vec) noexcept { return Wrench(vec); }

        /// Pure force (no torque)
        [[nodiscard]] static constexpr Wrench pure_force(const Vector3 &force) noexcept {
            return Wrench(force, Vector3{{T(0), T(0), T(0)}});
        }

        /// Pure force (no torque)
        [[nodiscard]] static constexpr Wrench pure_force(T fx, T fy, T fz) noexcept {
            return Wrench(Vector3{{fx, fy, fz}}, Vector3{{T(0), T(0), T(0)}});
        }

        /// Pure torque (no force)
        [[nodiscard]] static constexpr Wrench pure_torque(const Vector3 &torque) noexcept {
            return Wrench(Vector3{{T(0), T(0), T(0)}}, torque);
        }

        /// Pure torque (no force)
        [[nodiscard]] static constexpr Wrench pure_torque(T tx, T ty, T tz) noexcept {
            return Wrench(Vector3{{T(0), T(0), T(0)}}, Vector3{{tx, ty, tz}});
        }

        // ===== ACCESSORS =====

        /// Get force vector
        [[nodiscard]] constexpr const Vector3 &force() const noexcept { return force_; }
        [[nodiscard]] constexpr Vector3 &force() noexcept { return force_; }

        /// Get torque vector
        [[nodiscard]] constexpr const Vector3 &torque() const noexcept { return torque_; }
        [[nodiscard]] constexpr Vector3 &torque() noexcept { return torque_; }

        /// Get as Vector6 [f; τ]
        [[nodiscard]] Vector6 vector() const noexcept {
            return Vector6{{force_[0], force_[1], force_[2], torque_[0], torque_[1], torque_[2]}};
        }

        /// Get individual components
        [[nodiscard]] constexpr T fx() const noexcept { return force_[0]; }
        [[nodiscard]] constexpr T fy() const noexcept { return force_[1]; }
        [[nodiscard]] constexpr T fz() const noexcept { return force_[2]; }
        [[nodiscard]] constexpr T tx() const noexcept { return torque_[0]; }
        [[nodiscard]] constexpr T ty() const noexcept { return torque_[1]; }
        [[nodiscard]] constexpr T tz() const noexcept { return torque_[2]; }

        // ===== MUTATORS =====

        void set_force(const Vector3 &f) noexcept { force_ = f; }
        void set_torque(const Vector3 &t) noexcept { torque_ = t; }

        void set_force(T fx, T fy, T fz) noexcept {
            force_[0] = fx;
            force_[1] = fy;
            force_[2] = fz;
        }

        void set_torque(T tx, T ty, T tz) noexcept {
            torque_[0] = tx;
            torque_[1] = ty;
            torque_[2] = tz;
        }

        /// Set from Vector6
        void set_vector(const Vector6 &vec) noexcept {
            force_[0] = vec[0];
            force_[1] = vec[1];
            force_[2] = vec[2];
            torque_[0] = vec[3];
            torque_[1] = vec[4];
            torque_[2] = vec[5];
        }

        /// Set to zero
        Wrench &set_zero() noexcept {
            force_[0] = force_[1] = force_[2] = T(0);
            torque_[0] = torque_[1] = torque_[2] = T(0);
            return *this;
        }

        // ===== FRAME TRANSFORMATION =====

        /// Transform wrench to a new frame using SE3 transform.
        ///
        /// Uses the co-adjoint (adjoint transpose inverse):
        ///   wrench_new = Ad_T^{-T} * wrench_old
        ///
        /// This preserves the power pairing with twists:
        ///   twist_new · wrench_new = twist_old · wrench_old
        ///
        /// @param T_new_old Transform from old frame to new frame
        /// @return Wrench expressed in new frame
        [[nodiscard]] Wrench transform(const SE3<T> &T_new_old) const noexcept {
            // Co-adjoint is Ad^{-T} = (Ad^{-1})^T
            // For SE3: Ad^{-1} = Ad_{T^{-1}}
            // So we need (Ad_{T^{-1}})^T

            auto T_inv = T_new_old.inverse();
            auto Ad_inv = T_inv.Adj();

            // Transpose of Ad_inv and multiply
            auto w = vector();
            Vector6 result;
            for (int i = 0; i < 6; ++i) {
                T sum = T(0);
                for (int j = 0; j < 6; ++j) {
                    // Transpose: use Ad_inv(j, i) instead of Ad_inv(i, j)
                    sum += Ad_inv(j, i) * w[j];
                }
                result[i] = sum;
            }

            return Wrench(result);
        }

        // ===== POWER COMPUTATION =====

        /// Compute power: P = twist · wrench = v·f + ω·τ
        ///
        /// This is the rate of work done by the wrench when the body
        /// moves with the given twist.
        [[nodiscard]] T power(const LocalTwist<T> &twist) const noexcept {
            // P = v·f + ω·τ
            return twist.vx() * force_[0] + twist.vy() * force_[1] + twist.vz() * force_[2] + twist.wx() * torque_[0] +
                   twist.wy() * torque_[1] + twist.wz() * torque_[2];
        }

        /// Compute power with global twist
        [[nodiscard]] T power(const GlobalTwist<T> &twist) const noexcept {
            return twist.vx() * force_[0] + twist.vy() * force_[1] + twist.vz() * force_[2] + twist.wx() * torque_[0] +
                   twist.wy() * torque_[1] + twist.wz() * torque_[2];
        }

        /// Dot product with a 6-vector (same as power with twist vector)
        [[nodiscard]] T dot(const Vector6 &v) const noexcept {
            return v[0] * force_[0] + v[1] * force_[1] + v[2] * force_[2] + v[3] * torque_[0] + v[4] * torque_[1] +
                   v[5] * torque_[2];
        }

        // ===== ARITHMETIC OPERATORS =====

        [[nodiscard]] Wrench operator+(const Wrench &other) const noexcept {
            return Wrench(
                Vector3{{force_[0] + other.force_[0], force_[1] + other.force_[1], force_[2] + other.force_[2]}},
                Vector3{{torque_[0] + other.torque_[0], torque_[1] + other.torque_[1], torque_[2] + other.torque_[2]}});
        }

        [[nodiscard]] Wrench operator-(const Wrench &other) const noexcept {
            return Wrench(
                Vector3{{force_[0] - other.force_[0], force_[1] - other.force_[1], force_[2] - other.force_[2]}},
                Vector3{{torque_[0] - other.torque_[0], torque_[1] - other.torque_[1], torque_[2] - other.torque_[2]}});
        }

        [[nodiscard]] Wrench operator*(T scalar) const noexcept {
            return Wrench(Vector3{{force_[0] * scalar, force_[1] * scalar, force_[2] * scalar}},
                          Vector3{{torque_[0] * scalar, torque_[1] * scalar, torque_[2] * scalar}});
        }

        [[nodiscard]] Wrench operator/(T scalar) const noexcept {
            const T inv = T(1) / scalar;
            return Wrench(Vector3{{force_[0] * inv, force_[1] * inv, force_[2] * inv}},
                          Vector3{{torque_[0] * inv, torque_[1] * inv, torque_[2] * inv}});
        }

        [[nodiscard]] Wrench operator-() const noexcept {
            return Wrench(Vector3{{-force_[0], -force_[1], -force_[2]}},
                          Vector3{{-torque_[0], -torque_[1], -torque_[2]}});
        }

        Wrench &operator+=(const Wrench &other) noexcept {
            force_[0] += other.force_[0];
            force_[1] += other.force_[1];
            force_[2] += other.force_[2];
            torque_[0] += other.torque_[0];
            torque_[1] += other.torque_[1];
            torque_[2] += other.torque_[2];
            return *this;
        }

        Wrench &operator-=(const Wrench &other) noexcept {
            force_[0] -= other.force_[0];
            force_[1] -= other.force_[1];
            force_[2] -= other.force_[2];
            torque_[0] -= other.torque_[0];
            torque_[1] -= other.torque_[1];
            torque_[2] -= other.torque_[2];
            return *this;
        }

        Wrench &operator*=(T scalar) noexcept {
            force_[0] *= scalar;
            force_[1] *= scalar;
            force_[2] *= scalar;
            torque_[0] *= scalar;
            torque_[1] *= scalar;
            torque_[2] *= scalar;
            return *this;
        }

        Wrench &operator/=(T scalar) noexcept {
            const T inv = T(1) / scalar;
            force_[0] *= inv;
            force_[1] *= inv;
            force_[2] *= inv;
            torque_[0] *= inv;
            torque_[1] *= inv;
            torque_[2] *= inv;
            return *this;
        }

        // ===== COMPARISON =====

        [[nodiscard]] bool operator==(const Wrench &other) const noexcept {
            return std::abs(force_[0] - other.force_[0]) < epsilon<T> &&
                   std::abs(force_[1] - other.force_[1]) < epsilon<T> &&
                   std::abs(force_[2] - other.force_[2]) < epsilon<T> &&
                   std::abs(torque_[0] - other.torque_[0]) < epsilon<T> &&
                   std::abs(torque_[1] - other.torque_[1]) < epsilon<T> &&
                   std::abs(torque_[2] - other.torque_[2]) < epsilon<T>;
        }

        [[nodiscard]] bool operator!=(const Wrench &other) const noexcept { return !(*this == other); }

        [[nodiscard]] bool is_approx(const Wrench &other, T tol = epsilon<T>) const noexcept {
            return std::abs(force_[0] - other.force_[0]) < tol && std::abs(force_[1] - other.force_[1]) < tol &&
                   std::abs(force_[2] - other.force_[2]) < tol && std::abs(torque_[0] - other.torque_[0]) < tol &&
                   std::abs(torque_[1] - other.torque_[1]) < tol && std::abs(torque_[2] - other.torque_[2]) < tol;
        }

        [[nodiscard]] bool is_zero(T tol = epsilon<T>) const noexcept {
            return std::abs(force_[0]) < tol && std::abs(force_[1]) < tol && std::abs(force_[2]) < tol &&
                   std::abs(torque_[0]) < tol && std::abs(torque_[1]) < tol && std::abs(torque_[2]) < tol;
        }

        // ===== NORMS =====

        /// Force magnitude
        [[nodiscard]] T force_norm() const noexcept {
            return std::sqrt(force_[0] * force_[0] + force_[1] * force_[1] + force_[2] * force_[2]);
        }

        /// Torque magnitude
        [[nodiscard]] T torque_norm() const noexcept {
            return std::sqrt(torque_[0] * torque_[0] + torque_[1] * torque_[1] + torque_[2] * torque_[2]);
        }

        // ===== TYPE CONVERSION =====

        template <typename NewScalar> [[nodiscard]] Wrench<NewScalar> cast() const noexcept {
            return Wrench<NewScalar>(
                dp::mat::Vector<NewScalar, 3>{{static_cast<NewScalar>(force_[0]), static_cast<NewScalar>(force_[1]),
                                               static_cast<NewScalar>(force_[2])}},
                dp::mat::Vector<NewScalar, 3>{{static_cast<NewScalar>(torque_[0]), static_cast<NewScalar>(torque_[1]),
                                               static_cast<NewScalar>(torque_[2])}});
        }

      private:
        Vector3 force_;  // [fx, fy, fz] force vector
        Vector3 torque_; // [τx, τy, τz] torque vector
    };

    // ===== SCALAR * WRENCH =====

    template <typename T> [[nodiscard]] Wrench<T> operator*(T scalar, const Wrench<T> &wrench) noexcept {
        return wrench * scalar;
    }

    // ===== TYPE ALIASES =====

    using Wrenchf = Wrench<float>;
    using Wrenchd = Wrench<double>;

} // namespace optinum::lie
