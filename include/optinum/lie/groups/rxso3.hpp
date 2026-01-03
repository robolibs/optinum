#pragma once

#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/so3.hpp>

#include <datapod/matrix/math/quaternion.hpp>
#include <datapod/matrix/matrix.hpp>
#include <datapod/matrix/vector.hpp>

#include <cmath>
#include <random>
#include <type_traits>

namespace optinum::lie {

    namespace dp = ::datapod;

    // ===== RxSO3: Rotation + Scale in 3D =====
    //
    // RxSO3 = R+ x SO(3) represents 3D rotation with positive scale.
    // Internally stored as a non-unit quaternion:
    //   q = s * q_unit  where s = scale > 0, q_unit is unit quaternion
    //   |q|^2 = s^2
    //
    // Storage: dp::mat::Quaternion<T> (non-unit, norm = scale)
    // DoF: 4 (3 rotation + 1 log-scale)
    // NumParams: 4 (quaternion)
    //
    // Tangent space (Lie algebra):
    //   [sigma, omega] where sigma = log(scale), omega = rotation vector (R^3)
    //   Total: 4 DoF = 1 (scale) + 3 (rotation)
    //
    // exp([sigma, omega]) = exp(sigma) * SO3::exp(omega)
    // log(RxSO3) = [log(scale), SO3::log(rotation)]

    template <typename T = double> class RxSO3 {
        static_assert(std::is_floating_point_v<T>, "RxSO3 requires floating-point scalar type");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Tangent = dp::mat::Vector<T, 4>; // [sigma, wx, wy, wz]
        using Quaternion = dp::mat::Quaternion<T>;
        using Point = dp::mat::Vector<T, 3>;
        using Matrix = dp::mat::Matrix<T, 3, 3>;
        using AdjointMatrix = dp::mat::Matrix<T, 4, 4>;
        using Rotation = SO3<T>;

        // ===== CONSTANTS =====
        static constexpr std::size_t DoF = 4;
        static constexpr std::size_t NumParams = 4;

        // ===== CONSTRUCTORS =====

        // Default: identity (scale = 1, rotation = identity)
        constexpr RxSO3() noexcept : q_{T(1), T(0), T(0), T(0)} {}

        // From SO3 rotation (scale = 1)
        explicit RxSO3(const Rotation &R) noexcept : q_(R.unit_quaternion()) {}

        // From scale and SO3
        RxSO3(Scalar scale, const Rotation &R) noexcept {
            const auto &uq = R.unit_quaternion();
            q_.w = scale * uq.w;
            q_.x = scale * uq.x;
            q_.y = scale * uq.y;
            q_.z = scale * uq.z;
        }

        // From scale and axis-angle
        RxSO3(Scalar scale, const dp::mat::Vector<T, 3> &omega) noexcept : RxSO3(scale, Rotation::exp(omega)) {}

        // From quaternion (non-unit)
        explicit RxSO3(const Quaternion &q) noexcept : q_(q) {}

        // From quaternion components (w, x, y, z)
        RxSO3(Scalar w, Scalar x, Scalar y, Scalar z) noexcept : q_{w, x, y, z} {}

        // ===== STATIC FACTORY METHODS =====

        // Identity element (scale = 1, rotation = identity)
        [[nodiscard]] static constexpr RxSO3 identity() noexcept { return RxSO3(); }

        // Exponential map: [sigma, omega] -> RxSO3
        // exp([sigma, omega]) = exp(sigma) * q(omega)
        [[nodiscard]] static RxSO3 exp(const Tangent &tangent) noexcept {
            const T sigma = tangent[0]; // log(scale)
            dp::mat::Vector<T, 3> omega{tangent[1], tangent[2], tangent[3]};
            const T s = std::exp(sigma);
            return RxSO3(s, Rotation::exp(omega));
        }

        // Pure scale (no rotation)
        [[nodiscard]] static RxSO3 scale_only(Scalar s) noexcept { return RxSO3(s, T(0), T(0), T(0)); }

        // From axis-angle with scale = 1
        [[nodiscard]] static RxSO3 from_axis_angle(const dp::mat::Vector<T, 3> &omega) noexcept {
            return RxSO3(Rotation::exp(omega));
        }

        // Sample uniform random RxSO3
        template <typename RNG> [[nodiscard]] static RxSO3 sample_uniform(RNG &rng, T max_log_scale = T(1)) noexcept {
            std::uniform_real_distribution<T> scale_dist(-max_log_scale, max_log_scale);
            const T sigma = scale_dist(rng);
            return RxSO3(std::exp(sigma), Rotation::sample_uniform(rng));
        }

        // ===== CORE OPERATIONS =====

        // Logarithmic map: RxSO3 -> [sigma, omega]
        [[nodiscard]] Tangent log() const noexcept {
            const T s = scale();
            const auto R = so3();
            const auto omega = R.log();
            return Tangent{std::log(s), omega[0], omega[1], omega[2]};
        }

        // Inverse: q^-1 = conj(q) / |q|^2
        [[nodiscard]] RxSO3 inverse() const noexcept {
            const T norm_sq = q_.w * q_.w + q_.x * q_.x + q_.y * q_.y + q_.z * q_.z;
            return RxSO3(q_.w / norm_sq, -q_.x / norm_sq, -q_.y / norm_sq, -q_.z / norm_sq);
        }

        // Group composition: quaternion multiplication
        [[nodiscard]] RxSO3 operator*(const RxSO3 &other) const noexcept {
            // Hamilton product
            const T w = q_.w * other.q_.w - q_.x * other.q_.x - q_.y * other.q_.y - q_.z * other.q_.z;
            const T x = q_.w * other.q_.x + q_.x * other.q_.w + q_.y * other.q_.z - q_.z * other.q_.y;
            const T y = q_.w * other.q_.y - q_.x * other.q_.z + q_.y * other.q_.w + q_.z * other.q_.x;
            const T z = q_.w * other.q_.z + q_.x * other.q_.y - q_.y * other.q_.x + q_.z * other.q_.w;
            return RxSO3(w, x, y, z);
        }

        RxSO3 &operator*=(const RxSO3 &other) noexcept {
            *this = *this * other;
            return *this;
        }

        // Transform a 3D point: s * R * p
        [[nodiscard]] Point operator*(const Point &p) const noexcept {
            // Get the unit quaternion and scale
            const T s = scale();
            const T inv_s = T(1) / s;

            // Normalize to get unit quaternion
            const T qw = q_.w * inv_s;
            const T qx = q_.x * inv_s;
            const T qy = q_.y * inv_s;
            const T qz = q_.z * inv_s;

            // Apply rotation using unit quaternion formula:
            // t = 2 * (q_v x v)
            // v' = v + w*t + (q_v x t)
            const T tx = T(2) * (qy * p[2] - qz * p[1]);
            const T ty = T(2) * (qz * p[0] - qx * p[2]);
            const T tz = T(2) * (qx * p[1] - qy * p[0]);

            Point rotated;
            rotated[0] = p[0] + qw * tx + (qy * tz - qz * ty);
            rotated[1] = p[1] + qw * ty + (qz * tx - qx * tz);
            rotated[2] = p[2] + qw * tz + (qx * ty - qy * tx);

            // Apply scale
            rotated[0] *= s;
            rotated[1] *= s;
            rotated[2] *= s;

            return rotated;
        }

        // ===== MATRIX REPRESENTATION =====

        // Return 3x3 scaled rotation matrix: s * R
        [[nodiscard]] Matrix matrix() const noexcept {
            const T s = scale();
            auto R = so3().matrix();
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    R(i, j) *= s;
                }
            }
            return R;
        }

        // Rotation matrix only (unit scale)
        [[nodiscard]] Matrix rotation_matrix() const noexcept { return so3().matrix(); }

        // ===== LIE ALGEBRA =====

        // hat: [sigma, omega] -> 3x3 matrix
        // hat([sigma, omega]) = sigma * I + hat_so3(omega)
        [[nodiscard]] static Matrix hat(const Tangent &tangent) noexcept {
            Matrix M;
            const T sigma = tangent[0];
            M(0, 0) = sigma;
            M(0, 1) = -tangent[3]; // -wz
            M(0, 2) = tangent[2];  // wy
            M(1, 0) = tangent[3];  // wz
            M(1, 1) = sigma;
            M(1, 2) = -tangent[1]; // -wx
            M(2, 0) = -tangent[2]; // -wy
            M(2, 1) = tangent[1];  // wx
            M(2, 2) = sigma;
            return M;
        }

        // vee: 3x3 matrix -> [sigma, omega]
        [[nodiscard]] static Tangent vee(const Matrix &M) noexcept {
            const T sigma = (M(0, 0) + M(1, 1) + M(2, 2)) / T(3);
            return Tangent{sigma, (M(2, 1) - M(1, 2)) / T(2), (M(0, 2) - M(2, 0)) / T(2), (M(1, 0) - M(0, 1)) / T(2)};
        }

        // Adjoint representation
        [[nodiscard]] AdjointMatrix Adj() const noexcept {
            // For RxSO3, Adj = [[1, 0], [0, R]]
            auto R = so3().matrix();
            AdjointMatrix A;
            A(0, 0) = T(1);
            A(0, 1) = A(0, 2) = A(0, 3) = T(0);
            A(1, 0) = A(2, 0) = A(3, 0) = T(0);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    A(i + 1, j + 1) = R(i, j);
                }
            }
            return A;
        }

        // Lie bracket
        [[nodiscard]] static Tangent lie_bracket(const Tangent &a, const Tangent &b) noexcept {
            // [a, b] = [0, omega_a x omega_b] (scale part commutes)
            dp::mat::Vector<T, 3> omega_a{a[1], a[2], a[3]};
            dp::mat::Vector<T, 3> omega_b{b[1], b[2], b[3]};
            return Tangent{T(0), omega_a[1] * omega_b[2] - omega_a[2] * omega_b[1],
                           omega_a[2] * omega_b[0] - omega_a[0] * omega_b[2],
                           omega_a[0] * omega_b[1] - omega_a[1] * omega_b[0]};
        }

        // ===== ACCESSORS =====

        // Get scale factor: |q|
        [[nodiscard]] T scale() const noexcept {
            return std::sqrt(q_.w * q_.w + q_.x * q_.x + q_.y * q_.y + q_.z * q_.z);
        }

        // Get scale squared: |q|^2
        [[nodiscard]] T scale_squared() const noexcept { return q_.w * q_.w + q_.x * q_.x + q_.y * q_.y + q_.z * q_.z; }

        // Get SO3 rotation (unit quaternion)
        [[nodiscard]] Rotation so3() const noexcept {
            const T s = scale();
            return Rotation(q_.w / s, q_.x / s, q_.y / s, q_.z / s);
        }

        // Get quaternion (non-unit)
        [[nodiscard]] const Quaternion &quaternion() const noexcept { return q_; }

        // Get unit quaternion
        [[nodiscard]] Quaternion unit_quaternion() const noexcept {
            const T s = scale();
            return Quaternion{q_.w / s, q_.x / s, q_.y / s, q_.z / s};
        }

        // Quaternion components
        [[nodiscard]] T w() const noexcept { return q_.w; }
        [[nodiscard]] T x() const noexcept { return q_.x; }
        [[nodiscard]] T y() const noexcept { return q_.y; }
        [[nodiscard]] T z() const noexcept { return q_.z; }

        // Raw data pointer
        [[nodiscard]] T *data() noexcept { return &q_.w; }
        [[nodiscard]] const T *data() const noexcept { return &q_.w; }

        // ===== MUTATORS =====

        // Set scale (preserving rotation)
        void set_scale(T new_scale) noexcept {
            const T current_scale = scale();
            if (current_scale > epsilon<T>) {
                const T factor = new_scale / current_scale;
                q_.w *= factor;
                q_.x *= factor;
                q_.y *= factor;
                q_.z *= factor;
            } else {
                q_.w = new_scale;
                q_.x = q_.y = q_.z = T(0);
            }
        }

        // Set SO3 rotation (preserving scale)
        void set_so3(const Rotation &R) noexcept {
            const T s = scale();
            const auto &uq = R.unit_quaternion();
            q_.w = s * uq.w;
            q_.x = s * uq.x;
            q_.y = s * uq.y;
            q_.z = s * uq.z;
        }

        // ===== TYPE CONVERSION =====

        template <typename NewScalar> [[nodiscard]] RxSO3<NewScalar> cast() const noexcept {
            return RxSO3<NewScalar>(static_cast<NewScalar>(q_.w), static_cast<NewScalar>(q_.x),
                                    static_cast<NewScalar>(q_.y), static_cast<NewScalar>(q_.z));
        }

        // ===== COMPARISON =====

        [[nodiscard]] bool operator==(const RxSO3 &other) const noexcept {
            return std::abs(q_.w - other.q_.w) < epsilon<T> && std::abs(q_.x - other.q_.x) < epsilon<T> &&
                   std::abs(q_.y - other.q_.y) < epsilon<T> && std::abs(q_.z - other.q_.z) < epsilon<T>;
        }

        [[nodiscard]] bool operator!=(const RxSO3 &other) const noexcept { return !(*this == other); }

        [[nodiscard]] bool is_approx(const RxSO3 &other, T tol = epsilon<T>) const noexcept {
            return std::abs(q_.w - other.q_.w) < tol && std::abs(q_.x - other.q_.x) < tol &&
                   std::abs(q_.y - other.q_.y) < tol && std::abs(q_.z - other.q_.z) < tol;
        }

        [[nodiscard]] bool is_identity(T tol = epsilon<T>) const noexcept {
            return std::abs(q_.w - T(1)) < tol && std::abs(q_.x) < tol && std::abs(q_.y) < tol && std::abs(q_.z) < tol;
        }

      private:
        Quaternion q_; // Non-unit quaternion (|q| = scale)
    };

    // ===== TYPE ALIASES =====

    using RxSO3f = RxSO3<float>;
    using RxSO3d = RxSO3<double>;

    // ===== FREE FUNCTIONS =====

    // Geodesic interpolation
    template <typename T> [[nodiscard]] RxSO3<T> interpolate(const RxSO3<T> &a, const RxSO3<T> &b, T t) noexcept {
        auto tangent = (a.inverse() * b).log();
        tangent[0] *= t;
        tangent[1] *= t;
        tangent[2] *= t;
        tangent[3] *= t;
        return a * RxSO3<T>::exp(tangent);
    }

} // namespace optinum::lie
