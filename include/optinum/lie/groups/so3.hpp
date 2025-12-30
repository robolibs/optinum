#pragma once

#include <optinum/lie/core/constants.hpp>
#include <optinum/simd/backend/cross.hpp>
#include <optinum/simd/backend/matmul.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <datapod/matrix/math/quaternion.hpp>

#include <cmath>
#include <random>
#include <type_traits>

namespace optinum::lie {

    namespace dp = ::datapod;

    // ===== SO3: Special Orthogonal Group in 3D =====
    //
    // SO(3) represents 3D rotations. Internally stored as a unit quaternion.
    //
    // Quaternion convention: q = w + xi + yj + zk (scalar-first, Hamilton convention)
    // Storage: dp::mat::quaternion<T> with [w, x, y, z]
    // DoF: 3 (rotation vector / axis-angle)
    // NumParams: 4 (unit quaternion)
    //
    // Tangent space (Lie algebra so(3)):
    //   Vector omega in R^3 representing axis-angle
    //   hat(omega) -> 3x3 skew-symmetric matrix
    //
    // Key formulas:
    //   exp(omega) = cos(|omega|/2) + sin(|omega|/2) * omega_hat/|omega|
    //   log(q) = 2 * atan2(|v|, w) * v/|v|  where v = [x,y,z]

    template <typename T = double> class SO3 {
        static_assert(std::is_floating_point_v<T>, "SO3 requires floating-point scalar type");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Tangent = dp::mat::vector<T, 3>; // R^3 (rotation vector) - owning type
        using Quaternion = dp::mat::quaternion<T>;
        using Point = dp::mat::vector<T, 3>;             // 3D point - owning type
        using RotationMatrix = dp::mat::matrix<T, 3, 3>; // Owning type for return values
        using AdjointMatrix = dp::mat::matrix<T, 3, 3>;  // Owning type for return values

        // ===== CONSTANTS =====
        static constexpr std::size_t DoF = 3;
        static constexpr std::size_t NumParams = 4;

        // ===== CONSTRUCTORS =====

        // Default: identity rotation
        constexpr SO3() noexcept : q_{T(1), T(0), T(0), T(0)} {}

        // From unit quaternion (normalizes if needed)
        explicit SO3(const Quaternion &q) noexcept : q_(q) { normalize(); }

        // From quaternion components (w, x, y, z)
        SO3(T w, T x, T y, T z) noexcept : q_{w, x, y, z} { normalize(); }

        // From rotation matrix
        explicit SO3(const RotationMatrix &R) noexcept { set_rotation_matrix(R); }

        // From axis-angle (axis should be unit vector)
        static SO3 from_axis_angle(const Tangent &axis, T angle) noexcept {
            return SO3(Quaternion::from_axis_angle(axis[0], axis[1], axis[2], angle));
        }

        // ===== STATIC FACTORY METHODS =====

        // Identity element
        [[nodiscard]] static constexpr SO3 identity() noexcept { return SO3(); }

        // Rotation around X axis
        [[nodiscard]] static SO3 rot_x(T angle) noexcept {
            const T half = angle / T(2);
            return SO3(std::cos(half), std::sin(half), T(0), T(0));
        }

        // Rotation around Y axis
        [[nodiscard]] static SO3 rot_y(T angle) noexcept {
            const T half = angle / T(2);
            return SO3(std::cos(half), T(0), std::sin(half), T(0));
        }

        // Rotation around Z axis
        [[nodiscard]] static SO3 rot_z(T angle) noexcept {
            const T half = angle / T(2);
            return SO3(std::cos(half), T(0), T(0), std::sin(half));
        }

        // Exponential map: omega (axis-angle) -> SO3
        // omega = theta * axis, where axis is unit vector and theta is rotation angle
        [[nodiscard]] static SO3 exp(const Tangent &omega) noexcept {
            const T theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];

            if (theta_sq < epsilon<T> * epsilon<T>) {
                // Small angle: use Taylor expansion
                // q ≈ [1, omega/2] normalized
                const T half = T(0.5);
                Quaternion q{T(1), half * omega[0], half * omega[1], half * omega[2]};
                // Normalize (for very small angles, this is approximately 1)
                const T n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
                return SO3(q.w / n, q.x / n, q.y / n, q.z / n);
            }

            const T theta = std::sqrt(theta_sq);
            const T half_theta = theta / T(2);
            const T sin_half_over_theta = std::sin(half_theta) / theta;

            return SO3(std::cos(half_theta), sin_half_over_theta * omega[0], sin_half_over_theta * omega[1],
                       sin_half_over_theta * omega[2]);
        }

        // Exponential map that also returns the angle
        [[nodiscard]] static std::pair<SO3, T> exp_and_theta(const Tangent &omega) noexcept {
            const T theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
            const T theta = std::sqrt(theta_sq);

            if (theta_sq < epsilon<T> * epsilon<T>) {
                const T half = T(0.5);
                Quaternion q{T(1), half * omega[0], half * omega[1], half * omega[2]};
                const T n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
                return {SO3(q.w / n, q.x / n, q.y / n, q.z / n), theta};
            }

            const T half_theta = theta / T(2);
            const T sin_half_over_theta = std::sin(half_theta) / theta;

            return {SO3(std::cos(half_theta), sin_half_over_theta * omega[0], sin_half_over_theta * omega[1],
                        sin_half_over_theta * omega[2]),
                    theta};
        }

        // Sample uniform random rotation (uniform on SO3)
        template <typename RNG> [[nodiscard]] static SO3 sample_uniform(RNG &rng) noexcept {
            std::uniform_real_distribution<T> dist(T(0), T(1));

            // Generate uniform quaternion using Shoemake's method
            const T u1 = dist(rng);
            const T u2 = dist(rng) * two_pi<T>;
            const T u3 = dist(rng) * two_pi<T>;

            const T sqrt_u1 = std::sqrt(u1);
            const T sqrt_1_u1 = std::sqrt(T(1) - u1);

            return SO3(sqrt_1_u1 * std::sin(u2), sqrt_1_u1 * std::cos(u2), sqrt_u1 * std::sin(u3),
                       sqrt_u1 * std::cos(u3));
        }

        // Fit closest SO3 to arbitrary 3x3 matrix (via SVD/polar decomposition)
        [[nodiscard]] static SO3 fit_to_SO3(const RotationMatrix &M) noexcept {
            // Simple approach: treat as rotation matrix and convert
            // For a proper implementation, we'd use SVD
            return SO3(M);
        }

        /// Compute rotation that maps vector v1 to vector v2.
        ///
        /// Given two 3D vectors, computes the rotation R such that R * v1 is parallel to v2.
        /// The vectors do not need to be normalized - the method handles normalization internally.
        ///
        /// Edge cases:
        /// - Parallel vectors (same direction): returns identity rotation
        /// - Anti-parallel vectors (opposite direction): returns 180° rotation around an orthogonal axis
        ///
        /// @param v1 Source vector (will be rotated)
        /// @param v2 Target vector (rotation destination)
        /// @return SO3 rotation that maps v1 direction to v2 direction
        [[nodiscard]] static SO3 from_two_vectors(const Point &v1, const Point &v2) noexcept {
            // Compute norms
            const T norm1 = std::sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
            const T norm2 = std::sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);

            // Handle degenerate cases (zero vectors)
            if (norm1 < epsilon<T> || norm2 < epsilon<T>) {
                return identity();
            }

            // Normalize vectors
            const T inv_norm1 = T(1) / norm1;
            const T inv_norm2 = T(1) / norm2;
            const Point u1{{v1[0] * inv_norm1, v1[1] * inv_norm1, v1[2] * inv_norm1}};
            const Point u2{{v2[0] * inv_norm2, v2[1] * inv_norm2, v2[2] * inv_norm2}};

            // Compute dot product (cosine of angle)
            const T dot = u1[0] * u2[0] + u1[1] * u2[1] + u1[2] * u2[2];

            // Compute cross product (axis of rotation, scaled by sin(angle))
            const T cx = u1[1] * u2[2] - u1[2] * u2[1];
            const T cy = u1[2] * u2[0] - u1[0] * u2[2];
            const T cz = u1[0] * u2[1] - u1[1] * u2[0];
            const T cross_norm = std::sqrt(cx * cx + cy * cy + cz * cz);

            // Case 1: Vectors are nearly parallel (same direction)
            if (dot > T(1) - epsilon<T>) {
                return identity();
            }

            // Case 2: Vectors are nearly anti-parallel (opposite direction)
            // Need to find an orthogonal axis for 180° rotation
            if (dot < T(-1) + epsilon<T>) {
                // Find an axis orthogonal to v1
                // Try cross product with x-axis first, if too small use y-axis
                Point ortho;
                if (std::abs(u1[0]) < T(0.9)) {
                    // Cross with x-axis: [1,0,0] x u1 = [0, u1[2], -u1[1]]
                    ortho = Point{{T(0), u1[2], -u1[1]}};
                } else {
                    // Cross with y-axis: [0,1,0] x u1 = [-u1[2], 0, u1[0]]
                    ortho = Point{{-u1[2], T(0), u1[0]}};
                }

                // Normalize the orthogonal axis
                const T ortho_norm = std::sqrt(ortho[0] * ortho[0] + ortho[1] * ortho[1] + ortho[2] * ortho[2]);
                const T inv_ortho_norm = T(1) / ortho_norm;
                const Point axis{{ortho[0] * inv_ortho_norm, ortho[1] * inv_ortho_norm, ortho[2] * inv_ortho_norm}};

                // 180° rotation around the orthogonal axis
                return from_axis_angle(axis, pi<T>);
            }

            // Case 3: General case - use Rodrigues' rotation formula via quaternion
            // The rotation quaternion from v1 to v2 can be computed as:
            // q = [1 + dot, cross] normalized
            // This is equivalent to half-angle formula
            const T w = T(1) + dot;
            Quaternion q{w, cx, cy, cz};

            // Normalize the quaternion
            const T qnorm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
            const T inv_qnorm = T(1) / qnorm;
            return SO3(q.w * inv_qnorm, q.x * inv_qnorm, q.y * inv_qnorm, q.z * inv_qnorm);
        }

        // ===== CORE OPERATIONS =====

        // Logarithmic map: SO3 -> omega (axis-angle)
        [[nodiscard]] Tangent log() const noexcept {
            // v = [x, y, z], |v| = sin(theta/2)
            const T v_norm_sq = q_.x * q_.x + q_.y * q_.y + q_.z * q_.z;

            if (v_norm_sq < epsilon<T> * epsilon<T>) {
                // Near identity: omega ≈ 2 * v
                return Tangent{{T(2) * q_.x, T(2) * q_.y, T(2) * q_.z}};
            }

            const T v_norm = std::sqrt(v_norm_sq);
            // theta = 2 * atan2(|v|, w)
            const T theta = T(2) * std::atan2(v_norm, q_.w);

            // omega = theta * v / |v|
            const T scale = theta / v_norm;
            return Tangent{{scale * q_.x, scale * q_.y, scale * q_.z}};
        }

        // Log that also returns the angle
        [[nodiscard]] std::pair<Tangent, T> log_and_theta() const noexcept {
            const T v_norm_sq = q_.x * q_.x + q_.y * q_.y + q_.z * q_.z;

            if (v_norm_sq < epsilon<T> * epsilon<T>) {
                return {Tangent{{T(2) * q_.x, T(2) * q_.y, T(2) * q_.z}}, T(0)};
            }

            const T v_norm = std::sqrt(v_norm_sq);
            const T theta = T(2) * std::atan2(v_norm, q_.w);
            const T scale = theta / v_norm;

            return {Tangent{{scale * q_.x, scale * q_.y, scale * q_.z}}, theta};
        }

        // Inverse: quaternion conjugate (for unit quaternion)
        [[nodiscard]] SO3 inverse() const noexcept { return SO3(q_.w, -q_.x, -q_.y, -q_.z); }

        // Group composition: Hamilton product
        [[nodiscard]] SO3 operator*(const SO3 &other) const noexcept {
            // Hamilton product: q1 * q2
            const T w = q_.w * other.q_.w - q_.x * other.q_.x - q_.y * other.q_.y - q_.z * other.q_.z;
            const T x = q_.w * other.q_.x + q_.x * other.q_.w + q_.y * other.q_.z - q_.z * other.q_.y;
            const T y = q_.w * other.q_.y - q_.x * other.q_.z + q_.y * other.q_.w + q_.z * other.q_.x;
            const T z = q_.w * other.q_.z + q_.x * other.q_.y - q_.y * other.q_.x + q_.z * other.q_.w;
            return SO3(w, x, y, z);
        }

        SO3 &operator*=(const SO3 &other) noexcept {
            *this = *this * other;
            return *this;
        }

        // Rotate a 3D point: R * p  (equivalent to q * p * q^-1)
        [[nodiscard]] Point operator*(const Point &p) const noexcept {
            // Optimized rotation using quaternion with SIMD cross product utilities
            // v' = q * v * q^-1 = v + 2*w*(q_v x v) + 2*(q_v x (q_v x v))
            // where q_v = [x, y, z]

            // q_v as array for SIMD cross product
            const T q_v[3] = {q_.x, q_.y, q_.z};

            // t = 2 * (q_v x v)
            T t[3];
            simd::backend::cross_scale(q_v, p.data(), T(2), t);

            // t2 = q_v x t
            T t2[3];
            simd::backend::cross(q_v, t, t2);

            // v' = v + w*t + t2
            Point result;
            result[0] = p[0] + q_.w * t[0] + t2[0];
            result[1] = p[1] + q_.w * t[1] + t2[1];
            result[2] = p[2] + q_.w * t[2] + t2[2];

            return result;
        }

        // ===== ROTATION MATRIX =====

        // Return 3x3 rotation matrix
        [[nodiscard]] RotationMatrix matrix() const noexcept {
            const T w = q_.w, x = q_.x, y = q_.y, z = q_.z;

            const T xx = x * x, yy = y * y, zz = z * z;
            const T xy = x * y, xz = x * z, yz = y * z;
            const T wx = w * x, wy = w * y, wz = w * z;

            RotationMatrix R;
            R(0, 0) = T(1) - T(2) * (yy + zz);
            R(0, 1) = T(2) * (xy - wz);
            R(0, 2) = T(2) * (xz + wy);

            R(1, 0) = T(2) * (xy + wz);
            R(1, 1) = T(1) - T(2) * (xx + zz);
            R(1, 2) = T(2) * (yz - wx);

            R(2, 0) = T(2) * (xz - wy);
            R(2, 1) = T(2) * (yz + wx);
            R(2, 2) = T(1) - T(2) * (xx + yy);

            return R;
        }

        // ===== LIE ALGEBRA =====

        // hat: omega -> 3x3 skew-symmetric matrix
        // hat([wx, wy, wz]) = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]
        [[nodiscard]] static RotationMatrix hat(const Tangent &omega) noexcept {
            RotationMatrix Omega;
            Omega(0, 0) = T(0);
            Omega(0, 1) = -omega[2];
            Omega(0, 2) = omega[1];
            Omega(1, 0) = omega[2];
            Omega(1, 1) = T(0);
            Omega(1, 2) = -omega[0];
            Omega(2, 0) = -omega[1];
            Omega(2, 1) = omega[0];
            Omega(2, 2) = T(0);
            return Omega;
        }

        // vee: 3x3 skew-symmetric matrix -> omega
        [[nodiscard]] static Tangent vee(const RotationMatrix &Omega) noexcept {
            // Extract from skew-symmetric (average for numerical stability)
            return Tangent{{(Omega(2, 1) - Omega(1, 2)) / T(2), (Omega(0, 2) - Omega(2, 0)) / T(2),
                            (Omega(1, 0) - Omega(0, 1)) / T(2)}};
        }

        // Adjoint representation: Adj = R for SO3
        [[nodiscard]] AdjointMatrix Adj() const noexcept { return matrix(); }

        // Lie bracket [a, b] = a x b (cross product for so(3))
        [[nodiscard]] static Tangent lie_bracket(const Tangent &a, const Tangent &b) noexcept {
            return Tangent{{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}};
        }

        // Generator matrices (3 generators for SO3)
        [[nodiscard]] static RotationMatrix generator(std::size_t i) noexcept {
            Tangent e;
            e[0] = (i == 0) ? T(1) : T(0);
            e[1] = (i == 1) ? T(1) : T(0);
            e[2] = (i == 2) ? T(1) : T(0);
            return hat(e);
        }

        // ===== JACOBIANS =====

        // Left Jacobian J_l(omega)
        // J_l = I + (1-cos(theta))/theta^2 * hat(omega) + (theta - sin(theta))/theta^3 * hat(omega)^2
        [[nodiscard]] static AdjointMatrix left_jacobian(const Tangent &omega) noexcept {
            const T theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];

            AdjointMatrix J;

            if (theta_sq < epsilon<T> * epsilon<T>) {
                // Small angle: J ≈ I + 0.5 * hat(omega)
                const T half = T(0.5);
                J(0, 0) = T(1);
                J(0, 1) = -half * omega[2];
                J(0, 2) = half * omega[1];
                J(1, 0) = half * omega[2];
                J(1, 1) = T(1);
                J(1, 2) = -half * omega[0];
                J(2, 0) = -half * omega[1];
                J(2, 1) = half * omega[0];
                J(2, 2) = T(1);
                return J;
            }

            const T theta = std::sqrt(theta_sq);
            const T s = std::sin(theta);
            const T c = std::cos(theta);

            // Coefficients
            const T a = (T(1) - c) / theta_sq;            // (1-cos)/theta^2
            const T b = (theta - s) / (theta_sq * theta); // (theta-sin)/theta^3

            // hat(omega) using SIMD skew utility
            T Omega_data[9];
            simd::backend::skew(omega.data(), Omega_data);

            // hat(omega)^2 using SIMD matmul
            T Omega2_data[9];
            simd::backend::matmul<T, 3, 3, 3>(Omega2_data, Omega_data, Omega_data);

            // J = I + a * Omega + b * Omega^2
            // Column-major layout: J[col*3 + row]
            for (int col = 0; col < 3; ++col) {
                for (int row = 0; row < 3; ++row) {
                    const int idx = col * 3 + row;
                    J(row, col) = (row == col ? T(1) : T(0)) + a * Omega_data[idx] + b * Omega2_data[idx];
                }
            }

            return J;
        }

        // Inverse of left Jacobian
        // J_l^-1 = I - 0.5*hat(omega) + (1/theta^2 - (1+cos)/(2*theta*sin)) * hat(omega)^2
        [[nodiscard]] static AdjointMatrix left_jacobian_inverse(const Tangent &omega) noexcept {
            const T theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];

            AdjointMatrix J_inv;

            if (theta_sq < epsilon<T> * epsilon<T>) {
                // Small angle: J^-1 ≈ I - 0.5 * hat(omega)
                const T half = T(0.5);
                J_inv(0, 0) = T(1);
                J_inv(0, 1) = half * omega[2];
                J_inv(0, 2) = -half * omega[1];
                J_inv(1, 0) = -half * omega[2];
                J_inv(1, 1) = T(1);
                J_inv(1, 2) = half * omega[0];
                J_inv(2, 0) = half * omega[1];
                J_inv(2, 1) = -half * omega[0];
                J_inv(2, 2) = T(1);
                return J_inv;
            }

            const T theta = std::sqrt(theta_sq);
            const T half_theta = theta / T(2);

            // Coefficient for hat(omega)^2 term
            const T cot_half = T(1) / std::tan(half_theta);
            const T a = T(1) / theta_sq - cot_half / (T(2) * theta);

            // hat(omega) using SIMD skew utility
            T Omega_data[9];
            simd::backend::skew(omega.data(), Omega_data);

            // hat(omega)^2 using SIMD matmul
            T Omega2_data[9];
            simd::backend::matmul<T, 3, 3, 3>(Omega2_data, Omega_data, Omega_data);

            // J^-1 = I - 0.5*Omega + a*Omega^2
            // Column-major layout: J[col*3 + row]
            for (int col = 0; col < 3; ++col) {
                for (int row = 0; row < 3; ++row) {
                    const int idx = col * 3 + row;
                    J_inv(row, col) = (row == col ? T(1) : T(0)) - T(0.5) * Omega_data[idx] + a * Omega2_data[idx];
                }
            }

            return J_inv;
        }

        // ===== ACCESSORS =====

        // Get unit quaternion
        [[nodiscard]] constexpr const Quaternion &unit_quaternion() const noexcept { return q_; }

        // Get quaternion components
        [[nodiscard]] constexpr T w() const noexcept { return q_.w; }
        [[nodiscard]] constexpr T x() const noexcept { return q_.x; }
        [[nodiscard]] constexpr T y() const noexcept { return q_.y; }
        [[nodiscard]] constexpr T z() const noexcept { return q_.z; }

        // Get rotation angle (magnitude of rotation)
        [[nodiscard]] T angle() const noexcept {
            const T v_norm = std::sqrt(q_.x * q_.x + q_.y * q_.y + q_.z * q_.z);
            return T(2) * std::atan2(v_norm, std::abs(q_.w));
        }

        // Get rotation axis (unit vector, undefined for zero rotation)
        [[nodiscard]] Tangent axis() const noexcept {
            const T v_norm = std::sqrt(q_.x * q_.x + q_.y * q_.y + q_.z * q_.z);
            if (v_norm < epsilon<T>) {
                return Tangent{{T(1), T(0), T(0)}}; // Default to x-axis
            }
            return Tangent{{q_.x / v_norm, q_.y / v_norm, q_.z / v_norm}};
        }

        // Extract Euler angles (ZYX convention: roll, pitch, yaw)
        [[nodiscard]] Tangent to_euler() const noexcept {
            const T w = q_.w, x = q_.x, y = q_.y, z = q_.z;

            // Roll (x-axis rotation)
            const T sinr_cosp = T(2) * (w * x + y * z);
            const T cosr_cosp = T(1) - T(2) * (x * x + y * y);
            const T roll = std::atan2(sinr_cosp, cosr_cosp);

            // Pitch (y-axis rotation)
            const T sinp = T(2) * (w * y - z * x);
            T pitch;
            if (std::abs(sinp) >= T(1)) {
                pitch = std::copysign(half_pi<T>, sinp);
            } else {
                pitch = std::asin(sinp);
            }

            // Yaw (z-axis rotation)
            const T siny_cosp = T(2) * (w * z + x * y);
            const T cosy_cosp = T(1) - T(2) * (y * y + z * z);
            const T yaw = std::atan2(siny_cosp, cosy_cosp);

            return Tangent{{roll, pitch, yaw}};
        }

        // Raw data pointer
        [[nodiscard]] T *data() noexcept { return &q_.w; }
        [[nodiscard]] const T *data() const noexcept { return &q_.w; }

        // ===== MUTATORS =====

        // Normalize quaternion
        void normalize() noexcept {
            const T n = std::sqrt(q_.w * q_.w + q_.x * q_.x + q_.y * q_.y + q_.z * q_.z);
            if (n > epsilon<T>) {
                q_.w /= n;
                q_.x /= n;
                q_.y /= n;
                q_.z /= n;
            } else {
                q_ = Quaternion::identity();
            }
            // Ensure w >= 0 for canonical form (hemisphere constraint)
            if (q_.w < T(0)) {
                q_.w = -q_.w;
                q_.x = -q_.x;
                q_.y = -q_.y;
                q_.z = -q_.z;
            }
        }

        // Set from rotation matrix
        void set_rotation_matrix(const RotationMatrix &R) noexcept {
            // Shepperd's method for numerical stability
            const T trace = R(0, 0) + R(1, 1) + R(2, 2);

            if (trace > T(0)) {
                const T s = T(0.5) / std::sqrt(trace + T(1));
                q_.w = T(0.25) / s;
                q_.x = (R(2, 1) - R(1, 2)) * s;
                q_.y = (R(0, 2) - R(2, 0)) * s;
                q_.z = (R(1, 0) - R(0, 1)) * s;
            } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
                const T s = T(2) * std::sqrt(T(1) + R(0, 0) - R(1, 1) - R(2, 2));
                q_.w = (R(2, 1) - R(1, 2)) / s;
                q_.x = T(0.25) * s;
                q_.y = (R(0, 1) + R(1, 0)) / s;
                q_.z = (R(0, 2) + R(2, 0)) / s;
            } else if (R(1, 1) > R(2, 2)) {
                const T s = T(2) * std::sqrt(T(1) + R(1, 1) - R(0, 0) - R(2, 2));
                q_.w = (R(0, 2) - R(2, 0)) / s;
                q_.x = (R(0, 1) + R(1, 0)) / s;
                q_.y = T(0.25) * s;
                q_.z = (R(1, 2) + R(2, 1)) / s;
            } else {
                const T s = T(2) * std::sqrt(T(1) + R(2, 2) - R(0, 0) - R(1, 1));
                q_.w = (R(1, 0) - R(0, 1)) / s;
                q_.x = (R(0, 2) + R(2, 0)) / s;
                q_.y = (R(1, 2) + R(2, 1)) / s;
                q_.z = T(0.25) * s;
            }

            normalize();
        }

        // ===== TYPE CONVERSION =====

        template <typename NewScalar> [[nodiscard]] SO3<NewScalar> cast() const noexcept {
            return SO3<NewScalar>(static_cast<NewScalar>(q_.w), static_cast<NewScalar>(q_.x),
                                  static_cast<NewScalar>(q_.y), static_cast<NewScalar>(q_.z));
        }

        // ===== COMPARISON =====

        [[nodiscard]] bool operator==(const SO3 &other) const noexcept {
            // Account for q and -q representing the same rotation
            const T dot = q_.w * other.q_.w + q_.x * other.q_.x + q_.y * other.q_.y + q_.z * other.q_.z;
            return std::abs(std::abs(dot) - T(1)) < epsilon<T>;
        }

        [[nodiscard]] bool operator!=(const SO3 &other) const noexcept { return !(*this == other); }

        [[nodiscard]] bool is_approx(const SO3 &other, T tol = epsilon<T>) const noexcept {
            const T dot = q_.w * other.q_.w + q_.x * other.q_.x + q_.y * other.q_.y + q_.z * other.q_.z;
            return std::abs(std::abs(dot) - T(1)) < tol;
        }

        [[nodiscard]] bool is_identity(T tol = epsilon<T>) const noexcept {
            return std::abs(q_.w - T(1)) < tol && std::abs(q_.x) < tol && std::abs(q_.y) < tol && std::abs(q_.z) < tol;
        }

      private:
        Quaternion q_; // Unit quaternion [w, x, y, z]
    };

    // ===== TYPE ALIASES =====

    using SO3f = SO3<float>;
    using SO3d = SO3<double>;

    // ===== FREE FUNCTIONS =====

    // Geodesic interpolation (SLERP on SO3)
    template <typename T> [[nodiscard]] SO3<T> interpolate(const SO3<T> &a, const SO3<T> &b, T t) noexcept {
        // omega = log(a^-1 * b)
        auto omega = (a.inverse() * b).log();
        // Scale by t
        omega[0] *= t;
        omega[1] *= t;
        omega[2] *= t;
        // result = a * exp(t * omega)
        return a * SO3<T>::exp(omega);
    }

    // SLERP (direct quaternion interpolation)
    template <typename T> [[nodiscard]] SO3<T> slerp(const SO3<T> &a, const SO3<T> &b, T t) noexcept {
        const auto &qa = a.unit_quaternion();
        auto qb = b.unit_quaternion();

        // Ensure shortest path
        T dot = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z;
        if (dot < T(0)) {
            qb.w = -qb.w;
            qb.x = -qb.x;
            qb.y = -qb.y;
            qb.z = -qb.z;
            dot = -dot;
        }

        if (dot > T(1) - epsilon<T>) {
            // Linear interpolation for very close quaternions
            dp::mat::quaternion<T> result{qa.w + t * (qb.w - qa.w), qa.x + t * (qb.x - qa.x), qa.y + t * (qb.y - qa.y),
                                          qa.z + t * (qb.z - qa.z)};
            return SO3<T>(result);
        }

        const T theta = std::acos(dot);
        const T sin_theta = std::sin(theta);
        const T wa = std::sin((T(1) - t) * theta) / sin_theta;
        const T wb = std::sin(t * theta) / sin_theta;

        return SO3<T>(wa * qa.w + wb * qb.w, wa * qa.x + wb * qb.x, wa * qa.y + wb * qb.y, wa * qa.z + wb * qb.z);
    }

} // namespace optinum::lie
