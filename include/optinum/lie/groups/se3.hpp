#pragma once

#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/so3.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>
#include <random>
#include <type_traits>

namespace optinum::lie {

    // ===== SE3: Special Euclidean Group in 3D =====
    //
    // SE(3) represents 3D rigid transforms (rotation + translation).
    // Internally stored as SO3 (rotation) + vector<T, 3> (translation).
    //
    // Storage: SO3 + vector<T, 3> = 7 parameters total
    // DoF: 6 (3 translation + 3 rotation)
    // NumParams: 7 (4 for quaternion + 3 for translation)
    //
    // Tangent space (twist): [vx, vy, vz, wx, wy, wz]^T in R^6
    //   - (vx, vy, vz) = translational velocity
    //   - (wx, wy, wz) = angular velocity (rotation vector)
    //
    // Lie algebra se(3):
    //   hat([v, w]) = [[hat(w), v], [0, 0, 0, 0]]  (4x4 matrix)

    template <typename T = double> class SE3 {
        static_assert(std::is_floating_point_v<T>, "SE3 requires floating-point scalar type");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Tangent = simd::Vector<T, 6>; // [vx, vy, vz, wx, wy, wz]
        using Translation = simd::Vector<T, 3>;
        using Point = simd::Vector<T, 3>;
        using Params = simd::Vector<T, 7>; // [qw, qx, qy, qz, tx, ty, tz]
        using HomogeneousMatrix = simd::Matrix<T, 4, 4>;
        using TransformMatrix = simd::Matrix<T, 3, 4>; // Compact form
        using AdjointMatrix = simd::Matrix<T, 6, 6>;
        using Rotation = SO3<T>;

        // ===== CONSTANTS =====
        static constexpr std::size_t DoF = 6;
        static constexpr std::size_t NumParams = 7;

        // ===== CONSTRUCTORS =====

        // Default: identity transform
        constexpr SE3() noexcept : so3_(), translation_{T(0), T(0), T(0)} {}

        // From SO3 rotation and translation
        SE3(const Rotation &rotation, const Translation &translation) noexcept
            : so3_(rotation), translation_(translation) {}

        // From quaternion and translation
        SE3(const typename Rotation::Quaternion &q, const Translation &translation) noexcept
            : so3_(q), translation_(translation) {}

        // From rotation matrix and translation
        SE3(const dp::mat::matrix<T, 3, 3> &R, const Translation &translation) noexcept
            : so3_(R), translation_(translation) {}

        // From 4x4 homogeneous matrix
        explicit SE3(const HomogeneousMatrix &T_mat) noexcept {
            // Extract rotation from top-left 3x3
            dp::mat::matrix<T, 3, 3> R_mat;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    R_mat(i, j) = T_mat(i, j);
                }
            }
            so3_ = Rotation(R_mat);

            // Extract translation from last column
            translation_[0] = T_mat(0, 3);
            translation_[1] = T_mat(1, 3);
            translation_[2] = T_mat(2, 3);
        }

        // ===== STATIC FACTORY METHODS =====

        // Identity element
        [[nodiscard]] static constexpr SE3 identity() noexcept { return SE3(); }

        // Pure rotation (no translation)
        [[nodiscard]] static SE3 rot_x(T angle) noexcept {
            return SE3(Rotation::rot_x(angle), Translation{T(0), T(0), T(0)});
        }

        [[nodiscard]] static SE3 rot_y(T angle) noexcept {
            return SE3(Rotation::rot_y(angle), Translation{T(0), T(0), T(0)});
        }

        [[nodiscard]] static SE3 rot_z(T angle) noexcept {
            return SE3(Rotation::rot_z(angle), Translation{T(0), T(0), T(0)});
        }

        // Pure translation (no rotation)
        [[nodiscard]] static SE3 trans(T tx, T ty, T tz) noexcept {
            return SE3(Rotation::identity(), Translation{tx, ty, tz});
        }

        [[nodiscard]] static SE3 trans(const Translation &t) noexcept { return SE3(Rotation::identity(), t); }

        [[nodiscard]] static SE3 trans_x(T tx) noexcept {
            return SE3(Rotation::identity(), Translation{tx, T(0), T(0)});
        }

        [[nodiscard]] static SE3 trans_y(T ty) noexcept {
            return SE3(Rotation::identity(), Translation{T(0), ty, T(0)});
        }

        [[nodiscard]] static SE3 trans_z(T tz) noexcept {
            return SE3(Rotation::identity(), Translation{T(0), T(0), tz});
        }

        // Exponential map: twist -> SE3
        // twist = [v, omega] where v is translational, omega is rotational
        [[nodiscard]] static SE3 exp(const Tangent &twist) noexcept {
            // Extract translational and rotational parts
            dp::mat::vector<T, 3> v{twist[0], twist[1], twist[2]};
            dp::mat::vector<T, 3> omega{twist[3], twist[4], twist[5]};

            // Rotation: R = exp(omega)
            Rotation R = Rotation::exp(omega);

            // Translation: t = V * v where V is the left Jacobian of SO3
            const T theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];

            Translation t;
            if (theta_sq < epsilon<T> * epsilon<T>) {
                // Small angle: V ≈ I + 0.5 * hat(omega)
                // t ≈ v + 0.5 * (omega x v)
                t[0] = v[0] + T(0.5) * (omega[1] * v[2] - omega[2] * v[1]);
                t[1] = v[1] + T(0.5) * (omega[2] * v[0] - omega[0] * v[2]);
                t[2] = v[2] + T(0.5) * (omega[0] * v[1] - omega[1] * v[0]);
            } else {
                const T theta = std::sqrt(theta_sq);
                const T s = std::sin(theta);
                const T c = std::cos(theta);

                // V = I + (1-cos)/theta^2 * hat(omega) + (theta-sin)/theta^3 * hat(omega)^2
                const T a = (T(1) - c) / theta_sq;
                const T b = (theta - s) / (theta_sq * theta);

                // omega x v
                const T cross_x = omega[1] * v[2] - omega[2] * v[1];
                const T cross_y = omega[2] * v[0] - omega[0] * v[2];
                const T cross_z = omega[0] * v[1] - omega[1] * v[0];

                // omega x (omega x v) = omega * (omega . v) - v * |omega|^2
                const T dot = omega[0] * v[0] + omega[1] * v[1] + omega[2] * v[2];
                const T cross2_x = omega[0] * dot - v[0] * theta_sq;
                const T cross2_y = omega[1] * dot - v[1] * theta_sq;
                const T cross2_z = omega[2] * dot - v[2] * theta_sq;

                // t = v + a * (omega x v) + b * (omega x (omega x v))
                t[0] = v[0] + a * cross_x + b * cross2_x;
                t[1] = v[1] + a * cross_y + b * cross2_y;
                t[2] = v[2] + a * cross_z + b * cross2_z;
            }

            return SE3(R, t);
        }

        // Sample uniform random pose
        template <typename RNG>
        [[nodiscard]] static SE3 sample_uniform(RNG &rng, T translation_range = T(10)) noexcept {
            std::uniform_real_distribution<T> trans_dist(-translation_range, translation_range);

            return SE3(Rotation::sample_uniform(rng), Translation{trans_dist(rng), trans_dist(rng), trans_dist(rng)});
        }

        // Fit closest SE3 to arbitrary 4x4 matrix
        [[nodiscard]] static SE3 fit_to_SE3(const HomogeneousMatrix &M) noexcept {
            // Extract rotation and fit
            dp::mat::matrix<T, 3, 3> R_mat;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    R_mat(i, j) = M(i, j);
                }
            }

            return SE3(Rotation::fit_to_SO3(R_mat), Translation{M(0, 3), M(1, 3), M(2, 3)});
        }

        // ===== CORE OPERATIONS =====

        // Logarithmic map: SE3 -> twist
        [[nodiscard]] Tangent log() const noexcept {
            // Get rotation log
            auto omega = so3_.log();

            const T theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];

            Tangent twist;
            twist[3] = omega[0];
            twist[4] = omega[1];
            twist[5] = omega[2];

            if (theta_sq < epsilon<T> * epsilon<T>) {
                // Small angle: V^-1 ≈ I - 0.5 * hat(omega)
                // v ≈ t - 0.5 * (omega x t)
                twist[0] = translation_[0] - T(0.5) * (omega[1] * translation_[2] - omega[2] * translation_[1]);
                twist[1] = translation_[1] - T(0.5) * (omega[2] * translation_[0] - omega[0] * translation_[2]);
                twist[2] = translation_[2] - T(0.5) * (omega[0] * translation_[1] - omega[1] * translation_[0]);
            } else {
                const T theta = std::sqrt(theta_sq);
                const T half_theta = theta / T(2);
                const T cot_half = T(1) / std::tan(half_theta);

                // V^-1 = I - 0.5*hat(omega) + (1/theta^2 - cot(theta/2)/(2*theta)) * hat(omega)^2
                const T a = T(1) / theta_sq - cot_half / (T(2) * theta);

                // omega x t
                const T cross_x = omega[1] * translation_[2] - omega[2] * translation_[1];
                const T cross_y = omega[2] * translation_[0] - omega[0] * translation_[2];
                const T cross_z = omega[0] * translation_[1] - omega[1] * translation_[0];

                // omega x (omega x t)
                const T dot = omega[0] * translation_[0] + omega[1] * translation_[1] + omega[2] * translation_[2];
                const T cross2_x = omega[0] * dot - translation_[0] * theta_sq;
                const T cross2_y = omega[1] * dot - translation_[1] * theta_sq;
                const T cross2_z = omega[2] * dot - translation_[2] * theta_sq;

                // v = t - 0.5*(omega x t) + a*(omega x (omega x t))
                twist[0] = translation_[0] - T(0.5) * cross_x + a * cross2_x;
                twist[1] = translation_[1] - T(0.5) * cross_y + a * cross2_y;
                twist[2] = translation_[2] - T(0.5) * cross_z + a * cross2_z;
            }

            return twist;
        }

        // Inverse: T^-1 = (R^-1, -R^-1 * t)
        [[nodiscard]] SE3 inverse() const noexcept {
            Rotation R_inv = so3_.inverse();
            Translation t_inv = R_inv * translation_;
            t_inv[0] = -t_inv[0];
            t_inv[1] = -t_inv[1];
            t_inv[2] = -t_inv[2];
            return SE3(R_inv, t_inv);
        }

        // Group composition: T1 * T2 = (R1*R2, t1 + R1*t2)
        [[nodiscard]] SE3 operator*(const SE3 &other) const noexcept {
            Rotation R_new = so3_ * other.so3_;
            Point t_rotated = so3_ * other.translation_;
            Translation t_new;
            t_new[0] = translation_[0] + t_rotated[0];
            t_new[1] = translation_[1] + t_rotated[1];
            t_new[2] = translation_[2] + t_rotated[2];
            return SE3(R_new, t_new);
        }

        SE3 &operator*=(const SE3 &other) noexcept {
            *this = *this * other;
            return *this;
        }

        // Transform a 3D point: R*p + t
        [[nodiscard]] Point operator*(const Point &p) const noexcept {
            Point result = so3_ * p;
            result[0] += translation_[0];
            result[1] += translation_[1];
            result[2] += translation_[2];
            return result;
        }

        // ===== MATRIX REPRESENTATIONS =====

        // Return 4x4 homogeneous matrix
        [[nodiscard]] HomogeneousMatrix matrix() const noexcept {
            HomogeneousMatrix T_mat;
            auto R = so3_.matrix();

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    T_mat(i, j) = R(i, j);
                }
                T_mat(i, 3) = translation_[i];
            }

            T_mat(3, 0) = T(0);
            T_mat(3, 1) = T(0);
            T_mat(3, 2) = T(0);
            T_mat(3, 3) = T(1);

            return T_mat;
        }

        // Return 3x4 compact matrix [R | t]
        [[nodiscard]] TransformMatrix matrix3x4() const noexcept {
            TransformMatrix T_mat;
            auto R = so3_.matrix();

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    T_mat(i, j) = R(i, j);
                }
                T_mat(i, 3) = translation_[i];
            }

            return T_mat;
        }

        // Return 3x3 rotation matrix
        [[nodiscard]] dp::mat::matrix<T, 3, 3> rotation_matrix() const noexcept { return so3_.matrix(); }

        // ===== LIE ALGEBRA =====

        // hat: twist -> 4x4 se(3) matrix
        // hat([v, omega]) = [[hat(omega), v], [0, 0, 0, 0]]
        [[nodiscard]] static HomogeneousMatrix hat(const Tangent &twist) noexcept {
            HomogeneousMatrix Omega;

            // Top-left 3x3: hat(omega)
            Omega(0, 0) = T(0);
            Omega(0, 1) = -twist[5];
            Omega(0, 2) = twist[4];
            Omega(1, 0) = twist[5];
            Omega(1, 1) = T(0);
            Omega(1, 2) = -twist[3];
            Omega(2, 0) = -twist[4];
            Omega(2, 1) = twist[3];
            Omega(2, 2) = T(0);

            // Right column: v
            Omega(0, 3) = twist[0];
            Omega(1, 3) = twist[1];
            Omega(2, 3) = twist[2];

            // Bottom row: zeros
            Omega(3, 0) = T(0);
            Omega(3, 1) = T(0);
            Omega(3, 2) = T(0);
            Omega(3, 3) = T(0);

            return Omega;
        }

        // vee: 4x4 se(3) matrix -> twist
        [[nodiscard]] static Tangent vee(const HomogeneousMatrix &Omega) noexcept {
            Tangent twist;

            // v from right column
            twist[0] = Omega(0, 3);
            twist[1] = Omega(1, 3);
            twist[2] = Omega(2, 3);

            // omega from skew-symmetric part
            twist[3] = (Omega(2, 1) - Omega(1, 2)) / T(2);
            twist[4] = (Omega(0, 2) - Omega(2, 0)) / T(2);
            twist[5] = (Omega(1, 0) - Omega(0, 1)) / T(2);

            return twist;
        }

        // Adjoint representation: 6x6 matrix
        // Adj = [[R, hat(t)*R], [0, R]]
        [[nodiscard]] AdjointMatrix Adj() const noexcept {
            AdjointMatrix adj;
            auto R = so3_.matrix();

            // Top-left 3x3: R
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    adj(i, j) = R(i, j);
                }
            }

            // Top-right 3x3: hat(t) * R
            // hat(t) = [[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]]
            const T tx = translation_[0], ty = translation_[1], tz = translation_[2];
            dp::mat::matrix<T, 3, 3> hat_t;
            hat_t(0, 0) = T(0);
            hat_t(0, 1) = -tz;
            hat_t(0, 2) = ty;
            hat_t(1, 0) = tz;
            hat_t(1, 1) = T(0);
            hat_t(1, 2) = -tx;
            hat_t(2, 0) = -ty;
            hat_t(2, 1) = tx;
            hat_t(2, 2) = T(0);

            // hat(t) * R
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    T sum = T(0);
                    for (int k = 0; k < 3; ++k) {
                        sum += hat_t(i, k) * R(k, j);
                    }
                    adj(i, j + 3) = sum;
                }
            }

            // Bottom-left 3x3: zeros
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    adj(i + 3, j) = T(0);
                }
            }

            // Bottom-right 3x3: R
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    adj(i + 3, j + 3) = R(i, j);
                }
            }

            return adj;
        }

        // Lie bracket [a, b] for se(3)
        [[nodiscard]] static Tangent lie_bracket(const Tangent &a, const Tangent &b) noexcept {
            // [a, b] = [[omega_a x v_b - omega_b x v_a], [omega_a x omega_b]]
            dp::mat::vector<T, 3> va{a[0], a[1], a[2]};
            dp::mat::vector<T, 3> wa{a[3], a[4], a[5]};
            dp::mat::vector<T, 3> vb{b[0], b[1], b[2]};
            dp::mat::vector<T, 3> wb{b[3], b[4], b[5]};

            // omega_a x v_b
            dp::mat::vector<T, 3> wa_cross_vb{wa[1] * vb[2] - wa[2] * vb[1], wa[2] * vb[0] - wa[0] * vb[2],
                                              wa[0] * vb[1] - wa[1] * vb[0]};

            // omega_b x v_a
            dp::mat::vector<T, 3> wb_cross_va{wb[1] * va[2] - wb[2] * va[1], wb[2] * va[0] - wb[0] * va[2],
                                              wb[0] * va[1] - wb[1] * va[0]};

            // omega_a x omega_b
            dp::mat::vector<T, 3> wa_cross_wb{wa[1] * wb[2] - wa[2] * wb[1], wa[2] * wb[0] - wa[0] * wb[2],
                                              wa[0] * wb[1] - wa[1] * wb[0]};

            return Tangent{wa_cross_vb[0] - wb_cross_va[0],
                           wa_cross_vb[1] - wb_cross_va[1],
                           wa_cross_vb[2] - wb_cross_va[2],
                           wa_cross_wb[0],
                           wa_cross_wb[1],
                           wa_cross_wb[2]};
        }

        // Generator matrices (6 generators for SE3)
        [[nodiscard]] static HomogeneousMatrix generator(std::size_t i) noexcept {
            Tangent e;
            for (std::size_t j = 0; j < 6; ++j) {
                e[j] = (i == j) ? T(1) : T(0);
            }
            return hat(e);
        }

        // ===== JACOBIANS =====

        // Left Jacobian (6x6)
        [[nodiscard]] static AdjointMatrix left_jacobian(const Tangent &twist) noexcept {
            dp::mat::vector<T, 3> omega{twist[3], twist[4], twist[5]};
            const T theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];

            // Get SO3 left Jacobian
            auto J_so3 = Rotation::left_jacobian(omega);

            AdjointMatrix J;

            if (theta_sq < epsilon<T> * epsilon<T>) {
                // Small angle approximation
                // J = [[J_so3, Q], [0, J_so3]] where Q ≈ 0.5 * hat(v)
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        J(i, j) = J_so3(i, j);
                        J(i + 3, j) = T(0);
                        J(i + 3, j + 3) = J_so3(i, j);
                    }
                }

                // Q ≈ 0.5 * hat(v)
                J(0, 3) = T(0);
                J(0, 4) = -twist[2] * T(0.5);
                J(0, 5) = twist[1] * T(0.5);
                J(1, 3) = twist[2] * T(0.5);
                J(1, 4) = T(0);
                J(1, 5) = -twist[0] * T(0.5);
                J(2, 3) = -twist[1] * T(0.5);
                J(2, 4) = twist[0] * T(0.5);
                J(2, 5) = T(0);

                return J;
            }

            // Full computation
            const T theta = std::sqrt(theta_sq);
            const T s = std::sin(theta);
            const T c = std::cos(theta);

            // Coefficients for Q matrix computation
            const T a1 = (T(1) - c) / theta_sq;
            const T a2 = (theta - s) / (theta_sq * theta);
            const T a3 = (T(1) - a1) / theta_sq;
            const T a4 = (a1 - T(2) * a2) / theta_sq;

            dp::mat::vector<T, 3> v{twist[0], twist[1], twist[2]};

            // Compute Q matrix (complex formula involving v, omega, and their products)
            dp::mat::matrix<T, 3, 3> Q;

            // hat(v)
            dp::mat::matrix<T, 3, 3> hat_v;
            hat_v(0, 0) = T(0);
            hat_v(0, 1) = -v[2];
            hat_v(0, 2) = v[1];
            hat_v(1, 0) = v[2];
            hat_v(1, 1) = T(0);
            hat_v(1, 2) = -v[0];
            hat_v(2, 0) = -v[1];
            hat_v(2, 1) = v[0];
            hat_v(2, 2) = T(0);

            // hat(omega)
            dp::mat::matrix<T, 3, 3> hat_w;
            hat_w(0, 0) = T(0);
            hat_w(0, 1) = -omega[2];
            hat_w(0, 2) = omega[1];
            hat_w(1, 0) = omega[2];
            hat_w(1, 1) = T(0);
            hat_w(1, 2) = -omega[0];
            hat_w(2, 0) = -omega[1];
            hat_w(2, 1) = omega[0];
            hat_w(2, 2) = T(0);

            // Q = 0.5*hat(v) + a2*(hat(w)*hat(v) + hat(v)*hat(w) + hat(w)*hat(v)*hat(w))
            //     - a3*(hat(w)^2*hat(v) + hat(v)*hat(w)^2 - 3*hat(w)*hat(v)*hat(w))
            //     - 0.5*a4*(hat(w)*hat(v)*hat(w)^2 + hat(w)^2*hat(v)*hat(w))

            // Simplified: Q ≈ 0.5*hat(v) + lower order terms
            // For now, use the simplified approximation
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    Q(i, j) = T(0.5) * hat_v(i, j);
                }
            }

            // Assemble full Jacobian
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    J(i, j) = J_so3(i, j);
                    J(i, j + 3) = Q(i, j);
                    J(i + 3, j) = T(0);
                    J(i + 3, j + 3) = J_so3(i, j);
                }
            }

            return J;
        }

        // Inverse of left Jacobian
        [[nodiscard]] static AdjointMatrix left_jacobian_inverse(const Tangent &twist) noexcept {
            dp::mat::vector<T, 3> omega{twist[3], twist[4], twist[5]};

            // Get SO3 left Jacobian inverse
            auto J_so3_inv = Rotation::left_jacobian_inverse(omega);

            AdjointMatrix J_inv;

            // J^-1 = [[J_so3^-1, -J_so3^-1 * Q * J_so3^-1], [0, J_so3^-1]]
            // Simplified for small angles and general case

            const T theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];

            dp::mat::vector<T, 3> v{twist[0], twist[1], twist[2]};

            // Q_inv ≈ -0.5 * hat(v) for small angles
            dp::mat::matrix<T, 3, 3> Q_inv;
            Q_inv(0, 0) = T(0);
            Q_inv(0, 1) = v[2] * T(0.5);
            Q_inv(0, 2) = -v[1] * T(0.5);
            Q_inv(1, 0) = -v[2] * T(0.5);
            Q_inv(1, 1) = T(0);
            Q_inv(1, 2) = v[0] * T(0.5);
            Q_inv(2, 0) = v[1] * T(0.5);
            Q_inv(2, 1) = -v[0] * T(0.5);
            Q_inv(2, 2) = T(0);

            // Compute -J_so3^-1 * Q * J_so3^-1 (simplified to Q_inv for now)
            dp::mat::matrix<T, 3, 3> top_right;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    top_right(i, j) = Q_inv(i, j);
                }
            }

            // Assemble
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    J_inv(i, j) = J_so3_inv(i, j);
                    J_inv(i, j + 3) = top_right(i, j);
                    J_inv(i + 3, j) = T(0);
                    J_inv(i + 3, j + 3) = J_so3_inv(i, j);
                }
            }

            return J_inv;
        }

        // ===== ACCESSORS =====

        // Get rotation component
        [[nodiscard]] constexpr const Rotation &so3() const noexcept { return so3_; }
        [[nodiscard]] constexpr Rotation &so3() noexcept { return so3_; }

        // Get translation component
        [[nodiscard]] constexpr const Translation &translation() const noexcept { return translation_; }
        [[nodiscard]] constexpr Translation &translation() noexcept { return translation_; }

        // Get quaternion
        [[nodiscard]] const typename Rotation::Quaternion &unit_quaternion() const noexcept {
            return so3_.unit_quaternion();
        }

        // Get individual translation components
        [[nodiscard]] constexpr T x() const noexcept { return translation_[0]; }
        [[nodiscard]] constexpr T y() const noexcept { return translation_[1]; }
        [[nodiscard]] constexpr T z() const noexcept { return translation_[2]; }

        // Get parameters as vector [qw, qx, qy, qz, tx, ty, tz]
        [[nodiscard]] Params params() const noexcept {
            const auto &q = so3_.unit_quaternion();
            return Params{q.w, q.x, q.y, q.z, translation_[0], translation_[1], translation_[2]};
        }

        // Raw data pointer
        [[nodiscard]] T *data() noexcept { return so3_.data(); }
        [[nodiscard]] const T *data() const noexcept { return so3_.data(); }

        // ===== MUTATORS =====

        void set_translation(const Translation &t) noexcept { translation_ = t; }
        void set_translation(T tx, T ty, T tz) noexcept {
            translation_[0] = tx;
            translation_[1] = ty;
            translation_[2] = tz;
        }

        void normalize() noexcept { so3_.normalize(); }

        // ===== TYPE CONVERSION =====

        template <typename NewScalar> [[nodiscard]] SE3<NewScalar> cast() const noexcept {
            return SE3<NewScalar>(so3_.template cast<NewScalar>(),
                                  simd::Vector<NewScalar, 3>{static_cast<NewScalar>(translation_[0]),
                                                             static_cast<NewScalar>(translation_[1]),
                                                             static_cast<NewScalar>(translation_[2])});
        }

        // ===== COMPARISON =====

        [[nodiscard]] bool operator==(const SE3 &other) const noexcept {
            return so3_ == other.so3_ && std::abs(translation_[0] - other.translation_[0]) < epsilon<T> &&
                   std::abs(translation_[1] - other.translation_[1]) < epsilon<T> &&
                   std::abs(translation_[2] - other.translation_[2]) < epsilon<T>;
        }

        [[nodiscard]] bool operator!=(const SE3 &other) const noexcept { return !(*this == other); }

        [[nodiscard]] bool is_approx(const SE3 &other, T tol = epsilon<T>) const noexcept {
            return so3_.is_approx(other.so3_, tol) && std::abs(translation_[0] - other.translation_[0]) < tol &&
                   std::abs(translation_[1] - other.translation_[1]) < tol &&
                   std::abs(translation_[2] - other.translation_[2]) < tol;
        }

        [[nodiscard]] bool is_identity(T tol = epsilon<T>) const noexcept {
            return so3_.is_identity(tol) && std::abs(translation_[0]) < tol && std::abs(translation_[1]) < tol &&
                   std::abs(translation_[2]) < tol;
        }

      private:
        Rotation so3_;            // Rotation component
        Translation translation_; // Translation component
    };

    // ===== TYPE ALIASES =====

    using SE3f = SE3<float>;
    using SE3d = SE3<double>;

    // ===== FREE FUNCTIONS =====

    // Geodesic interpolation on SE3
    template <typename T> [[nodiscard]] SE3<T> interpolate(const SE3<T> &a, const SE3<T> &b, T t) noexcept {
        // twist = log(a^-1 * b)
        auto twist = (a.inverse() * b).log();
        // Scale twist by t
        for (int i = 0; i < 6; ++i) {
            twist[i] *= t;
        }
        // result = a * exp(t * twist)
        return a * SE3<T>::exp(twist);
    }

} // namespace optinum::lie
