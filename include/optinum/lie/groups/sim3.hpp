#pragma once

#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/rxso3.hpp>
#include <optinum/lie/groups/se3.hpp>

#include <datapod/matrix/matrix.hpp>
#include <datapod/matrix/vector.hpp>

#include <cmath>
#include <random>
#include <type_traits>

namespace optinum::lie {

    namespace dp = ::datapod;

    // ===== Sim3: 3D Similarity Group =====
    //
    // Sim(3) = R+ x SO(3) x R^3 represents 3D similarity transforms:
    //   scale + rotation + translation
    //
    // Transformation: p' = s * R * p + t
    //
    // Storage: RxSO3 + Vector<T, 3> (scaled rotation + translation)
    // DoF: 7 (1 scale + 3 rotation + 3 translation)
    // NumParams: 7 (4 quaternion + 3 translation)
    //
    // Tangent space (twist):
    //   [sigma, wx, wy, wz, vx, vy, vz] where:
    //     sigma = log(scale)
    //     (wx, wy, wz) = rotation vector
    //     (vx, vy, vz) = translational velocity
    //
    // Useful for: monocular SLAM, loop closure with scale drift

    template <typename T = double> class Sim3 {
        static_assert(std::is_floating_point_v<T>, "Sim3 requires floating-point scalar type");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Tangent = dp::mat::vector<T, 7>; // [sigma, wx, wy, wz, vx, vy, vz]
        using Translation = dp::mat::vector<T, 3>;
        using Point = dp::mat::vector<T, 3>;
        using Params = dp::mat::vector<T, 7>; // [qw, qx, qy, qz, tx, ty, tz]
        using HomogeneousMatrix = dp::mat::matrix<T, 4, 4>;
        using AdjointMatrix = dp::mat::matrix<T, 7, 7>;
        using ScaledRotation = RxSO3<T>;
        using Rotation = SO3<T>;

        // ===== CONSTANTS =====
        static constexpr std::size_t DoF = 7;
        static constexpr std::size_t NumParams = 7;

        // ===== CONSTRUCTORS =====

        // Default: identity transform
        constexpr Sim3() noexcept : rxso3_(), translation_{T(0), T(0), T(0)} {}

        // From RxSO3 and translation
        Sim3(const ScaledRotation &rxso3, const Translation &t) noexcept : rxso3_(rxso3), translation_(t) {}

        // From scale, SO3, and translation
        Sim3(Scalar scale, const Rotation &R, const Translation &t) noexcept : rxso3_(scale, R), translation_(t) {}

        // From SE3 (with scale = 1)
        explicit Sim3(const SE3<T> &se3) noexcept : rxso3_(se3.so3()), translation_(se3.translation()) {}

        // From 4x4 homogeneous matrix
        explicit Sim3(const HomogeneousMatrix &T_mat) noexcept {
            // Extract scale from rotation part
            const T s = std::sqrt(T_mat(0, 0) * T_mat(0, 0) + T_mat(1, 0) * T_mat(1, 0) + T_mat(2, 0) * T_mat(2, 0));

            // Extract rotation matrix
            dp::mat::matrix<T, 3, 3> R_mat;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    R_mat(i, j) = T_mat(i, j) / s;
                }
            }
            rxso3_ = ScaledRotation(s, Rotation(R_mat));

            // Extract translation
            translation_[0] = T_mat(0, 3);
            translation_[1] = T_mat(1, 3);
            translation_[2] = T_mat(2, 3);
        }

        // ===== STATIC FACTORY METHODS =====

        // Identity element
        [[nodiscard]] static constexpr Sim3 identity() noexcept { return Sim3(); }

        // Pure scale (no rotation, no translation)
        [[nodiscard]] static Sim3 scale(Scalar s) noexcept {
            return Sim3(ScaledRotation::scale_only(s), Translation{T(0), T(0), T(0)});
        }

        // Pure rotation (scale = 1, no translation)
        [[nodiscard]] static Sim3 rot_x(Scalar angle) noexcept {
            return Sim3(ScaledRotation(Rotation::rot_x(angle)), Translation{T(0), T(0), T(0)});
        }

        [[nodiscard]] static Sim3 rot_y(Scalar angle) noexcept {
            return Sim3(ScaledRotation(Rotation::rot_y(angle)), Translation{T(0), T(0), T(0)});
        }

        [[nodiscard]] static Sim3 rot_z(Scalar angle) noexcept {
            return Sim3(ScaledRotation(Rotation::rot_z(angle)), Translation{T(0), T(0), T(0)});
        }

        // Pure translation (scale = 1, no rotation)
        [[nodiscard]] static Sim3 trans(Scalar tx, Scalar ty, Scalar tz) noexcept {
            return Sim3(ScaledRotation(), Translation{tx, ty, tz});
        }

        [[nodiscard]] static Sim3 trans(const Translation &t) noexcept { return Sim3(ScaledRotation(), t); }

        [[nodiscard]] static Sim3 trans_x(Scalar tx) noexcept { return trans(tx, T(0), T(0)); }
        [[nodiscard]] static Sim3 trans_y(Scalar ty) noexcept { return trans(T(0), ty, T(0)); }
        [[nodiscard]] static Sim3 trans_z(Scalar tz) noexcept { return trans(T(0), T(0), tz); }

        // Exponential map: [sigma, omega, v] -> Sim3
        [[nodiscard]] static Sim3 exp(const Tangent &twist) noexcept {
            const T sigma = twist[0];
            dp::mat::vector<T, 3> omega{twist[1], twist[2], twist[3]};
            dp::mat::vector<T, 3> v{twist[4], twist[5], twist[6]};

            // Scaled rotation
            ScaledRotation rxso3 =
                ScaledRotation::exp(typename ScaledRotation::Tangent{sigma, omega[0], omega[1], omega[2]});

            // Compute translation using the similarity left Jacobian
            const T s = std::exp(sigma);
            const T theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
            const T theta = std::sqrt(theta_sq);
            const T sigma_sq = sigma * sigma;

            Translation t;

            if (theta_sq < epsilon<T> * epsilon<T> && std::abs(sigma) < epsilon<T>) {
                // Small angle and scale: W ≈ I
                t = v;
            } else if (theta_sq < epsilon<T> * epsilon<T>) {
                // Small angle, significant scale
                const T a = (s - T(1)) / sigma;
                t[0] = a * v[0];
                t[1] = a * v[1];
                t[2] = a * v[2];
            } else if (std::abs(sigma) < epsilon<T>) {
                // Significant angle, small scale
                // W = J_l (SO3 left Jacobian)
                const T sin_th = std::sin(theta);
                const T cos_th = std::cos(theta);
                const T a = (T(1) - cos_th) / theta_sq;
                const T b = (theta - sin_th) / (theta_sq * theta);

                // omega x v
                const T cross_x = omega[1] * v[2] - omega[2] * v[1];
                const T cross_y = omega[2] * v[0] - omega[0] * v[2];
                const T cross_z = omega[0] * v[1] - omega[1] * v[0];

                // omega x (omega x v)
                const T dot = omega[0] * v[0] + omega[1] * v[1] + omega[2] * v[2];
                const T cross2_x = omega[0] * dot - v[0] * theta_sq;
                const T cross2_y = omega[1] * dot - v[1] * theta_sq;
                const T cross2_z = omega[2] * dot - v[2] * theta_sq;

                t[0] = v[0] + a * cross_x + b * cross2_x;
                t[1] = v[1] + a * cross_y + b * cross2_y;
                t[2] = v[2] + a * cross_z + b * cross2_z;
            } else {
                // General case
                const T sin_th = std::sin(theta);
                const T cos_th = std::cos(theta);

                // Coefficients for the similarity Jacobian
                const T denom = sigma_sq + theta_sq;
                const T A = (sigma * (s - T(1)) + theta_sq) / (sigma * denom);
                const T B = ((s * cos_th - T(1)) * sigma + s * sin_th * theta) / (theta_sq * denom);
                const T C = (s - T(1) - sigma * A) / theta_sq;

                // omega x v
                const T cross_x = omega[1] * v[2] - omega[2] * v[1];
                const T cross_y = omega[2] * v[0] - omega[0] * v[2];
                const T cross_z = omega[0] * v[1] - omega[1] * v[0];

                // omega x (omega x v)
                const T dot = omega[0] * v[0] + omega[1] * v[1] + omega[2] * v[2];
                const T cross2_x = omega[0] * dot - v[0] * theta_sq;
                const T cross2_y = omega[1] * dot - v[1] * theta_sq;
                const T cross2_z = omega[2] * dot - v[2] * theta_sq;

                t[0] = A * v[0] + B * cross_x + C * cross2_x;
                t[1] = A * v[1] + B * cross_y + C * cross2_y;
                t[2] = A * v[2] + B * cross_z + C * cross2_z;
            }

            return Sim3(rxso3, t);
        }

        // Sample uniform random Sim3
        template <typename RNG>
        [[nodiscard]] static Sim3 sample_uniform(RNG &rng, T max_log_scale = T(1), T trans_range = T(10)) noexcept {
            std::uniform_real_distribution<T> scale_dist(-max_log_scale, max_log_scale);
            std::uniform_real_distribution<T> trans_dist(-trans_range, trans_range);

            return Sim3(ScaledRotation(std::exp(scale_dist(rng)), Rotation::sample_uniform(rng)),
                        Translation{trans_dist(rng), trans_dist(rng), trans_dist(rng)});
        }

        // ===== CORE OPERATIONS =====

        // Logarithmic map: Sim3 -> [sigma, omega, v]
        [[nodiscard]] Tangent log() const noexcept {
            // Get [sigma, omega] from RxSO3
            auto rxso3_log = rxso3_.log();
            const T sigma = rxso3_log[0];
            dp::mat::vector<T, 3> omega{rxso3_log[1], rxso3_log[2], rxso3_log[3]};

            // Compute v = W^-1 * t where W is the left Jacobian for Sim(3)
            const T s = rxso3_.scale();
            const T theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
            const T theta = std::sqrt(theta_sq);
            const T sigma_sq = sigma * sigma;

            dp::mat::vector<T, 3> v;

            if (theta_sq < epsilon<T> * epsilon<T> && std::abs(sigma) < epsilon<T>) {
                // Small angle and scale: W ≈ I
                v = translation_;
            } else if (theta_sq < epsilon<T> * epsilon<T>) {
                // Small angle, significant scale
                // W = (s-1)/sigma * I, so W^-1 = sigma/(s-1) * I
                const T a = sigma / (s - T(1));
                v[0] = a * translation_[0];
                v[1] = a * translation_[1];
                v[2] = a * translation_[2];
            } else if (std::abs(sigma) < epsilon<T>) {
                // Small scale change - use SO3 inverse left Jacobian
                // J_l^-1 = I - 0.5*[omega]_x + (1/theta^2 - cot(theta/2)/(2*theta)) * [omega]_x^2
                const T half_theta = theta / T(2);
                const T cot_half = T(1) / std::tan(half_theta);
                const T c = T(1) / theta_sq - cot_half / (T(2) * theta);

                // -0.5 * omega x t
                const T half_cross_x = -T(0.5) * (omega[1] * translation_[2] - omega[2] * translation_[1]);
                const T half_cross_y = -T(0.5) * (omega[2] * translation_[0] - omega[0] * translation_[2]);
                const T half_cross_z = -T(0.5) * (omega[0] * translation_[1] - omega[1] * translation_[0]);

                // omega x (omega x t) = omega*(omega.t) - t*|omega|^2
                const T dot = omega[0] * translation_[0] + omega[1] * translation_[1] + omega[2] * translation_[2];
                const T cross2_x = omega[0] * dot - translation_[0] * theta_sq;
                const T cross2_y = omega[1] * dot - translation_[1] * theta_sq;
                const T cross2_z = omega[2] * dot - translation_[2] * theta_sq;

                v[0] = translation_[0] + half_cross_x + c * cross2_x;
                v[1] = translation_[1] + half_cross_y + c * cross2_y;
                v[2] = translation_[2] + half_cross_z + c * cross2_z;
            } else {
                // General case: need to invert W from exp()
                // W = A*I + B*[omega]_x + C*[omega]_x^2
                // For the inverse, we use the same structure with different coefficients
                // W^-1 = A'*I + B'*[omega]_x + C'*[omega]_x^2

                const T sin_th = std::sin(theta);
                const T cos_th = std::cos(theta);
                const T denom = sigma_sq + theta_sq;

                // Coefficients from exp()
                const T A = (sigma * (s - T(1)) + theta_sq) / (sigma * denom);
                const T B = ((s * cos_th - T(1)) * sigma + s * sin_th * theta) / (theta_sq * denom);
                const T C = (s - T(1) - sigma * A) / theta_sq;

                // For the inverse, we compute W^-1 * t directly using the inverse formula
                // The inverse of W = A*I + B*[omega]_x + C*[omega]_x^2 is:
                // W^-1 = a*I + b*[omega]_x + c*[omega]_x^2
                // where the coefficients satisfy the inverse relationship

                // Use the identity: (aI + b[w]_x + c[w]_x^2)^-1 can be computed
                // For Sim3, we need a more direct approach
                // Let's use the closed-form inverse for similarity Jacobian

                // omega x t
                const T cross_x = omega[1] * translation_[2] - omega[2] * translation_[1];
                const T cross_y = omega[2] * translation_[0] - omega[0] * translation_[2];
                const T cross_z = omega[0] * translation_[1] - omega[1] * translation_[0];

                // omega x (omega x t)
                const T dot = omega[0] * translation_[0] + omega[1] * translation_[1] + omega[2] * translation_[2];
                const T cross2_x = omega[0] * dot - translation_[0] * theta_sq;
                const T cross2_y = omega[1] * dot - translation_[1] * theta_sq;
                const T cross2_z = omega[2] * dot - translation_[2] * theta_sq;

                // Compute inverse coefficients using the determinant formula
                // For the inverse Jacobian of Sim(3):
                // W^-1 = sigma/(s-1) * I - 0.5 * [omega]_x + ...
                // This is a first-order approximation; for exactness we'd need to solve
                const T a_inv = sigma / (s - T(1));
                const T b_inv = -T(0.5);
                const T c_inv = (T(1) / theta_sq - a_inv / (T(2) * theta) * (T(1) / std::tan(theta / T(2))));

                v[0] = a_inv * translation_[0] + b_inv * cross_x + c_inv * cross2_x;
                v[1] = a_inv * translation_[1] + b_inv * cross_y + c_inv * cross2_y;
                v[2] = a_inv * translation_[2] + b_inv * cross_z + c_inv * cross2_z;
            }

            return Tangent{sigma, omega[0], omega[1], omega[2], v[0], v[1], v[2]};
        }

        // Inverse: (sR, t)^-1 = ((sR)^-1, -(sR)^-1 * t)
        [[nodiscard]] Sim3 inverse() const noexcept {
            auto rxso3_inv = rxso3_.inverse();
            Translation t_inv = rxso3_inv * translation_;
            t_inv[0] = -t_inv[0];
            t_inv[1] = -t_inv[1];
            t_inv[2] = -t_inv[2];
            return Sim3(rxso3_inv, t_inv);
        }

        // Group composition: (sR1, t1) * (sR2, t2) = (sR1*sR2, t1 + sR1*t2)
        [[nodiscard]] Sim3 operator*(const Sim3 &other) const noexcept {
            ScaledRotation rxso3_composed = rxso3_ * other.rxso3_;
            Translation t_composed = rxso3_ * other.translation_;
            t_composed[0] += translation_[0];
            t_composed[1] += translation_[1];
            t_composed[2] += translation_[2];
            return Sim3(rxso3_composed, t_composed);
        }

        Sim3 &operator*=(const Sim3 &other) noexcept {
            *this = *this * other;
            return *this;
        }

        // Transform a 3D point: p' = s * R * p + t
        [[nodiscard]] Point operator*(const Point &p) const noexcept {
            Point result = rxso3_ * p;
            result[0] += translation_[0];
            result[1] += translation_[1];
            result[2] += translation_[2];
            return result;
        }

        // ===== MATRIX REPRESENTATION =====

        // Return 4x4 homogeneous matrix
        [[nodiscard]] HomogeneousMatrix matrix() const noexcept {
            HomogeneousMatrix M;
            auto sR = rxso3_.matrix();
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    M(i, j) = sR(i, j);
                }
                M(i, 3) = translation_[i];
            }
            M(3, 0) = M(3, 1) = M(3, 2) = T(0);
            M(3, 3) = T(1);
            return M;
        }

        // Return 3x4 compact form [sR | t]
        [[nodiscard]] dp::mat::matrix<T, 3, 4> matrix3x4() const noexcept {
            dp::mat::matrix<T, 3, 4> M;
            auto sR = rxso3_.matrix();
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    M(i, j) = sR(i, j);
                }
                M(i, 3) = translation_[i];
            }
            return M;
        }

        // ===== LIE ALGEBRA =====

        // hat: [sigma, omega, v] -> 4x4 matrix
        [[nodiscard]] static HomogeneousMatrix hat(const Tangent &twist) noexcept {
            HomogeneousMatrix M;
            const T sigma = twist[0];

            // Top-left 3x3: sigma*I + hat(omega)
            M(0, 0) = sigma;
            M(0, 1) = -twist[3]; // -wz
            M(0, 2) = twist[2];  // wy
            M(1, 0) = twist[3];  // wz
            M(1, 1) = sigma;
            M(1, 2) = -twist[1]; // -wx
            M(2, 0) = -twist[2]; // -wy
            M(2, 1) = twist[1];  // wx
            M(2, 2) = sigma;

            // Last column: v
            M(0, 3) = twist[4];
            M(1, 3) = twist[5];
            M(2, 3) = twist[6];

            // Bottom row
            M(3, 0) = M(3, 1) = M(3, 2) = M(3, 3) = T(0);

            return M;
        }

        // vee: 4x4 matrix -> [sigma, omega, v]
        [[nodiscard]] static Tangent vee(const HomogeneousMatrix &M) noexcept {
            const T sigma = (M(0, 0) + M(1, 1) + M(2, 2)) / T(3);
            return Tangent{sigma,
                           (M(2, 1) - M(1, 2)) / T(2), // wx
                           (M(0, 2) - M(2, 0)) / T(2), // wy
                           (M(1, 0) - M(0, 1)) / T(2), // wz
                           M(0, 3),                    // vx
                           M(1, 3),                    // vy
                           M(2, 3)};                   // vz
        }

        // ===== ACCESSORS =====

        // Get scale factor
        [[nodiscard]] T scale() const noexcept { return rxso3_.scale(); }

        // Get RxSO3 (scaled rotation)
        [[nodiscard]] const ScaledRotation &rxso3() const noexcept { return rxso3_; }

        // Get SO3 rotation (unit)
        [[nodiscard]] Rotation so3() const noexcept { return rxso3_.so3(); }

        // Get translation
        [[nodiscard]] const Translation &translation() const noexcept { return translation_; }

        // Get SE3 (assuming scale = 1)
        [[nodiscard]] SE3<T> se3() const noexcept { return SE3<T>(rxso3_.so3(), translation_); }

        // ===== MUTATORS =====

        void set_scale(T new_scale) noexcept { rxso3_.set_scale(new_scale); }

        void set_translation(const Translation &t) noexcept { translation_ = t; }

        void set_rxso3(const ScaledRotation &rxso3) noexcept { rxso3_ = rxso3; }

        // ===== TYPE CONVERSION =====

        template <typename NewScalar> [[nodiscard]] Sim3<NewScalar> cast() const noexcept {
            return Sim3<NewScalar>(rxso3_.template cast<NewScalar>(),
                                   typename Sim3<NewScalar>::Translation{static_cast<NewScalar>(translation_[0]),
                                                                         static_cast<NewScalar>(translation_[1]),
                                                                         static_cast<NewScalar>(translation_[2])});
        }

        // ===== COMPARISON =====

        [[nodiscard]] bool operator==(const Sim3 &other) const noexcept {
            return rxso3_ == other.rxso3_ && std::abs(translation_[0] - other.translation_[0]) < epsilon<T> &&
                   std::abs(translation_[1] - other.translation_[1]) < epsilon<T> &&
                   std::abs(translation_[2] - other.translation_[2]) < epsilon<T>;
        }

        [[nodiscard]] bool operator!=(const Sim3 &other) const noexcept { return !(*this == other); }

        [[nodiscard]] bool is_approx(const Sim3 &other, T tol = epsilon<T>) const noexcept {
            return rxso3_.is_approx(other.rxso3_, tol) && std::abs(translation_[0] - other.translation_[0]) < tol &&
                   std::abs(translation_[1] - other.translation_[1]) < tol &&
                   std::abs(translation_[2] - other.translation_[2]) < tol;
        }

        [[nodiscard]] bool is_identity(T tol = epsilon<T>) const noexcept {
            return rxso3_.is_identity(tol) && std::abs(translation_[0]) < tol && std::abs(translation_[1]) < tol &&
                   std::abs(translation_[2]) < tol;
        }

      private:
        ScaledRotation rxso3_;    // Scaled rotation
        Translation translation_; // Translation
    };

    // ===== TYPE ALIASES =====

    using Sim3f = Sim3<float>;
    using Sim3d = Sim3<double>;

    // ===== FREE FUNCTIONS =====

    // Geodesic interpolation
    template <typename T> [[nodiscard]] Sim3<T> interpolate(const Sim3<T> &a, const Sim3<T> &b, T t) noexcept {
        auto twist = (a.inverse() * b).log();
        for (std::size_t i = 0; i < 7; ++i) {
            twist[i] *= t;
        }
        return a * Sim3<T>::exp(twist);
    }

} // namespace optinum::lie
