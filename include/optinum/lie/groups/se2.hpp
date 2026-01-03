#pragma once

#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/so2.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <datapod/matrix/matrix.hpp>
#include <datapod/matrix/vector.hpp>

#include <cmath>
#include <random>
#include <type_traits>

namespace optinum::lie {

    namespace dp = ::datapod;

    // ===== SE2: Special Euclidean Group in 2D =====
    //
    // SE(2) represents 2D rigid transforms (rotation + translation).
    // Internally stored as SO2 (rotation) + Vector2 (translation).
    //
    // Storage: SO2 + Vector<T, 2> = 4 parameters total
    // DoF: 3 (2 translation + 1 rotation)
    // NumParams: 4 (2 for unit complex + 2 for translation)
    //
    // Tangent space (twist): [vx, vy, theta]^T in R^3
    //   - (vx, vy) = translational velocity
    //   - theta = angular velocity
    //
    // Lie algebra se(2):
    //   hat([vx, vy, theta]) = [[0, -theta, vx], [theta, 0, vy], [0, 0, 0]]

    template <typename T = double> class SE2 {
        static_assert(std::is_floating_point_v<T>, "SE2 requires floating-point scalar type");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Tangent = dp::mat::Vector<T, 3>;              // [vx, vy, theta] - owning
        using Translation = dp::mat::Vector<T, 2>;          // owning
        using Point = dp::mat::Vector<T, 2>;                // owning
        using Params = dp::mat::Vector<T, 4>;               // [cos, sin, tx, ty] - owning
        using HomogeneousMatrix = dp::mat::Matrix<T, 3, 3>; // owning
        using TransformMatrix = dp::mat::Matrix<T, 2, 3>;   // Compact form - owning
        using AdjointMatrix = dp::mat::Matrix<T, 3, 3>;     // owning
        using Rotation = SO2<T>;

        // ===== CONSTANTS =====
        static constexpr std::size_t DoF = 3;
        static constexpr std::size_t NumParams = 4;

        // ===== CONSTRUCTORS =====

        // Default: identity transform
        constexpr SE2() noexcept : so2_(), translation_{{T(0), T(0)}} {}

        // From SO2 rotation and translation
        SE2(const Rotation &rotation, const Translation &translation) noexcept
            : so2_(rotation), translation_(translation) {}

        // From angle and translation
        SE2(Scalar theta, const Translation &translation) noexcept : so2_(theta), translation_(translation) {}

        // From angle and translation components
        SE2(Scalar theta, Scalar tx, Scalar ty) noexcept : so2_(theta), translation_{{tx, ty}} {}

        // From 3x3 homogeneous matrix
        explicit SE2(const HomogeneousMatrix &T_mat) noexcept {
            // Extract rotation from top-left 2x2
            dp::mat::Matrix<T, 2, 2> R_mat;
            R_mat(0, 0) = T_mat(0, 0);
            R_mat(0, 1) = T_mat(0, 1);
            R_mat(1, 0) = T_mat(1, 0);
            R_mat(1, 1) = T_mat(1, 1);
            so2_ = Rotation(R_mat);

            // Extract translation from last column
            translation_[0] = T_mat(0, 2);
            translation_[1] = T_mat(1, 2);
        }

        // ===== STATIC FACTORY METHODS =====

        // Identity element
        [[nodiscard]] static constexpr SE2 identity() noexcept { return SE2(); }

        // Pure rotation (no translation)
        [[nodiscard]] static SE2 rot(Scalar theta) noexcept { return SE2(theta, Translation{{T(0), T(0)}}); }

        // Pure translation (no rotation)
        [[nodiscard]] static SE2 trans(Scalar tx, Scalar ty) noexcept { return SE2(T(0), Translation{{tx, ty}}); }

        [[nodiscard]] static SE2 trans(const Translation &t) noexcept { return SE2(T(0), t); }

        // Translation along X axis
        [[nodiscard]] static SE2 trans_x(Scalar tx) noexcept { return SE2(T(0), Translation{{tx, T(0)}}); }

        // Translation along Y axis
        [[nodiscard]] static SE2 trans_y(Scalar ty) noexcept { return SE2(T(0), Translation{{T(0), ty}}); }

        // Exponential map: twist -> SE2
        // exp([vx, vy, theta]) with proper handling of small angles
        [[nodiscard]] static SE2 exp(const Tangent &twist) noexcept {
            const T vx = twist[0];
            const T vy = twist[1];
            const T theta = twist[2];

            Rotation R = Rotation::exp(theta);

            Translation t;
            if (std::abs(theta) < epsilon<T>) {
                // Small angle: use Taylor expansion
                // V ≈ I + theta/2 * [[0, -1], [1, 0]] + O(theta^2)
                // t ≈ [vx - theta*vy/2, vy + theta*vx/2]
                const T half_theta = theta / T(2);
                t[0] = vx - half_theta * vy;
                t[1] = vy + half_theta * vx;
            } else {
                // Full formula: t = V * [vx, vy]
                // V = [[sin(theta)/theta, (1-cos(theta))/theta],
                //      [(cos(theta)-1)/theta, sin(theta)/theta]]
                const T s = std::sin(theta);
                const T c = std::cos(theta);
                const T one_minus_c = T(1) - c;

                // V * v = (sin/theta) * v + ((1-cos)/theta) * v_perp
                // where v_perp = [-vy, vx]
                const T sin_over_theta = s / theta;
                const T one_minus_cos_over_theta = one_minus_c / theta;

                t[0] = sin_over_theta * vx - one_minus_cos_over_theta * vy;
                t[1] = one_minus_cos_over_theta * vx + sin_over_theta * vy;
            }

            return SE2(R, t);
        }

        // Sample uniform random pose
        template <typename RNG>
        [[nodiscard]] static SE2 sample_uniform(RNG &rng, T translation_range = T(10)) noexcept {
            std::uniform_real_distribution<T> angle_dist(T(0), two_pi<T>);
            std::uniform_real_distribution<T> trans_dist(-translation_range, translation_range);

            return SE2(angle_dist(rng), Translation{{trans_dist(rng), trans_dist(rng)}});
        }

        // Fit closest SE2 to arbitrary 3x3 matrix
        [[nodiscard]] static SE2 fit_to_SE2(const HomogeneousMatrix &M) noexcept {
            // Extract and normalize rotation
            dp::mat::Matrix<T, 2, 2> R_mat;
            R_mat(0, 0) = M(0, 0);
            R_mat(0, 1) = M(0, 1);
            R_mat(1, 0) = M(1, 0);
            R_mat(1, 1) = M(1, 1);

            return SE2(Rotation::fit_to_SO2(R_mat), Translation{{M(0, 2), M(1, 2)}});
        }

        // ===== CORE OPERATIONS =====

        // Logarithmic map: SE2 -> twist
        [[nodiscard]] Tangent log() const noexcept {
            const T theta = so2_.log();

            Tangent twist;
            twist[2] = theta;

            if (std::abs(theta) < epsilon<T>) {
                // Small angle: V^-1 ≈ I - theta/2 * [[0, -1], [1, 0]]
                const T half_theta = theta / T(2);
                twist[0] = translation_[0] + half_theta * translation_[1];
                twist[1] = translation_[1] - half_theta * translation_[0];
            } else {
                // Full formula: v = V^-1 * t
                // V^-1 = [[sin(theta)/theta, -(1-cos(theta))/theta],
                //         [(1-cos(theta))/theta, sin(theta)/theta]] * (theta / (2*sin(theta/2))^2)
                // Simplified: V^-1 = (theta/2) * cot(theta/2) * I + (theta/2) * [[0, 1], [-1, 0]]
                const T half_theta = theta / T(2);
                const T half_theta_cot = half_theta / std::tan(half_theta);

                twist[0] = half_theta_cot * translation_[0] + half_theta * translation_[1];
                twist[1] = half_theta_cot * translation_[1] - half_theta * translation_[0];
            }

            return twist;
        }

        // Inverse: T^-1 = (R^-1, -R^-1 * t)
        [[nodiscard]] SE2 inverse() const noexcept {
            Rotation R_inv = so2_.inverse();
            Translation t_inv = R_inv * translation_;
            t_inv[0] = -t_inv[0];
            t_inv[1] = -t_inv[1];
            return SE2(R_inv, t_inv);
        }

        // Group composition: T1 * T2 = (R1*R2, t1 + R1*t2)
        [[nodiscard]] SE2 operator*(const SE2 &other) const noexcept {
            Rotation R_new = so2_ * other.so2_;
            Point t_rotated = so2_ * other.translation_;
            Translation t_new;
            t_new[0] = translation_[0] + t_rotated[0];
            t_new[1] = translation_[1] + t_rotated[1];
            return SE2(R_new, t_new);
        }

        SE2 &operator*=(const SE2 &other) noexcept {
            *this = *this * other;
            return *this;
        }

        // Transform a 2D point: R*p + t
        [[nodiscard]] Point operator*(const Point &p) const noexcept {
            Point result = so2_ * p;
            result[0] += translation_[0];
            result[1] += translation_[1];
            return result;
        }

        // ===== MATRIX REPRESENTATIONS =====

        // Return 3x3 homogeneous matrix
        [[nodiscard]] HomogeneousMatrix matrix() const noexcept {
            HomogeneousMatrix T_mat;
            auto R = so2_.matrix();

            T_mat(0, 0) = R(0, 0);
            T_mat(0, 1) = R(0, 1);
            T_mat(0, 2) = translation_[0];
            T_mat(1, 0) = R(1, 0);
            T_mat(1, 1) = R(1, 1);
            T_mat(1, 2) = translation_[1];
            T_mat(2, 0) = T(0);
            T_mat(2, 1) = T(0);
            T_mat(2, 2) = T(1);

            return T_mat;
        }

        // Return 2x3 compact matrix [R | t]
        [[nodiscard]] TransformMatrix matrix2x3() const noexcept {
            TransformMatrix T_mat;
            auto R = so2_.matrix();

            T_mat(0, 0) = R(0, 0);
            T_mat(0, 1) = R(0, 1);
            T_mat(0, 2) = translation_[0];
            T_mat(1, 0) = R(1, 0);
            T_mat(1, 1) = R(1, 1);
            T_mat(1, 2) = translation_[1];

            return T_mat;
        }

        // Return 2x2 rotation matrix
        [[nodiscard]] dp::mat::Matrix<T, 2, 2> rotation_matrix() const noexcept { return so2_.matrix(); }

        // ===== LIE ALGEBRA =====

        // hat: twist -> 3x3 se(2) matrix
        // hat([vx, vy, theta]) = [[0, -theta, vx], [theta, 0, vy], [0, 0, 0]]
        [[nodiscard]] static HomogeneousMatrix hat(const Tangent &twist) noexcept {
            HomogeneousMatrix Omega;
            Omega(0, 0) = T(0);
            Omega(0, 1) = -twist[2];
            Omega(0, 2) = twist[0];
            Omega(1, 0) = twist[2];
            Omega(1, 1) = T(0);
            Omega(1, 2) = twist[1];
            Omega(2, 0) = T(0);
            Omega(2, 1) = T(0);
            Omega(2, 2) = T(0);
            return Omega;
        }

        // vee: 3x3 se(2) matrix -> twist
        [[nodiscard]] static Tangent vee(const HomogeneousMatrix &Omega) noexcept {
            Tangent twist;
            twist[0] = Omega(0, 2);                        // vx
            twist[1] = Omega(1, 2);                        // vy
            twist[2] = (Omega(1, 0) - Omega(0, 1)) / T(2); // theta (average for stability)
            return twist;
        }

        // Adjoint representation: 3x3 matrix
        // Adj(T) = [[R, [t]_x], [0, 1]] where [t]_x = [-ty, tx]
        // For SE2: Adj = [[R, t_perp], [0, 0, 1]]
        [[nodiscard]] AdjointMatrix Adj() const noexcept {
            AdjointMatrix adj;
            const T c = so2_.real();
            const T s = so2_.imag();

            // Top-left 2x2: rotation matrix
            adj(0, 0) = c;
            adj(0, 1) = -s;
            adj(1, 0) = s;
            adj(1, 1) = c;

            // Right column: t_perp = [-ty, tx]
            adj(0, 2) = -translation_[1];
            adj(1, 2) = translation_[0];

            // Bottom row: [0, 0, 1]
            adj(2, 0) = T(0);
            adj(2, 1) = T(0);
            adj(2, 2) = T(1);

            return adj;
        }

        // Lie bracket [a, b] for se(2)
        [[nodiscard]] static Tangent lie_bracket(const Tangent &a, const Tangent &b) noexcept {
            // [a, b] = hat(a) * hat(b) - hat(b) * hat(a)
            // For se(2): [a, b] = [a_theta * b_y - b_theta * a_y,
            //                     -a_theta * b_x + b_theta * a_x,
            //                     0]
            Tangent result;
            result[0] = a[2] * b[1] - b[2] * a[1];
            result[1] = -a[2] * b[0] + b[2] * a[0];
            result[2] = T(0);
            return result;
        }

        // Generator matrices (3 generators for SE2)
        [[nodiscard]] static HomogeneousMatrix generator(std::size_t i) noexcept {
            Tangent e;
            e[0] = (i == 0) ? T(1) : T(0);
            e[1] = (i == 1) ? T(1) : T(0);
            e[2] = (i == 2) ? T(1) : T(0);
            return hat(e);
        }

        // ===== JACOBIANS =====

        // Left Jacobian J_l(twist)
        // For SE2, this is a 3x3 matrix
        [[nodiscard]] static AdjointMatrix left_jacobian(const Tangent &twist) noexcept {
            const T theta = twist[2];
            AdjointMatrix J;

            if (std::abs(theta) < epsilon<T>) {
                // Small angle approximation: J ≈ I
                J(0, 0) = T(1);
                J(0, 1) = T(0);
                J(0, 2) = -twist[1] / T(2);
                J(1, 0) = T(0);
                J(1, 1) = T(1);
                J(1, 2) = twist[0] / T(2);
                J(2, 0) = T(0);
                J(2, 1) = T(0);
                J(2, 2) = T(1);
            } else {
                const T s = std::sin(theta);
                const T c = std::cos(theta);
                const T theta_sq = theta * theta;

                // V matrix (2x2 part)
                const T a = s / theta;
                const T b = (T(1) - c) / theta;

                // Top-left 2x2: V matrix
                J(0, 0) = a;
                J(0, 1) = -b;
                J(1, 0) = b;
                J(1, 1) = a;

                // Right column: more complex expression
                const T vx = twist[0];
                const T vy = twist[1];
                J(0, 2) = (vx * (theta - s) + vy * (c - T(1))) / theta_sq;
                J(1, 2) = (vx * (T(1) - c) + vy * (theta - s)) / theta_sq;

                // Bottom row
                J(2, 0) = T(0);
                J(2, 1) = T(0);
                J(2, 2) = T(1);
            }

            return J;
        }

        // Inverse of left Jacobian
        [[nodiscard]] static AdjointMatrix left_jacobian_inverse(const Tangent &twist) noexcept {
            const T theta = twist[2];
            AdjointMatrix J_inv;

            if (std::abs(theta) < epsilon<T>) {
                // Small angle approximation
                J_inv(0, 0) = T(1);
                J_inv(0, 1) = T(0);
                J_inv(0, 2) = twist[1] / T(2);
                J_inv(1, 0) = T(0);
                J_inv(1, 1) = T(1);
                J_inv(1, 2) = -twist[0] / T(2);
                J_inv(2, 0) = T(0);
                J_inv(2, 1) = T(0);
                J_inv(2, 2) = T(1);
            } else {
                const T half_theta = theta / T(2);
                const T cot_half = T(1) / std::tan(half_theta);

                // V^-1 2x2 part
                const T a = half_theta * cot_half;
                const T b = half_theta;

                J_inv(0, 0) = a;
                J_inv(0, 1) = b;
                J_inv(1, 0) = -b;
                J_inv(1, 1) = a;

                // For the right column, we need -V^-1 * w where w is the right column of J
                // First compute w (right column of J)
                const T s = std::sin(theta);
                const T c = std::cos(theta);
                const T theta_sq = theta * theta;
                const T vx = twist[0];
                const T vy = twist[1];

                const T w0 = (vx * (theta - s) + vy * (c - T(1))) / theta_sq;
                const T w1 = (vx * (T(1) - c) + vy * (theta - s)) / theta_sq;

                // J_inv right column = -V^-1 * w
                J_inv(0, 2) = -(a * w0 + b * w1);
                J_inv(1, 2) = -(-b * w0 + a * w1);

                // Bottom row
                J_inv(2, 0) = T(0);
                J_inv(2, 1) = T(0);
                J_inv(2, 2) = T(1);
            }

            return J_inv;
        }

        // ===== ACCESSORS =====

        // Get rotation component
        [[nodiscard]] constexpr const Rotation &so2() const noexcept { return so2_; }
        [[nodiscard]] constexpr Rotation &so2() noexcept { return so2_; }

        // Get translation component
        [[nodiscard]] constexpr const Translation &translation() const noexcept { return translation_; }
        [[nodiscard]] constexpr Translation &translation() noexcept { return translation_; }

        // Get angle
        [[nodiscard]] Scalar angle() const noexcept { return so2_.angle(); }

        // Get x translation
        [[nodiscard]] constexpr Scalar x() const noexcept { return translation_[0]; }

        // Get y translation
        [[nodiscard]] constexpr Scalar y() const noexcept { return translation_[1]; }

        // Get parameters as vector [cos, sin, tx, ty]
        [[nodiscard]] Params params() const noexcept {
            Params p;
            p[0] = so2_.real();
            p[1] = so2_.imag();
            p[2] = translation_[0];
            p[3] = translation_[1];
            return p;
        }

        // Raw data pointer (to rotation, then translation)
        [[nodiscard]] T *data() noexcept { return so2_.data(); }
        [[nodiscard]] const T *data() const noexcept { return so2_.data(); }

        // ===== MUTATORS =====

        // Set rotation from angle
        void set_angle(Scalar theta) noexcept { so2_.set_angle(theta); }

        // Set translation
        void set_translation(const Translation &t) noexcept { translation_ = t; }
        void set_translation(Scalar tx, Scalar ty) noexcept {
            translation_[0] = tx;
            translation_[1] = ty;
        }

        // Normalize rotation
        void normalize() noexcept { so2_.normalize(); }

        // ===== TYPE CONVERSION =====

        // Cast to different scalar type
        template <typename NewScalar> [[nodiscard]] SE2<NewScalar> cast() const noexcept {
            return SE2<NewScalar>(so2_.template cast<NewScalar>(),
                                  dp::mat::Vector<NewScalar, 2>{{static_cast<NewScalar>(translation_[0]),
                                                                 static_cast<NewScalar>(translation_[1])}});
        }

        // ===== COMPARISON =====

        [[nodiscard]] bool operator==(const SE2 &other) const noexcept {
            return so2_ == other.so2_ && std::abs(translation_[0] - other.translation_[0]) < epsilon<T> &&
                   std::abs(translation_[1] - other.translation_[1]) < epsilon<T>;
        }

        [[nodiscard]] bool operator!=(const SE2 &other) const noexcept { return !(*this == other); }

        // Check if close to another SE2
        [[nodiscard]] bool is_approx(const SE2 &other, Scalar tol = epsilon<T>) const noexcept {
            return so2_.is_approx(other.so2_, tol) && std::abs(translation_[0] - other.translation_[0]) < tol &&
                   std::abs(translation_[1] - other.translation_[1]) < tol;
        }

        // Check if approximately identity
        [[nodiscard]] bool is_identity(Scalar tol = epsilon<T>) const noexcept {
            return so2_.is_identity(tol) && std::abs(translation_[0]) < tol && std::abs(translation_[1]) < tol;
        }

      private:
        Rotation so2_;            // Rotation component
        Translation translation_; // Translation component
    };

    // ===== TYPE ALIASES =====

    using SE2f = SE2<float>;
    using SE2d = SE2<double>;

    // ===== FREE FUNCTIONS =====

    // Geodesic interpolation on SE2
    template <typename T> [[nodiscard]] SE2<T> interpolate(const SE2<T> &a, const SE2<T> &b, T t) noexcept {
        // twist = log(a^-1 * b)
        auto twist = (a.inverse() * b).log();
        // Scale twist by t
        twist[0] *= t;
        twist[1] *= t;
        twist[2] *= t;
        // result = a * exp(t * twist)
        return a * SE2<T>::exp(twist);
    }

} // namespace optinum::lie
