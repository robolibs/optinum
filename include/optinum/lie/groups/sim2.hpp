#pragma once

#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/rxso2.hpp>
#include <optinum/lie/groups/se2.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>
#include <random>
#include <type_traits>

namespace optinum::lie {

    // ===== Sim2: 2D Similarity Group =====
    //
    // Sim(2) = R+ x SO(2) x R^2 represents 2D similarity transforms:
    //   scale + rotation + translation
    //
    // Transformation: p' = s * R * p + t
    //
    // Storage: RxSO2 + Vector<T, 2> (scaled rotation + translation)
    // DoF: 4 (1 scale + 1 rotation + 2 translation)
    // NumParams: 4 (2 complex + 2 translation)
    //
    // Tangent space (twist):
    //   [sigma, theta, vx, vy] where:
    //     sigma = log(scale)
    //     theta = rotation angle
    //     (vx, vy) = translational velocity
    //
    // Useful for: monocular SLAM, scale-invariant registration

    template <typename T = double> class Sim2 {
        static_assert(std::is_floating_point_v<T>, "Sim2 requires floating-point scalar type");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Tangent = simd::Vector<T, 4>; // [sigma, theta, vx, vy]
        using Translation = simd::Vector<T, 2>;
        using Point = simd::Vector<T, 2>;
        using Params = simd::Vector<T, 4>; // [s*cos, s*sin, tx, ty]
        using HomogeneousMatrix = simd::Matrix<T, 3, 3>;
        using AdjointMatrix = simd::Matrix<T, 4, 4>;
        using ScaledRotation = RxSO2<T>;
        using Rotation = SO2<T>;

        // ===== CONSTANTS =====
        static constexpr std::size_t DoF = 4;
        static constexpr std::size_t NumParams = 4;

        // ===== CONSTRUCTORS =====

        // Default: identity transform
        constexpr Sim2() noexcept : rxso2_(), translation_{T(0), T(0)} {}

        // From RxSO2 and translation
        Sim2(const ScaledRotation &rxso2, const Translation &t) noexcept : rxso2_(rxso2), translation_(t) {}

        // From scale, angle, and translation
        Sim2(Scalar scale, Scalar theta, const Translation &t) noexcept : rxso2_(scale, theta), translation_(t) {}

        // From scale, SO2, and translation
        Sim2(Scalar scale, const Rotation &R, const Translation &t) noexcept : rxso2_(scale, R), translation_(t) {}

        // From SE2 (with scale = 1)
        explicit Sim2(const SE2<T> &se2) noexcept : rxso2_(se2.so2()), translation_(se2.translation()) {}

        // From 3x3 homogeneous matrix
        explicit Sim2(const HomogeneousMatrix &T_mat) noexcept {
            // Extract scaled rotation from top-left 2x2
            const T s = std::sqrt(T_mat(0, 0) * T_mat(0, 0) + T_mat(1, 0) * T_mat(1, 0));
            rxso2_ = ScaledRotation(T_mat(0, 0), T_mat(1, 0), true);
            translation_[0] = T_mat(0, 2);
            translation_[1] = T_mat(1, 2);
        }

        // ===== STATIC FACTORY METHODS =====

        // Identity element
        [[nodiscard]] static constexpr Sim2 identity() noexcept { return Sim2(); }

        // Pure scale (no rotation, no translation)
        [[nodiscard]] static Sim2 scale(Scalar s) noexcept {
            return Sim2(ScaledRotation(s, T(0)), Translation{T(0), T(0)});
        }

        // Pure rotation (scale = 1, no translation)
        [[nodiscard]] static Sim2 rot(Scalar theta) noexcept {
            return Sim2(ScaledRotation(T(1), theta), Translation{T(0), T(0)});
        }

        // Pure translation (scale = 1, no rotation)
        [[nodiscard]] static Sim2 trans(Scalar tx, Scalar ty) noexcept {
            return Sim2(ScaledRotation(), Translation{tx, ty});
        }

        [[nodiscard]] static Sim2 trans(const Translation &t) noexcept { return Sim2(ScaledRotation(), t); }

        [[nodiscard]] static Sim2 trans_x(Scalar tx) noexcept { return trans(tx, T(0)); }
        [[nodiscard]] static Sim2 trans_y(Scalar ty) noexcept { return trans(T(0), ty); }

        // Exponential map: [sigma, theta, vx, vy] -> Sim2
        [[nodiscard]] static Sim2 exp(const Tangent &twist) noexcept {
            const T sigma = twist[0];
            const T theta = twist[1];
            const T vx = twist[2];
            const T vy = twist[3];

            // Scaled rotation: exp([sigma, theta])
            ScaledRotation rxso2 = ScaledRotation::exp(typename ScaledRotation::Tangent{sigma, theta});

            // Compute translation using the left Jacobian
            // For Sim2, the Jacobian is more complex due to scale
            Translation t;

            const T s = std::exp(sigma);
            const T theta_sq = theta * theta;
            const T sigma_sq = sigma * sigma;

            if (std::abs(theta) < epsilon<T> && std::abs(sigma) < epsilon<T>) {
                // Small angle and small scale: V ≈ I
                t[0] = vx;
                t[1] = vy;
            } else if (std::abs(theta) < epsilon<T>) {
                // Small angle, significant scale
                const T a = (s - T(1)) / sigma;
                t[0] = a * vx;
                t[1] = a * vy;
            } else if (std::abs(sigma) < epsilon<T>) {
                // Significant angle, small scale
                const T sin_th = std::sin(theta);
                const T cos_th = std::cos(theta);
                const T a = sin_th / theta;
                const T b = (T(1) - cos_th) / theta;
                t[0] = a * vx - b * vy;
                t[1] = b * vx + a * vy;
            } else {
                // General case
                const T sin_th = std::sin(theta);
                const T cos_th = std::cos(theta);
                const T denom = sigma_sq + theta_sq;

                // V = (s-1)/denom * (sigma*I + theta*J) + 1/denom * ((s*cos-1)*I + s*sin*J)
                // where J = [[0, -1], [1, 0]]
                const T a = (sigma * (s * cos_th - T(1)) + theta * s * sin_th) / denom;
                const T b = (sigma * s * sin_th - theta * (s * cos_th - T(1))) / denom;

                t[0] = a * vx - b * vy;
                t[1] = b * vx + a * vy;
            }

            return Sim2(rxso2, t);
        }

        // Sample uniform random Sim2
        template <typename RNG>
        [[nodiscard]] static Sim2 sample_uniform(RNG &rng, T max_log_scale = T(1), T trans_range = T(10)) noexcept {
            std::uniform_real_distribution<T> scale_dist(-max_log_scale, max_log_scale);
            std::uniform_real_distribution<T> angle_dist(T(0), two_pi<T>);
            std::uniform_real_distribution<T> trans_dist(-trans_range, trans_range);

            return Sim2(ScaledRotation::exp(typename ScaledRotation::Tangent{scale_dist(rng), angle_dist(rng)}),
                        Translation{trans_dist(rng), trans_dist(rng)});
        }

        // ===== CORE OPERATIONS =====

        // Logarithmic map: Sim2 -> [sigma, theta, vx, vy]
        [[nodiscard]] Tangent log() const noexcept {
            // Get [sigma, theta] from RxSO2
            auto rxso2_log = rxso2_.log();
            const T sigma = rxso2_log[0];
            const T theta = rxso2_log[1];

            // Compute v = V^-1 * t where V is the left Jacobian for Sim(2)
            // We need to invert the V matrix from exp()
            simd::Vector<T, 2> v;
            const T s = rxso2_.scale();
            const T theta_sq = theta * theta;
            const T sigma_sq = sigma * sigma;

            if (std::abs(theta) < epsilon<T> && std::abs(sigma) < epsilon<T>) {
                // Small angle and scale: V ≈ I, so V^-1 ≈ I
                v[0] = translation_[0];
                v[1] = translation_[1];
            } else if (std::abs(theta) < epsilon<T>) {
                // Small angle, significant scale
                // V = (s-1)/sigma * I, so V^-1 = sigma/(s-1) * I
                const T a = sigma / (s - T(1));
                v[0] = a * translation_[0];
                v[1] = a * translation_[1];
            } else if (std::abs(sigma) < epsilon<T>) {
                // Significant angle, small scale change
                // V = [[sin/theta, -(1-cos)/theta], [(1-cos)/theta, sin/theta]]
                // V^-1 = [[a, b], [-b, a]] where a = theta*sin/(2*(1-cos)), b = theta/2
                // Using half-angle: a = cot(theta/2) * theta/2
                const T half_theta = theta / T(2);
                const T cot_half = T(1) / std::tan(half_theta);
                const T a = half_theta * cot_half;
                const T b = half_theta;
                v[0] = a * translation_[0] + b * translation_[1];
                v[1] = -b * translation_[0] + a * translation_[1];
            } else {
                // General case: compute V^-1 directly
                // V = [[a, -b], [b, a]] from exp()
                // V^-1 = 1/(a^2 + b^2) * [[a, b], [-b, a]]
                const T sin_th = std::sin(theta);
                const T cos_th = std::cos(theta);
                const T denom = sigma_sq + theta_sq;

                // From exp: a, b are the components of V
                const T a = (sigma * (s * cos_th - T(1)) + theta * s * sin_th) / denom;
                const T b = (sigma * s * sin_th - theta * (s * cos_th - T(1))) / denom;

                // Invert the 2x2 matrix [[a, -b], [b, a]]
                const T det = a * a + b * b;
                const T inv_det = T(1) / det;

                v[0] = inv_det * (a * translation_[0] + b * translation_[1]);
                v[1] = inv_det * (-b * translation_[0] + a * translation_[1]);
            }

            return Tangent{sigma, theta, v[0], v[1]};
        }

        // Inverse: (sR, t)^-1 = ((sR)^-1, -(sR)^-1 * t)
        [[nodiscard]] Sim2 inverse() const noexcept {
            auto rxso2_inv = rxso2_.inverse();
            Translation t_inv = rxso2_inv * translation_;
            t_inv[0] = -t_inv[0];
            t_inv[1] = -t_inv[1];
            return Sim2(rxso2_inv, t_inv);
        }

        // Group composition: (sR1, t1) * (sR2, t2) = (sR1*sR2, t1 + sR1*t2)
        [[nodiscard]] Sim2 operator*(const Sim2 &other) const noexcept {
            ScaledRotation rxso2_composed = rxso2_ * other.rxso2_;
            Translation t_composed = rxso2_ * other.translation_;
            t_composed[0] += translation_[0];
            t_composed[1] += translation_[1];
            return Sim2(rxso2_composed, t_composed);
        }

        Sim2 &operator*=(const Sim2 &other) noexcept {
            *this = *this * other;
            return *this;
        }

        // Transform a 2D point: p' = s * R * p + t
        [[nodiscard]] Point operator*(const Point &p) const noexcept {
            Point result = rxso2_ * p;
            result[0] += translation_[0];
            result[1] += translation_[1];
            return result;
        }

        // ===== MATRIX REPRESENTATION =====

        // Return 3x3 homogeneous matrix
        [[nodiscard]] HomogeneousMatrix matrix() const noexcept {
            HomogeneousMatrix M;
            auto sR = rxso2_.matrix();
            M(0, 0) = sR(0, 0);
            M(0, 1) = sR(0, 1);
            M(0, 2) = translation_[0];
            M(1, 0) = sR(1, 0);
            M(1, 1) = sR(1, 1);
            M(1, 2) = translation_[1];
            M(2, 0) = T(0);
            M(2, 1) = T(0);
            M(2, 2) = T(1);
            return M;
        }

        // Return 2x3 compact form [sR | t]
        [[nodiscard]] simd::Matrix<T, 2, 3> matrix2x3() const noexcept {
            simd::Matrix<T, 2, 3> M;
            auto sR = rxso2_.matrix();
            M(0, 0) = sR(0, 0);
            M(0, 1) = sR(0, 1);
            M(0, 2) = translation_[0];
            M(1, 0) = sR(1, 0);
            M(1, 1) = sR(1, 1);
            M(1, 2) = translation_[1];
            return M;
        }

        // ===== LIE ALGEBRA =====

        // hat: [sigma, theta, vx, vy] -> 3x3 matrix
        [[nodiscard]] static HomogeneousMatrix hat(const Tangent &twist) noexcept {
            HomogeneousMatrix M;
            M(0, 0) = twist[0];  // sigma
            M(0, 1) = -twist[1]; // -theta
            M(0, 2) = twist[2];  // vx
            M(1, 0) = twist[1];  // theta
            M(1, 1) = twist[0];  // sigma
            M(1, 2) = twist[3];  // vy
            M(2, 0) = T(0);
            M(2, 1) = T(0);
            M(2, 2) = T(0);
            return M;
        }

        // vee: 3x3 matrix -> [sigma, theta, vx, vy]
        [[nodiscard]] static Tangent vee(const HomogeneousMatrix &M) noexcept {
            return Tangent{(M(0, 0) + M(1, 1)) / T(2), (M(1, 0) - M(0, 1)) / T(2), M(0, 2), M(1, 2)};
        }

        // Adjoint representation
        [[nodiscard]] AdjointMatrix Adj() const noexcept {
            AdjointMatrix A;
            const T c = rxso2_.real();
            const T s_sin = rxso2_.imag();
            const T tx = translation_[0];
            const T ty = translation_[1];

            // Adj = [[1, 0, 0, 0], [0, 1, 0, 0], [ty, tx, c, -s], [-tx, ty, s, c]]
            A(0, 0) = T(1);
            A(0, 1) = A(0, 2) = A(0, 3) = T(0);
            A(1, 0) = T(0);
            A(1, 1) = T(1);
            A(1, 2) = A(1, 3) = T(0);
            A(2, 0) = ty;
            A(2, 1) = tx;
            A(2, 2) = c;
            A(2, 3) = -s_sin;
            A(3, 0) = -tx;
            A(3, 1) = ty;
            A(3, 2) = s_sin;
            A(3, 3) = c;
            return A;
        }

        // ===== ACCESSORS =====

        // Get scale factor
        [[nodiscard]] T scale() const noexcept { return rxso2_.scale(); }

        // Get rotation angle
        [[nodiscard]] T angle() const noexcept { return rxso2_.angle(); }

        // Get RxSO2 (scaled rotation)
        [[nodiscard]] const ScaledRotation &rxso2() const noexcept { return rxso2_; }

        // Get SO2 rotation (unit)
        [[nodiscard]] Rotation so2() const noexcept { return rxso2_.so2(); }

        // Get translation
        [[nodiscard]] const Translation &translation() const noexcept { return translation_; }

        // Get SE2 (assuming scale = 1)
        [[nodiscard]] SE2<T> se2() const noexcept { return SE2<T>(rxso2_.so2(), translation_); }

        // ===== MUTATORS =====

        void set_scale(T new_scale) noexcept { rxso2_.set_scale(new_scale); }

        void set_translation(const Translation &t) noexcept { translation_ = t; }

        void set_rxso2(const ScaledRotation &rxso2) noexcept { rxso2_ = rxso2; }

        // ===== TYPE CONVERSION =====

        template <typename NewScalar> [[nodiscard]] Sim2<NewScalar> cast() const noexcept {
            return Sim2<NewScalar>(rxso2_.template cast<NewScalar>(),
                                   typename Sim2<NewScalar>::Translation{static_cast<NewScalar>(translation_[0]),
                                                                         static_cast<NewScalar>(translation_[1])});
        }

        // ===== COMPARISON =====

        [[nodiscard]] bool operator==(const Sim2 &other) const noexcept {
            return rxso2_ == other.rxso2_ && std::abs(translation_[0] - other.translation_[0]) < epsilon<T> &&
                   std::abs(translation_[1] - other.translation_[1]) < epsilon<T>;
        }

        [[nodiscard]] bool operator!=(const Sim2 &other) const noexcept { return !(*this == other); }

        [[nodiscard]] bool is_approx(const Sim2 &other, T tol = epsilon<T>) const noexcept {
            return rxso2_.is_approx(other.rxso2_, tol) && std::abs(translation_[0] - other.translation_[0]) < tol &&
                   std::abs(translation_[1] - other.translation_[1]) < tol;
        }

        [[nodiscard]] bool is_identity(T tol = epsilon<T>) const noexcept {
            return rxso2_.is_identity(tol) && std::abs(translation_[0]) < tol && std::abs(translation_[1]) < tol;
        }

      private:
        ScaledRotation rxso2_;    // Scaled rotation
        Translation translation_; // Translation
    };

    // ===== TYPE ALIASES =====

    using Sim2f = Sim2<float>;
    using Sim2d = Sim2<double>;

    // ===== FREE FUNCTIONS =====

    // Geodesic interpolation
    template <typename T> [[nodiscard]] Sim2<T> interpolate(const Sim2<T> &a, const Sim2<T> &b, T t) noexcept {
        auto twist = (a.inverse() * b).log();
        twist[0] *= t;
        twist[1] *= t;
        twist[2] *= t;
        twist[3] *= t;
        return a * Sim2<T>::exp(twist);
    }

} // namespace optinum::lie
