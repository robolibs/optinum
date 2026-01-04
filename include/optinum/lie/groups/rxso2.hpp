#pragma once

#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/so2.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <datapod/datapod.hpp>

#include <cmath>
#include <random>
#include <type_traits>

namespace optinum::lie {

    // ===== RxSO2: Rotation + Scale in 2D =====
    //
    // RxSO2 = R+ x SO(2) represents 2D rotation with positive scale.
    // Internally stored as a non-unit complex number:
    //   z = s * (cos(theta) + i*sin(theta)) = (s*cos, s*sin)
    //   where s = scale > 0, theta = rotation angle
    //
    // Storage: Vector<T, 2> where [0] = s*cos(theta), [1] = s*sin(theta)
    // DoF: 2 (rotation angle + log(scale))
    // NumParams: 2 (complex number)
    //
    // Tangent space (Lie algebra):
    //   [sigma, theta] where sigma = log(scale), theta = rotation
    //   The tangent has 2 components: (sigma, theta)
    //
    // exp([sigma, theta]) = exp(sigma) * SO2(theta)
    // log(RxSO2) = [log(scale), theta]

    template <typename T = double> class RxSO2 {
        static_assert(std::is_floating_point_v<T>, "RxSO2 requires floating-point scalar type");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Tangent = dp::mat::Vector<T, 2>; // [sigma, theta] = [log(scale), angle]
        using Params = dp::mat::Vector<T, 2>;  // Complex (s*cos, s*sin)
        using Point = dp::mat::Vector<T, 2>;   // 2D point
        using Matrix = dp::mat::Matrix<T, 2, 2>;
        using AdjointMatrix = dp::mat::Matrix<T, 2, 2>;
        using Rotation = SO2<T>;

        // ===== CONSTANTS =====
        static constexpr std::size_t DoF = 2;
        static constexpr std::size_t NumParams = 2;

        // ===== CONSTRUCTORS =====

        // Default: identity (scale = 1, angle = 0)
        constexpr RxSO2() noexcept : z_{{T(1), T(0)}} {}

        // From SO2 rotation (scale = 1)
        explicit RxSO2(const Rotation &R) noexcept : z_{R.real(), R.imag()} {}

        // From scale and angle
        RxSO2(Scalar scale, Scalar theta) noexcept : z_{scale * std::cos(theta), scale * std::sin(theta)} {}

        // From scale and SO2
        RxSO2(Scalar scale, const Rotation &R) noexcept : z_{scale * R.real(), scale * R.imag()} {}

        // From complex number (real, imag)
        RxSO2(Scalar real, Scalar imag, bool /*disambiguate*/) noexcept : z_{real, imag} {}

        // From parameters
        explicit RxSO2(const Params &z) noexcept : z_(z) {}

        // ===== STATIC FACTORY METHODS =====

        // Identity element (scale = 1, rotation = 0)
        [[nodiscard]] static constexpr RxSO2 identity() noexcept { return RxSO2(); }

        // Exponential map: [sigma, theta] -> RxSO2
        // exp([sigma, theta]) = exp(sigma) * (cos(theta) + i*sin(theta))
        [[nodiscard]] static RxSO2 exp(const Tangent &tangent) noexcept {
            const T sigma = tangent[0]; // log(scale)
            const T theta = tangent[1]; // rotation angle
            const T s = std::exp(sigma);
            return RxSO2(s * std::cos(theta), s * std::sin(theta), true);
        }

        // Sample uniform random RxSO2
        template <typename RNG> [[nodiscard]] static RxSO2 sample_uniform(RNG &rng, T max_log_scale = T(1)) noexcept {
            std::uniform_real_distribution<T> angle_dist(T(0), two_pi<T>);
            std::uniform_real_distribution<T> scale_dist(-max_log_scale, max_log_scale);
            return exp(Tangent{{scale_dist(rng), angle_dist(rng)}});
        }

        // ===== CORE OPERATIONS =====

        // Logarithmic map: RxSO2 -> [sigma, theta]
        [[nodiscard]] Tangent log() const noexcept {
            const T s = scale();
            const T theta = std::atan2(z_[1], z_[0]);
            return Tangent{{std::log(s), theta}};
        }

        // Inverse: z^-1 = conj(z) / |z|^2
        [[nodiscard]] RxSO2 inverse() const noexcept {
            const T norm_sq = z_[0] * z_[0] + z_[1] * z_[1];
            return RxSO2(z_[0] / norm_sq, -z_[1] / norm_sq, true);
        }

        // Group composition: complex multiplication
        [[nodiscard]] RxSO2 operator*(const RxSO2 &other) const noexcept {
            return RxSO2(z_[0] * other.z_[0] - z_[1] * other.z_[1], z_[0] * other.z_[1] + z_[1] * other.z_[0], true);
        }

        RxSO2 &operator*=(const RxSO2 &other) noexcept {
            *this = *this * other;
            return *this;
        }

        // Transform a 2D point: s * R * p
        [[nodiscard]] Point operator*(const Point &p) const noexcept {
            Point result;
            result[0] = z_[0] * p[0] - z_[1] * p[1];
            result[1] = z_[1] * p[0] + z_[0] * p[1];
            return result;
        }

        // ===== MATRIX REPRESENTATION =====

        // Return 2x2 scaled rotation matrix: s * R
        [[nodiscard]] Matrix matrix() const noexcept {
            Matrix M;
            M(0, 0) = z_[0];
            M(0, 1) = -z_[1];
            M(1, 0) = z_[1];
            M(1, 1) = z_[0];
            return M;
        }

        // ===== LIE ALGEBRA =====

        // hat: [sigma, theta] -> 2x2 matrix
        // hat([sigma, theta]) = [[sigma, -theta], [theta, sigma]]
        [[nodiscard]] static Matrix hat(const Tangent &tangent) noexcept {
            Matrix M;
            M(0, 0) = tangent[0];  // sigma
            M(0, 1) = -tangent[1]; // -theta
            M(1, 0) = tangent[1];  // theta
            M(1, 1) = tangent[0];  // sigma
            return M;
        }

        // vee: 2x2 matrix -> [sigma, theta]
        [[nodiscard]] static Tangent vee(const Matrix &M) noexcept {
            return Tangent{{(M(0, 0) + M(1, 1)) / T(2), (M(1, 0) - M(0, 1)) / T(2)}};
        }

        // Adjoint representation
        // For RxSO2, Adj = [[1, 0], [0, 1]] = I (commutative)
        [[nodiscard]] AdjointMatrix Adj() const noexcept {
            AdjointMatrix A;
            A(0, 0) = T(1);
            A(0, 1) = T(0);
            A(1, 0) = T(0);
            A(1, 1) = T(1);
            return A;
        }

        // Lie bracket [a, b] = 0 (commutative group)
        [[nodiscard]] static Tangent lie_bracket(const Tangent & /*a*/, const Tangent & /*b*/) noexcept {
            return Tangent{{T(0), T(0)}};
        }

        // ===== ACCESSORS =====

        // Get scale factor: |z|
        [[nodiscard]] T scale() const noexcept { return std::sqrt(z_[0] * z_[0] + z_[1] * z_[1]); }

        // Get scale squared: |z|^2
        [[nodiscard]] T scale_squared() const noexcept { return z_[0] * z_[0] + z_[1] * z_[1]; }

        // Get rotation angle
        [[nodiscard]] T angle() const noexcept { return std::atan2(z_[1], z_[0]); }

        // Get SO2 rotation (unit complex)
        [[nodiscard]] Rotation so2() const noexcept {
            const T s = scale();
            return Rotation(z_[0] / s, z_[1] / s);
        }

        // Get complex components
        [[nodiscard]] T real() const noexcept { return z_[0]; }
        [[nodiscard]] T imag() const noexcept { return z_[1]; }

        // Get raw parameters
        [[nodiscard]] const Params &params() const noexcept { return z_; }

        // Raw data pointer
        [[nodiscard]] T *data() noexcept { return z_.data(); }
        [[nodiscard]] const T *data() const noexcept { return z_.data(); }

        // ===== MUTATORS =====

        // Set scale (preserving rotation)
        void set_scale(T new_scale) noexcept {
            const T current_scale = scale();
            if (current_scale > epsilon<T>) {
                const T factor = new_scale / current_scale;
                z_[0] *= factor;
                z_[1] *= factor;
            } else {
                z_[0] = new_scale;
                z_[1] = T(0);
            }
        }

        // Set rotation (preserving scale)
        void set_so2(const Rotation &R) noexcept {
            const T s = scale();
            z_[0] = s * R.real();
            z_[1] = s * R.imag();
        }

        // ===== TYPE CONVERSION =====

        template <typename NewScalar> [[nodiscard]] RxSO2<NewScalar> cast() const noexcept {
            return RxSO2<NewScalar>(static_cast<NewScalar>(z_[0]), static_cast<NewScalar>(z_[1]), true);
        }

        // ===== COMPARISON =====

        [[nodiscard]] bool operator==(const RxSO2 &other) const noexcept {
            return std::abs(z_[0] - other.z_[0]) < epsilon<T> && std::abs(z_[1] - other.z_[1]) < epsilon<T>;
        }

        [[nodiscard]] bool operator!=(const RxSO2 &other) const noexcept { return !(*this == other); }

        [[nodiscard]] bool is_approx(const RxSO2 &other, T tol = epsilon<T>) const noexcept {
            return std::abs(z_[0] - other.z_[0]) < tol && std::abs(z_[1] - other.z_[1]) < tol;
        }

        [[nodiscard]] bool is_identity(T tol = epsilon<T>) const noexcept {
            return std::abs(z_[0] - T(1)) < tol && std::abs(z_[1]) < tol;
        }

      private:
        Params z_; // Complex number (s*cos, s*sin)
    };

    // ===== TYPE ALIASES =====

    using RxSO2f = RxSO2<float>;
    using RxSO2d = RxSO2<double>;

    // ===== FREE FUNCTIONS =====

    // Geodesic interpolation
    template <typename T> [[nodiscard]] RxSO2<T> interpolate(const RxSO2<T> &a, const RxSO2<T> &b, T t) noexcept {
        auto tangent = (a.inverse() * b).log();
        tangent[0] *= t;
        tangent[1] *= t;
        return a * RxSO2<T>::exp(tangent);
    }

} // namespace optinum::lie
