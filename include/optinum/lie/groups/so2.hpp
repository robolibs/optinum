#pragma once

#include <optinum/lie/core/constants.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <datapod/matrix/math/complex.hpp>
#include <datapod/matrix/matrix.hpp>
#include <datapod/matrix/vector.hpp>

#include <cmath>
#include <random>
#include <type_traits>

namespace optinum::lie {

    namespace dp = ::datapod;

    // ===== SO2: Special Orthogonal Group in 2D =====
    //
    // SO(2) represents 2D rotations. Internally stored as a unit complex number:
    //   z = cos(theta) + i*sin(theta) = (real, imag) = (cos, sin)
    //
    // Storage: dp::mat::complex<T> where real = cos(theta), imag = sin(theta)
    // DoF: 1 (rotation angle theta)
    // NumParams: 2 (unit complex number)
    //
    // Lie algebra so(2):
    //   Tangent space is R^1 (just the angle theta)
    //   hat(theta) = [[0, -theta], [theta, 0]] (skew-symmetric 2x2)

    template <typename T = double> class SO2 {
        static_assert(std::is_floating_point_v<T>, "SO2 requires floating-point scalar type");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Tangent = T;                               // R^1, just the angle
        using Complex = dp::mat::complex<T>;             // Unit complex (cos, sin) - internal storage
        using Params = dp::mat::vector<T, 2>;            // For compatibility - owning
        using Point = dp::mat::vector<T, 2>;             // 2D point - owning
        using RotationMatrix = dp::mat::matrix<T, 2, 2>; // owning
        using AdjointMatrix = T;                         // 1x1 matrix = scalar

        // ===== CONSTANTS =====
        static constexpr std::size_t DoF = 1;
        static constexpr std::size_t NumParams = 2;

        // ===== CONSTRUCTORS =====

        // Default: identity rotation (theta = 0)
        constexpr SO2() noexcept : z_{T(1), T(0)} {}

        // From angle (theta in radians)
        explicit SO2(Scalar theta) noexcept : z_{std::cos(theta), std::sin(theta)} {}

        // From unit complex number (cos, sin) - normalizes if needed
        SO2(Scalar real, Scalar imag) noexcept : z_{real, imag} { normalize(); }

        // From 2x2 rotation matrix
        explicit SO2(const RotationMatrix &R) noexcept {
            // Extract cos and sin from R = [[cos, -sin], [sin, cos]]
            z_.real = R(0, 0); // cos
            z_.imag = R(1, 0); // sin
            normalize();
        }

        // From parameters (unit complex as vector)
        explicit SO2(const Params &z) noexcept : z_{z[0], z[1]} { normalize(); }

        // From complex directly
        explicit SO2(const Complex &z) noexcept : z_(z) { normalize(); }

        // ===== STATIC FACTORY METHODS =====

        // Identity element
        [[nodiscard]] static constexpr SO2 identity() noexcept { return SO2(); }

        // Exponential map: theta -> SO2
        // exp(theta) = (cos(theta), sin(theta))
        [[nodiscard]] static SO2 exp(Tangent theta) noexcept { return SO2(theta); }

        // Sample uniform random rotation
        template <typename RNG> [[nodiscard]] static SO2 sample_uniform(RNG &rng) noexcept {
            std::uniform_real_distribution<T> dist(T(0), two_pi<T>);
            return SO2(dist(rng));
        }

        // Fit closest SO2 to arbitrary 2x2 matrix (via SVD/normalization)
        [[nodiscard]] static SO2 fit_to_SO2(const RotationMatrix &M) noexcept {
            // For 2x2, we can just take the first column and normalize
            const T c = M(0, 0);
            const T s = M(1, 0);
            return SO2(c, s); // Constructor normalizes
        }

        // ===== CORE OPERATIONS =====

        // Logarithmic map: SO2 -> theta
        // log() = atan2(sin, cos) = phase of complex number
        [[nodiscard]] Tangent log() const noexcept { return z_.phase(); }

        // Inverse: conjugate of unit complex
        // (cos, sin)^-1 = (cos, -sin)
        [[nodiscard]] SO2 inverse() const noexcept {
            SO2 result;
            result.z_ = z_.conjugate();
            return result;
        }

        // Group composition: complex multiplication
        // (c1 + i*s1) * (c2 + i*s2) = (c1*c2 - s1*s2) + i*(c1*s2 + s1*c2)
        [[nodiscard]] SO2 operator*(const SO2 &other) const noexcept {
            SO2 result;
            result.z_ = z_ * other.z_;
            return result;
        }

        SO2 &operator*=(const SO2 &other) noexcept {
            z_ *= other.z_;
            return *this;
        }

        // Rotate a 2D point: R * p
        // [[cos, -sin], [sin, cos]] * [x, y]^T = [cos*x - sin*y, sin*x + cos*y]
        [[nodiscard]] Point operator*(const Point &p) const noexcept {
            Point result;
            result[0] = z_.real * p[0] - z_.imag * p[1];
            result[1] = z_.imag * p[0] + z_.real * p[1];
            return result;
        }

        // ===== ROTATION MATRIX =====

        // Return 2x2 rotation matrix
        [[nodiscard]] RotationMatrix matrix() const noexcept {
            RotationMatrix R;
            R(0, 0) = z_.real;
            R(0, 1) = -z_.imag;
            R(1, 0) = z_.imag;
            R(1, 1) = z_.real;
            return R;
        }

        // ===== LIE ALGEBRA =====

        // hat: theta -> skew-symmetric 2x2 matrix
        // hat(theta) = [[0, -theta], [theta, 0]]
        [[nodiscard]] static RotationMatrix hat(Tangent theta) noexcept {
            RotationMatrix Omega;
            Omega(0, 0) = T(0);
            Omega(0, 1) = -theta;
            Omega(1, 0) = theta;
            Omega(1, 1) = T(0);
            return Omega;
        }

        // vee: skew-symmetric 2x2 matrix -> theta
        // vee([[0, -theta], [theta, 0]]) = theta
        [[nodiscard]] static Tangent vee(const RotationMatrix &Omega) noexcept {
            // Take average of (1,0) and -(0,1) for numerical stability
            return (Omega(1, 0) - Omega(0, 1)) / T(2);
        }

        // Adjoint representation (for SO2, Adj = 1 since it's commutative)
        [[nodiscard]] constexpr AdjointMatrix Adj() const noexcept { return T(1); }

        // Lie bracket [a, b] = 0 for SO2 (commutative)
        [[nodiscard]] static constexpr Tangent lie_bracket(Tangent /*a*/, Tangent /*b*/) noexcept { return T(0); }

        // Generator (only one for SO2)
        // d/dt exp(t * theta) at t=0 = hat(1)
        [[nodiscard]] static RotationMatrix generator() noexcept { return hat(T(1)); }

        // ===== DERIVATIVES =====

        // Derivative of exp(x) with respect to x
        // For SO2: d/dx exp(x) = [-sin(x), cos(x)]^T (2x1 Jacobian in params)
        // But as a map from tangent (R^1) to tangent (R^1): d/dx exp(x) = 1
        [[nodiscard]] static Params Dx_exp_x(Tangent theta) noexcept {
            Params J;
            J[0] = -std::sin(theta); // d(cos)/d(theta)
            J[1] = std::cos(theta);  // d(sin)/d(theta)
            return J;
        }

        // Derivative of exp at x=0
        [[nodiscard]] static Params Dx_exp_x_at_0() noexcept {
            Params J;
            J[0] = T(0); // d(cos)/d(theta) at theta=0
            J[1] = T(1); // d(sin)/d(theta) at theta=0
            return J;
        }

        // Derivative of this * exp(x) at x=0 with respect to x
        // Result: 2x1 Jacobian in params space
        [[nodiscard]] Params Dx_this_mul_exp_x_at_0() const noexcept {
            // d/dx (this * exp(x)) at x=0 = this * d/dx(exp(x)) at x=0
            // = [[cos, -sin], [sin, cos]] * [0, 1]^T = [-sin, cos]^T
            Params J;
            J[0] = -z_.imag; // -sin(this)
            J[1] = z_.real;  // cos(this)
            return J;
        }

        // Derivative of log(this^-1 * x) at x=this
        // For SO2, this is simply 1
        [[nodiscard]] constexpr Scalar Dx_log_this_inv_by_x_at_this() const noexcept { return T(1); }

        // ===== ACCESSORS =====

        // Get unit complex number
        [[nodiscard]] constexpr const Complex &unit_complex() const noexcept { return z_; }

        // Get cosine component
        [[nodiscard]] constexpr Scalar real() const noexcept { return z_.real; }

        // Get sine component
        [[nodiscard]] constexpr Scalar imag() const noexcept { return z_.imag; }

        // Get angle in radians
        [[nodiscard]] Scalar angle() const noexcept { return log(); }

        // Raw data pointer (for compatibility)
        [[nodiscard]] T *data() noexcept { return &z_.real; }
        [[nodiscard]] const T *data() const noexcept { return &z_.real; }

        // Parameters as vector (for compatibility)
        [[nodiscard]] Params params() const noexcept { return Params{{z_.real, z_.imag}}; }

        // ===== MUTATORS =====

        // Set from angle
        void set_angle(Scalar theta) noexcept {
            z_.real = std::cos(theta);
            z_.imag = std::sin(theta);
        }

        // Set from complex (normalizes)
        void set_complex(Scalar real, Scalar imag) noexcept {
            z_.real = real;
            z_.imag = imag;
            normalize();
        }

        // Set from rotation matrix
        void set_rotation_matrix(const RotationMatrix &R) noexcept {
            z_.real = R(0, 0);
            z_.imag = R(1, 0);
            normalize();
        }

        // Normalize to ensure unit complex
        void normalize() noexcept {
            const Scalar norm = z_.magnitude();
            if (norm > epsilon<T>) {
                z_.real /= norm;
                z_.imag /= norm;
            } else {
                z_.real = T(1);
                z_.imag = T(0);
            }
        }

        // ===== TYPE CONVERSION =====

        // Cast to different scalar type
        template <typename NewScalar> [[nodiscard]] SO2<NewScalar> cast() const noexcept {
            return SO2<NewScalar>(static_cast<NewScalar>(z_.real), static_cast<NewScalar>(z_.imag));
        }

        // ===== COMPARISON =====

        [[nodiscard]] bool operator==(const SO2 &other) const noexcept {
            return std::abs(z_.real - other.z_.real) < epsilon<T> && std::abs(z_.imag - other.z_.imag) < epsilon<T>;
        }

        [[nodiscard]] bool operator!=(const SO2 &other) const noexcept { return !(*this == other); }

        // Check if close to another SO2 (within tolerance)
        [[nodiscard]] bool is_approx(const SO2 &other, Scalar tol = epsilon<T>) const noexcept {
            return std::abs(z_.real - other.z_.real) < tol && std::abs(z_.imag - other.z_.imag) < tol;
        }

        // Check if approximately identity
        [[nodiscard]] bool is_identity(Scalar tol = epsilon<T>) const noexcept {
            return std::abs(z_.real - T(1)) < tol && std::abs(z_.imag) < tol;
        }

      private:
        Complex z_; // Unit complex number: cos(theta) + i*sin(theta)
    };

    // ===== TYPE ALIASES =====

    using SO2f = SO2<float>;
    using SO2d = SO2<double>;

    // ===== FREE FUNCTIONS =====

    // Geodesic interpolation on SO2
    // interpolate(a, b, t) = a * exp(t * log(a^-1 * b))
    template <typename T> [[nodiscard]] SO2<T> interpolate(const SO2<T> &a, const SO2<T> &b, T t) noexcept {
        // theta = log(a^-1 * b)
        const T theta = (a.inverse() * b).log();
        return a * SO2<T>::exp(t * theta);
    }

    // Average of multiple SO2 elements (closed-form for SO2)
    template <typename Iterator> [[nodiscard]] auto average(Iterator begin, Iterator end) {
        using SO2Type = std::remove_cvref_t<decltype(*begin)>;
        using T = typename SO2Type::Scalar;

        if (begin == end) {
            return SO2Type::identity();
        }

        // For SO2, we can average the complex numbers directly
        T sum_cos = T(0);
        T sum_sin = T(0);
        std::size_t count = 0;

        for (auto it = begin; it != end; ++it) {
            sum_cos += it->real();
            sum_sin += it->imag();
            ++count;
        }

        // Normalize the average
        return SO2Type(sum_cos / static_cast<T>(count), sum_sin / static_cast<T>(count));
    }

} // namespace optinum::lie
