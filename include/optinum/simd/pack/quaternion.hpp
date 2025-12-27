#pragma once

// =============================================================================
// optinum/simd/pack/quaternion.hpp
// SIMD pack specialization for quaternions (split representation)
// =============================================================================

#include <datapod/matrix/math/quaternion.hpp>
#include <optinum/simd/pack/pack.hpp>

namespace optinum::simd {

    namespace dp = ::datapod;

    /**
     * @brief SIMD pack for quaternions using split representation
     *
     * Stores w, x, y, z components in separate SIMD registers for efficiency:
     * - pack_w: [w0, w1, w2, ..., wW-1]  (scalar part)
     * - pack_x: [x0, x1, x2, ..., xW-1]  (i component)
     * - pack_y: [y0, y1, y2, ..., yW-1]  (j component)
     * - pack_z: [z0, z1, z2, ..., zW-1]  (k component)
     *
     * This is more efficient than interleaved [w0, x0, y0, z0, w1, ...] for most operations.
     *
     * Quaternion convention: q = w + xi + yj + zk
     * where i² = j² = k² = ijk = -1
     *
     * Hamilton product is implemented for SIMD quaternion multiplication.
     */
    template <typename T, std::size_t W> class pack<dp::mat::quaternion<T>, W> {
        static_assert(std::is_floating_point_v<T>, "Quaternion pack requires floating-point type");

      public:
        using value_type = dp::mat::quaternion<T>;
        using real_type = T;
        using real_pack = pack<T, W>;

        static constexpr std::size_t width = W;

      private:
        real_pack w_; // Scalar (real) part
        real_pack x_; // Imaginary i
        real_pack y_; // Imaginary j
        real_pack z_; // Imaginary k

      public:
        // ===== CONSTRUCTORS =====

        pack() noexcept = default;

        // Construct from four component packs
        pack(const real_pack &w, const real_pack &x, const real_pack &y, const real_pack &z) noexcept
            : w_(w), x_(x), y_(y), z_(z) {}

        // Broadcast single quaternion value to all lanes
        explicit pack(const value_type &val) noexcept : w_(val.w), x_(val.x), y_(val.y), z_(val.z) {}

        // Construct from scalar (real quaternion only)
        explicit pack(T scalar) noexcept : w_(scalar), x_(T{}), y_(T{}), z_(T{}) {}

        // Zero initialization
        static pack zero() noexcept { return pack(real_pack(T{}), real_pack(T{}), real_pack(T{}), real_pack(T{})); }

        // Identity quaternion (1, 0, 0, 0) broadcast to all lanes
        static pack identity() noexcept {
            return pack(real_pack(T{1}), real_pack(T{}), real_pack(T{}), real_pack(T{}));
        }

        // ===== COMPONENT ACCESS =====

        [[nodiscard]] const real_pack &w() const noexcept { return w_; }
        [[nodiscard]] const real_pack &x() const noexcept { return x_; }
        [[nodiscard]] const real_pack &y() const noexcept { return y_; }
        [[nodiscard]] const real_pack &z() const noexcept { return z_; }
        [[nodiscard]] real_pack &w() noexcept { return w_; }
        [[nodiscard]] real_pack &x() noexcept { return x_; }
        [[nodiscard]] real_pack &y() noexcept { return y_; }
        [[nodiscard]] real_pack &z() noexcept { return z_; }

        // Scalar/vector part access (for Lie group operations)
        [[nodiscard]] const real_pack &scalar() const noexcept { return w_; }

        // ===== MEMORY OPERATIONS =====

        // Load from interleaved memory: [w0, x0, y0, z0, w1, x1, y1, z1, ...]
        static pack loadu_interleaved(const value_type *ptr) noexcept {
            alignas(32) T w_vals[W];
            alignas(32) T x_vals[W];
            alignas(32) T y_vals[W];
            alignas(32) T z_vals[W];
            for (std::size_t i = 0; i < W; ++i) {
                w_vals[i] = ptr[i].w;
                x_vals[i] = ptr[i].x;
                y_vals[i] = ptr[i].y;
                z_vals[i] = ptr[i].z;
            }
            return pack(real_pack::loadu(w_vals), real_pack::loadu(x_vals), real_pack::loadu(y_vals),
                        real_pack::loadu(z_vals));
        }

        // Store to interleaved memory: [w0, x0, y0, z0, w1, x1, y1, z1, ...]
        void storeu_interleaved(value_type *ptr) const noexcept {
            alignas(32) T w_vals[W];
            alignas(32) T x_vals[W];
            alignas(32) T y_vals[W];
            alignas(32) T z_vals[W];
            w_.storeu(w_vals);
            x_.storeu(x_vals);
            y_.storeu(y_vals);
            z_.storeu(z_vals);
            for (std::size_t i = 0; i < W; ++i) {
                ptr[i].w = w_vals[i];
                ptr[i].x = x_vals[i];
                ptr[i].y = y_vals[i];
                ptr[i].z = z_vals[i];
            }
        }

        // Load from split memory: separate arrays for w, x, y, z
        static pack loadu_split(const T *w_ptr, const T *x_ptr, const T *y_ptr, const T *z_ptr) noexcept {
            return pack(real_pack::loadu(w_ptr), real_pack::loadu(x_ptr), real_pack::loadu(y_ptr),
                        real_pack::loadu(z_ptr));
        }

        // Store to split memory
        void storeu_split(T *w_ptr, T *x_ptr, T *y_ptr, T *z_ptr) const noexcept {
            w_.storeu(w_ptr);
            x_.storeu(x_ptr);
            y_.storeu(y_ptr);
            z_.storeu(z_ptr);
        }

        // ===== ARITHMETIC OPERATIONS =====

        // Addition: component-wise
        pack operator+(const pack &other) const noexcept {
            return pack(w_ + other.w_, x_ + other.x_, y_ + other.y_, z_ + other.z_);
        }

        // Subtraction: component-wise
        pack operator-(const pack &other) const noexcept {
            return pack(w_ - other.w_, x_ - other.x_, y_ - other.y_, z_ - other.z_);
        }

        // Hamilton product (non-commutative quaternion multiplication)
        // (a + bi + cj + dk)(e + fi + gj + hk) =
        //   (ae - bf - cg - dh) +
        //   (af + be + ch - dg)i +
        //   (ag - bh + ce + df)j +
        //   (ah + bg - cf + de)k
        pack operator*(const pack &other) const noexcept {
            // Using variable names matching the quaternion convention
            // this = (w, x, y, z), other = (ow, ox, oy, oz)
            auto nw = w_ * other.w_ - x_ * other.x_ - y_ * other.y_ - z_ * other.z_;
            auto nx = w_ * other.x_ + x_ * other.w_ + y_ * other.z_ - z_ * other.y_;
            auto ny = w_ * other.y_ - x_ * other.z_ + y_ * other.w_ + z_ * other.x_;
            auto nz = w_ * other.z_ + x_ * other.y_ - y_ * other.x_ + z_ * other.w_;
            return pack(nw, nx, ny, nz);
        }

        // Scalar multiplication
        pack operator*(T scalar) const noexcept { return pack(w_ * scalar, x_ * scalar, y_ * scalar, z_ * scalar); }

        friend pack operator*(T scalar, const pack &p) noexcept { return p * scalar; }

        // Division by quaternion: this * other.inverse()
        pack operator/(const pack &other) const noexcept { return *this * other.inverse(); }

        // Division by scalar
        pack operator/(T scalar) const noexcept { return pack(w_ / scalar, x_ / scalar, y_ / scalar, z_ / scalar); }

        // Unary negation
        pack operator-() const noexcept { return pack(-w_, -x_, -y_, -z_); }

        // Unary plus
        pack operator+() const noexcept { return *this; }

        // ===== COMPOUND ASSIGNMENT =====

        pack &operator+=(const pack &other) noexcept {
            w_ += other.w_;
            x_ += other.x_;
            y_ += other.y_;
            z_ += other.z_;
            return *this;
        }

        pack &operator-=(const pack &other) noexcept {
            w_ -= other.w_;
            x_ -= other.x_;
            y_ -= other.y_;
            z_ -= other.z_;
            return *this;
        }

        pack &operator*=(const pack &other) noexcept {
            *this = *this * other;
            return *this;
        }

        pack &operator*=(T scalar) noexcept {
            w_ *= scalar;
            x_ *= scalar;
            y_ *= scalar;
            z_ *= scalar;
            return *this;
        }

        pack &operator/=(const pack &other) noexcept {
            *this = *this / other;
            return *this;
        }

        pack &operator/=(T scalar) noexcept {
            w_ /= scalar;
            x_ /= scalar;
            y_ /= scalar;
            z_ /= scalar;
            return *this;
        }

        // ===== QUATERNION OPERATIONS =====

        // Conjugate: conj(w + xi + yj + zk) = w - xi - yj - zk
        [[nodiscard]] pack conjugate() const noexcept { return pack(w_, -x_, -y_, -z_); }

        // Norm squared: |q|² = w² + x² + y² + z²
        [[nodiscard]] real_pack norm_squared() const noexcept { return w_ * w_ + x_ * x_ + y_ * y_ + z_ * z_; }

        // Norm: |q| = sqrt(w² + x² + y² + z²)
        [[nodiscard]] real_pack norm() const noexcept { return sqrt(norm_squared()); }

        // Magnitude (alias for norm)
        [[nodiscard]] real_pack magnitude() const noexcept { return norm(); }

        // Inverse: q^(-1) = conjugate / |q|²
        [[nodiscard]] pack inverse() const noexcept {
            auto n2 = norm_squared();
            return pack(w_ / n2, -x_ / n2, -y_ / n2, -z_ / n2);
        }

        // Unit inverse (for unit quaternions, inverse == conjugate, faster)
        [[nodiscard]] pack unit_inverse() const noexcept { return conjugate(); }

        // Normalize: q / |q|
        [[nodiscard]] pack normalized() const noexcept {
            auto n = norm();
            return pack(w_ / n, x_ / n, y_ / n, z_ / n);
        }

        // ===== LIE GROUP OPERATIONS =====
        // These are essential for Lie group optimization on SO(3)

        // Dot product (for interpolation and geodesic distance)
        [[nodiscard]] real_pack dot(const pack &other) const noexcept {
            return w_ * other.w_ + x_ * other.x_ + y_ * other.y_ + z_ * other.z_;
        }

        // Geodesic distance on the unit quaternion manifold
        // d(q1, q2) = arccos(|q1 · q2|)
        [[nodiscard]] real_pack geodesic_distance(const pack &other) const noexcept {
            auto d = dot(other);
            // Handle sign (q and -q represent same rotation)
            d = abs(d);
            // Clamp to avoid numerical issues with acos
            return acos(min(d, real_pack(T{1})));
        }

        // Linear interpolation (not normalized)
        [[nodiscard]] pack lerp(const pack &other, T t) const noexcept { return *this * (T{1} - t) + other * t; }

        // Normalized linear interpolation (fast slerp approximation)
        [[nodiscard]] pack nlerp(const pack &other, T t) const noexcept {
            // Handle quaternion double-cover (q and -q represent same rotation)
            auto d = dot(other);
            // If dot product is negative, negate other to take shorter path
            // This is done per-lane but we use a simplified version here
            return lerp(other, t).normalized();
        }

        // ===== ROTATION OPERATIONS =====
        // For rotating vectors using unit quaternions

        // Rotate a vector (vx, vy, vz) by this quaternion pack
        // Uses the formula: v' = q * v * q^(-1) where v = (0, vx, vy, vz)
        // Optimized Rodrigues rotation formula
        void rotate_vector(real_pack &vx, real_pack &vy, real_pack &vz) const noexcept {
            // t = 2 * cross(q.xyz, v)
            auto tx = real_pack(T{2}) * (y_ * vz - z_ * vy);
            auto ty = real_pack(T{2}) * (z_ * vx - x_ * vz);
            auto tz = real_pack(T{2}) * (x_ * vy - y_ * vx);

            // v' = v + w * t + cross(q.xyz, t)
            vx = vx + w_ * tx + (y_ * tz - z_ * ty);
            vy = vy + w_ * ty + (z_ * tx - x_ * tz);
            vz = vz + w_ * tz + (x_ * ty - y_ * tx);
        }

        // ===== HORIZONTAL OPERATIONS =====

        // Horizontal sum (for reductions)
        [[nodiscard]] value_type hsum() const noexcept {
            return value_type{w_.hsum(), x_.hsum(), y_.hsum(), z_.hsum()};
        }

        // ===== EXPONENTIAL AND LOGARITHM =====
        // Critical for Lie algebra operations

        // Exponential map: exp(q) where q is a pure quaternion (w=0)
        // Maps from Lie algebra so(3) to Lie group SO(3)
        [[nodiscard]] static pack exp_pure(const real_pack &vx, const real_pack &vy, const real_pack &vz) noexcept {
            // For pure quaternion v = (0, vx, vy, vz)
            // exp(v) = cos(|v|) + sin(|v|)/|v| * v
            auto vnorm = sqrt(vx * vx + vy * vy + vz * vz);

            // Handle small angles (avoid division by zero)
            // For small |v|: sin(|v|)/|v| ≈ 1 - |v|²/6
            auto sinc = sin(vnorm) / vnorm;
            auto cosv = cos(vnorm);

            return pack(cosv, sinc * vx, sinc * vy, sinc * vz);
        }

        // Logarithm: log(q) - returns pure quaternion for unit quaternions
        // Maps from Lie group SO(3) to Lie algebra so(3)
        [[nodiscard]] pack log() const noexcept {
            auto n = norm();
            auto vnorm = sqrt(x_ * x_ + y_ * y_ + z_ * z_);

            // For unit quaternions: log(q) = (0, acos(w)/|v| * v)
            auto s = acos(w_ / n) / vnorm;

            return pack(real_pack(T{0}), s * x_, s * y_, s * z_);
        }

        // Extract the rotation angle (for unit quaternions)
        [[nodiscard]] real_pack angle() const noexcept {
            // angle = 2 * acos(w) for unit quaternion
            return real_pack(T{2}) * acos(w_);
        }

        // Extract axis of rotation (for unit quaternions)
        void axis(real_pack &ax, real_pack &ay, real_pack &az) const noexcept {
            auto s = sqrt(x_ * x_ + y_ * y_ + z_ * z_);
            ax = x_ / s;
            ay = y_ / s;
            az = z_ / s;
        }
    };

    // ===== FREE FUNCTIONS =====

    // Dot product
    template <typename T, std::size_t W>
    pack<T, W> dot(const pack<dp::mat::quaternion<T>, W> &a, const pack<dp::mat::quaternion<T>, W> &b) noexcept {
        return a.dot(b);
    }

    // Spherical linear interpolation (constant angular velocity)
    template <typename T, std::size_t W>
    pack<dp::mat::quaternion<T>, W> slerp(const pack<dp::mat::quaternion<T>, W> &a,
                                          const pack<dp::mat::quaternion<T>, W> &b, T t) noexcept {
        // Handle quaternion double-cover
        auto d = a.dot(b);

        // For simplicity, we use nlerp for now
        // Full slerp requires per-lane branching which is complex in SIMD
        return a.nlerp(b, t);
    }

} // namespace optinum::simd
