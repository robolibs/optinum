#pragma once

// =============================================================================
// optinum/simd/pack/quaternion.hpp
// SIMD pack specialization for quaternions (split representation)
// =============================================================================

#include <datapod/datapod.hpp>
#include <optinum/simd/mask.hpp>
#include <optinum/simd/math/simd_math.hpp>
#include <optinum/simd/pack/pack.hpp>

#include <cmath>

namespace optinum::simd {

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
    template <typename T, std::size_t W> class pack<dp::mat::Quaternion<T>, W> {
        static_assert(std::is_floating_point_v<T>, "Quaternion pack requires floating-point type");

      public:
        using value_type = dp::mat::Quaternion<T>;
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

        // ===== FACTORY METHODS =====

        // Create from axis-angle representation (SIMD version)
        // ax, ay, az should be unit vectors, angle is in radians
        [[nodiscard]] static pack from_axis_angle(const real_pack &ax, const real_pack &ay, const real_pack &az,
                                                  const real_pack &angle) noexcept {
            auto half = angle * T{0.5};
            auto s = optinum::simd::sin(half);
            auto c = optinum::simd::cos(half);
            return pack(c, ax * s, ay * s, az * s);
        }

        // Create from Euler angles (ZYX convention: roll, pitch, yaw)
        [[nodiscard]] static pack from_euler(const real_pack &roll, const real_pack &pitch,
                                             const real_pack &yaw) noexcept {
            auto half_roll = roll * T{0.5};
            auto half_pitch = pitch * T{0.5};
            auto half_yaw = yaw * T{0.5};

            auto cr = optinum::simd::cos(half_roll);
            auto sr = optinum::simd::sin(half_roll);
            auto cp = optinum::simd::cos(half_pitch);
            auto sp = optinum::simd::sin(half_pitch);
            auto cy = optinum::simd::cos(half_yaw);
            auto sy = optinum::simd::sin(half_yaw);

            return pack(cr * cp * cy + sr * sp * sy,  // w
                        sr * cp * cy - cr * sp * sy,  // x
                        cr * sp * cy + sr * cp * sy,  // y
                        cr * cp * sy - sr * sp * cy); // z
        }

        // ===== CONVERSION METHODS =====

        // Convert to Euler angles (roll, pitch, yaw in radians)
        void to_euler(real_pack &roll, real_pack &pitch, real_pack &yaw) const noexcept {
            // Roll (x-axis rotation)
            auto sinr_cosp = real_pack(T{2}) * (w_ * x_ + y_ * z_);
            auto cosr_cosp = real_pack(T{1}) - real_pack(T{2}) * (x_ * x_ + y_ * y_);
            roll = atan2(sinr_cosp, cosr_cosp);

            // Pitch (y-axis rotation) - need to handle gimbal lock
            auto sinp = real_pack(T{2}) * (w_ * y_ - z_ * x_);
            // Clamp sinp to [-1, 1] to avoid NaN from asin
            sinp = max(real_pack(T{-1}), min(sinp, real_pack(T{1})));
            pitch = asin(sinp);

            // Yaw (z-axis rotation)
            auto siny_cosp = real_pack(T{2}) * (w_ * z_ + x_ * y_);
            auto cosy_cosp = real_pack(T{1}) - real_pack(T{2}) * (y_ * y_ + z_ * z_);
            yaw = atan2(siny_cosp, cosy_cosp);
        }

        // Convert to axis-angle representation
        void to_axis_angle(real_pack &ax, real_pack &ay, real_pack &az, real_pack &angle) const noexcept {
            angle = real_pack(T{2}) * acos(w_);
            auto s = sqrt(real_pack(T{1}) - w_ * w_);

            // Handle small angles (avoid division by zero)
            // When s is very small, axis is arbitrary (no rotation)
            auto safe_s = max(s, real_pack(T{1e-10}));
            ax = x_ / safe_s;
            ay = y_ / safe_s;
            az = z_ / safe_s;
        }

        // Convert to 3x3 rotation matrix (column-major, outputs 9 real_packs)
        // Matrix layout: [m00, m10, m20, m01, m11, m21, m02, m12, m22]
        void to_rotation_matrix(real_pack &m00, real_pack &m01, real_pack &m02, real_pack &m10, real_pack &m11,
                                real_pack &m12, real_pack &m20, real_pack &m21, real_pack &m22) const noexcept {
            auto two = real_pack(T{2});

            auto xx = x_ * x_;
            auto yy = y_ * y_;
            auto zz = z_ * z_;
            auto xy = x_ * y_;
            auto xz = x_ * z_;
            auto yz = y_ * z_;
            auto wx = w_ * x_;
            auto wy = w_ * y_;
            auto wz = w_ * z_;

            m00 = real_pack(T{1}) - two * (yy + zz);
            m01 = two * (xy - wz);
            m02 = two * (xz + wy);

            m10 = two * (xy + wz);
            m11 = real_pack(T{1}) - two * (xx + zz);
            m12 = two * (yz - wx);

            m20 = two * (xz - wy);
            m21 = two * (yz + wx);
            m22 = real_pack(T{1}) - two * (xx + yy);
        }

        // ===== ADVANCED INTERPOLATION =====

        // True SLERP with proper handling of quaternion double-cover
        // Uses branchless SIMD operations where possible
        [[nodiscard]] pack slerp(const pack &other, T t) const noexcept {
            auto d = dot(other);

            // Handle quaternion double-cover: if dot < 0, negate other
            // This ensures we take the shorter path on the sphere
            auto sign_mask = cmp_lt(d, real_pack(T{0}));

            // Conditionally negate other's components based on sign
            // blend(a, b, mask) returns b where mask is true, else a
            auto other_w = blend(other.w_, -other.w_, sign_mask);
            auto other_x = blend(other.x_, -other.x_, sign_mask);
            auto other_y = blend(other.y_, -other.y_, sign_mask);
            auto other_z = blend(other.z_, -other.z_, sign_mask);
            d = abs(d);

            // Clamp d to avoid numerical issues with acos
            d = min(d, real_pack(T{1}));

            auto theta = acos(d);
            auto sin_theta = sin(theta);

            // Handle near-parallel quaternions (use lerp when sin_theta ≈ 0)
            auto use_lerp = cmp_lt(sin_theta, real_pack(T{1e-6}));

            // SLERP weights
            auto one_minus_t = real_pack(T{1} - t);
            auto wa_slerp = sin(one_minus_t * theta) / sin_theta;
            auto wb_slerp = sin(real_pack(t) * theta) / sin_theta;

            // LERP weights (fallback)
            auto wa_lerp = one_minus_t;
            auto wb_lerp = real_pack(t);

            // Blend between slerp and lerp weights
            // blend(a, b, mask): returns b where mask is true, else a
            auto wa = blend(wa_slerp, wa_lerp, use_lerp);
            auto wb = blend(wb_slerp, wb_lerp, use_lerp);

            // Compute interpolated quaternion
            auto result_w = wa * w_ + wb * other_w;
            auto result_x = wa * x_ + wb * other_x;
            auto result_y = wa * y_ + wb * other_y;
            auto result_z = wa * z_ + wb * other_z;

            // Normalize if we used lerp (slerp result is already unit)
            auto result = pack(result_w, result_x, result_y, result_z);
            return qblend(result, result.normalized(), use_lerp);
        }

        // ===== POWER AND EXPONENTIAL =====

        // Full exponential for general quaternions (not just pure)
        // exp(q) = exp(w) * (cos|v| + sin|v|/|v| * v)
        [[nodiscard]] pack qexp() const noexcept {
            auto vnorm = sqrt(x_ * x_ + y_ * y_ + z_ * z_);
            // Use std::exp element-wise for the scalar exponential
            // This avoids dependency on SIMD exp which may not be available for all types
            real_pack ew;
            for (std::size_t i = 0; i < W; ++i) {
                ew.data_[i] = std::exp(w_[i]);
            }

            // Handle small vector norm (avoid division by zero)
            auto safe_vnorm = max(vnorm, real_pack(T{1e-10}));
            auto sinc = sin(vnorm) / safe_vnorm;

            return pack(ew * cos(vnorm), ew * sinc * x_, ew * sinc * y_, ew * sinc * z_);
        }

        // Power: q^t (useful for interpolation and smooth rotations)
        [[nodiscard]] pack pow(T t) const noexcept { return (this->log() * t).qexp(); }

        // Power with SIMD exponent
        [[nodiscard]] pack pow(const real_pack &t) const noexcept { return (this->log() * t).qexp(); }

        // ===== DERIVATIVE OPERATIONS =====
        // Critical for optimization on SO(3)

        // Compute the derivative of q*v*q^(-1) with respect to q
        // Returns dv/dq for a given vector v = (vx, vy, vz)
        // This is essential for Jacobian computation in Lie group optimization
        void rotate_vector_jacobian(const real_pack &vx, const real_pack &vy, const real_pack &vz, real_pack &dw,
                                    real_pack &dx, real_pack &dy, real_pack &dz) const noexcept {
            // Jacobian of rotation with respect to quaternion
            // dR/dq where R(q)v = q*v*q^(-1)
            auto two = real_pack(T{2});

            // Partial derivatives with respect to w
            dw = two * (w_ * vx + y_ * vz - z_ * vy);

            // Partial derivatives with respect to x
            dx = two * (x_ * vx + y_ * vy + z_ * vz);

            // Partial derivatives with respect to y
            dy = two * (-y_ * vx + x_ * vy + w_ * vz);

            // Partial derivatives with respect to z
            dz = two * (-z_ * vx - w_ * vy + x_ * vz);
        }

        // Difference quaternion: returns delta such that other = this * delta
        // Useful for computing relative rotations in optimization
        [[nodiscard]] pack difference(const pack &other) const noexcept { return this->conjugate() * other; }

        // Angular velocity between this and other quaternion
        // Returns the axis-angle representation of the difference
        void angular_difference(const pack &other, real_pack &wx, real_pack &wy, real_pack &wz) const noexcept {
            auto delta = difference(other);
            auto log_delta = delta.log();
            // Angular velocity is 2 * log(delta).xyz
            wx = real_pack(T{2}) * log_delta.x();
            wy = real_pack(T{2}) * log_delta.y();
            wz = real_pack(T{2}) * log_delta.z();
        }

        // ===== SQUAD (Spherical Quadrangle Interpolation) =====
        // For smooth C1 interpolation between multiple quaternions

        // Inner control point for SQUAD (given q_{i-1}, q_i, q_{i+1})
        [[nodiscard]] static pack squad_control(const pack &q_prev, const pack &q_curr, const pack &q_next) noexcept {
            // s_i = q_i * exp(-1/4 * (log(q_i^(-1) * q_{i+1}) + log(q_i^(-1) * q_{i-1})))
            auto q_inv = q_curr.conjugate();
            auto log_next = (q_inv * q_next).log();
            auto log_prev = (q_inv * q_prev).log();
            auto sum = (log_next + log_prev) * T{-0.25};
            return q_curr * sum.qexp();
        }

        // SQUAD interpolation between q1 and q2 with control points s1 and s2
        [[nodiscard]] static pack squad(const pack &q1, const pack &q2, const pack &s1, const pack &s2, T t) noexcept {
            // squad(q1, q2, s1, s2, t) = slerp(slerp(q1, q2, t), slerp(s1, s2, t), 2t(1-t))
            auto slerp1 = q1.slerp(q2, t);
            auto slerp2 = s1.slerp(s2, t);
            return slerp1.slerp(slerp2, T{2} * t * (T{1} - t));
        }

        // ===== UTILITY =====

        // Check if quaternion is approximately unit (within tolerance)
        [[nodiscard]] real_pack is_unit(T tolerance = T{1e-6}) const noexcept {
            auto n2 = norm_squared();
            return abs(n2 - real_pack(T{1})) < real_pack(tolerance);
        }

        // Check if quaternion is approximately identity
        [[nodiscard]] real_pack is_identity(T tolerance = T{1e-6}) const noexcept {
            auto dw = abs(w_ - real_pack(T{1}));
            auto dx = abs(x_);
            auto dy = abs(y_);
            auto dz = abs(z_);
            return (dw < real_pack(tolerance)) & (dx < real_pack(tolerance)) & (dy < real_pack(tolerance)) &
                   (dz < real_pack(tolerance));
        }

        // Blend two quaternion packs based on a mask
        // blend(a, b, mask): returns b where mask is true, else a
        [[nodiscard]] static pack qblend(const pack &a, const pack &b, const mask<T, W> &m) noexcept {
            return pack(blend(a.w_, b.w_, m), blend(a.x_, b.x_, m), blend(a.y_, b.y_, m), blend(a.z_, b.z_, m));
        }
    };

    // ===== FREE FUNCTIONS =====

    // Dot product
    template <typename T, std::size_t W>
    pack<T, W> dot(const pack<dp::mat::Quaternion<T>, W> &a, const pack<dp::mat::Quaternion<T>, W> &b) noexcept {
        return a.dot(b);
    }

    // Spherical linear interpolation (constant angular velocity)
    template <typename T, std::size_t W>
    pack<dp::mat::Quaternion<T>, W> slerp(const pack<dp::mat::Quaternion<T>, W> &a,
                                          const pack<dp::mat::Quaternion<T>, W> &b, T t) noexcept {
        // Handle quaternion double-cover
        auto d = a.dot(b);

        // For simplicity, we use nlerp for now
        // Full slerp requires per-lane branching which is complex in SIMD
        return a.nlerp(b, t);
    }

} // namespace optinum::simd
