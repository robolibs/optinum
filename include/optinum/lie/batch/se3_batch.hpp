#pragma once

// =============================================================================
// optinum/lie/batch/se3_batch.hpp
// SE3Batch<T, N> - Batched SE3 rigid transforms with transparent SIMD acceleration
// =============================================================================

#include <optinum/lie/batch/so3_batch.hpp>
#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/se3.hpp>
#include <optinum/simd/math/cos.hpp>
#include <optinum/simd/math/sin.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/pack/pack.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace optinum::lie {

    namespace dp = ::datapod;

    // ===== SE3Batch: Batched 3D Rigid Transforms with SIMD =====
    //
    // Stores N SE3 poses (rotation + translation) and provides SIMD-accelerated operations.
    // Uses SO3Batch internally for rotation operations.
    //
    // Storage: N quaternions (via SO3Batch) + N*3 translation components
    //
    // Usage:
    //   SE3Batch<double, 8> poses;
    //   poses[0] = SE3d::trans(1, 2, 3) * SE3d::rot_z(0.5);
    //   ...
    //   poses.transform(px, py, pz);  // SIMD transform 8 points
    //
    // The SIMD width is auto-detected via SO3Batch.

    template <typename T, std::size_t N> class SE3Batch {
        static_assert(std::is_floating_point_v<T>, "SE3Batch requires floating-point type");
        static_assert(N > 0, "SE3Batch requires N > 0");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Quaternion = dp::mat::quaternion<T>;
        using Translation = dp::mat::vector<T, 3>;
        using Tangent = dp::mat::vector<T, 6>; // [vx, vy, vz, wx, wy, wz]
        using Point = dp::mat::vector<T, 3>;
        using Element = SE3<T>;
        using Rotation = SO3<T>;
        using RotationBatch = SO3Batch<T, N>;

        static constexpr std::size_t size = N;
        static constexpr std::size_t DoF = 6;
        static constexpr std::size_t NumParams = 7;

      private:
        RotationBatch rotations_;
        std::array<T, N> tx_; // Translation x components
        std::array<T, N> ty_; // Translation y components
        std::array<T, N> tz_; // Translation z components

      public:
        // ===== CONSTRUCTORS =====

        // Default: all identity transforms
        SE3Batch() noexcept {
            tx_.fill(T(0));
            ty_.fill(T(0));
            tz_.fill(T(0));
        }

        // From SO3Batch and translation arrays
        SE3Batch(const RotationBatch &rotations, const T *tx, const T *ty, const T *tz) noexcept
            : rotations_(rotations) {
            for (std::size_t i = 0; i < N; ++i) {
                tx_[i] = tx[i];
                ty_[i] = ty[i];
                tz_[i] = tz[i];
            }
        }

        // From array of SE3 elements
        explicit SE3Batch(const std::array<Element, N> &elements) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                rotations_.set(i, elements[i].so3());
                const auto &t = elements[i].translation();
                tx_[i] = t[0];
                ty_[i] = t[1];
                tz_[i] = t[2];
            }
        }

        // Broadcast single SE3 to all lanes
        explicit SE3Batch(const Element &elem) noexcept : rotations_(elem.so3()) {
            const auto &t = elem.translation();
            tx_.fill(t[0]);
            ty_.fill(t[1]);
            tz_.fill(t[2]);
        }

        // ===== STATIC FACTORY METHODS =====

        // All identity transforms
        [[nodiscard]] static SE3Batch identity() noexcept { return SE3Batch(); }

        // Exponential map from N twists
        // twists: array of N twist vectors [vx, vy, vz, wx, wy, wz]
        [[nodiscard]] static SE3Batch exp(const std::array<Tangent, N> &twists) noexcept {
            SE3Batch result;
            for (std::size_t i = 0; i < N; ++i) {
                auto elem = Element::exp(twists[i]);
                result.rotations_.set(i, elem.so3());
                const auto &t = elem.translation();
                result.tx_[i] = t[0];
                result.ty_[i] = t[1];
                result.tz_[i] = t[2];
            }
            return result;
        }

        // Exponential map from separate arrays (SIMD-friendly layout)
        // True SIMD implementation using pack operations
        [[nodiscard]] static SE3Batch exp(const T *vx, const T *vy, const T *vz, const T *wx, const T *wy,
                                          const T *wz) noexcept {
            SE3Batch result;

            // First compute rotations using SO3Batch SIMD exp
            result.rotations_ = RotationBatch::exp(wx, wy, wz);

            // SIMD width for this type
            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4; // float: 8, double: 4 for AVX
            using Pack = simd::pack<T, W>;

            // Process W elements at a time using SIMD for translation computation
            // Translation: t = V * v where V is the left Jacobian of SO3
            // V = I + (1-cos(θ))/θ² * [ω]× + (θ-sin(θ))/θ³ * [ω]×²
            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                // Load omega (rotation) and v (translation velocity)
                auto omega_x = Pack::loadu(wx + i);
                auto omega_y = Pack::loadu(wy + i);
                auto omega_z = Pack::loadu(wz + i);
                auto v_x = Pack::loadu(vx + i);
                auto v_y = Pack::loadu(vy + i);
                auto v_z = Pack::loadu(vz + i);

                // Compute theta_sq = wx² + wy² + wz²
                auto theta_sq = omega_x * omega_x + omega_y * omega_y + omega_z * omega_z;
                auto theta = simd::sqrt(theta_sq);

                // Compute sin(theta) and cos(theta)
                auto sin_theta = simd::sin(theta);
                auto cos_theta = simd::cos(theta);

                // Compute coefficients for left Jacobian V
                // a = (1 - cos(θ)) / θ²
                // b = (θ - sin(θ)) / θ³
                auto eps_sq = Pack(epsilon<T> * epsilon<T>);
                auto small_mask = simd::cmp_lt(theta_sq, eps_sq);

                // Safe division (add epsilon to avoid div by zero)
                auto safe_theta_sq = theta_sq + Pack(epsilon<T>);
                auto safe_theta_cubed = safe_theta_sq * theta + Pack(epsilon<T>);

                auto a = (Pack(T(1)) - cos_theta) / safe_theta_sq;
                auto b = (theta - sin_theta) / safe_theta_cubed;

                // For small angles: a ≈ 0.5, b ≈ 1/6
                auto a_small = Pack(T(0.5));
                auto b_small = Pack(T(1.0 / 6.0));
                a = simd::blend(a, a_small, small_mask);
                b = simd::blend(b, b_small, small_mask);

                // Compute omega × v (cross product)
                auto cross_x = omega_y * v_z - omega_z * v_y;
                auto cross_y = omega_z * v_x - omega_x * v_z;
                auto cross_z = omega_x * v_y - omega_y * v_x;

                // Compute omega × (omega × v) = omega * (omega · v) - v * |omega|²
                auto dot = omega_x * v_x + omega_y * v_y + omega_z * v_z;
                auto cross2_x = omega_x * dot - v_x * theta_sq;
                auto cross2_y = omega_y * dot - v_y * theta_sq;
                auto cross2_z = omega_z * dot - v_z * theta_sq;

                // t = v + a * (omega × v) + b * (omega × (omega × v))
                auto t_x = v_x + a * cross_x + b * cross2_x;
                auto t_y = v_y + a * cross_y + b * cross2_y;
                auto t_z = v_z + a * cross_z + b * cross2_z;

                // Store translations
                t_x.storeu(result.tx_.data() + i);
                t_y.storeu(result.ty_.data() + i);
                t_z.storeu(result.tz_.data() + i);
            }

            // Handle remaining elements with scalar fallback
            for (; i < N; ++i) {
                Tangent twist{{vx[i], vy[i], vz[i], wx[i], wy[i], wz[i]}};
                auto elem = Element::exp(twist);
                // Rotation already set by SO3Batch::exp above
                const auto &t = elem.translation();
                result.tx_[i] = t[0];
                result.ty_[i] = t[1];
                result.tz_[i] = t[2];
            }

            return result;
        }

        // Pure rotations (no translation)
        [[nodiscard]] static SE3Batch rot_x(T angle) noexcept {
            SE3Batch result;
            result.rotations_ = RotationBatch::rot_x(angle);
            return result;
        }

        [[nodiscard]] static SE3Batch rot_y(T angle) noexcept {
            SE3Batch result;
            result.rotations_ = RotationBatch::rot_y(angle);
            return result;
        }

        [[nodiscard]] static SE3Batch rot_z(T angle) noexcept {
            SE3Batch result;
            result.rotations_ = RotationBatch::rot_z(angle);
            return result;
        }

        // Pure translations (no rotation)
        [[nodiscard]] static SE3Batch trans(T tx, T ty, T tz) noexcept {
            SE3Batch result;
            result.tx_.fill(tx);
            result.ty_.fill(ty);
            result.tz_.fill(tz);
            return result;
        }

        [[nodiscard]] static SE3Batch trans(const Translation &t) noexcept { return trans(t[0], t[1], t[2]); }

        [[nodiscard]] static SE3Batch trans_x(T tx) noexcept { return trans(tx, T(0), T(0)); }
        [[nodiscard]] static SE3Batch trans_y(T ty) noexcept { return trans(T(0), ty, T(0)); }
        [[nodiscard]] static SE3Batch trans_z(T tz) noexcept { return trans(T(0), T(0), tz); }

        // ===== ELEMENT ACCESS =====

        [[nodiscard]] Element operator[](std::size_t i) const noexcept {
            return Element(rotations_[i], Translation{{tx_[i], ty_[i], tz_[i]}});
        }

        // Set element
        void set(std::size_t i, const Element &elem) noexcept {
            rotations_.set(i, elem.so3());
            const auto &t = elem.translation();
            tx_[i] = t[0];
            ty_[i] = t[1];
            tz_[i] = t[2];
        }

        // Component access
        [[nodiscard]] const RotationBatch &rotations() const noexcept { return rotations_; }
        [[nodiscard]] RotationBatch &rotations() noexcept { return rotations_; }

        [[nodiscard]] const T *tx() const noexcept { return tx_.data(); }
        [[nodiscard]] const T *ty() const noexcept { return ty_.data(); }
        [[nodiscard]] const T *tz() const noexcept { return tz_.data(); }

        [[nodiscard]] T *tx() noexcept { return tx_.data(); }
        [[nodiscard]] T *ty() noexcept { return ty_.data(); }
        [[nodiscard]] T *tz() noexcept { return tz_.data(); }

        // Get translation at index
        [[nodiscard]] Translation translation(std::size_t i) const noexcept {
            return Translation{{tx_[i], ty_[i], tz_[i]}};
        }

        // ===== CORE OPERATIONS =====

        // Logarithmic map: all transforms -> array of twists
        [[nodiscard]] std::array<Tangent, N> log() const noexcept {
            std::array<Tangent, N> result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = (*this)[i].log();
            }
            return result;
        }

        // Log to separate arrays (SIMD-friendly layout)
        void log(T *vx, T *vy, T *vz, T *wx, T *wy, T *wz) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                auto twist = (*this)[i].log();
                vx[i] = twist[0];
                vy[i] = twist[1];
                vz[i] = twist[2];
                wx[i] = twist[3];
                wy[i] = twist[4];
                wz[i] = twist[5];
            }
        }

        // Inverse all transforms (SIMD)
        // T^-1 = (R^-1, -R^-1 * t)
        [[nodiscard]] SE3Batch inverse() const noexcept {
            SE3Batch result;

            // R_inv = R^T (conjugate for quaternions)
            result.rotations_ = rotations_.inverse();

            // t_inv = -R^-1 * t
            // Copy translations
            for (std::size_t i = 0; i < N; ++i) {
                result.tx_[i] = -tx_[i];
                result.ty_[i] = -ty_[i];
                result.tz_[i] = -tz_[i];
            }
            // Rotate by inverse rotation
            result.rotations_.rotate(result.tx_.data(), result.ty_.data(), result.tz_.data());

            return result;
        }

        // Inverse in place
        void inverse_inplace() noexcept { *this = inverse(); }

        // Group composition: this * other (SIMD)
        // (R1, t1) * (R2, t2) = (R1*R2, t1 + R1*t2)
        [[nodiscard]] SE3Batch operator*(const SE3Batch &other) const noexcept {
            SE3Batch result;

            // R_result = R1 * R2
            result.rotations_ = rotations_ * other.rotations_;

            // t_result = t1 + R1 * t2
            // First, copy other's translation
            for (std::size_t i = 0; i < N; ++i) {
                result.tx_[i] = other.tx_[i];
                result.ty_[i] = other.ty_[i];
                result.tz_[i] = other.tz_[i];
            }
            // Rotate by R1
            rotations_.rotate(result.tx_.data(), result.ty_.data(), result.tz_.data());

            // Add t1 (SIMD)
            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                auto r_tx = Pack::loadu(result.tx_.data() + i);
                auto r_ty = Pack::loadu(result.ty_.data() + i);
                auto r_tz = Pack::loadu(result.tz_.data() + i);
                auto t_x = Pack::loadu(tx_.data() + i);
                auto t_y = Pack::loadu(ty_.data() + i);
                auto t_z = Pack::loadu(tz_.data() + i);

                (r_tx + t_x).storeu(result.tx_.data() + i);
                (r_ty + t_y).storeu(result.ty_.data() + i);
                (r_tz + t_z).storeu(result.tz_.data() + i);
            }

            // Scalar fallback
            for (; i < N; ++i) {
                result.tx_[i] += tx_[i];
                result.ty_[i] += ty_[i];
                result.tz_[i] += tz_[i];
            }

            return result;
        }

        SE3Batch &operator*=(const SE3Batch &other) noexcept {
            *this = *this * other;
            return *this;
        }

        // Transform N points (SIMD): p' = R*p + t
        // px, py, pz are arrays of N components, transformed in place
        void transform(T *px, T *py, T *pz) const noexcept {
            // First rotate
            rotations_.rotate(px, py, pz);

            // Then translate (SIMD)
            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                auto p_x = Pack::loadu(px + i);
                auto p_y = Pack::loadu(py + i);
                auto p_z = Pack::loadu(pz + i);
                auto t_x = Pack::loadu(tx_.data() + i);
                auto t_y = Pack::loadu(ty_.data() + i);
                auto t_z = Pack::loadu(tz_.data() + i);

                (p_x + t_x).storeu(px + i);
                (p_y + t_y).storeu(py + i);
                (p_z + t_z).storeu(pz + i);
            }

            // Scalar fallback
            for (; i < N; ++i) {
                px[i] += tx_[i];
                py[i] += ty_[i];
                pz[i] += tz_[i];
            }
        }

        // Transform N points, returning new arrays
        void transform(const T *px_in, const T *py_in, const T *pz_in, T *px_out, T *py_out, T *pz_out) const noexcept {
            // Copy to output first
            for (std::size_t i = 0; i < N; ++i) {
                px_out[i] = px_in[i];
                py_out[i] = py_in[i];
                pz_out[i] = pz_in[i];
            }
            // Transform in place
            transform(px_out, py_out, pz_out);
        }

        // Inverse transform: p' = R^-1 * (p - t) (SIMD)
        void inverse_transform(T *px, T *py, T *pz) const noexcept {
            // First subtract translation (SIMD)
            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                auto p_x = Pack::loadu(px + i);
                auto p_y = Pack::loadu(py + i);
                auto p_z = Pack::loadu(pz + i);
                auto t_x = Pack::loadu(tx_.data() + i);
                auto t_y = Pack::loadu(ty_.data() + i);
                auto t_z = Pack::loadu(tz_.data() + i);

                (p_x - t_x).storeu(px + i);
                (p_y - t_y).storeu(py + i);
                (p_z - t_z).storeu(pz + i);
            }

            // Scalar fallback
            for (; i < N; ++i) {
                px[i] -= tx_[i];
                py[i] -= ty_[i];
                pz[i] -= tz_[i];
            }

            // Then rotate by inverse
            auto r_inv = rotations_.inverse();
            r_inv.rotate(px, py, pz);
        }

        // ===== INTERPOLATION =====

        // Geodesic interpolation (via log/exp)
        [[nodiscard]] SE3Batch interpolate(const SE3Batch &other, T t) const noexcept {
            SE3Batch result;

            for (std::size_t i = 0; i < N; ++i) {
                auto a = (*this)[i];
                auto b = other[i];
                auto interp = lie::interpolate(a, b, t);
                result.set(i, interp);
            }

            return result;
        }

        // Linear interpolation (faster, but not geodesic)
        // Rotation: slerp, Translation: lerp (SIMD)
        [[nodiscard]] SE3Batch lerp(const SE3Batch &other, T t) const noexcept {
            SE3Batch result;

            // Rotation: slerp
            result.rotations_ = rotations_.slerp(other.rotations_, t);

            // Translation: linear interpolation (SIMD)
            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            const Pack t_pack(t);
            const Pack one_minus_t(T(1) - t);

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                auto tx1 = Pack::loadu(tx_.data() + i);
                auto ty1 = Pack::loadu(ty_.data() + i);
                auto tz1 = Pack::loadu(tz_.data() + i);
                auto tx2 = Pack::loadu(other.tx_.data() + i);
                auto ty2 = Pack::loadu(other.ty_.data() + i);
                auto tz2 = Pack::loadu(other.tz_.data() + i);

                (one_minus_t * tx1 + t_pack * tx2).storeu(result.tx_.data() + i);
                (one_minus_t * ty1 + t_pack * ty2).storeu(result.ty_.data() + i);
                (one_minus_t * tz1 + t_pack * tz2).storeu(result.tz_.data() + i);
            }

            // Scalar fallback
            for (; i < N; ++i) {
                result.tx_[i] = (T(1) - t) * tx_[i] + t * other.tx_[i];
                result.ty_[i] = (T(1) - t) * ty_[i] + t * other.ty_[i];
                result.tz_[i] = (T(1) - t) * tz_[i] + t * other.tz_[i];
            }

            return result;
        }

        // ===== NORMALIZATION =====

        // Normalize all rotation quaternions (SIMD)
        void normalize_inplace() noexcept { rotations_.normalize_inplace(); }

        // Return normalized copy
        [[nodiscard]] SE3Batch normalized() const noexcept {
            SE3Batch result = *this;
            result.normalize_inplace();
            return result;
        }

        // ===== REDUCTION OPERATIONS =====

        // Compute geodesic distances to another batch
        void geodesic_distance(const SE3Batch &other, T *out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                // SE3 geodesic distance = sqrt(||log(T1^-1 * T2)||^2)
                auto delta = ((*this)[i].inverse() * other[i]).log();
                T dist_sq = T(0);
                for (std::size_t j = 0; j < 6; ++j) {
                    dist_sq += delta[j] * delta[j];
                }
                out[i] = std::sqrt(dist_sq);
            }
        }

        // Compute translation norms (SIMD)
        void translation_norms(T *out) const noexcept {
            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                auto tx = Pack::loadu(tx_.data() + i);
                auto ty = Pack::loadu(ty_.data() + i);
                auto tz = Pack::loadu(tz_.data() + i);
                auto norm_sq = tx * tx + ty * ty + tz * tz;
                simd::sqrt(norm_sq).storeu(out + i);
            }

            // Scalar fallback
            for (; i < N; ++i) {
                out[i] = std::sqrt(tx_[i] * tx_[i] + ty_[i] * ty_[i] + tz_[i] * tz_[i]);
            }
        }

        // ===== ITERATORS =====

        class Iterator {
            const SE3Batch *batch_;
            std::size_t idx_;

          public:
            Iterator(const SE3Batch *b, std::size_t i) : batch_(b), idx_(i) {}

            Element operator*() const { return (*batch_)[idx_]; }
            Iterator &operator++() {
                ++idx_;
                return *this;
            }
            bool operator!=(const Iterator &other) const { return idx_ != other.idx_; }
        };

        [[nodiscard]] Iterator begin() const { return Iterator(this, 0); }
        [[nodiscard]] Iterator end() const { return Iterator(this, N); }
    };

    // ===== TYPE ALIASES =====

    template <std::size_t N> using SE3Batchf = SE3Batch<float, N>;
    template <std::size_t N> using SE3Batchd = SE3Batch<double, N>;

    // Common sizes
    using SE3Batch4f = SE3Batch<float, 4>;
    using SE3Batch8f = SE3Batch<float, 8>;
    using SE3Batch16f = SE3Batch<float, 16>;

    using SE3Batch4d = SE3Batch<double, 4>;
    using SE3Batch8d = SE3Batch<double, 8>;
    using SE3Batch16d = SE3Batch<double, 16>;

    // ===== FREE FUNCTIONS =====

    // Geodesic interpolation for batches
    template <typename T, std::size_t N>
    [[nodiscard]] SE3Batch<T, N> interpolate(const SE3Batch<T, N> &a, const SE3Batch<T, N> &b, T t) noexcept {
        return a.interpolate(b, t);
    }

} // namespace optinum::lie
