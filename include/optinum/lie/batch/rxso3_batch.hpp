#pragma once

// =============================================================================
// optinum/lie/batch/rxso3_batch.hpp
// RxSO3Batch<T, N> - Batched scaled 3D rotations with transparent SIMD acceleration
// =============================================================================

#include <optinum/lie/batch/so3_batch.hpp>
#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/rxso3.hpp>
#include <optinum/simd/math/abs.hpp>
#include <optinum/simd/math/acos.hpp>
#include <optinum/simd/math/cos.hpp>
#include <optinum/simd/math/exp.hpp>
#include <optinum/simd/math/log.hpp>
#include <optinum/simd/math/sin.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/pack/pack.hpp>
#include <optinum/simd/view/quaternion_view.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace optinum::lie {

    namespace dp = ::datapod;

    // ===== RxSO3Batch: Batched Scaled 3D Rotations with SIMD =====
    //
    // Stores N RxSO3 transforms (rotation + positive scale) and provides SIMD-accelerated operations.
    // Uses quaternion_view internally for transparent SIMD dispatch.
    //
    // Storage: Non-unit quaternions where |q| = scale factor
    //   q = scale * unit_quaternion
    //
    // Usage:
    //   RxSO3Batch<double, 8> transforms;
    //   transforms[0] = RxSO3d(2.0, SO3d::rot_x(0.1));  // scale=2, rotation around X
    //   ...
    //   transforms.transform(vx, vy, vz);  // SIMD scaled rotation of 8 vectors
    //
    // The SIMD width is auto-detected:
    //   - AVX: processes 4 doubles at a time
    //   - SSE: processes 2 doubles at a time
    //   - Scalar fallback if N < SIMD width

    template <typename T, std::size_t N> class RxSO3Batch {
        static_assert(std::is_floating_point_v<T>, "RxSO3Batch requires floating-point type");
        static_assert(N > 0, "RxSO3Batch requires N > 0");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Quaternion = dp::mat::quaternion<T>;
        using Tangent = dp::mat::vector<T, 4>; // [sigma, wx, wy, wz] where sigma = log(scale)
        using Point = dp::mat::vector<T, 3>;
        using Matrix = dp::mat::matrix<T, 3, 3>;
        using Element = RxSO3<T>;
        using Rotation = SO3<T>;

        // SIMD width selection (auto-detect based on architecture)
        static constexpr std::size_t SimdWidth = sizeof(T) == 4 ? 8 : 4; // float: 8, double: 4 for AVX
        using View = simd::quaternion_view<T, SimdWidth>;

        static constexpr std::size_t size = N;
        static constexpr std::size_t DoF = 4;       // 3 rotation + 1 log-scale
        static constexpr std::size_t NumParams = 4; // quaternion components

      private:
        std::array<Quaternion, N> quats_; // Non-unit quaternions (|q| = scale)

      public:
        // ===== CONSTRUCTORS =====

        /// Default: all identity (scale=1, rotation=identity) -> quaternion (1,0,0,0)
        RxSO3Batch() noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                quats_[i] = Quaternion{T(1), T(0), T(0), T(0)};
            }
        }

        /// From array of RxSO3 elements
        explicit RxSO3Batch(const std::array<Element, N> &elements) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                quats_[i] = elements[i].quaternion();
            }
        }

        /// From array of quaternions (non-unit, |q| = scale)
        explicit RxSO3Batch(const std::array<Quaternion, N> &quaternions) noexcept : quats_(quaternions) {}

        /// Broadcast single RxSO3 to all lanes
        explicit RxSO3Batch(const Element &elem) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                quats_[i] = elem.quaternion();
            }
        }

        // ===== STATIC FACTORY METHODS =====

        /// All identity transforms (scale=1, rotation=identity)
        [[nodiscard]] static RxSO3Batch identity() noexcept { return RxSO3Batch(); }

        /// Exponential map from N tangent vectors [sigma, wx, wy, wz]
        /// sigma is log(scale), omega = [wx, wy, wz] is rotation vector
        /// True SIMD implementation using pack operations
        [[nodiscard]] static RxSO3Batch exp(const T *sigma, const T *wx, const T *wy, const T *wz) noexcept {
            RxSO3Batch result;

            // SIMD width for this type
            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4; // float: 8, double: 4 for AVX
            using Pack = simd::pack<T, W>;

            // Process W elements at a time using SIMD
            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                // Load sigma (log-scale) and omega components
                auto sig = Pack::loadu(sigma + i);
                auto omega_x = Pack::loadu(wx + i);
                auto omega_y = Pack::loadu(wy + i);
                auto omega_z = Pack::loadu(wz + i);

                // Compute scale = exp(sigma)
                auto scale = simd::exp(sig);

                // Compute theta_sq = wx² + wy² + wz²
                auto theta_sq = omega_x * omega_x + omega_y * omega_y + omega_z * omega_z;

                // Compute theta = sqrt(theta_sq)
                auto theta = simd::sqrt(theta_sq);

                // Compute half_theta = theta / 2
                auto half_theta = theta * Pack(T(0.5));

                // Compute sin(half_theta) and cos(half_theta)
                auto sin_half = simd::sin(half_theta);
                auto cos_half = simd::cos(half_theta);

                // Compute sinc_half = sin(half_theta) / theta
                // For small theta, use Taylor: sin(t/2)/t ≈ 0.5 - t²/48 + ...
                auto eps_sq = Pack(epsilon<T> * epsilon<T>);
                auto small_mask = simd::cmp_lt(theta_sq, eps_sq);

                // Safe division: sin(half_theta) / theta (avoid div by zero)
                auto safe_theta = theta + Pack(epsilon<T>);
                auto sinc_half = sin_half / safe_theta;

                // For small angles: sinc_half ≈ 0.5 (first term of Taylor)
                auto sinc_half_small = Pack(T(0.5));
                sinc_half = simd::blend(sinc_half, sinc_half_small, small_mask);

                // For small angles: cos(half_theta) ≈ 1
                auto cos_half_small = Pack(T(1));
                cos_half = simd::blend(cos_half, cos_half_small, small_mask);

                // Compute unit quaternion components: q_unit = [cos(θ/2), sin(θ/2)/θ * ω]
                // Then multiply by scale: q = scale * q_unit
                auto qw = scale * cos_half;
                auto qx = scale * sinc_half * omega_x;
                auto qy = scale * sinc_half * omega_y;
                auto qz = scale * sinc_half * omega_z;

                // Store to quaternions (w, x, y, z layout)
                alignas(32) T qw_arr[W], qx_arr[W], qy_arr[W], qz_arr[W];
                qw.storeu(qw_arr);
                qx.storeu(qx_arr);
                qy.storeu(qy_arr);
                qz.storeu(qz_arr);

                for (std::size_t j = 0; j < W; ++j) {
                    result.quats_[i + j].w = qw_arr[j];
                    result.quats_[i + j].x = qx_arr[j];
                    result.quats_[i + j].y = qy_arr[j];
                    result.quats_[i + j].z = qz_arr[j];
                }
            }

            // Handle remaining elements with scalar fallback
            for (; i < N; ++i) {
                Tangent tangent{{sigma[i], wx[i], wy[i], wz[i]}};
                result.set(i, Element::exp(tangent));
            }

            return result;
        }

        /// Create with pure scale (no rotation) broadcast to all lanes
        [[nodiscard]] static RxSO3Batch scale_only(T s) noexcept {
            RxSO3Batch result;
            for (std::size_t i = 0; i < N; ++i) {
                result.quats_[i] = Quaternion{s, T(0), T(0), T(0)};
            }
            return result;
        }

        // ===== ELEMENT ACCESS =====

        /// Get element at index i
        [[nodiscard]] Element operator[](std::size_t i) const noexcept { return Element(quats_[i]); }

        /// Set element at index i
        void set(std::size_t i, const Element &elem) noexcept { quats_[i] = elem.quaternion(); }

        /// Direct quaternion access (non-unit, |q| = scale)
        [[nodiscard]] Quaternion &quat(std::size_t i) noexcept { return quats_[i]; }
        [[nodiscard]] const Quaternion &quat(std::size_t i) const noexcept { return quats_[i]; }

        /// Raw data pointer
        [[nodiscard]] Quaternion *data() noexcept { return quats_.data(); }
        [[nodiscard]] const Quaternion *data() const noexcept { return quats_.data(); }

        // ===== VIEW ACCESS =====

        /// Get quaternion_view for SIMD operations
        [[nodiscard]] View view() noexcept { return View(quats_.data(), N); }
        [[nodiscard]] View view() const noexcept { return View(quats_.data(), N); }

        // ===== CORE OPERATIONS (SIMD) =====

        /// Logarithmic map: all transforms -> [sigma, wx, wy, wz] arrays
        /// sigma = log(scale), omega = rotation vector
        void log(T *sigma, T *omega_x, T *omega_y, T *omega_z) const noexcept {
            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                // Load quaternion components
                alignas(32) T w_vals[W], x_vals[W], y_vals[W], z_vals[W];
                for (std::size_t j = 0; j < W; ++j) {
                    w_vals[j] = quats_[i + j].w;
                    x_vals[j] = quats_[i + j].x;
                    y_vals[j] = quats_[i + j].y;
                    z_vals[j] = quats_[i + j].z;
                }

                auto qw = Pack::loadu(w_vals);
                auto qx = Pack::loadu(x_vals);
                auto qy = Pack::loadu(y_vals);
                auto qz = Pack::loadu(z_vals);

                // Compute scale = |q| = sqrt(w² + x² + y² + z²)
                auto scale = simd::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);

                // sigma = log(scale)
                auto sig = simd::log(scale);

                // Normalize to get unit quaternion: q_unit = q / scale
                auto inv_scale = Pack(T(1)) / scale;
                auto uw = qw * inv_scale;
                auto ux = qx * inv_scale;
                auto uy = qy * inv_scale;
                auto uz = qz * inv_scale;

                // Compute rotation vector from unit quaternion
                // angle = 2 * acos(w), axis = (x,y,z) / sin(angle/2)
                // omega = axis * angle

                // Clamp w to [-1, 1] for numerical stability
                uw = Pack::min(Pack::max(uw, Pack(T(-1))), Pack(T(1)));
                auto half_angle = simd::acos(uw); // acos(w) = half_angle
                auto angle = half_angle * Pack(T(2));

                // sin(half_angle) = sqrt(1 - w²) = sqrt(x² + y² + z²)
                auto sin_half = simd::sqrt(ux * ux + uy * uy + uz * uz);

                // For small angles: omega ≈ 2 * (x, y, z)
                auto eps = Pack(epsilon<T>);
                auto small_mask = simd::cmp_lt(sin_half, eps);

                // Safe sinc computation: angle / sin(half_angle)
                auto safe_sin_half = simd::blend(sin_half, Pack(T(1)), small_mask);
                auto factor = angle / safe_sin_half;
                auto factor_small = Pack(T(2));
                factor = simd::blend(factor, factor_small, small_mask);

                // omega = factor * (x, y, z) of unit quaternion
                auto wx = factor * ux;
                auto wy = factor * uy;
                auto wz = factor * uz;

                // Store results
                sig.storeu(sigma + i);
                wx.storeu(omega_x + i);
                wy.storeu(omega_y + i);
                wz.storeu(omega_z + i);
            }

            // Scalar fallback for remaining elements
            for (; i < N; ++i) {
                auto tangent = (*this)[i].log();
                sigma[i] = tangent[0];
                omega_x[i] = tangent[1];
                omega_y[i] = tangent[2];
                omega_z[i] = tangent[3];
            }
        }

        /// Inverse all transforms (SIMD)
        /// For RxSO3: q_inv = conj(q) / |q|² = (w, -x, -y, -z) / (w² + x² + y² + z²)
        [[nodiscard]] RxSO3Batch inverse() const noexcept {
            RxSO3Batch result;

            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                // Load quaternion components
                alignas(32) T w_vals[W], x_vals[W], y_vals[W], z_vals[W];
                for (std::size_t j = 0; j < W; ++j) {
                    w_vals[j] = quats_[i + j].w;
                    x_vals[j] = quats_[i + j].x;
                    y_vals[j] = quats_[i + j].y;
                    z_vals[j] = quats_[i + j].z;
                }

                auto qw = Pack::loadu(w_vals);
                auto qx = Pack::loadu(x_vals);
                auto qy = Pack::loadu(y_vals);
                auto qz = Pack::loadu(z_vals);

                // Compute norm squared
                auto norm_sq = qw * qw + qx * qx + qy * qy + qz * qz;

                // Inverse: conj(q) / norm_sq
                auto inv_norm_sq = Pack(T(1)) / norm_sq;
                auto rw = qw * inv_norm_sq;
                auto rx = -qx * inv_norm_sq;
                auto ry = -qy * inv_norm_sq;
                auto rz = -qz * inv_norm_sq;

                // Store results
                alignas(32) T rw_arr[W], rx_arr[W], ry_arr[W], rz_arr[W];
                rw.storeu(rw_arr);
                rx.storeu(rx_arr);
                ry.storeu(ry_arr);
                rz.storeu(rz_arr);

                for (std::size_t j = 0; j < W; ++j) {
                    result.quats_[i + j].w = rw_arr[j];
                    result.quats_[i + j].x = rx_arr[j];
                    result.quats_[i + j].y = ry_arr[j];
                    result.quats_[i + j].z = rz_arr[j];
                }
            }

            // Scalar fallback
            for (; i < N; ++i) {
                result.quats_[i] = (*this)[i].inverse().quaternion();
            }

            return result;
        }

        /// Group composition: this * other (SIMD Hamilton product)
        /// Scales multiply: |q1 * q2| = |q1| * |q2|
        [[nodiscard]] RxSO3Batch operator*(const RxSO3Batch &other) const noexcept {
            RxSO3Batch result;
            (void)view().multiply_to(other.view(), result.quats_.data());
            return result;
        }

        RxSO3Batch &operator*=(const RxSO3Batch &other) noexcept {
            RxSO3Batch temp;
            (void)view().multiply_to(other.view(), temp.quats_.data());
            quats_ = temp.quats_;
            return *this;
        }

        /// Transform N 3D points: scaled rotation p' = s * R * p (SIMD)
        /// vx, vy, vz are arrays of N components, transformed in place
        void transform(T *vx, T *vy, T *vz) const noexcept {
            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                // Load quaternion components
                alignas(32) T w_vals[W], x_vals[W], y_vals[W], z_vals[W];
                for (std::size_t j = 0; j < W; ++j) {
                    w_vals[j] = quats_[i + j].w;
                    x_vals[j] = quats_[i + j].x;
                    y_vals[j] = quats_[i + j].y;
                    z_vals[j] = quats_[i + j].z;
                }

                auto qw = Pack::loadu(w_vals);
                auto qx = Pack::loadu(x_vals);
                auto qy = Pack::loadu(y_vals);
                auto qz = Pack::loadu(z_vals);

                // Compute scale = |q|
                auto scale = simd::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);

                // Normalize to get unit quaternion
                auto inv_scale = Pack(T(1)) / scale;
                auto uw = qw * inv_scale;
                auto ux = qx * inv_scale;
                auto uy = qy * inv_scale;
                auto uz = qz * inv_scale;

                // Load vectors
                auto px = Pack::loadu(vx + i);
                auto py = Pack::loadu(vy + i);
                auto pz = Pack::loadu(vz + i);

                // Rotate using unit quaternion (Rodrigues formula)
                // t = 2 * cross(q.xyz, v)
                auto two = Pack(T(2));
                auto tx = two * (uy * pz - uz * py);
                auto ty = two * (uz * px - ux * pz);
                auto tz = two * (ux * py - uy * px);

                // v' = v + w * t + cross(q.xyz, t)
                auto rx = px + uw * tx + (uy * tz - uz * ty);
                auto ry = py + uw * ty + (uz * tx - ux * tz);
                auto rz = pz + uw * tz + (ux * ty - uy * tx);

                // Apply scale
                rx = rx * scale;
                ry = ry * scale;
                rz = rz * scale;

                // Store results
                rx.storeu(vx + i);
                ry.storeu(vy + i);
                rz.storeu(vz + i);
            }

            // Scalar fallback
            for (; i < N; ++i) {
                Point p{{vx[i], vy[i], vz[i]}};
                p = (*this)[i] * p;
                vx[i] = p[0];
                vy[i] = p[1];
                vz[i] = p[2];
            }
        }

        // ===== SCALE/ROTATION EXTRACTION =====

        /// Extract scale factors to output array (SIMD)
        /// scale = |q| = sqrt(w² + x² + y² + z²)
        void scales(T *out) const noexcept {
            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                alignas(32) T w_vals[W], x_vals[W], y_vals[W], z_vals[W];
                for (std::size_t j = 0; j < W; ++j) {
                    w_vals[j] = quats_[i + j].w;
                    x_vals[j] = quats_[i + j].x;
                    y_vals[j] = quats_[i + j].y;
                    z_vals[j] = quats_[i + j].z;
                }

                auto qw = Pack::loadu(w_vals);
                auto qx = Pack::loadu(x_vals);
                auto qy = Pack::loadu(y_vals);
                auto qz = Pack::loadu(z_vals);

                auto scale = simd::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
                scale.storeu(out + i);
            }

            // Scalar fallback
            for (; i < N; ++i) {
                out[i] = (*this)[i].scale();
            }
        }

        /// Extract SO3 rotations (normalized quaternions) as SO3Batch
        [[nodiscard]] SO3Batch<T, N> so3() const noexcept {
            SO3Batch<T, N> result;

            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                alignas(32) T w_vals[W], x_vals[W], y_vals[W], z_vals[W];
                for (std::size_t j = 0; j < W; ++j) {
                    w_vals[j] = quats_[i + j].w;
                    x_vals[j] = quats_[i + j].x;
                    y_vals[j] = quats_[i + j].y;
                    z_vals[j] = quats_[i + j].z;
                }

                auto qw = Pack::loadu(w_vals);
                auto qx = Pack::loadu(x_vals);
                auto qy = Pack::loadu(y_vals);
                auto qz = Pack::loadu(z_vals);

                // Normalize
                auto norm = simd::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
                auto inv_norm = Pack(T(1)) / norm;

                auto uw = qw * inv_norm;
                auto ux = qx * inv_norm;
                auto uy = qy * inv_norm;
                auto uz = qz * inv_norm;

                // Store to result
                alignas(32) T uw_arr[W], ux_arr[W], uy_arr[W], uz_arr[W];
                uw.storeu(uw_arr);
                ux.storeu(ux_arr);
                uy.storeu(uy_arr);
                uz.storeu(uz_arr);

                for (std::size_t j = 0; j < W; ++j) {
                    result.quat(i + j).w = uw_arr[j];
                    result.quat(i + j).x = ux_arr[j];
                    result.quat(i + j).y = uy_arr[j];
                    result.quat(i + j).z = uz_arr[j];
                }
            }

            // Scalar fallback
            for (; i < N; ++i) {
                result.set(i, (*this)[i].so3());
            }

            return result;
        }

        /// Set all scales to a single value (preserving rotations)
        void set_scale(T s) noexcept {
            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                alignas(32) T w_vals[W], x_vals[W], y_vals[W], z_vals[W];
                for (std::size_t j = 0; j < W; ++j) {
                    w_vals[j] = quats_[i + j].w;
                    x_vals[j] = quats_[i + j].x;
                    y_vals[j] = quats_[i + j].y;
                    z_vals[j] = quats_[i + j].z;
                }

                auto qw = Pack::loadu(w_vals);
                auto qx = Pack::loadu(x_vals);
                auto qy = Pack::loadu(y_vals);
                auto qz = Pack::loadu(z_vals);

                // Compute current scale and normalize, then multiply by new scale
                auto current_scale = simd::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
                auto factor = Pack(s) / current_scale;

                auto nw = qw * factor;
                auto nx = qx * factor;
                auto ny = qy * factor;
                auto nz = qz * factor;

                // Store
                alignas(32) T nw_arr[W], nx_arr[W], ny_arr[W], nz_arr[W];
                nw.storeu(nw_arr);
                nx.storeu(nx_arr);
                ny.storeu(ny_arr);
                nz.storeu(nz_arr);

                for (std::size_t j = 0; j < W; ++j) {
                    quats_[i + j].w = nw_arr[j];
                    quats_[i + j].x = nx_arr[j];
                    quats_[i + j].y = ny_arr[j];
                    quats_[i + j].z = nz_arr[j];
                }
            }

            // Scalar fallback
            for (; i < N; ++i) {
                (*this)[i].set_scale(s);
                quats_[i] = Element(quats_[i]).quaternion();
            }
        }

        /// Set individual scales from array (preserving rotations)
        void set_scales(const T *s) noexcept {
            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                alignas(32) T w_vals[W], x_vals[W], y_vals[W], z_vals[W];
                for (std::size_t j = 0; j < W; ++j) {
                    w_vals[j] = quats_[i + j].w;
                    x_vals[j] = quats_[i + j].x;
                    y_vals[j] = quats_[i + j].y;
                    z_vals[j] = quats_[i + j].z;
                }

                auto qw = Pack::loadu(w_vals);
                auto qx = Pack::loadu(x_vals);
                auto qy = Pack::loadu(y_vals);
                auto qz = Pack::loadu(z_vals);

                // Load new scales
                auto new_scale = Pack::loadu(s + i);

                // Compute current scale and adjust
                auto current_scale = simd::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
                auto factor = new_scale / current_scale;

                auto nw = qw * factor;
                auto nx = qx * factor;
                auto ny = qy * factor;
                auto nz = qz * factor;

                // Store
                alignas(32) T nw_arr[W], nx_arr[W], ny_arr[W], nz_arr[W];
                nw.storeu(nw_arr);
                nx.storeu(nx_arr);
                ny.storeu(ny_arr);
                nz.storeu(nz_arr);

                for (std::size_t j = 0; j < W; ++j) {
                    quats_[i + j].w = nw_arr[j];
                    quats_[i + j].x = nx_arr[j];
                    quats_[i + j].y = ny_arr[j];
                    quats_[i + j].z = nz_arr[j];
                }
            }

            // Scalar fallback
            for (; i < N; ++i) {
                auto elem = (*this)[i];
                elem.set_scale(s[i]);
                quats_[i] = elem.quaternion();
            }
        }

        // ===== INTERPOLATION (SIMD) =====

        /// SLERP interpolation with another batch (interpolates both scale and rotation)
        [[nodiscard]] RxSO3Batch slerp(const RxSO3Batch &other, T t) const noexcept {
            RxSO3Batch result;

            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                // Load this quaternions
                alignas(32) T w1[W], x1[W], y1[W], z1[W];
                alignas(32) T w2[W], x2[W], y2[W], z2[W];
                for (std::size_t j = 0; j < W; ++j) {
                    w1[j] = quats_[i + j].w;
                    x1[j] = quats_[i + j].x;
                    y1[j] = quats_[i + j].y;
                    z1[j] = quats_[i + j].z;
                    w2[j] = other.quats_[i + j].w;
                    x2[j] = other.quats_[i + j].x;
                    y2[j] = other.quats_[i + j].y;
                    z2[j] = other.quats_[i + j].z;
                }

                auto qw1 = Pack::loadu(w1);
                auto qx1 = Pack::loadu(x1);
                auto qy1 = Pack::loadu(y1);
                auto qz1 = Pack::loadu(z1);
                auto qw2 = Pack::loadu(w2);
                auto qx2 = Pack::loadu(x2);
                auto qy2 = Pack::loadu(y2);
                auto qz2 = Pack::loadu(z2);

                // Extract scales
                auto scale1 = simd::sqrt(qw1 * qw1 + qx1 * qx1 + qy1 * qy1 + qz1 * qz1);
                auto scale2 = simd::sqrt(qw2 * qw2 + qx2 * qx2 + qy2 * qy2 + qz2 * qz2);

                // Interpolate scales linearly
                auto one_minus_t = Pack(T(1) - t);
                auto t_pack = Pack(t);
                auto scale_interp = one_minus_t * scale1 + t_pack * scale2;

                // Normalize to get unit quaternions
                auto inv_s1 = Pack(T(1)) / scale1;
                auto inv_s2 = Pack(T(1)) / scale2;
                auto uw1 = qw1 * inv_s1;
                auto ux1 = qx1 * inv_s1;
                auto uy1 = qy1 * inv_s1;
                auto uz1 = qz1 * inv_s1;
                auto uw2 = qw2 * inv_s2;
                auto ux2 = qx2 * inv_s2;
                auto uy2 = qy2 * inv_s2;
                auto uz2 = qz2 * inv_s2;

                // Dot product for SLERP
                auto dot = uw1 * uw2 + ux1 * ux2 + uy1 * uy2 + uz1 * uz2;

                // Handle quaternion double-cover
                auto sign_mask = simd::cmp_lt(dot, Pack(T(0)));
                uw2 = simd::blend(uw2, -uw2, sign_mask);
                ux2 = simd::blend(ux2, -ux2, sign_mask);
                uy2 = simd::blend(uy2, -uy2, sign_mask);
                uz2 = simd::blend(uz2, -uz2, sign_mask);
                dot = simd::abs(dot);

                // Clamp dot to avoid acos issues
                dot = Pack::min(dot, Pack(T(1)));

                auto theta = simd::acos(dot);
                auto sin_theta = simd::sin(theta);

                // Handle near-parallel quaternions (use lerp)
                auto use_lerp = simd::cmp_lt(sin_theta, Pack(T(1e-6)));

                // SLERP weights
                auto wa_slerp = simd::sin(one_minus_t * theta) / sin_theta;
                auto wb_slerp = simd::sin(t_pack * theta) / sin_theta;

                // LERP weights (fallback)
                auto wa_lerp = one_minus_t;
                auto wb_lerp = t_pack;

                // Blend between slerp and lerp weights
                auto wa = simd::blend(wa_slerp, wa_lerp, use_lerp);
                auto wb = simd::blend(wb_slerp, wb_lerp, use_lerp);

                // Compute interpolated unit quaternion
                auto rw = wa * uw1 + wb * uw2;
                auto rx = wa * ux1 + wb * ux2;
                auto ry = wa * uy1 + wb * uy2;
                auto rz = wa * uz1 + wb * uz2;

                // Normalize if we used lerp
                auto norm_r = simd::sqrt(rw * rw + rx * rx + ry * ry + rz * rz);
                auto inv_norm_r = Pack(T(1)) / norm_r;
                auto rw_norm = rw * inv_norm_r;
                auto rx_norm = rx * inv_norm_r;
                auto ry_norm = ry * inv_norm_r;
                auto rz_norm = rz * inv_norm_r;

                rw = simd::blend(rw, rw_norm, use_lerp);
                rx = simd::blend(rx, rx_norm, use_lerp);
                ry = simd::blend(ry, ry_norm, use_lerp);
                rz = simd::blend(rz, rz_norm, use_lerp);

                // Apply interpolated scale
                rw = rw * scale_interp;
                rx = rx * scale_interp;
                ry = ry * scale_interp;
                rz = rz * scale_interp;

                // Store results
                alignas(32) T rw_arr[W], rx_arr[W], ry_arr[W], rz_arr[W];
                rw.storeu(rw_arr);
                rx.storeu(rx_arr);
                ry.storeu(ry_arr);
                rz.storeu(rz_arr);

                for (std::size_t j = 0; j < W; ++j) {
                    result.quats_[i + j].w = rw_arr[j];
                    result.quats_[i + j].x = rx_arr[j];
                    result.quats_[i + j].y = ry_arr[j];
                    result.quats_[i + j].z = rz_arr[j];
                }
            }

            // Scalar fallback
            for (; i < N; ++i) {
                result.set(i, interpolate((*this)[i], other[i], t));
            }

            return result;
        }

        /// Geodesic interpolation (via log/exp)
        [[nodiscard]] RxSO3Batch interpolate(const RxSO3Batch &other, T t) const noexcept {
            // For RxSO3, geodesic interpolation uses the slerp approach
            return slerp(other, t);
        }

        // ===== ITERATORS =====

        /// For range-based for loop (iterates over RxSO3 elements)
        class Iterator {
            const RxSO3Batch *batch_;
            std::size_t idx_;

          public:
            Iterator(const RxSO3Batch *b, std::size_t i) : batch_(b), idx_(i) {}

            Element operator*() const { return (*batch_)[idx_]; }
            Iterator &operator++() {
                ++idx_;
                return *this;
            }
            bool operator!=(const Iterator &other) const { return idx_ != other.idx_; }
        };

        [[nodiscard]] Iterator begin() const { return Iterator(this, 0); }
        [[nodiscard]] Iterator end() const { return Iterator(this, N); }

        // ===== MATRIX OPERATIONS =====

        /// Get all scaled rotation matrices (3x3)
        void matrices(Matrix *out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = (*this)[i].matrix();
            }
        }

        /// Get all rotation matrices (unit scale)
        void rotation_matrices(Matrix *out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = (*this)[i].rotation_matrix();
            }
        }

        // ===== VALIDATION =====

        /// Check if all quaternions have valid positive scale
        [[nodiscard]] bool all_valid(T tolerance = epsilon<T>) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                T s = (*this)[i].scale();
                if (s <= tolerance) {
                    return false;
                }
            }
            return true;
        }

        /// Check if all are approximately identity (scale=1, rotation=identity)
        [[nodiscard]] bool all_identity(T tolerance = epsilon<T>) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                if (!(*this)[i].is_identity(tolerance)) {
                    return false;
                }
            }
            return true;
        }
    };

    // ===== TYPE ALIASES =====

    template <std::size_t N> using RxSO3Batchf = RxSO3Batch<float, N>;
    template <std::size_t N> using RxSO3Batchd = RxSO3Batch<double, N>;

    // Common sizes
    using RxSO3Batch4f = RxSO3Batch<float, 4>;
    using RxSO3Batch8f = RxSO3Batch<float, 8>;
    using RxSO3Batch16f = RxSO3Batch<float, 16>;

    using RxSO3Batch4d = RxSO3Batch<double, 4>;
    using RxSO3Batch8d = RxSO3Batch<double, 8>;
    using RxSO3Batch16d = RxSO3Batch<double, 16>;

    // ===== FREE FUNCTIONS =====

    /// Geodesic interpolation for batches
    template <typename T, std::size_t N>
    [[nodiscard]] RxSO3Batch<T, N> interpolate(const RxSO3Batch<T, N> &a, const RxSO3Batch<T, N> &b, T t) noexcept {
        return a.interpolate(b, t);
    }

    /// SLERP for batches
    template <typename T, std::size_t N>
    [[nodiscard]] RxSO3Batch<T, N> slerp(const RxSO3Batch<T, N> &a, const RxSO3Batch<T, N> &b, T t) noexcept {
        return a.slerp(b, t);
    }

} // namespace optinum::lie
