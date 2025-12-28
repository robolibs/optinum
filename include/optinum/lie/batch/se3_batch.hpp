#pragma once

// =============================================================================
// optinum/lie/batch/se3_batch.hpp
// SE3Batch<T, N> - Batched SE3 rigid transforms with transparent SIMD acceleration
// =============================================================================

#include <optinum/lie/batch/so3_batch.hpp>
#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/se3.hpp>

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
        using Translation = simd::Vector<T, 3>;
        using Tangent = simd::Vector<T, 6>; // [vx, vy, vz, wx, wy, wz]
        using Point = simd::Vector<T, 3>;
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
        [[nodiscard]] static SE3Batch exp(const T *vx, const T *vy, const T *vz, const T *wx, const T *wy,
                                          const T *wz) noexcept {
            SE3Batch result;

            // For each pose, compute SE3::exp
            for (std::size_t i = 0; i < N; ++i) {
                Tangent twist{vx[i], vy[i], vz[i], wx[i], wy[i], wz[i]};
                auto elem = Element::exp(twist);
                result.rotations_.set(i, elem.so3());
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
            return Element(rotations_[i], Translation{tx_[i], ty_[i], tz_[i]});
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
            return Translation{tx_[i], ty_[i], tz_[i]};
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
            // Add t1
            for (std::size_t i = 0; i < N; ++i) {
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
            // Then translate
            for (std::size_t i = 0; i < N; ++i) {
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

        // Inverse transform: p' = R^-1 * (p - t)
        void inverse_transform(T *px, T *py, T *pz) const noexcept {
            // First subtract translation
            for (std::size_t i = 0; i < N; ++i) {
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
        // Rotation: slerp, Translation: lerp
        [[nodiscard]] SE3Batch lerp(const SE3Batch &other, T t) const noexcept {
            SE3Batch result;

            // Rotation: slerp
            result.rotations_ = rotations_.slerp(other.rotations_, t);

            // Translation: linear interpolation
            const T one_minus_t = T(1) - t;
            for (std::size_t i = 0; i < N; ++i) {
                result.tx_[i] = one_minus_t * tx_[i] + t * other.tx_[i];
                result.ty_[i] = one_minus_t * ty_[i] + t * other.ty_[i];
                result.tz_[i] = one_minus_t * tz_[i] + t * other.tz_[i];
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

        // Compute translation norms
        void translation_norms(T *out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
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
