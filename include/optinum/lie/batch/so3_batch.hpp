#pragma once

// =============================================================================
// optinum/lie/batch/so3_batch.hpp
// SO3Batch<T, N> - Batched SO3 rotations with transparent SIMD acceleration
// =============================================================================

#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/so3.hpp>
#include <optinum/simd/view/quaternion_view.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace optinum::lie {

    namespace dp = ::datapod;

    // ===== SO3Batch: Batched 3D Rotations with SIMD =====
    //
    // Stores N SO3 rotations and provides SIMD-accelerated operations.
    // Uses quaternion_view internally for transparent SIMD dispatch.
    //
    // Usage:
    //   SO3Batch<double, 8> rotations;
    //   rotations[0] = SO3d::rot_x(0.1);
    //   ...
    //   rotations.normalize_inplace();  // SIMD normalize all 8
    //   rotations.rotate(vx, vy, vz);   // SIMD rotate 8 vectors
    //
    // The SIMD width is auto-detected:
    //   - AVX: processes 4 doubles at a time
    //   - SSE: processes 2 doubles at a time
    //   - Scalar fallback if N < SIMD width

    template <typename T, std::size_t N> class SO3Batch {
        static_assert(std::is_floating_point_v<T>, "SO3Batch requires floating-point type");
        static_assert(N > 0, "SO3Batch requires N > 0");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Quaternion = dp::mat::quaternion<T>;
        using Tangent = simd::Vector<T, 3>;
        using Point = simd::Vector<T, 3>;
        using RotationMatrix = simd::Matrix<T, 3, 3>;
        using Element = SO3<T>;

        // SIMD width selection (auto-detect based on architecture)
        static constexpr std::size_t SimdWidth = sizeof(T) == 4 ? 8 : 4; // float: 8, double: 4 for AVX
        using View = simd::quaternion_view<T, SimdWidth>;

        static constexpr std::size_t size = N;
        static constexpr std::size_t DoF = 3;
        static constexpr std::size_t NumParams = 4;

      private:
        std::array<Quaternion, N> quats_;

      public:
        // ===== CONSTRUCTORS =====

        // Default: all identity rotations
        SO3Batch() noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                quats_[i] = Quaternion{T(1), T(0), T(0), T(0)};
            }
        }

        // From array of SO3 elements
        explicit SO3Batch(const std::array<Element, N> &elements) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                quats_[i] = elements[i].unit_quaternion();
            }
        }

        // From array of quaternions
        explicit SO3Batch(const std::array<Quaternion, N> &quaternions) noexcept : quats_(quaternions) {}

        // Broadcast single SO3 to all lanes
        explicit SO3Batch(const Element &elem) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                quats_[i] = elem.unit_quaternion();
            }
        }

        // ===== STATIC FACTORY METHODS =====

        // All identity rotations
        [[nodiscard]] static SO3Batch identity() noexcept { return SO3Batch(); }

        // Exponential map from N rotation vectors
        // omegas: array of N rotation vectors (axis-angle)
        [[nodiscard]] static SO3Batch exp(const std::array<Tangent, N> &omegas) noexcept {
            SO3Batch result;
            for (std::size_t i = 0; i < N; ++i) {
                result.set(i, Element::exp(omegas[i]));
            }
            return result;
        }

        // Exponential map from separate x, y, z arrays (SIMD-friendly layout)
        // Uses SO3::exp for each element (scalar for now, SIMD optimization possible)
        [[nodiscard]] static SO3Batch exp(const T *omega_x, const T *omega_y, const T *omega_z) noexcept {
            SO3Batch result;

            // Use scalar SO3::exp for correctness
            // The quaternion exp_pure uses full angle, but SO3::exp uses half-angle
            for (std::size_t i = 0; i < N; ++i) {
                Tangent omega{omega_x[i], omega_y[i], omega_z[i]};
                result.set(i, Element::exp(omega));
            }

            return result;
        }

        // Create from Euler angles (arrays)
        [[nodiscard]] static SO3Batch from_euler(const T *roll, const T *pitch, const T *yaw) noexcept {
            SO3Batch result;
            View::from_euler(roll, pitch, yaw, result.quats_.data(), N);
            return result;
        }

        // All rotations around X axis
        [[nodiscard]] static SO3Batch rot_x(T angle) noexcept { return SO3Batch(Element::rot_x(angle)); }

        // All rotations around Y axis
        [[nodiscard]] static SO3Batch rot_y(T angle) noexcept { return SO3Batch(Element::rot_y(angle)); }

        // All rotations around Z axis
        [[nodiscard]] static SO3Batch rot_z(T angle) noexcept { return SO3Batch(Element::rot_z(angle)); }

        // ===== ELEMENT ACCESS =====

        [[nodiscard]] Element operator[](std::size_t i) const noexcept { return Element(quats_[i]); }

        // Set element (assigns and normalizes)
        void set(std::size_t i, const Element &elem) noexcept { quats_[i] = elem.unit_quaternion(); }

        // Direct quaternion access (for advanced users)
        [[nodiscard]] Quaternion &quat(std::size_t i) noexcept { return quats_[i]; }
        [[nodiscard]] const Quaternion &quat(std::size_t i) const noexcept { return quats_[i]; }

        // Raw data pointer
        [[nodiscard]] Quaternion *data() noexcept { return quats_.data(); }
        [[nodiscard]] const Quaternion *data() const noexcept { return quats_.data(); }

        // ===== VIEW ACCESS =====

        // Get quaternion_view for SIMD operations
        [[nodiscard]] View view() noexcept { return View(quats_.data(), N); }
        [[nodiscard]] View view() const noexcept { return View(quats_.data(), N); }

        // ===== CORE OPERATIONS (SIMD) =====

        // Logarithmic map: all rotations -> array of rotation vectors
        [[nodiscard]] std::array<Tangent, N> log() const noexcept {
            std::array<Tangent, N> result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = (*this)[i].log();
            }
            return result;
        }

        // Log to separate arrays (SIMD-friendly layout)
        void log(T *omega_x, T *omega_y, T *omega_z) const noexcept {
            // Use axis-angle extraction from quaternion_view
            alignas(32) T ax[N], ay[N], az[N], angles[N];
            view().to_axis_angle(ax, ay, az, angles);

            // omega = angle * axis
            for (std::size_t i = 0; i < N; ++i) {
                omega_x[i] = angles[i] * ax[i];
                omega_y[i] = angles[i] * ay[i];
                omega_z[i] = angles[i] * az[i];
            }
        }

        // Inverse all rotations (SIMD conjugate)
        [[nodiscard]] SO3Batch inverse() const noexcept {
            SO3Batch result;
            (void)view().conjugate_to(result.quats_.data());
            return result;
        }

        // Inverse in place
        void inverse_inplace() noexcept { view().conjugate_inplace(); }

        // Group composition: this * other (SIMD Hamilton product)
        [[nodiscard]] SO3Batch operator*(const SO3Batch &other) const noexcept {
            SO3Batch result;
            (void)view().multiply_to(other.view(), result.quats_.data());
            return result;
        }

        SO3Batch &operator*=(const SO3Batch &other) noexcept {
            SO3Batch temp;
            (void)view().multiply_to(other.view(), temp.quats_.data());
            quats_ = temp.quats_;
            return *this;
        }

        // Rotate N vectors (SIMD)
        // vx, vy, vz are arrays of N components, rotated in place
        void rotate(T *vx, T *vy, T *vz) const noexcept { view().rotate_vectors(vx, vy, vz); }

        // Rotate N points, returning new arrays
        void rotate(const T *vx_in, const T *vy_in, const T *vz_in, T *vx_out, T *vy_out, T *vz_out) const noexcept {
            // Copy to output first
            for (std::size_t i = 0; i < N; ++i) {
                vx_out[i] = vx_in[i];
                vy_out[i] = vy_in[i];
                vz_out[i] = vz_in[i];
            }
            // Rotate in place
            view().rotate_vectors(vx_out, vy_out, vz_out);
        }

        // ===== INTERPOLATION (SIMD) =====

        // SLERP interpolation with another batch
        [[nodiscard]] SO3Batch slerp(const SO3Batch &other, T t) const noexcept {
            SO3Batch result;
            (void)view().slerp_to(other.view(), t, result.quats_.data());
            return result;
        }

        // NLERP interpolation (faster, normalized linear)
        [[nodiscard]] SO3Batch nlerp(const SO3Batch &other, T t) const noexcept {
            SO3Batch result;
            (void)view().nlerp_to(other.view(), t, result.quats_.data());
            return result;
        }

        // Geodesic interpolation (via log/exp)
        [[nodiscard]] SO3Batch interpolate(const SO3Batch &other, T t) const noexcept {
            // For SO3, slerp IS the geodesic interpolation
            return slerp(other, t);
        }

        // ===== NORMALIZATION =====

        // Normalize all quaternions (SIMD)
        void normalize_inplace() noexcept { view().normalize_inplace(); }

        // Return normalized copy
        [[nodiscard]] SO3Batch normalized() const noexcept {
            SO3Batch result;
            (void)view().normalized_to(result.quats_.data());
            return result;
        }

        // ===== CONVERSIONS =====

        // Convert all to Euler angles (arrays)
        void to_euler(T *roll, T *pitch, T *yaw) const noexcept { view().to_euler(roll, pitch, yaw); }

        // Convert all to axis-angle (arrays)
        void to_axis_angle(T *ax, T *ay, T *az, T *angle) const noexcept { view().to_axis_angle(ax, ay, az, angle); }

        // ===== REDUCTION OPERATIONS =====

        // Compute geodesic distances to another batch
        void geodesic_distance(const SO3Batch &other, T *out) const noexcept {
            // d(q1, q2) = 2 * arccos(|q1 · q2|)
            alignas(32) T dots[N];
            view().dot_to(other.view(), dots);

            for (std::size_t i = 0; i < N; ++i) {
                out[i] = T(2) * std::acos(std::min(std::abs(dots[i]), T(1)));
            }
        }

        // Check if all are approximately unit quaternions
        [[nodiscard]] bool all_unit(T tolerance = epsilon<T>) const noexcept {
            alignas(32) T norms[N];
            view().norms_to(norms);

            for (std::size_t i = 0; i < N; ++i) {
                if (std::abs(norms[i] - T(1)) > tolerance) {
                    return false;
                }
            }
            return true;
        }

        // ===== ITERATORS =====

        // For range-based for loop (iterates over SO3 elements)
        class Iterator {
            const SO3Batch *batch_;
            std::size_t idx_;

          public:
            Iterator(const SO3Batch *b, std::size_t i) : batch_(b), idx_(i) {}

            Element operator*() const { return (*batch_)[idx_]; }
            Iterator &operator++() {
                ++idx_;
                return *this;
            }
            bool operator!=(const Iterator &other) const { return idx_ != other.idx_; }
        };

        [[nodiscard]] Iterator begin() const { return Iterator(this, 0); }
        [[nodiscard]] Iterator end() const { return Iterator(this, N); }

        // ===== JACOBIANS (batched) =====

        // Get all rotation matrices (for Adjoint = R in SO3)
        void matrices(simd::Matrix<T, 3, 3> *out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = (*this)[i].matrix();
            }
        }
    };

    // ===== TYPE ALIASES =====

    template <std::size_t N> using SO3Batchf = SO3Batch<float, N>;
    template <std::size_t N> using SO3Batchd = SO3Batch<double, N>;

    // Common sizes
    using SO3Batch4f = SO3Batch<float, 4>;
    using SO3Batch8f = SO3Batch<float, 8>;
    using SO3Batch16f = SO3Batch<float, 16>;

    using SO3Batch4d = SO3Batch<double, 4>;
    using SO3Batch8d = SO3Batch<double, 8>;
    using SO3Batch16d = SO3Batch<double, 16>;

    // ===== FREE FUNCTIONS =====

    // Geodesic interpolation for batches
    template <typename T, std::size_t N>
    [[nodiscard]] SO3Batch<T, N> interpolate(const SO3Batch<T, N> &a, const SO3Batch<T, N> &b, T t) noexcept {
        return a.interpolate(b, t);
    }

    // SLERP for batches
    template <typename T, std::size_t N>
    [[nodiscard]] SO3Batch<T, N> slerp(const SO3Batch<T, N> &a, const SO3Batch<T, N> &b, T t) noexcept {
        return a.slerp(b, t);
    }

} // namespace optinum::lie
