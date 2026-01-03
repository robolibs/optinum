#pragma once

// =============================================================================
// optinum/lie/batch/so2_batch.hpp
// SO2Batch<T, N> - Batched SO2 rotations with transparent SIMD acceleration
// =============================================================================

#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/so2.hpp>
#include <optinum/simd/math/atan2.hpp>
#include <optinum/simd/math/cos.hpp>
#include <optinum/simd/math/sin.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/pack/complex.hpp>
#include <optinum/simd/pack/pack.hpp>
#include <optinum/simd/view/complex_view.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace optinum::lie {

    namespace dp = ::datapod;

    // ===== SO2Batch: Batched 2D Rotations with SIMD =====
    //
    // Stores N SO2 rotations as unit complex numbers (cos, sin) and provides
    // SIMD-accelerated operations via complex_view.
    //
    // Usage:
    //   SO2Batch<double, 8> rotations;
    //   rotations[0] = SO2d(0.1);
    //   ...
    //   rotations.normalize_inplace();  // SIMD normalize all 8
    //   rotations.rotate(vx, vy);       // SIMD rotate 8 2D vectors
    //
    // The SIMD width is auto-detected:
    //   - AVX: processes 4 doubles at a time
    //   - SSE: processes 2 doubles at a time
    //   - Scalar fallback if N < SIMD width

    template <typename T, std::size_t N> class SO2Batch {
        static_assert(std::is_floating_point_v<T>, "SO2Batch requires floating-point type");
        static_assert(N > 0, "SO2Batch requires N > 0");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Complex = dp::mat::Complex<T>;
        using Tangent = T; // R^1, just the angle for SO2
        using Point = dp::mat::Vector<T, 2>;
        using RotationMatrix = dp::mat::Matrix<T, 2, 2>;
        using Element = SO2<T>;

        // SIMD width selection (auto-detect based on architecture)
        static constexpr std::size_t SimdWidth = sizeof(T) == 4 ? 8 : 4; // float: 8, double: 4 for AVX
        using View = simd::complex_view<T, SimdWidth>;

        static constexpr std::size_t size = N;
        static constexpr std::size_t DoF = 1;
        static constexpr std::size_t NumParams = 2;

      private:
        std::array<Complex, N> complexes_;

      public:
        // ===== CONSTRUCTORS =====

        /// Default: all identity rotations (1, 0)
        SO2Batch() noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                complexes_[i] = Complex{T(1), T(0)};
            }
        }

        /// From array of SO2 elements
        explicit SO2Batch(const std::array<Element, N> &elements) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                complexes_[i] = Complex{elements[i].real(), elements[i].imag()};
            }
        }

        /// From array of complex numbers
        explicit SO2Batch(const std::array<Complex, N> &complexes) noexcept : complexes_(complexes) {}

        /// Broadcast single SO2 to all lanes
        explicit SO2Batch(const Element &elem) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                complexes_[i] = Complex{elem.real(), elem.imag()};
            }
        }

        // ===== STATIC FACTORY METHODS =====

        /// All identity rotations
        [[nodiscard]] static SO2Batch identity() noexcept { return SO2Batch(); }

        /// Exponential map from N angles (SIMD sin/cos)
        /// theta_array: array of N angles in radians
        [[nodiscard]] static SO2Batch exp(const T *theta_array) noexcept {
            SO2Batch result;

            // SIMD width for this type
            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4; // float: 8, double: 4 for AVX
            using Pack = simd::pack<T, W>;

            // Process W elements at a time using SIMD
            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                // Load angles
                auto theta = Pack::loadu(theta_array + i);

                // Compute cos and sin using SIMD
                auto cos_theta = simd::cos(theta);
                auto sin_theta = simd::sin(theta);

                // Store to complex numbers
                alignas(32) T cos_arr[W], sin_arr[W];
                cos_theta.storeu(cos_arr);
                sin_theta.storeu(sin_arr);

                for (std::size_t j = 0; j < W; ++j) {
                    result.complexes_[i + j].real = cos_arr[j];
                    result.complexes_[i + j].imag = sin_arr[j];
                }
            }

            // Handle remaining elements with scalar fallback
            for (; i < N; ++i) {
                result.complexes_[i].real = std::cos(theta_array[i]);
                result.complexes_[i].imag = std::sin(theta_array[i]);
            }

            return result;
        }

        /// Exponential map from std::array
        [[nodiscard]] static SO2Batch exp(const std::array<Tangent, N> &thetas) noexcept { return exp(thetas.data()); }

        /// Broadcast single angle to all rotations
        [[nodiscard]] static SO2Batch rot(T angle) noexcept { return SO2Batch(Element(angle)); }

        // ===== ELEMENT ACCESS =====

        /// Get element at index i
        [[nodiscard]] Element operator[](std::size_t i) const noexcept {
            return Element(complexes_[i].real, complexes_[i].imag);
        }

        /// Set element at index i
        void set(std::size_t i, const Element &elem) noexcept {
            complexes_[i].real = elem.real();
            complexes_[i].imag = elem.imag();
        }

        /// Direct complex access
        [[nodiscard]] Complex &complex(std::size_t i) noexcept { return complexes_[i]; }
        [[nodiscard]] const Complex &complex(std::size_t i) const noexcept { return complexes_[i]; }

        /// Raw data pointer
        [[nodiscard]] Complex *data() noexcept { return complexes_.data(); }
        [[nodiscard]] const Complex *data() const noexcept { return complexes_.data(); }

        // ===== VIEW ACCESS =====

        /// Get complex_view for SIMD operations
        [[nodiscard]] View view() noexcept { return View(complexes_.data(), N); }
        [[nodiscard]] View view() const noexcept { return View(complexes_.data(), N); }

        // ===== CORE OPERATIONS (SIMD) =====

        /// Logarithmic map: all rotations -> array of angles
        [[nodiscard]] std::array<Tangent, N> log() const noexcept {
            std::array<Tangent, N> result;
            log(result.data());
            return result;
        }

        /// Log to output array (SIMD atan2)
        void log(T *theta_out) const noexcept {
            // SIMD width for this type
            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            // Process W elements at a time using SIMD
            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                // Load real and imaginary parts
                alignas(32) T real_arr[W], imag_arr[W];
                for (std::size_t j = 0; j < W; ++j) {
                    real_arr[j] = complexes_[i + j].real;
                    imag_arr[j] = complexes_[i + j].imag;
                }

                auto real_pack = Pack::loadu(real_arr);
                auto imag_pack = Pack::loadu(imag_arr);

                // theta = atan2(imag, real)
                auto theta = simd::atan2(imag_pack, real_pack);
                theta.storeu(theta_out + i);
            }

            // Scalar fallback for remaining elements
            for (; i < N; ++i) {
                theta_out[i] = std::atan2(complexes_[i].imag, complexes_[i].real);
            }
        }

        /// Inverse all rotations (SIMD conjugate: negate imaginary part)
        [[nodiscard]] SO2Batch inverse() const noexcept {
            SO2Batch result;
            (void)view().conjugate_to(result.complexes_.data());
            return result;
        }

        /// Inverse in place
        void inverse_inplace() noexcept { view().conjugate_inplace(); }

        /// Group composition: this * other (SIMD complex multiplication)
        /// (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        [[nodiscard]] SO2Batch operator*(const SO2Batch &other) const noexcept {
            SO2Batch result;
            (void)view().multiply_to(other.view(), result.complexes_.data());
            return result;
        }

        SO2Batch &operator*=(const SO2Batch &other) noexcept {
            SO2Batch temp;
            (void)view().multiply_to(other.view(), temp.complexes_.data());
            complexes_ = temp.complexes_;
            return *this;
        }

        /// Rotate N 2D vectors (SIMD)
        /// vx, vy are arrays of N components, rotated in place
        /// Rotation: x' = cos*x - sin*y, y' = sin*x + cos*y
        void rotate(T *vx, T *vy) const noexcept {
            // SIMD width for this type
            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            // Process W elements at a time using SIMD
            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                // Load cos and sin
                alignas(32) T cos_arr[W], sin_arr[W];
                for (std::size_t j = 0; j < W; ++j) {
                    cos_arr[j] = complexes_[i + j].real;
                    sin_arr[j] = complexes_[i + j].imag;
                }

                auto cos_pack = Pack::loadu(cos_arr);
                auto sin_pack = Pack::loadu(sin_arr);

                // Load vectors
                auto x = Pack::loadu(vx + i);
                auto y = Pack::loadu(vy + i);

                // Rotate: x' = cos*x - sin*y, y' = sin*x + cos*y
                auto x_new = cos_pack * x - sin_pack * y;
                auto y_new = sin_pack * x + cos_pack * y;

                // Store results
                x_new.storeu(vx + i);
                y_new.storeu(vy + i);
            }

            // Scalar fallback for remaining elements
            for (; i < N; ++i) {
                T c = complexes_[i].real;
                T s = complexes_[i].imag;
                T x = vx[i];
                T y = vy[i];
                vx[i] = c * x - s * y;
                vy[i] = s * x + c * y;
            }
        }

        /// Rotate N points, returning new arrays
        void rotate(const T *vx_in, const T *vy_in, T *vx_out, T *vy_out) const noexcept {
            // Copy to output first
            for (std::size_t i = 0; i < N; ++i) {
                vx_out[i] = vx_in[i];
                vy_out[i] = vy_in[i];
            }
            // Rotate in place
            rotate(vx_out, vy_out);
        }

        // ===== INTERPOLATION (SIMD) =====

        /// SLERP interpolation with another batch
        /// For SO2, slerp is just angle interpolation: exp((1-t)*log(a) + t*log(b))
        [[nodiscard]] SO2Batch slerp(const SO2Batch &other, T t) const noexcept {
            // Get angles from both
            alignas(32) T theta_a[N], theta_b[N];
            log(theta_a);
            other.log(theta_b);

            // Interpolate angles: theta = (1-t)*theta_a + t*theta_b
            // But handle angle wrapping: use delta = theta_b - theta_a, wrap to [-pi, pi]
            alignas(32) T theta_interp[N];

            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                auto a = Pack::loadu(theta_a + i);
                auto b = Pack::loadu(theta_b + i);

                // Compute delta and wrap to [-pi, pi]
                auto delta = b - a;

                // Wrap delta: while delta > pi, delta -= 2*pi; while delta < -pi, delta += 2*pi
                // Using modular arithmetic: delta = delta - 2*pi * round(delta / (2*pi))
                auto two_pi_pack = Pack(two_pi<T>);
                auto pi_pack = Pack(pi<T>);

                // Simple wrapping using conditional logic
                // For angles close to each other, this should work
                auto wrap_pos = simd::cmp_gt(delta, pi_pack);
                auto wrap_neg = simd::cmp_lt(delta, Pack(-pi<T>));
                delta = simd::blend(delta, delta - two_pi_pack, wrap_pos);
                delta = simd::blend(delta, delta + two_pi_pack, wrap_neg);

                // Interpolate
                auto result = a + Pack(t) * delta;
                result.storeu(theta_interp + i);
            }

            // Scalar fallback
            for (; i < N; ++i) {
                T delta = theta_b[i] - theta_a[i];
                // Wrap to [-pi, pi]
                while (delta > pi<T>)
                    delta -= two_pi<T>;
                while (delta < -pi<T>)
                    delta += two_pi<T>;
                theta_interp[i] = theta_a[i] + t * delta;
            }

            return exp(theta_interp);
        }

        /// NLERP interpolation (normalized linear interpolation)
        /// Faster than slerp, good approximation for small angles
        [[nodiscard]] SO2Batch nlerp(const SO2Batch &other, T t) const noexcept {
            SO2Batch result;

            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                // Load complex numbers
                alignas(32) T real_a[W], imag_a[W], real_b[W], imag_b[W];
                for (std::size_t j = 0; j < W; ++j) {
                    real_a[j] = complexes_[i + j].real;
                    imag_a[j] = complexes_[i + j].imag;
                    real_b[j] = other.complexes_[i + j].real;
                    imag_b[j] = other.complexes_[i + j].imag;
                }

                auto ra = Pack::loadu(real_a);
                auto ia = Pack::loadu(imag_a);
                auto rb = Pack::loadu(real_b);
                auto ib = Pack::loadu(imag_b);

                // Linear interpolation
                auto one_minus_t = Pack(T(1) - t);
                auto t_pack = Pack(t);
                auto real_interp = one_minus_t * ra + t_pack * rb;
                auto imag_interp = one_minus_t * ia + t_pack * ib;

                // Normalize
                auto norm_sq = real_interp * real_interp + imag_interp * imag_interp;
                auto norm = simd::sqrt(norm_sq);
                real_interp = real_interp / norm;
                imag_interp = imag_interp / norm;

                // Store
                alignas(32) T real_out[W], imag_out[W];
                real_interp.storeu(real_out);
                imag_interp.storeu(imag_out);

                for (std::size_t j = 0; j < W; ++j) {
                    result.complexes_[i + j].real = real_out[j];
                    result.complexes_[i + j].imag = imag_out[j];
                }
            }

            // Scalar fallback
            for (; i < N; ++i) {
                T ra = complexes_[i].real;
                T ia = complexes_[i].imag;
                T rb = other.complexes_[i].real;
                T ib = other.complexes_[i].imag;

                T real_interp = (T(1) - t) * ra + t * rb;
                T imag_interp = (T(1) - t) * ia + t * ib;

                T norm = std::sqrt(real_interp * real_interp + imag_interp * imag_interp);
                result.complexes_[i].real = real_interp / norm;
                result.complexes_[i].imag = imag_interp / norm;
            }

            return result;
        }

        /// Geodesic interpolation (via log/exp) - same as slerp for SO2
        [[nodiscard]] SO2Batch interpolate(const SO2Batch &other, T t) const noexcept { return slerp(other, t); }

        // ===== NORMALIZATION =====

        /// Normalize all complex numbers to unit magnitude (SIMD)
        void normalize_inplace() noexcept { view().normalize_inplace(); }

        /// Return normalized copy
        [[nodiscard]] SO2Batch normalized() const noexcept {
            SO2Batch result;
            (void)view().normalized_to(result.complexes_.data());
            return result;
        }

        // ===== CONVERSIONS =====

        /// Convert all to angles (same as log)
        void to_angles(T *angles) const noexcept { log(angles); }

        /// Convert all to rotation matrices
        void matrices(RotationMatrix *out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = (*this)[i].matrix();
            }
        }

        // ===== REDUCTION OPERATIONS =====

        /// Compute angular distances to another batch
        void angular_distance(const SO2Batch &other, T *out) const noexcept {
            // Distance = |angle_diff|, wrapped to [0, pi]
            alignas(32) T theta_a[N], theta_b[N];
            log(theta_a);
            other.log(theta_b);

            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                auto a = Pack::loadu(theta_a + i);
                auto b = Pack::loadu(theta_b + i);

                auto delta = b - a;

                // Wrap to [-pi, pi]
                auto two_pi_pack = Pack(two_pi<T>);
                auto pi_pack = Pack(pi<T>);
                auto wrap_pos = simd::cmp_gt(delta, pi_pack);
                auto wrap_neg = simd::cmp_lt(delta, Pack(-pi<T>));
                delta = simd::blend(delta, delta - two_pi_pack, wrap_pos);
                delta = simd::blend(delta, delta + two_pi_pack, wrap_neg);

                // Take absolute value
                auto abs_delta = simd::abs(delta);
                abs_delta.storeu(out + i);
            }

            // Scalar fallback
            for (; i < N; ++i) {
                T delta = theta_b[i] - theta_a[i];
                while (delta > pi<T>)
                    delta -= two_pi<T>;
                while (delta < -pi<T>)
                    delta += two_pi<T>;
                out[i] = std::abs(delta);
            }
        }

        /// Check if all are approximately unit complex numbers
        [[nodiscard]] bool all_unit(T tolerance = epsilon<T>) const noexcept {
            alignas(32) T mags[N];
            view().magnitudes_to(mags);

            for (std::size_t i = 0; i < N; ++i) {
                if (std::abs(mags[i] - T(1)) > tolerance) {
                    return false;
                }
            }
            return true;
        }

        // ===== ITERATORS =====

        /// For range-based for loop (iterates over SO2 elements)
        class Iterator {
            const SO2Batch *batch_;
            std::size_t idx_;

          public:
            Iterator(const SO2Batch *b, std::size_t i) : batch_(b), idx_(i) {}

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

        /// Get all rotation matrices (for Adjoint = 1 in SO2, matrices are useful for other purposes)
        void matrices(std::array<RotationMatrix, N> &out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = (*this)[i].matrix();
            }
        }
    };

    // ===== TYPE ALIASES =====

    template <std::size_t N> using SO2Batchf = SO2Batch<float, N>;
    template <std::size_t N> using SO2Batchd = SO2Batch<double, N>;

    // Common sizes
    using SO2Batch4f = SO2Batch<float, 4>;
    using SO2Batch8f = SO2Batch<float, 8>;
    using SO2Batch16f = SO2Batch<float, 16>;

    using SO2Batch4d = SO2Batch<double, 4>;
    using SO2Batch8d = SO2Batch<double, 8>;
    using SO2Batch16d = SO2Batch<double, 16>;

    // ===== FREE FUNCTIONS =====

    /// Geodesic interpolation for batches
    template <typename T, std::size_t N>
    [[nodiscard]] SO2Batch<T, N> interpolate(const SO2Batch<T, N> &a, const SO2Batch<T, N> &b, T t) noexcept {
        return a.interpolate(b, t);
    }

    /// SLERP for batches
    template <typename T, std::size_t N>
    [[nodiscard]] SO2Batch<T, N> slerp(const SO2Batch<T, N> &a, const SO2Batch<T, N> &b, T t) noexcept {
        return a.slerp(b, t);
    }

    /// NLERP for batches
    template <typename T, std::size_t N>
    [[nodiscard]] SO2Batch<T, N> nlerp(const SO2Batch<T, N> &a, const SO2Batch<T, N> &b, T t) noexcept {
        return a.nlerp(b, t);
    }

} // namespace optinum::lie
