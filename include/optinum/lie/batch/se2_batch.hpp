#pragma once

// =============================================================================
// optinum/lie/batch/se2_batch.hpp
// SE2Batch<T, N> - Batched SE2 rigid transforms with transparent SIMD acceleration
// =============================================================================

#include <optinum/lie/batch/so2_batch.hpp>
#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/se2.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/math/cos.hpp>
#include <optinum/simd/math/sin.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/pack/pack.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace optinum::lie {

    namespace dp = ::datapod;

    // ===== SE2Batch: Batched 2D Rigid Transforms with SIMD =====
    //
    // Stores N SE2 poses (rotation + translation) and provides SIMD-accelerated operations.
    // Uses SO2Batch internally for rotation operations.
    //
    // Storage: N complex numbers (via SO2Batch) + N*2 translation components
    //
    // Usage:
    //   SE2Batch<double, 8> poses;
    //   poses[0] = SE2d::trans(1, 2) * SE2d::rot(0.5);
    //   ...
    //   poses.transform(px, py);  // SIMD transform 8 points

    template <typename T, std::size_t N> class SE2Batch {
        static_assert(std::is_floating_point_v<T>, "SE2Batch requires floating-point type");
        static_assert(N > 0, "SE2Batch requires N > 0");

      public:
        // ===== TYPE ALIASES =====
        using Scalar = T;
        using Complex = dp::mat::Complex<T>;
        using Translation = dp::mat::Vector<T, 2>;
        using Tangent = dp::mat::Vector<T, 3>; // [vx, vy, theta]
        using Point = dp::mat::Vector<T, 2>;
        using Element = SE2<T>;
        using Rotation = SO2<T>;
        using RotationBatch = SO2Batch<T, N>;

        static constexpr std::size_t size = N;
        static constexpr std::size_t DoF = 3;
        static constexpr std::size_t NumParams = 4;

      private:
        RotationBatch rotations_;
        std::array<T, N> tx_; // Translation x components
        std::array<T, N> ty_; // Translation y components

      public:
        // ===== CONSTRUCTORS =====

        /// Default: all identity transforms
        SE2Batch() noexcept {
            tx_.fill(T(0));
            ty_.fill(T(0));
        }

        /// From SO2Batch and translation arrays
        SE2Batch(const RotationBatch &rotations, const T *tx, const T *ty) noexcept : rotations_(rotations) {
            for (std::size_t i = 0; i < N; ++i) {
                tx_[i] = tx[i];
                ty_[i] = ty[i];
            }
        }

        /// From array of SE2 elements
        explicit SE2Batch(const std::array<Element, N> &elements) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                rotations_.set(i, elements[i].so2());
                const auto &t = elements[i].translation();
                tx_[i] = t[0];
                ty_[i] = t[1];
            }
        }

        /// Broadcast single SE2 to all lanes
        explicit SE2Batch(const Element &elem) noexcept : rotations_(elem.so2()) {
            const auto &t = elem.translation();
            tx_.fill(t[0]);
            ty_.fill(t[1]);
        }

        // ===== STATIC FACTORY METHODS =====

        /// All identity transforms
        [[nodiscard]] static SE2Batch identity() noexcept { return SE2Batch(); }

        /// Exponential map from N twists
        [[nodiscard]] static SE2Batch exp(const std::array<Tangent, N> &twists) noexcept {
            SE2Batch result;
            for (std::size_t i = 0; i < N; ++i) {
                auto elem = Element::exp(twists[i]);
                result.rotations_.set(i, elem.so2());
                const auto &t = elem.translation();
                result.tx_[i] = t[0];
                result.ty_[i] = t[1];
            }
            return result;
        }

        /// Exponential map from separate arrays (SIMD-friendly layout)
        [[nodiscard]] static SE2Batch exp(const T *vx, const T *vy, const T *theta) noexcept {
            SE2Batch result;
            result.rotations_ = RotationBatch::exp(theta);

            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                auto theta_pack = Pack::loadu(theta + i);
                auto v_x = Pack::loadu(vx + i);
                auto v_y = Pack::loadu(vy + i);

                auto sin_theta = simd::sin(theta_pack);
                auto cos_theta = simd::cos(theta_pack);

                auto eps = Pack(epsilon<T>);
                auto theta_abs = simd::abs(theta_pack);
                auto small_mask = simd::cmp_lt(theta_abs, eps);

                auto safe_theta = theta_pack + simd::blend(Pack(T(0)), eps, small_mask);

                auto a = sin_theta / safe_theta;
                auto b = (Pack(T(1)) - cos_theta) / safe_theta;

                a = simd::blend(a, Pack(T(1)), small_mask);
                b = simd::blend(b, theta_pack * Pack(T(0.5)), small_mask);

                auto t_x = a * v_x - b * v_y;
                auto t_y = b * v_x + a * v_y;

                t_x.storeu(result.tx_.data() + i);
                t_y.storeu(result.ty_.data() + i);
            }

            for (; i < N; ++i) {
                Tangent twist{{vx[i], vy[i], theta[i]}};
                auto elem = Element::exp(twist);
                const auto &t = elem.translation();
                result.tx_[i] = t[0];
                result.ty_[i] = t[1];
            }

            return result;
        }

        /// Pure rotation batch
        [[nodiscard]] static SE2Batch rot(const T *theta) noexcept {
            SE2Batch result;
            result.rotations_ = RotationBatch::exp(theta);
            return result;
        }

        /// Pure translation batch
        [[nodiscard]] static SE2Batch trans(const T *tx, const T *ty) noexcept {
            SE2Batch result;
            simd::backend::copy_runtime<T>(result.tx_.data(), tx, N);
            simd::backend::copy_runtime<T>(result.ty_.data(), ty, N);
            return result;
        }

        /// Broadcast single angle
        [[nodiscard]] static SE2Batch rot(T angle) noexcept { return SE2Batch(Element::rot(angle)); }

        /// Broadcast single translation
        [[nodiscard]] static SE2Batch trans(T tx, T ty) noexcept { return SE2Batch(Element::trans(tx, ty)); }

        /// Broadcast x-axis translation
        [[nodiscard]] static SE2Batch trans_x(T tx) noexcept { return trans(tx, T(0)); }

        /// Broadcast y-axis translation
        [[nodiscard]] static SE2Batch trans_y(T ty) noexcept { return trans(T(0), ty); }

        // ===== ELEMENT ACCESS =====

        [[nodiscard]] Element operator[](std::size_t i) const noexcept {
            return Element(rotations_[i], Translation{{tx_[i], ty_[i]}});
        }

        void set(std::size_t i, const Element &elem) noexcept {
            rotations_.set(i, elem.so2());
            const auto &t = elem.translation();
            tx_[i] = t[0];
            ty_[i] = t[1];
        }

        [[nodiscard]] RotationBatch &rotations() noexcept { return rotations_; }
        [[nodiscard]] const RotationBatch &rotations() const noexcept { return rotations_; }

        [[nodiscard]] T *tx() noexcept { return tx_.data(); }
        [[nodiscard]] const T *tx() const noexcept { return tx_.data(); }
        [[nodiscard]] T *ty() noexcept { return ty_.data(); }
        [[nodiscard]] const T *ty() const noexcept { return ty_.data(); }

        // ===== CORE OPERATIONS (SIMD) =====

        [[nodiscard]] std::array<Tangent, N> log() const noexcept {
            std::array<Tangent, N> result;
            log(result);
            return result;
        }

        void log(std::array<Tangent, N> &out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = (*this)[i].log();
            }
        }

        [[nodiscard]] SE2Batch inverse() const noexcept {
            SE2Batch result;
            result.rotations_ = rotations_.inverse();

            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                // Use the INVERTED rotation (R^-1) to compute -R^-1 * t
                // R^-1 = (cos, -sin) where original R = (cos, sin)
                // R^-1 * t = [cos*tx + sin*ty, -sin*tx + cos*ty] (using inverted sin = -sin_orig)
                // But result.rotations_ already has (cos, -sin_orig), so:
                // cos_inv = cos, sin_inv = -sin_orig
                // R^-1 * t = [cos_inv*tx - sin_inv*ty, sin_inv*tx + cos_inv*ty]
                //          = [cos*tx - (-sin_orig)*ty, (-sin_orig)*tx + cos*ty]
                //          = [cos*tx + sin_orig*ty, -sin_orig*tx + cos*ty]
                // -R^-1 * t = [-(cos*tx + sin_orig*ty), -(-sin_orig*tx + cos*ty)]
                //           = [-cos*tx - sin_orig*ty, sin_orig*tx - cos*ty]
                // Using inverted values: sin_inv = -sin_orig, so sin_orig = -sin_inv
                // -R^-1 * t = [-cos*tx + sin_inv*ty, -sin_inv*tx - cos*ty]
                alignas(32) T cos_arr[W], sin_arr[W];
                for (std::size_t j = 0; j < W; ++j) {
                    cos_arr[j] = result.rotations_.complex(i + j).real;
                    sin_arr[j] = result.rotations_.complex(i + j).imag; // This is -sin_original
                }

                auto cos_pack = Pack::loadu(cos_arr);
                auto sin_pack = Pack::loadu(sin_arr); // sin_pack = -sin_original
                auto t_x = Pack::loadu(tx_.data() + i);
                auto t_y = Pack::loadu(ty_.data() + i);

                // -R^-1 * t where R^-1 has (cos, sin_inv) with sin_inv = -sin_orig
                // new_tx = -(cos*tx - sin_inv*ty) = -cos*tx + sin_inv*ty
                // new_ty = -(sin_inv*tx + cos*ty) = -sin_inv*tx - cos*ty
                auto new_tx = Pack(T(-1)) * cos_pack * t_x + sin_pack * t_y;
                auto new_ty = Pack(T(-1)) * sin_pack * t_x - cos_pack * t_y;

                new_tx.storeu(result.tx_.data() + i);
                new_ty.storeu(result.ty_.data() + i);
            }

            for (; i < N; ++i) {
                auto elem = (*this)[i].inverse();
                const auto &t = elem.translation();
                result.tx_[i] = t[0];
                result.ty_[i] = t[1];
            }

            return result;
        }

        [[nodiscard]] SE2Batch operator*(const SE2Batch &other) const noexcept {
            SE2Batch result;
            result.rotations_ = rotations_ * other.rotations_;

            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                alignas(32) T cos_arr[W], sin_arr[W];
                for (std::size_t j = 0; j < W; ++j) {
                    cos_arr[j] = rotations_.complex(i + j).real;
                    sin_arr[j] = rotations_.complex(i + j).imag;
                }

                auto cos_pack = Pack::loadu(cos_arr);
                auto sin_pack = Pack::loadu(sin_arr);

                auto t1_x = Pack::loadu(tx_.data() + i);
                auto t1_y = Pack::loadu(ty_.data() + i);
                auto t2_x = Pack::loadu(other.tx_.data() + i);
                auto t2_y = Pack::loadu(other.ty_.data() + i);

                auto rot_t2_x = cos_pack * t2_x - sin_pack * t2_y;
                auto rot_t2_y = sin_pack * t2_x + cos_pack * t2_y;

                auto new_tx = t1_x + rot_t2_x;
                auto new_ty = t1_y + rot_t2_y;

                new_tx.storeu(result.tx_.data() + i);
                new_ty.storeu(result.ty_.data() + i);
            }

            for (; i < N; ++i) {
                auto elem = (*this)[i] * other[i];
                const auto &t = elem.translation();
                result.tx_[i] = t[0];
                result.ty_[i] = t[1];
            }

            return result;
        }

        SE2Batch &operator*=(const SE2Batch &other) noexcept {
            *this = *this * other;
            return *this;
        }

        void transform(T *px, T *py) const noexcept {
            rotations_.rotate(px, py);
            simd::backend::add_runtime<T>(px, px, tx_.data(), N);
            simd::backend::add_runtime<T>(py, py, ty_.data(), N);
        }

        void transform(const T *px_in, const T *py_in, T *px_out, T *py_out) const noexcept {
            simd::backend::copy_runtime<T>(px_out, px_in, N);
            simd::backend::copy_runtime<T>(py_out, py_in, N);
            transform(px_out, py_out);
        }

        void inverse_transform(T *px, T *py) const noexcept {
            simd::backend::sub_runtime<T>(px, px, tx_.data(), N);
            simd::backend::sub_runtime<T>(py, py, ty_.data(), N);
            auto inv_rot = rotations_.inverse();
            inv_rot.rotate(px, py);
        }

        // ===== INTERPOLATION (SIMD) =====

        [[nodiscard]] SE2Batch interpolate(const SE2Batch &other, T t) const noexcept {
            SE2Batch result;
            result.rotations_ = rotations_.interpolate(other.rotations_, t);

            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            auto one_minus_t = Pack(T(1) - t);
            auto t_pack = Pack(t);

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                auto tx1 = Pack::loadu(tx_.data() + i);
                auto ty1 = Pack::loadu(ty_.data() + i);
                auto tx2 = Pack::loadu(other.tx_.data() + i);
                auto ty2 = Pack::loadu(other.ty_.data() + i);

                auto new_tx = one_minus_t * tx1 + t_pack * tx2;
                auto new_ty = one_minus_t * ty1 + t_pack * ty2;

                new_tx.storeu(result.tx_.data() + i);
                new_ty.storeu(result.ty_.data() + i);
            }

            for (; i < N; ++i) {
                result.tx_[i] = (T(1) - t) * tx_[i] + t * other.tx_[i];
                result.ty_[i] = (T(1) - t) * ty_[i] + t * other.ty_[i];
            }

            return result;
        }

        [[nodiscard]] SE2Batch slerp(const SE2Batch &other, T t) const noexcept { return interpolate(other, t); }

        /// Linear interpolation (alias for interpolate)
        [[nodiscard]] SE2Batch lerp(const SE2Batch &other, T t) const noexcept { return interpolate(other, t); }

        // ===== UTILITY METHODS =====

        /// Compute translation norms for all poses
        void translation_norms(T *out) const noexcept {
            constexpr std::size_t W = sizeof(T) == 4 ? 8 : 4;
            using Pack = simd::pack<T, W>;

            std::size_t i = 0;
            for (; i + W <= N; i += W) {
                auto tx_pack = Pack::loadu(tx_.data() + i);
                auto ty_pack = Pack::loadu(ty_.data() + i);

                auto norm_sq = tx_pack * tx_pack + ty_pack * ty_pack;
                auto norm = simd::sqrt(norm_sq);
                norm.storeu(out + i);
            }

            for (; i < N; ++i) {
                out[i] = std::sqrt(tx_[i] * tx_[i] + ty_[i] * ty_[i]);
            }
        }

        // ===== ITERATORS =====

        class Iterator {
            const SE2Batch *batch_;
            std::size_t idx_;

          public:
            Iterator(const SE2Batch *b, std::size_t i) : batch_(b), idx_(i) {}
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

    template <std::size_t N> using SE2Batchf = SE2Batch<float, N>;
    template <std::size_t N> using SE2Batchd = SE2Batch<double, N>;

    using SE2Batch4f = SE2Batch<float, 4>;
    using SE2Batch8f = SE2Batch<float, 8>;
    using SE2Batch16f = SE2Batch<float, 16>;

    using SE2Batch4d = SE2Batch<double, 4>;
    using SE2Batch8d = SE2Batch<double, 8>;
    using SE2Batch16d = SE2Batch<double, 16>;

    // ===== FREE FUNCTIONS =====

    template <typename T, std::size_t N>
    [[nodiscard]] SE2Batch<T, N> interpolate(const SE2Batch<T, N> &a, const SE2Batch<T, N> &b, T t) noexcept {
        return a.interpolate(b, t);
    }

    template <typename T, std::size_t N>
    [[nodiscard]] SE2Batch<T, N> slerp(const SE2Batch<T, N> &a, const SE2Batch<T, N> &b, T t) noexcept {
        return a.slerp(b, t);
    }

} // namespace optinum::lie
