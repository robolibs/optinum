#pragma once

// =============================================================================
// optinum/simd/pack/neon.hpp
// ARM NEON specializations for pack<float,4>, pack<double,2>, pack<int32_t,4>, pack<int64_t,2>
// =============================================================================

#include <optinum/simd/pack/pack.hpp>

#ifdef OPTINUM_HAS_NEON

#include <optinum/simd/mask.hpp>

namespace optinum::simd {

    namespace detail {

        // =====================================================================
        // Horizontal operations for NEON (no native reduce instructions)
        // =====================================================================

        // Horizontal sum for float32x4_t
        OPTINUM_INLINE float hsum_f32x4(float32x4_t v) noexcept {
            float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
            sum = vpadd_f32(sum, sum);
            return vget_lane_f32(sum, 0);
        }

        // Horizontal min for float32x4_t
        OPTINUM_INLINE float hmin_f32x4(float32x4_t v) noexcept {
            float32x2_t m = vmin_f32(vget_low_f32(v), vget_high_f32(v));
            m = vpmin_f32(m, m);
            return vget_lane_f32(m, 0);
        }

        // Horizontal max for float32x4_t
        OPTINUM_INLINE float hmax_f32x4(float32x4_t v) noexcept {
            float32x2_t m = vmax_f32(vget_low_f32(v), vget_high_f32(v));
            m = vpmax_f32(m, m);
            return vget_lane_f32(m, 0);
        }

        // Horizontal product for float32x4_t
        OPTINUM_INLINE float hprod_f32x4(float32x4_t v) noexcept {
            float32x2_t prod = vmul_f32(vget_low_f32(v), vget_high_f32(v));
            prod = vpadd_f32(prod, prod); // Not multiply, but we need custom
            // Manual implementation
            float temp[2];
            vst1_f32(temp, prod);
            return temp[0] * temp[1];
        }

#ifdef __aarch64__
        // Horizontal sum for float64x2_t (ARM64 only)
        OPTINUM_INLINE double hsum_f64x2(float64x2_t v) noexcept {
            return vaddvq_f64(v); // ARM64 native reduction
        }

        // Horizontal min for float64x2_t
        OPTINUM_INLINE double hmin_f64x2(float64x2_t v) noexcept { return vminvq_f64(v); }

        // Horizontal max for float64x2_t
        OPTINUM_INLINE double hmax_f64x2(float64x2_t v) noexcept { return vmaxvq_f64(v); }

        // Horizontal product for float64x2_t
        OPTINUM_INLINE double hprod_f64x2(float64x2_t v) noexcept {
            double temp[2];
            vst1q_f64(temp, v);
            return temp[0] * temp[1];
        }
#endif

        // Horizontal sum for int32x4_t
        OPTINUM_INLINE int32_t hsum_s32x4(int32x4_t v) noexcept {
#ifdef __aarch64__
            return vaddvq_s32(v); // ARM64 native reduction
#else
            int32x2_t sum = vadd_s32(vget_low_s32(v), vget_high_s32(v));
            sum = vpadd_s32(sum, sum);
            return vget_lane_s32(sum, 0);
#endif
        }

        // Horizontal min for int32x4_t
        OPTINUM_INLINE int32_t hmin_s32x4(int32x4_t v) noexcept {
#ifdef __aarch64__
            return vminvq_s32(v);
#else
            int32x2_t m = vmin_s32(vget_low_s32(v), vget_high_s32(v));
            m = vpmin_s32(m, m);
            return vget_lane_s32(m, 0);
#endif
        }

        // Horizontal max for int32x4_t
        OPTINUM_INLINE int32_t hmax_s32x4(int32x4_t v) noexcept {
#ifdef __aarch64__
            return vmaxvq_s32(v);
#else
            int32x2_t m = vmax_s32(vget_low_s32(v), vget_high_s32(v));
            m = vpmax_s32(m, m);
            return vget_lane_s32(m, 0);
#endif
        }

        // Horizontal product for int32x4_t
        OPTINUM_INLINE int32_t hprod_s32x4(int32x4_t v) noexcept {
            int32_t temp[4];
            vst1q_s32(temp, v);
            return temp[0] * temp[1] * temp[2] * temp[3];
        }

#ifdef __aarch64__
        // Horizontal sum for int64x2_t
        OPTINUM_INLINE int64_t hsum_s64x2(int64x2_t v) noexcept { return vaddvq_s64(v); }

        // Horizontal min for int64x2_t
        OPTINUM_INLINE int64_t hmin_s64x2(int64x2_t v) noexcept {
            int64_t temp[2];
            vst1q_s64(temp, v);
            return temp[0] < temp[1] ? temp[0] : temp[1];
        }

        // Horizontal max for int64x2_t
        OPTINUM_INLINE int64_t hmax_s64x2(int64x2_t v) noexcept {
            int64_t temp[2];
            vst1q_s64(temp, v);
            return temp[0] > temp[1] ? temp[0] : temp[1];
        }

        // Horizontal product for int64x2_t
        OPTINUM_INLINE int64_t hprod_s64x2(int64x2_t v) noexcept {
            int64_t temp[2];
            vst1q_s64(temp, v);
            return temp[0] * temp[1];
        }
#endif

        // Negate float32x4_t using XOR (flip sign bit)
        OPTINUM_INLINE float32x4_t neg_f32x4(float32x4_t v) noexcept {
            return vnegq_f32(v); // NEON has native negate
        }

#ifdef __aarch64__
        // Negate float64x2_t
        OPTINUM_INLINE float64x2_t neg_f64x2(float64x2_t v) noexcept { return vnegq_f64(v); }
#endif

        // Negate int32x4_t
        OPTINUM_INLINE int32x4_t neg_s32x4(int32x4_t v) noexcept { return vnegq_s32(v); }

#ifdef __aarch64__
        // Negate int64x2_t
        OPTINUM_INLINE int64x2_t neg_s64x2(int64x2_t v) noexcept { return vnegq_s64(v); }
#endif

    } // namespace detail

    // =============================================================================
    // pack<float, 4> - ARM NEON (4 x 32-bit float)
    // =============================================================================

    template <> struct pack<float, 4> {
        using value_type = float;
        using native_type = float32x4_t;
        static constexpr std::size_t width = 4;

        native_type data_;

        // Constructors
        OPTINUM_INLINE pack() noexcept : data_(vdupq_n_f32(0.0f)) {}
        OPTINUM_INLINE explicit pack(float val) noexcept : data_(vdupq_n_f32(val)) {}
        OPTINUM_INLINE explicit pack(native_type v) noexcept : data_(v) {}

        OPTINUM_INLINE pack(float v0, float v1, float v2, float v3) noexcept {
            alignas(16) float temp[4] = {v0, v1, v2, v3};
            data_ = vld1q_f32(temp);
        }

        // Load/Store
        static OPTINUM_INLINE pack load_aligned(const float *ptr) noexcept { return pack(vld1q_f32(ptr)); }

        static OPTINUM_INLINE pack load_unaligned(const float *ptr) noexcept { return pack(vld1q_f32(ptr)); }

        OPTINUM_INLINE void store_aligned(float *ptr) const noexcept { vst1q_f32(ptr, data_); }

        OPTINUM_INLINE void store_unaligned(float *ptr) const noexcept { vst1q_f32(ptr, data_); }

        // Accessors
        OPTINUM_INLINE native_type native() const noexcept { return data_; }
        OPTINUM_INLINE native_type &native() noexcept { return data_; }

        // Element access
        template <int I> OPTINUM_INLINE float get() const noexcept {
            static_assert(I >= 0 && I < 4, "Index out of range");
            return vgetq_lane_f32(data_, I);
        }

        OPTINUM_INLINE float operator[](std::size_t i) const noexcept {
            alignas(16) float temp[4];
            vst1q_f32(temp, data_);
            return temp[i];
        }

        // Arithmetic operators
        OPTINUM_INLINE pack operator+(const pack &other) const noexcept { return pack(vaddq_f32(data_, other.data_)); }

        OPTINUM_INLINE pack operator-(const pack &other) const noexcept { return pack(vsubq_f32(data_, other.data_)); }

        OPTINUM_INLINE pack operator*(const pack &other) const noexcept { return pack(vmulq_f32(data_, other.data_)); }

        OPTINUM_INLINE pack operator/(const pack &other) const noexcept {
#ifdef __aarch64__
            return pack(vdivq_f32(data_, other.data_)); // ARMv8 only
#else
            // Scalar fallback for ARMv7
            alignas(16) float a[4], b[4], result[4];
            vst1q_f32(a, data_);
            vst1q_f32(b, other.data_);
            for (int i = 0; i < 4; ++i)
                result[i] = a[i] / b[i];
            return pack(vld1q_f32(result));
#endif
        }

        OPTINUM_INLINE pack operator-() const noexcept { return pack(detail::neg_f32x4(data_)); }

        // Compound assignment
        OPTINUM_INLINE pack &operator+=(const pack &other) noexcept {
            data_ = vaddq_f32(data_, other.data_);
            return *this;
        }

        OPTINUM_INLINE pack &operator-=(const pack &other) noexcept {
            data_ = vsubq_f32(data_, other.data_);
            return *this;
        }

        OPTINUM_INLINE pack &operator*=(const pack &other) noexcept {
            data_ = vmulq_f32(data_, other.data_);
            return *this;
        }

        OPTINUM_INLINE pack &operator/=(const pack &other) noexcept {
#ifdef __aarch64__
            data_ = vdivq_f32(data_, other.data_);
#else
            *this = *this / other; // Use operator/ with fallback
#endif
            return *this;
        }

        // Horizontal operations
        OPTINUM_INLINE float hsum() const noexcept { return detail::hsum_f32x4(data_); }

        OPTINUM_INLINE float hmin() const noexcept { return detail::hmin_f32x4(data_); }

        OPTINUM_INLINE float hmax() const noexcept { return detail::hmax_f32x4(data_); }

        OPTINUM_INLINE float hprod() const noexcept { return detail::hprod_f32x4(data_); }

        // Math operations
        OPTINUM_INLINE pack sqrt() const noexcept {
#ifdef __aarch64__
            return pack(vsqrtq_f32(data_)); // ARMv8 only
#else
            // Scalar fallback
            alignas(16) float temp[4];
            vst1q_f32(temp, data_);
            for (int i = 0; i < 4; ++i)
                temp[i] = std::sqrt(temp[i]);
            return pack(vld1q_f32(temp));
#endif
        }

        OPTINUM_INLINE pack rsqrt() const noexcept {
            float32x4_t estimate = vrsqrteq_f32(data_);
            // Newton-Raphson refinement: x1 = x0 * (3 - a * x0^2) / 2
            estimate = vmulq_f32(estimate, vrsqrtsq_f32(vmulq_f32(data_, estimate), estimate));
            return pack(estimate);
        }

        OPTINUM_INLINE pack abs() const noexcept { return pack(vabsq_f32(data_)); }

        OPTINUM_INLINE pack rcp() const noexcept {
            float32x4_t estimate = vrecpeq_f32(data_);
            // Newton-Raphson refinement
            estimate = vmulq_f32(estimate, vrecpsq_f32(data_, estimate));
            return pack(estimate);
        }

        // Min/Max
        OPTINUM_INLINE pack min(const pack &other) const noexcept { return pack(vminq_f32(data_, other.data_)); }

        OPTINUM_INLINE pack max(const pack &other) const noexcept { return pack(vmaxq_f32(data_, other.data_)); }

        // FMA/FMS
        OPTINUM_INLINE pack fmadd(const pack &b, const pack &c) const noexcept {
#ifdef __aarch64__
            return pack(vfmaq_f32(c.data_, data_, b.data_)); // c + a * b
#else
            return *this * b + c;
#endif
        }

        OPTINUM_INLINE pack fmsub(const pack &b, const pack &c) const noexcept {
#ifdef __aarch64__
            return pack(vfmsq_f32(c.data_, data_, b.data_)); // c - a * b
#else
            return c - *this * b;
#endif
        }

        // Dot product
        OPTINUM_INLINE float dot(const pack &other) const noexcept { return (*this * other).hsum(); }

        // Factory functions
        static OPTINUM_INLINE pack set(float v0, float v1, float v2, float v3) noexcept { return pack(v0, v1, v2, v3); }

        static OPTINUM_INLINE pack set_sequential(float start = 0.0f, float step = 1.0f) noexcept {
            return pack(start, start + step, start + 2 * step, start + 3 * step);
        }

        // Permutations
        OPTINUM_INLINE pack reverse() const noexcept {
            // Reverse using vrev64q + vextq
            float32x4_t rev64 = vrev64q_f32(data_);  // [1,0,3,2]
            return pack(vextq_f32(rev64, rev64, 2)); // [3,2,1,0]
        }

        template <int N> OPTINUM_INLINE pack rotate() const noexcept {
            constexpr int shift = N & 3; // Modulo 4
            if constexpr (shift == 0) {
                return *this;
            } else if constexpr (shift == 1) {
                return pack(vextq_f32(data_, data_, 1)); // [1,2,3,0]
            } else if constexpr (shift == 2) {
                return pack(vextq_f32(data_, data_, 2)); // [2,3,0,1]
            } else {                                     // shift == 3
                return pack(vextq_f32(data_, data_, 3)); // [3,0,1,2]
            }
        }

        template <int N> OPTINUM_INLINE pack shift() const noexcept {
            constexpr int shift = N & 3;
            if constexpr (shift == 0) {
                return *this;
            } else if constexpr (shift == 1) {
                float32x4_t zero = vdupq_n_f32(0.0f);
                return pack(vextq_f32(zero, data_, 3)); // [0,0,1,2]
            } else if constexpr (shift == 2) {
                float32x4_t zero = vdupq_n_f32(0.0f);
                return pack(vextq_f32(zero, data_, 2)); // [0,0,0,1]
            } else {                                    // shift == 3
                float32x4_t zero = vdupq_n_f32(0.0f);
                return pack(vextq_f32(zero, data_, 1)); // [0,0,0,0]
            }
        }

        // Gather/Scatter - NEON doesn't have native gather/scatter, use scalar fallback
        static OPTINUM_INLINE pack gather(const float *base, const pack<int32_t, 4> &indices) noexcept {
            alignas(16) int32_t idx[4];
            vst1q_s32(idx, indices.native());
            alignas(16) float result[4];
            for (int i = 0; i < 4; ++i)
                result[i] = base[idx[i]];
            return pack(vld1q_f32(result));
        }

        OPTINUM_INLINE void scatter(float *base, const pack<int32_t, 4> &indices) const noexcept {
            alignas(16) int32_t idx[4];
            alignas(16) float values[4];
            vst1q_s32(idx, indices.native());
            vst1q_f32(values, data_);
            for (int i = 0; i < 4; ++i)
                base[idx[i]] = values[i];
        }

        // Cast to int
        OPTINUM_INLINE pack<int32_t, 4> cast_to_int() const noexcept { return pack<int32_t, 4>(vcvtq_s32_f32(data_)); }
    };

#ifdef __aarch64__
    // =============================================================================
    // pack<double, 2> - ARM NEON (2 x 64-bit double) - ARM64 only
    // =============================================================================

    template <> struct pack<double, 2> {
        using value_type = double;
        using native_type = float64x2_t;
        static constexpr std::size_t width = 2;

        native_type data_;

        // Constructors
        OPTINUM_INLINE pack() noexcept : data_(vdupq_n_f64(0.0)) {}
        OPTINUM_INLINE explicit pack(double val) noexcept : data_(vdupq_n_f64(val)) {}
        OPTINUM_INLINE explicit pack(native_type v) noexcept : data_(v) {}

        OPTINUM_INLINE pack(double v0, double v1) noexcept {
            alignas(16) double temp[2] = {v0, v1};
            data_ = vld1q_f64(temp);
        }

        // Load/Store
        static OPTINUM_INLINE pack load_aligned(const double *ptr) noexcept { return pack(vld1q_f64(ptr)); }

        static OPTINUM_INLINE pack load_unaligned(const double *ptr) noexcept { return pack(vld1q_f64(ptr)); }

        OPTINUM_INLINE void store_aligned(double *ptr) const noexcept { vst1q_f64(ptr, data_); }

        OPTINUM_INLINE void store_unaligned(double *ptr) const noexcept { vst1q_f64(ptr, data_); }

        // Accessors
        OPTINUM_INLINE native_type native() const noexcept { return data_; }
        OPTINUM_INLINE native_type &native() noexcept { return data_; }

        // Element access
        template <int I> OPTINUM_INLINE double get() const noexcept {
            static_assert(I >= 0 && I < 2, "Index out of range");
            return vgetq_lane_f64(data_, I);
        }

        OPTINUM_INLINE double operator[](std::size_t i) const noexcept {
            alignas(16) double temp[2];
            vst1q_f64(temp, data_);
            return temp[i];
        }

        // Arithmetic operators
        OPTINUM_INLINE pack operator+(const pack &other) const noexcept { return pack(vaddq_f64(data_, other.data_)); }

        OPTINUM_INLINE pack operator-(const pack &other) const noexcept { return pack(vsubq_f64(data_, other.data_)); }

        OPTINUM_INLINE pack operator*(const pack &other) const noexcept { return pack(vmulq_f64(data_, other.data_)); }

        OPTINUM_INLINE pack operator/(const pack &other) const noexcept { return pack(vdivq_f64(data_, other.data_)); }

        OPTINUM_INLINE pack operator-() const noexcept { return pack(detail::neg_f64x2(data_)); }

        // Compound assignment
        OPTINUM_INLINE pack &operator+=(const pack &other) noexcept {
            data_ = vaddq_f64(data_, other.data_);
            return *this;
        }

        OPTINUM_INLINE pack &operator-=(const pack &other) noexcept {
            data_ = vsubq_f64(data_, other.data_);
            return *this;
        }

        OPTINUM_INLINE pack &operator*=(const pack &other) noexcept {
            data_ = vmulq_f64(data_, other.data_);
            return *this;
        }

        OPTINUM_INLINE pack &operator/=(const pack &other) noexcept {
            data_ = vdivq_f64(data_, other.data_);
            return *this;
        }

        // Horizontal operations
        OPTINUM_INLINE double hsum() const noexcept { return detail::hsum_f64x2(data_); }

        OPTINUM_INLINE double hmin() const noexcept { return detail::hmin_f64x2(data_); }

        OPTINUM_INLINE double hmax() const noexcept { return detail::hmax_f64x2(data_); }

        OPTINUM_INLINE double hprod() const noexcept { return detail::hprod_f64x2(data_); }

        // Math operations
        OPTINUM_INLINE pack sqrt() const noexcept { return pack(vsqrtq_f64(data_)); }

        OPTINUM_INLINE pack rsqrt() const noexcept {
            // NEON doesn't have f64 rsqrt estimate, use 1/sqrt
            return pack(vdivq_f64(vdupq_n_f64(1.0), vsqrtq_f64(data_)));
        }

        OPTINUM_INLINE pack abs() const noexcept { return pack(vabsq_f64(data_)); }

        OPTINUM_INLINE pack rcp() const noexcept {
            // NEON doesn't have f64 rcp estimate
            return pack(vdivq_f64(vdupq_n_f64(1.0), data_));
        }

        // Min/Max
        OPTINUM_INLINE pack min(const pack &other) const noexcept { return pack(vminq_f64(data_, other.data_)); }

        OPTINUM_INLINE pack max(const pack &other) const noexcept { return pack(vmaxq_f64(data_, other.data_)); }

        // FMA/FMS
        OPTINUM_INLINE pack fmadd(const pack &b, const pack &c) const noexcept {
            return pack(vfmaq_f64(c.data_, data_, b.data_)); // c + a * b
        }

        OPTINUM_INLINE pack fmsub(const pack &b, const pack &c) const noexcept {
            return pack(vfmsq_f64(c.data_, data_, b.data_)); // c - a * b
        }

        // Dot product
        OPTINUM_INLINE double dot(const pack &other) const noexcept { return (*this * other).hsum(); }

        // Factory functions
        static OPTINUM_INLINE pack set(double v0, double v1) noexcept { return pack(v0, v1); }

        static OPTINUM_INLINE pack set_sequential(double start = 0.0, double step = 1.0) noexcept {
            return pack(start, start + step);
        }

        // Permutations
        OPTINUM_INLINE pack reverse() const noexcept {
            alignas(16) double temp[2];
            vst1q_f64(temp, data_);
            return pack(temp[1], temp[0]);
        }

        template <int N> OPTINUM_INLINE pack rotate() const noexcept {
            if constexpr ((N & 1) == 0) {
                return *this;
            } else {
                return reverse();
            }
        }

        template <int N> OPTINUM_INLINE pack shift() const noexcept {
            if constexpr ((N & 1) == 0) {
                return *this;
            } else {
                return pack(0.0, vgetq_lane_f64(data_, 0));
            }
        }

        // Gather/Scatter - scalar fallback
        static OPTINUM_INLINE pack gather(const double *base, const pack<int64_t, 2> &indices) noexcept {
            alignas(16) int64_t idx[2];
            vst1q_s64(idx, indices.native());
            return pack(base[idx[0]], base[idx[1]]);
        }

        OPTINUM_INLINE void scatter(double *base, const pack<int64_t, 2> &indices) const noexcept {
            alignas(16) int64_t idx[2];
            alignas(16) double values[2];
            vst1q_s64(idx, indices.native());
            vst1q_f64(values, data_);
            base[idx[0]] = values[0];
            base[idx[1]] = values[1];
        }

        // Cast to int
        OPTINUM_INLINE pack<int64_t, 2> cast_to_int() const noexcept { return pack<int64_t, 2>(vcvtq_s64_f64(data_)); }
    };
#endif // __aarch64__

    // =============================================================================
    // pack<int32_t, 4> - ARM NEON (4 x 32-bit int)
    // =============================================================================

    template <> struct pack<int32_t, 4> {
        using value_type = int32_t;
        using native_type = int32x4_t;
        static constexpr std::size_t width = 4;

        native_type data_;

        // Constructors
        OPTINUM_INLINE pack() noexcept : data_(vdupq_n_s32(0)) {}
        OPTINUM_INLINE explicit pack(int32_t val) noexcept : data_(vdupq_n_s32(val)) {}
        OPTINUM_INLINE explicit pack(native_type v) noexcept : data_(v) {}

        OPTINUM_INLINE pack(int32_t v0, int32_t v1, int32_t v2, int32_t v3) noexcept {
            alignas(16) int32_t temp[4] = {v0, v1, v2, v3};
            data_ = vld1q_s32(temp);
        }

        // Load/Store
        static OPTINUM_INLINE pack load_aligned(const int32_t *ptr) noexcept { return pack(vld1q_s32(ptr)); }

        static OPTINUM_INLINE pack load_unaligned(const int32_t *ptr) noexcept { return pack(vld1q_s32(ptr)); }

        OPTINUM_INLINE void store_aligned(int32_t *ptr) const noexcept { vst1q_s32(ptr, data_); }

        OPTINUM_INLINE void store_unaligned(int32_t *ptr) const noexcept { vst1q_s32(ptr, data_); }

        // Accessors
        OPTINUM_INLINE native_type native() const noexcept { return data_; }
        OPTINUM_INLINE native_type &native() noexcept { return data_; }

        // Element access
        template <int I> OPTINUM_INLINE int32_t get() const noexcept {
            static_assert(I >= 0 && I < 4, "Index out of range");
            return vgetq_lane_s32(data_, I);
        }

        OPTINUM_INLINE int32_t operator[](std::size_t i) const noexcept {
            alignas(16) int32_t temp[4];
            vst1q_s32(temp, data_);
            return temp[i];
        }

        // Arithmetic operators
        OPTINUM_INLINE pack operator+(const pack &other) const noexcept { return pack(vaddq_s32(data_, other.data_)); }

        OPTINUM_INLINE pack operator-(const pack &other) const noexcept { return pack(vsubq_s32(data_, other.data_)); }

        OPTINUM_INLINE pack operator*(const pack &other) const noexcept { return pack(vmulq_s32(data_, other.data_)); }

        OPTINUM_INLINE pack operator-() const noexcept { return pack(detail::neg_s32x4(data_)); }

        // Compound assignment
        OPTINUM_INLINE pack &operator+=(const pack &other) noexcept {
            data_ = vaddq_s32(data_, other.data_);
            return *this;
        }

        OPTINUM_INLINE pack &operator-=(const pack &other) noexcept {
            data_ = vsubq_s32(data_, other.data_);
            return *this;
        }

        OPTINUM_INLINE pack &operator*=(const pack &other) noexcept {
            data_ = vmulq_s32(data_, other.data_);
            return *this;
        }

        // Bitwise operators
        OPTINUM_INLINE pack operator&(const pack &other) const noexcept { return pack(vandq_s32(data_, other.data_)); }

        OPTINUM_INLINE pack operator|(const pack &other) const noexcept { return pack(vorrq_s32(data_, other.data_)); }

        OPTINUM_INLINE pack operator^(const pack &other) const noexcept { return pack(veorq_s32(data_, other.data_)); }

        OPTINUM_INLINE pack operator~() const noexcept { return pack(vmvnq_s32(data_)); }

        // Bit shifts
        OPTINUM_INLINE pack operator<<(int n) const noexcept { return pack(vshlq_n_s32(data_, n)); }

        OPTINUM_INLINE pack operator>>(int n) const noexcept {
            return pack(vshrq_n_s32(data_, n)); // Arithmetic shift
        }

        OPTINUM_INLINE pack shr_logical(int n) const noexcept {
            return pack(vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(data_), n)));
        }

        // Horizontal operations
        OPTINUM_INLINE int32_t hsum() const noexcept { return detail::hsum_s32x4(data_); }

        OPTINUM_INLINE int32_t hmin() const noexcept { return detail::hmin_s32x4(data_); }

        OPTINUM_INLINE int32_t hmax() const noexcept { return detail::hmax_s32x4(data_); }

        OPTINUM_INLINE int32_t hprod() const noexcept { return detail::hprod_s32x4(data_); }

        // Math operations
        OPTINUM_INLINE pack abs() const noexcept { return pack(vabsq_s32(data_)); }

        OPTINUM_INLINE pack min(const pack &other) const noexcept { return pack(vminq_s32(data_, other.data_)); }

        OPTINUM_INLINE pack max(const pack &other) const noexcept { return pack(vmaxq_s32(data_, other.data_)); }

        // Dot product
        OPTINUM_INLINE int32_t dot(const pack &other) const noexcept { return (*this * other).hsum(); }

        // Factory functions
        static OPTINUM_INLINE pack set(int32_t v0, int32_t v1, int32_t v2, int32_t v3) noexcept {
            return pack(v0, v1, v2, v3);
        }

        static OPTINUM_INLINE pack set_sequential(int32_t start = 0, int32_t step = 1) noexcept {
            return pack(start, start + step, start + 2 * step, start + 3 * step);
        }

        // Permutations
        OPTINUM_INLINE pack reverse() const noexcept {
            int32x4_t rev64 = vrev64q_s32(data_);
            return pack(vextq_s32(rev64, rev64, 2));
        }

        template <int N> OPTINUM_INLINE pack rotate() const noexcept {
            constexpr int shift = N & 3;
            if constexpr (shift == 0) {
                return *this;
            } else if constexpr (shift == 1) {
                return pack(vextq_s32(data_, data_, 1));
            } else if constexpr (shift == 2) {
                return pack(vextq_s32(data_, data_, 2));
            } else {
                return pack(vextq_s32(data_, data_, 3));
            }
        }

        template <int N> OPTINUM_INLINE pack shift() const noexcept {
            constexpr int shift = N & 3;
            if constexpr (shift == 0) {
                return *this;
            } else if constexpr (shift == 1) {
                int32x4_t zero = vdupq_n_s32(0);
                return pack(vextq_s32(zero, data_, 3));
            } else if constexpr (shift == 2) {
                int32x4_t zero = vdupq_n_s32(0);
                return pack(vextq_s32(zero, data_, 2));
            } else {
                int32x4_t zero = vdupq_n_s32(0);
                return pack(vextq_s32(zero, data_, 1));
            }
        }

        // Cast to float
        OPTINUM_INLINE pack<float, 4> cast_to_float() const noexcept { return pack<float, 4>(vcvtq_f32_s32(data_)); }
    };

#ifdef __aarch64__
    // =============================================================================
    // pack<int64_t, 2> - ARM NEON (2 x 64-bit int) - ARM64 only
    // =============================================================================

    template <> struct pack<int64_t, 2> {
        using value_type = int64_t;
        using native_type = int64x2_t;
        static constexpr std::size_t width = 2;

        native_type data_;

        // Constructors
        OPTINUM_INLINE pack() noexcept : data_(vdupq_n_s64(0)) {}
        OPTINUM_INLINE explicit pack(int64_t val) noexcept : data_(vdupq_n_s64(val)) {}
        OPTINUM_INLINE explicit pack(native_type v) noexcept : data_(v) {}

        OPTINUM_INLINE pack(int64_t v0, int64_t v1) noexcept {
            alignas(16) int64_t temp[2] = {v0, v1};
            data_ = vld1q_s64(temp);
        }

        // Load/Store
        static OPTINUM_INLINE pack load_aligned(const int64_t *ptr) noexcept { return pack(vld1q_s64(ptr)); }

        static OPTINUM_INLINE pack load_unaligned(const int64_t *ptr) noexcept { return pack(vld1q_s64(ptr)); }

        OPTINUM_INLINE void store_aligned(int64_t *ptr) const noexcept { vst1q_s64(ptr, data_); }

        OPTINUM_INLINE void store_unaligned(int64_t *ptr) const noexcept { vst1q_s64(ptr, data_); }

        // Accessors
        OPTINUM_INLINE native_type native() const noexcept { return data_; }
        OPTINUM_INLINE native_type &native() noexcept { return data_; }

        // Element access
        template <int I> OPTINUM_INLINE int64_t get() const noexcept {
            static_assert(I >= 0 && I < 2, "Index out of range");
            return vgetq_lane_s64(data_, I);
        }

        OPTINUM_INLINE int64_t operator[](std::size_t i) const noexcept {
            alignas(16) int64_t temp[2];
            vst1q_s64(temp, data_);
            return temp[i];
        }

        // Arithmetic operators
        OPTINUM_INLINE pack operator+(const pack &other) const noexcept { return pack(vaddq_s64(data_, other.data_)); }

        OPTINUM_INLINE pack operator-(const pack &other) const noexcept { return pack(vsubq_s64(data_, other.data_)); }

        OPTINUM_INLINE pack operator*(const pack &other) const noexcept {
            // NEON doesn't have native 64-bit multiply - scalar fallback
            alignas(16) int64_t a[2], b[2], result[2];
            vst1q_s64(a, data_);
            vst1q_s64(b, other.data_);
            result[0] = a[0] * b[0];
            result[1] = a[1] * b[1];
            return pack(vld1q_s64(result));
        }

        OPTINUM_INLINE pack operator-() const noexcept { return pack(detail::neg_s64x2(data_)); }

        // Compound assignment
        OPTINUM_INLINE pack &operator+=(const pack &other) noexcept {
            data_ = vaddq_s64(data_, other.data_);
            return *this;
        }

        OPTINUM_INLINE pack &operator-=(const pack &other) noexcept {
            data_ = vsubq_s64(data_, other.data_);
            return *this;
        }

        OPTINUM_INLINE pack &operator*=(const pack &other) noexcept {
            *this = *this * other;
            return *this;
        }

        // Bitwise operators
        OPTINUM_INLINE pack operator&(const pack &other) const noexcept { return pack(vandq_s64(data_, other.data_)); }

        OPTINUM_INLINE pack operator|(const pack &other) const noexcept { return pack(vorrq_s64(data_, other.data_)); }

        OPTINUM_INLINE pack operator^(const pack &other) const noexcept { return pack(veorq_s64(data_, other.data_)); }

        OPTINUM_INLINE pack operator~() const noexcept {
            return pack(vreinterpretq_s64_u64(vmvnq_u64(vreinterpretq_u64_s64(data_))));
        }

        // Bit shifts
        OPTINUM_INLINE pack operator<<(int n) const noexcept { return pack(vshlq_n_s64(data_, n)); }

        OPTINUM_INLINE pack operator>>(int n) const noexcept {
            // NEON doesn't have native 64-bit arithmetic right shift - scalar fallback
            alignas(16) int64_t temp[2];
            vst1q_s64(temp, data_);
            temp[0] >>= n;
            temp[1] >>= n;
            return pack(vld1q_s64(temp));
        }

        OPTINUM_INLINE pack shr_logical(int n) const noexcept {
            return pack(vreinterpretq_s64_u64(vshrq_n_u64(vreinterpretq_u64_s64(data_), n)));
        }

        // Horizontal operations
        OPTINUM_INLINE int64_t hsum() const noexcept { return detail::hsum_s64x2(data_); }

        OPTINUM_INLINE int64_t hmin() const noexcept { return detail::hmin_s64x2(data_); }

        OPTINUM_INLINE int64_t hmax() const noexcept { return detail::hmax_s64x2(data_); }

        OPTINUM_INLINE int64_t hprod() const noexcept { return detail::hprod_s64x2(data_); }

        // Math operations
        OPTINUM_INLINE pack abs() const noexcept { return pack(vabsq_s64(data_)); }

        OPTINUM_INLINE pack min(const pack &other) const noexcept {
            // NEON doesn't have native 64-bit min - scalar fallback
            alignas(16) int64_t a[2], b[2], result[2];
            vst1q_s64(a, data_);
            vst1q_s64(b, other.data_);
            result[0] = a[0] < b[0] ? a[0] : b[0];
            result[1] = a[1] < b[1] ? a[1] : b[1];
            return pack(vld1q_s64(result));
        }

        OPTINUM_INLINE pack max(const pack &other) const noexcept {
            alignas(16) int64_t a[2], b[2], result[2];
            vst1q_s64(a, data_);
            vst1q_s64(b, other.data_);
            result[0] = a[0] > b[0] ? a[0] : b[0];
            result[1] = a[1] > b[1] ? a[1] : b[1];
            return pack(vld1q_s64(result));
        }

        // Dot product
        OPTINUM_INLINE int64_t dot(const pack &other) const noexcept { return (*this * other).hsum(); }

        // Factory functions
        static OPTINUM_INLINE pack set(int64_t v0, int64_t v1) noexcept { return pack(v0, v1); }

        static OPTINUM_INLINE pack set_sequential(int64_t start = 0, int64_t step = 1) noexcept {
            return pack(start, start + step);
        }

        // Permutations
        OPTINUM_INLINE pack reverse() const noexcept {
            alignas(16) int64_t temp[2];
            vst1q_s64(temp, data_);
            return pack(temp[1], temp[0]);
        }

        template <int N> OPTINUM_INLINE pack rotate() const noexcept {
            if constexpr ((N & 1) == 0) {
                return *this;
            } else {
                return reverse();
            }
        }

        template <int N> OPTINUM_INLINE pack shift() const noexcept {
            if constexpr ((N & 1) == 0) {
                return *this;
            } else {
                return pack(0, vgetq_lane_s64(data_, 0));
            }
        }

        // Cast to double
        OPTINUM_INLINE pack<double, 2> cast_to_float() const noexcept { return pack<double, 2>(vcvtq_f64_s64(data_)); }
    };
#endif // __aarch64__

} // namespace optinum::simd

#endif // OPTINUM_HAS_NEON
