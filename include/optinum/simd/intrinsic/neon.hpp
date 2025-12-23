#pragma once

// =============================================================================
// optinum/simd/intrinsic/neon.hpp
// ARM NEON specializations for SIMDVec<float,4> (and SIMDVec<double,2> when available)
// =============================================================================

#include <optinum/simd/intrinsic/simd_vec.hpp>

#ifdef OPTINUM_HAS_NEON

namespace optinum::simd {

    namespace detail {

        OPTINUM_INLINE float hsum_f32x4(float32x4_t v) noexcept {
#if defined(__aarch64__)
            return vaddvq_f32(v);
#else
            float32x2_t sum2 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
            sum2 = vpadd_f32(sum2, sum2);
            return vget_lane_f32(sum2, 0);
#endif
        }

        OPTINUM_INLINE float hmin_f32x4(float32x4_t v) noexcept {
#if defined(__aarch64__)
            return vminvq_f32(v);
#else
            float32x2_t min2 = vmin_f32(vget_low_f32(v), vget_high_f32(v));
            min2 = vpmin_f32(min2, min2);
            return vget_lane_f32(min2, 0);
#endif
        }

        OPTINUM_INLINE float hmax_f32x4(float32x4_t v) noexcept {
#if defined(__aarch64__)
            return vmaxvq_f32(v);
#else
            float32x2_t max2 = vmax_f32(vget_low_f32(v), vget_high_f32(v));
            max2 = vpmax_f32(max2, max2);
            return vget_lane_f32(max2, 0);
#endif
        }

#if defined(__aarch64__) || defined(__ARM_FEATURE_FP64)
        OPTINUM_INLINE double hsum_f64x2(float64x2_t v) noexcept {
#if defined(__aarch64__)
            return vaddvq_f64(v);
#else
            return vgetq_lane_f64(v, 0) + vgetq_lane_f64(v, 1);
#endif
        }
#endif

    } // namespace detail

    template <> struct SIMDVec<float, 4> {
        using value_type = float;
        using native_type = float32x4_t;
        static constexpr std::size_t width = 4;

        native_type value;

        OPTINUM_INLINE SIMDVec() noexcept : value(vdupq_n_f32(0.f)) {}
        OPTINUM_INLINE explicit SIMDVec(float val) noexcept : value(vdupq_n_f32(val)) {}
        OPTINUM_INLINE explicit SIMDVec(native_type v) noexcept : value(v) {}

        OPTINUM_INLINE SIMDVec &operator=(float val) noexcept {
            value = vdupq_n_f32(val);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator=(native_type v) noexcept {
            value = v;
            return *this;
        }

        OPTINUM_INLINE static SIMDVec load(const float *ptr) noexcept { return SIMDVec(vld1q_f32(ptr)); }
        OPTINUM_INLINE static SIMDVec loadu(const float *ptr) noexcept { return load(ptr); }
        OPTINUM_INLINE void store(float *ptr) const noexcept { vst1q_f32(ptr, value); }
        OPTINUM_INLINE void storeu(float *ptr) const noexcept { store(ptr); }

        OPTINUM_INLINE float operator[](std::size_t i) const noexcept {
            return vgetq_lane_f32(value, static_cast<int>(i));
        }

        OPTINUM_INLINE SIMDVec operator+(const SIMDVec &rhs) const noexcept {
            return SIMDVec(vaddq_f32(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-(const SIMDVec &rhs) const noexcept {
            return SIMDVec(vsubq_f32(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator*(const SIMDVec &rhs) const noexcept {
            return SIMDVec(vmulq_f32(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator/(const SIMDVec &rhs) const noexcept {
#if defined(__aarch64__)
            return SIMDVec(vdivq_f32(value, rhs.value));
#else
            // Approx reciprocal with Newton-Raphson refinement
            float32x4_t recip = vrecpeq_f32(rhs.value);
            recip = vmulq_f32(vrecpsq_f32(rhs.value, recip), recip);
            recip = vmulq_f32(vrecpsq_f32(rhs.value, recip), recip);
            return SIMDVec(vmulq_f32(value, recip));
#endif
        }

        OPTINUM_INLINE SIMDVec operator-() const noexcept { return SIMDVec(vnegq_f32(value)); }

        OPTINUM_INLINE SIMDVec operator+(float rhs) const noexcept {
            return SIMDVec(vaddq_f32(value, vdupq_n_f32(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator-(float rhs) const noexcept {
            return SIMDVec(vsubq_f32(value, vdupq_n_f32(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator*(float rhs) const noexcept {
            return SIMDVec(vmulq_f32(value, vdupq_n_f32(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator/(float rhs) const noexcept { return (*this) / SIMDVec(rhs); }

        OPTINUM_INLINE SIMDVec &operator+=(const SIMDVec &rhs) noexcept {
            value = vaddq_f32(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator-=(const SIMDVec &rhs) noexcept {
            value = vsubq_f32(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator*=(const SIMDVec &rhs) noexcept {
            value = vmulq_f32(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator/=(const SIMDVec &rhs) noexcept {
            *this = (*this) / rhs;
            return *this;
        }

        OPTINUM_INLINE float hsum() const noexcept { return detail::hsum_f32x4(value); }
        OPTINUM_INLINE float hmin() const noexcept { return detail::hmin_f32x4(value); }
        OPTINUM_INLINE float hmax() const noexcept { return detail::hmax_f32x4(value); }

        OPTINUM_INLINE SIMDVec sqrt() const noexcept { return SIMDVec(vsqrtq_f32(value)); }
        OPTINUM_INLINE SIMDVec rsqrt() const noexcept {
            float32x4_t r = vrsqrteq_f32(value);
            r = vmulq_f32(vrsqrtsq_f32(vmulq_f32(value, r), r), r);
            r = vmulq_f32(vrsqrtsq_f32(vmulq_f32(value, r), r), r);
            return SIMDVec(r);
        }

        OPTINUM_INLINE SIMDVec abs() const noexcept { return SIMDVec(vabsq_f32(value)); }
        OPTINUM_INLINE SIMDVec rcp() const noexcept {
            float32x4_t r = vrecpeq_f32(value);
            r = vmulq_f32(vrecpsq_f32(value, r), r);
            r = vmulq_f32(vrecpsq_f32(value, r), r);
            return SIMDVec(r);
        }

        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(vminq_f32(a.value, b.value));
        }
        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(vmaxq_f32(a.value, b.value));
        }

        OPTINUM_INLINE static SIMDVec fma(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
#if defined(__aarch64__)
            return SIMDVec(vfmaq_f32(c.value, a.value, b.value));
#else
            return SIMDVec(vaddq_f32(vmulq_f32(a.value, b.value), c.value));
#endif
        }

        OPTINUM_INLINE static SIMDVec fms(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
#if defined(__aarch64__)
            return SIMDVec(vfmsq_f32(c.value, a.value, b.value));
#else
            return SIMDVec(vsubq_f32(vmulq_f32(a.value, b.value), c.value));
#endif
        }

        OPTINUM_INLINE float dot(const SIMDVec &other) const noexcept {
            return detail::hsum_f32x4(vmulq_f32(value, other.value));
        }
    };

#if defined(__aarch64__) || defined(__ARM_FEATURE_FP64)
    template <> struct SIMDVec<double, 2> {
        using value_type = double;
        using native_type = float64x2_t;
        static constexpr std::size_t width = 2;

        native_type value;

        OPTINUM_INLINE SIMDVec() noexcept : value(vdupq_n_f64(0.0)) {}
        OPTINUM_INLINE explicit SIMDVec(double val) noexcept : value(vdupq_n_f64(val)) {}
        OPTINUM_INLINE explicit SIMDVec(native_type v) noexcept : value(v) {}

        OPTINUM_INLINE SIMDVec &operator=(double val) noexcept {
            value = vdupq_n_f64(val);
            return *this;
        }

        OPTINUM_INLINE static SIMDVec load(const double *ptr) noexcept { return SIMDVec(vld1q_f64(ptr)); }
        OPTINUM_INLINE static SIMDVec loadu(const double *ptr) noexcept { return load(ptr); }
        OPTINUM_INLINE void store(double *ptr) const noexcept { vst1q_f64(ptr, value); }
        OPTINUM_INLINE void storeu(double *ptr) const noexcept { store(ptr); }

        OPTINUM_INLINE double operator[](std::size_t i) const noexcept {
            return vgetq_lane_f64(value, static_cast<int>(i));
        }

        OPTINUM_INLINE SIMDVec operator+(const SIMDVec &rhs) const noexcept {
            return SIMDVec(vaddq_f64(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-(const SIMDVec &rhs) const noexcept {
            return SIMDVec(vsubq_f64(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator*(const SIMDVec &rhs) const noexcept {
            return SIMDVec(vmulq_f64(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator/(const SIMDVec &rhs) const noexcept {
#if defined(__aarch64__)
            return SIMDVec(vdivq_f64(value, rhs.value));
#else
            return SIMDVec(vmulq_f64(value, vrecpeq_f64(rhs.value)));
#endif
        }
        OPTINUM_INLINE SIMDVec operator-() const noexcept { return SIMDVec(vnegq_f64(value)); }

        OPTINUM_INLINE SIMDVec &operator+=(const SIMDVec &rhs) noexcept {
            value = vaddq_f64(value, rhs.value);
            return *this;
        }

        OPTINUM_INLINE double hsum() const noexcept { return detail::hsum_f64x2(value); }
        OPTINUM_INLINE double hmin() const noexcept { return std::min((*this)[0], (*this)[1]); }
        OPTINUM_INLINE double hmax() const noexcept { return std::max((*this)[0], (*this)[1]); }

        OPTINUM_INLINE SIMDVec sqrt() const noexcept {
#if defined(__aarch64__)
            return SIMDVec(vsqrtq_f64(value));
#else
            alignas(16) double tmp[2];
            store(tmp);
            tmp[0] = std::sqrt(tmp[0]);
            tmp[1] = std::sqrt(tmp[1]);
            return SIMDVec::load(tmp);
#endif
        }

        OPTINUM_INLINE SIMDVec rsqrt() const noexcept { return SIMDVec(1.0) / sqrt(); }
        OPTINUM_INLINE SIMDVec abs() const noexcept { return SIMDVec(vabsq_f64(value)); }
        OPTINUM_INLINE SIMDVec rcp() const noexcept { return SIMDVec(1.0) / (*this); }

        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(vminq_f64(a.value, b.value));
        }
        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(vmaxq_f64(a.value, b.value));
        }

        OPTINUM_INLINE static SIMDVec fma(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
#if defined(__aarch64__)
            return SIMDVec(vfmaq_f64(c.value, a.value, b.value));
#else
            return SIMDVec(vaddq_f64(vmulq_f64(a.value, b.value), c.value));
#endif
        }

        OPTINUM_INLINE static SIMDVec fms(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
#if defined(__aarch64__)
            return SIMDVec(vfmsq_f64(c.value, a.value, b.value));
#else
            return SIMDVec(vsubq_f64(vmulq_f64(a.value, b.value), c.value));
#endif
        }

        OPTINUM_INLINE double dot(const SIMDVec &other) const noexcept {
            return detail::hsum_f64x2(vmulq_f64(value, other.value));
        }
    };
#endif

} // namespace optinum::simd

#endif // OPTINUM_HAS_NEON
