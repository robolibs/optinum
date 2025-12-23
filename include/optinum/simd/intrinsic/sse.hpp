#pragma once

// =============================================================================
// optinum/simd/intrinsic/sse.hpp
// SSE specializations for SIMDVec<float,4> and SIMDVec<double,2>
// =============================================================================

#include <optinum/simd/intrinsic/simd_vec.hpp>

#ifdef OPTINUM_HAS_SSE2

namespace optinum::simd {

    namespace detail {

        OPTINUM_INLINE float hsum_ps(__m128 v) noexcept {
#ifdef OPTINUM_HAS_SSE3
            __m128 shuf = _mm_movehdup_ps(v);
            __m128 sums = _mm_add_ps(v, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
#else
            __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 sums = _mm_add_ps(v, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
#endif
        }

        OPTINUM_INLINE double hsum_pd(__m128d v) noexcept {
            __m128d shuf = _mm_shuffle_pd(v, v, 1);
            __m128d sums = _mm_add_pd(v, shuf);
            return _mm_cvtsd_f64(sums);
        }

        OPTINUM_INLINE __m128 neg_ps(__m128 v) noexcept { return _mm_xor_ps(v, _mm_set1_ps(-0.0f)); }
        OPTINUM_INLINE __m128d neg_pd(__m128d v) noexcept { return _mm_xor_pd(v, _mm_set1_pd(-0.0)); }

        OPTINUM_INLINE __m128 abs_ps(__m128 v) noexcept { return _mm_andnot_ps(_mm_set1_ps(-0.0f), v); }
        OPTINUM_INLINE __m128d abs_pd(__m128d v) noexcept { return _mm_andnot_pd(_mm_set1_pd(-0.0), v); }

    } // namespace detail

    template <> struct SIMDVec<float, 4> {
        using value_type = float;
        using native_type = __m128;
        static constexpr std::size_t width = 4;

        native_type value;

        OPTINUM_INLINE SIMDVec() noexcept : value(_mm_setzero_ps()) {}
        OPTINUM_INLINE explicit SIMDVec(float val) noexcept : value(_mm_set1_ps(val)) {}
        OPTINUM_INLINE explicit SIMDVec(native_type v) noexcept : value(v) {}
        OPTINUM_INLINE SIMDVec(float a, float b, float c, float d) noexcept : value(_mm_setr_ps(a, b, c, d)) {}

        OPTINUM_INLINE SIMDVec &operator=(float val) noexcept {
            value = _mm_set1_ps(val);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator=(native_type v) noexcept {
            value = v;
            return *this;
        }

        OPTINUM_INLINE static SIMDVec load(const float *ptr) noexcept { return SIMDVec(_mm_load_ps(ptr)); }
        OPTINUM_INLINE static SIMDVec loadu(const float *ptr) noexcept { return SIMDVec(_mm_loadu_ps(ptr)); }
        OPTINUM_INLINE void store(float *ptr) const noexcept { _mm_store_ps(ptr, value); }
        OPTINUM_INLINE void storeu(float *ptr) const noexcept { _mm_storeu_ps(ptr, value); }

        OPTINUM_INLINE float operator[](std::size_t i) const noexcept {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, value);
            return tmp[i];
        }

        OPTINUM_INLINE SIMDVec operator+(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_add_ps(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_sub_ps(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator*(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_mul_ps(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator/(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_div_ps(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-() const noexcept { return SIMDVec(detail::neg_ps(value)); }

        OPTINUM_INLINE SIMDVec operator+(float rhs) const noexcept {
            return SIMDVec(_mm_add_ps(value, _mm_set1_ps(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator-(float rhs) const noexcept {
            return SIMDVec(_mm_sub_ps(value, _mm_set1_ps(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator*(float rhs) const noexcept {
            return SIMDVec(_mm_mul_ps(value, _mm_set1_ps(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator/(float rhs) const noexcept {
            return SIMDVec(_mm_div_ps(value, _mm_set1_ps(rhs)));
        }

        OPTINUM_INLINE SIMDVec &operator+=(const SIMDVec &rhs) noexcept {
            value = _mm_add_ps(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator-=(const SIMDVec &rhs) noexcept {
            value = _mm_sub_ps(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator*=(const SIMDVec &rhs) noexcept {
            value = _mm_mul_ps(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator/=(const SIMDVec &rhs) noexcept {
            value = _mm_div_ps(value, rhs.value);
            return *this;
        }

        OPTINUM_INLINE float hsum() const noexcept { return detail::hsum_ps(value); }

        OPTINUM_INLINE float hmin() const noexcept {
            __m128 tmp = _mm_min_ps(value, _mm_shuffle_ps(value, value, _MM_SHUFFLE(2, 3, 0, 1)));
            tmp = _mm_min_ps(tmp, _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_cvtss_f32(tmp);
        }

        OPTINUM_INLINE float hmax() const noexcept {
            __m128 tmp = _mm_max_ps(value, _mm_shuffle_ps(value, value, _MM_SHUFFLE(2, 3, 0, 1)));
            tmp = _mm_max_ps(tmp, _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_cvtss_f32(tmp);
        }

        OPTINUM_INLINE SIMDVec sqrt() const noexcept { return SIMDVec(_mm_sqrt_ps(value)); }
        OPTINUM_INLINE SIMDVec rsqrt() const noexcept { return SIMDVec(_mm_rsqrt_ps(value)); }
        OPTINUM_INLINE SIMDVec abs() const noexcept { return SIMDVec(detail::abs_ps(value)); }
        OPTINUM_INLINE SIMDVec rcp() const noexcept { return SIMDVec(_mm_rcp_ps(value)); }

        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm_min_ps(a.value, b.value));
        }

        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm_max_ps(a.value, b.value));
        }

        OPTINUM_INLINE static SIMDVec fma(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
#ifdef OPTINUM_HAS_FMA
            return SIMDVec(_mm_fmadd_ps(a.value, b.value, c.value));
#else
            return SIMDVec(_mm_add_ps(_mm_mul_ps(a.value, b.value), c.value));
#endif
        }

        OPTINUM_INLINE static SIMDVec fms(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
#ifdef OPTINUM_HAS_FMA
            return SIMDVec(_mm_fmsub_ps(a.value, b.value, c.value));
#else
            return SIMDVec(_mm_sub_ps(_mm_mul_ps(a.value, b.value), c.value));
#endif
        }

        OPTINUM_INLINE float dot(const SIMDVec &other) const noexcept {
#ifdef OPTINUM_HAS_SSE41
            return _mm_cvtss_f32(_mm_dp_ps(value, other.value, 0xFF));
#else
            return detail::hsum_ps(_mm_mul_ps(value, other.value));
#endif
        }
    };

    template <> struct SIMDVec<double, 2> {
        using value_type = double;
        using native_type = __m128d;
        static constexpr std::size_t width = 2;

        native_type value;

        OPTINUM_INLINE SIMDVec() noexcept : value(_mm_setzero_pd()) {}
        OPTINUM_INLINE explicit SIMDVec(double val) noexcept : value(_mm_set1_pd(val)) {}
        OPTINUM_INLINE explicit SIMDVec(native_type v) noexcept : value(v) {}
        OPTINUM_INLINE SIMDVec(double a, double b) noexcept : value(_mm_setr_pd(a, b)) {}

        OPTINUM_INLINE SIMDVec &operator=(double val) noexcept {
            value = _mm_set1_pd(val);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator=(native_type v) noexcept {
            value = v;
            return *this;
        }

        OPTINUM_INLINE static SIMDVec load(const double *ptr) noexcept { return SIMDVec(_mm_load_pd(ptr)); }
        OPTINUM_INLINE static SIMDVec loadu(const double *ptr) noexcept { return SIMDVec(_mm_loadu_pd(ptr)); }
        OPTINUM_INLINE void store(double *ptr) const noexcept { _mm_store_pd(ptr, value); }
        OPTINUM_INLINE void storeu(double *ptr) const noexcept { _mm_storeu_pd(ptr, value); }

        OPTINUM_INLINE double operator[](std::size_t i) const noexcept {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, value);
            return tmp[i];
        }

        OPTINUM_INLINE SIMDVec operator+(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_add_pd(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_sub_pd(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator*(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_mul_pd(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator/(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_div_pd(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-() const noexcept { return SIMDVec(detail::neg_pd(value)); }

        OPTINUM_INLINE SIMDVec operator+(double rhs) const noexcept {
            return SIMDVec(_mm_add_pd(value, _mm_set1_pd(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator-(double rhs) const noexcept {
            return SIMDVec(_mm_sub_pd(value, _mm_set1_pd(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator*(double rhs) const noexcept {
            return SIMDVec(_mm_mul_pd(value, _mm_set1_pd(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator/(double rhs) const noexcept {
            return SIMDVec(_mm_div_pd(value, _mm_set1_pd(rhs)));
        }

        OPTINUM_INLINE SIMDVec &operator+=(const SIMDVec &rhs) noexcept {
            value = _mm_add_pd(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator-=(const SIMDVec &rhs) noexcept {
            value = _mm_sub_pd(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator*=(const SIMDVec &rhs) noexcept {
            value = _mm_mul_pd(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator/=(const SIMDVec &rhs) noexcept {
            value = _mm_div_pd(value, rhs.value);
            return *this;
        }

        OPTINUM_INLINE double hsum() const noexcept { return detail::hsum_pd(value); }

        OPTINUM_INLINE double hmin() const noexcept {
            __m128d tmp = _mm_min_pd(value, _mm_shuffle_pd(value, value, 1));
            return _mm_cvtsd_f64(tmp);
        }

        OPTINUM_INLINE double hmax() const noexcept {
            __m128d tmp = _mm_max_pd(value, _mm_shuffle_pd(value, value, 1));
            return _mm_cvtsd_f64(tmp);
        }

        OPTINUM_INLINE SIMDVec sqrt() const noexcept { return SIMDVec(_mm_sqrt_pd(value)); }

        OPTINUM_INLINE SIMDVec rsqrt() const noexcept {
            return SIMDVec(_mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(value)));
        }

        OPTINUM_INLINE SIMDVec abs() const noexcept { return SIMDVec(detail::abs_pd(value)); }

        OPTINUM_INLINE SIMDVec rcp() const noexcept { return SIMDVec(_mm_div_pd(_mm_set1_pd(1.0), value)); }

        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm_min_pd(a.value, b.value));
        }

        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm_max_pd(a.value, b.value));
        }

        OPTINUM_INLINE static SIMDVec fma(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
#ifdef OPTINUM_HAS_FMA
            return SIMDVec(_mm_fmadd_pd(a.value, b.value, c.value));
#else
            return SIMDVec(_mm_add_pd(_mm_mul_pd(a.value, b.value), c.value));
#endif
        }

        OPTINUM_INLINE static SIMDVec fms(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
#ifdef OPTINUM_HAS_FMA
            return SIMDVec(_mm_fmsub_pd(a.value, b.value, c.value));
#else
            return SIMDVec(_mm_sub_pd(_mm_mul_pd(a.value, b.value), c.value));
#endif
        }

        OPTINUM_INLINE double dot(const SIMDVec &other) const noexcept {
#ifdef OPTINUM_HAS_SSE41
            return _mm_cvtsd_f64(_mm_dp_pd(value, other.value, 0xFF));
#else
            return detail::hsum_pd(_mm_mul_pd(value, other.value));
#endif
        }
    };

} // namespace optinum::simd

#endif // OPTINUM_HAS_SSE2
