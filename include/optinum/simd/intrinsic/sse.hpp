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

    // =============================================================================
    // SIMDVec<int32_t, 4> - SSE2 integer (4 x 32-bit signed integers)
    // =============================================================================

    template <> struct SIMDVec<int32_t, 4> {
        using value_type = int32_t;
        using native_type = __m128i;
        static constexpr std::size_t width = 4;

        native_type value;

        OPTINUM_INLINE SIMDVec() noexcept : value(_mm_setzero_si128()) {}
        OPTINUM_INLINE explicit SIMDVec(int32_t val) noexcept : value(_mm_set1_epi32(val)) {}
        OPTINUM_INLINE explicit SIMDVec(native_type v) noexcept : value(v) {}
        OPTINUM_INLINE SIMDVec(int32_t a, int32_t b, int32_t c, int32_t d) noexcept
            : value(_mm_setr_epi32(a, b, c, d)) {}

        OPTINUM_INLINE SIMDVec &operator=(int32_t val) noexcept {
            value = _mm_set1_epi32(val);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator=(native_type v) noexcept {
            value = v;
            return *this;
        }

        OPTINUM_INLINE static SIMDVec load(const int32_t *ptr) noexcept {
            return SIMDVec(_mm_load_si128(reinterpret_cast<const __m128i *>(ptr)));
        }
        OPTINUM_INLINE static SIMDVec loadu(const int32_t *ptr) noexcept {
            return SIMDVec(_mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr)));
        }
        OPTINUM_INLINE void store(int32_t *ptr) const noexcept {
            _mm_store_si128(reinterpret_cast<__m128i *>(ptr), value);
        }
        OPTINUM_INLINE void storeu(int32_t *ptr) const noexcept {
            _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr), value);
        }

        OPTINUM_INLINE int32_t operator[](std::size_t i) const noexcept {
            alignas(16) int32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), value);
            return tmp[i];
        }

        // Arithmetic
        OPTINUM_INLINE SIMDVec operator+(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_add_epi32(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_sub_epi32(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator*(const SIMDVec &rhs) const noexcept {
#ifdef OPTINUM_HAS_SSE41
            return SIMDVec(_mm_mullo_epi32(value, rhs.value));
#else
            // SSE2 fallback: emulate 32-bit multiply
            __m128i a13 = _mm_shuffle_epi32(value, _MM_SHUFFLE(2, 3, 0, 1));
            __m128i b13 = _mm_shuffle_epi32(rhs.value, _MM_SHUFFLE(2, 3, 0, 1));
            __m128i prod02 = _mm_mul_epu32(value, rhs.value);
            __m128i prod13 = _mm_mul_epu32(a13, b13);
            __m128i prod01 = _mm_unpacklo_epi32(prod02, prod13);
            __m128i prod23 = _mm_unpackhi_epi32(prod02, prod13);
            return SIMDVec(_mm_unpacklo_epi64(prod01, prod23));
#endif
        }
        OPTINUM_INLINE SIMDVec operator-() const noexcept { return SIMDVec(_mm_sub_epi32(_mm_setzero_si128(), value)); }

        // Scalar operations
        OPTINUM_INLINE SIMDVec operator+(int32_t rhs) const noexcept {
            return SIMDVec(_mm_add_epi32(value, _mm_set1_epi32(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator-(int32_t rhs) const noexcept {
            return SIMDVec(_mm_sub_epi32(value, _mm_set1_epi32(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator*(int32_t rhs) const noexcept { return *this * SIMDVec(rhs); }

        // Compound assignment
        OPTINUM_INLINE SIMDVec &operator+=(const SIMDVec &rhs) noexcept {
            value = _mm_add_epi32(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator-=(const SIMDVec &rhs) noexcept {
            value = _mm_sub_epi32(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator*=(const SIMDVec &rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }

        // Bitwise
        OPTINUM_INLINE SIMDVec operator&(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_and_si128(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator|(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_or_si128(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator^(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_xor_si128(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator~() const noexcept { return SIMDVec(_mm_xor_si128(value, _mm_set1_epi32(-1))); }

        // Shifts
        OPTINUM_INLINE SIMDVec operator<<(int count) const noexcept { return SIMDVec(_mm_slli_epi32(value, count)); }
        OPTINUM_INLINE SIMDVec operator>>(int count) const noexcept {
            return SIMDVec(_mm_srai_epi32(value, count)); // Arithmetic shift
        }
        OPTINUM_INLINE SIMDVec shr_logical(int count) const noexcept {
            return SIMDVec(_mm_srli_epi32(value, count)); // Logical shift
        }

        // Reductions
        OPTINUM_INLINE int32_t hsum() const noexcept {
            __m128i sum = _mm_add_epi32(value, _mm_shuffle_epi32(value, _MM_SHUFFLE(2, 3, 0, 1)));
            sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_cvtsi128_si32(sum);
        }
        OPTINUM_INLINE int32_t hmin() const noexcept {
            alignas(16) int32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), value);
            int32_t r = tmp[0];
            for (int i = 1; i < 4; ++i)
                r = (tmp[i] < r) ? tmp[i] : r;
            return r;
        }
        OPTINUM_INLINE int32_t hmax() const noexcept {
            alignas(16) int32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), value);
            int32_t r = tmp[0];
            for (int i = 1; i < 4; ++i)
                r = (tmp[i] > r) ? tmp[i] : r;
            return r;
        }
        OPTINUM_INLINE int32_t hprod() const noexcept {
            alignas(16) int32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), value);
            return tmp[0] * tmp[1] * tmp[2] * tmp[3];
        }

        // Abs
        OPTINUM_INLINE SIMDVec abs() const noexcept {
#ifdef OPTINUM_HAS_SSSE3
            return SIMDVec(_mm_abs_epi32(value));
#else
            __m128i mask = _mm_cmplt_epi32(value, _mm_setzero_si128());
            return SIMDVec(_mm_sub_epi32(_mm_xor_si128(value, mask), mask));
#endif
        }

        // Min/Max
        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
#ifdef OPTINUM_HAS_SSE41
            return SIMDVec(_mm_min_epi32(a.value, b.value));
#else
            __m128i cmp = _mm_cmplt_epi32(a.value, b.value);
            return SIMDVec(_mm_or_si128(_mm_and_si128(cmp, a.value), _mm_andnot_si128(cmp, b.value)));
#endif
        }
        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
#ifdef OPTINUM_HAS_SSE41
            return SIMDVec(_mm_max_epi32(a.value, b.value));
#else
            __m128i cmp = _mm_cmpgt_epi32(a.value, b.value);
            return SIMDVec(_mm_or_si128(_mm_and_si128(cmp, a.value), _mm_andnot_si128(cmp, b.value)));
#endif
        }

        // Dot product
        OPTINUM_INLINE int32_t dot(const SIMDVec &other) const noexcept { return (*this * other).hsum(); }
    };

    // =============================================================================
    // SIMDVec<int64_t, 2> - SSE2 integer (2 x 64-bit signed integers)
    // =============================================================================

    template <> struct SIMDVec<int64_t, 2> {
        using value_type = int64_t;
        using native_type = __m128i;
        static constexpr std::size_t width = 2;

        native_type value;

        OPTINUM_INLINE SIMDVec() noexcept : value(_mm_setzero_si128()) {}
        OPTINUM_INLINE explicit SIMDVec(int64_t val) noexcept : value(_mm_set1_epi64x(val)) {}
        OPTINUM_INLINE explicit SIMDVec(native_type v) noexcept : value(v) {}
        OPTINUM_INLINE SIMDVec(int64_t a, int64_t b) noexcept : value(_mm_set_epi64x(b, a)) {}

        OPTINUM_INLINE SIMDVec &operator=(int64_t val) noexcept {
            value = _mm_set1_epi64x(val);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator=(native_type v) noexcept {
            value = v;
            return *this;
        }

        OPTINUM_INLINE static SIMDVec load(const int64_t *ptr) noexcept {
            return SIMDVec(_mm_load_si128(reinterpret_cast<const __m128i *>(ptr)));
        }
        OPTINUM_INLINE static SIMDVec loadu(const int64_t *ptr) noexcept {
            return SIMDVec(_mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr)));
        }
        OPTINUM_INLINE void store(int64_t *ptr) const noexcept {
            _mm_store_si128(reinterpret_cast<__m128i *>(ptr), value);
        }
        OPTINUM_INLINE void storeu(int64_t *ptr) const noexcept {
            _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr), value);
        }

        OPTINUM_INLINE int64_t operator[](std::size_t i) const noexcept {
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), value);
            return tmp[i];
        }

        // Arithmetic
        OPTINUM_INLINE SIMDVec operator+(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_add_epi64(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_sub_epi64(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator*(const SIMDVec &rhs) const noexcept {
            // No native 64-bit multiply in SSE2/SSE4.1, emulate
            alignas(16) int64_t a[2], b[2], r[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(a), value);
            _mm_store_si128(reinterpret_cast<__m128i *>(b), rhs.value);
            r[0] = a[0] * b[0];
            r[1] = a[1] * b[1];
            return SIMDVec(_mm_load_si128(reinterpret_cast<const __m128i *>(r)));
        }
        OPTINUM_INLINE SIMDVec operator-() const noexcept { return SIMDVec(_mm_sub_epi64(_mm_setzero_si128(), value)); }

        // Scalar operations
        OPTINUM_INLINE SIMDVec operator+(int64_t rhs) const noexcept {
            return SIMDVec(_mm_add_epi64(value, _mm_set1_epi64x(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator-(int64_t rhs) const noexcept {
            return SIMDVec(_mm_sub_epi64(value, _mm_set1_epi64x(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator*(int64_t rhs) const noexcept { return *this * SIMDVec(rhs); }

        // Compound assignment
        OPTINUM_INLINE SIMDVec &operator+=(const SIMDVec &rhs) noexcept {
            value = _mm_add_epi64(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator-=(const SIMDVec &rhs) noexcept {
            value = _mm_sub_epi64(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator*=(const SIMDVec &rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }

        // Bitwise
        OPTINUM_INLINE SIMDVec operator&(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_and_si128(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator|(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_or_si128(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator^(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm_xor_si128(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator~() const noexcept {
            return SIMDVec(_mm_xor_si128(value, _mm_set1_epi64x(-1LL)));
        }

        // Shifts
        OPTINUM_INLINE SIMDVec operator<<(int count) const noexcept { return SIMDVec(_mm_slli_epi64(value, count)); }
        OPTINUM_INLINE SIMDVec operator>>(int count) const noexcept {
            // SSE2 doesn't have arithmetic shift for 64-bit, emulate
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), value);
            tmp[0] >>= count;
            tmp[1] >>= count;
            return SIMDVec(_mm_load_si128(reinterpret_cast<const __m128i *>(tmp)));
        }
        OPTINUM_INLINE SIMDVec shr_logical(int count) const noexcept { return SIMDVec(_mm_srli_epi64(value, count)); }

        // Reductions
        OPTINUM_INLINE int64_t hsum() const noexcept {
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), value);
            return tmp[0] + tmp[1];
        }
        OPTINUM_INLINE int64_t hmin() const noexcept {
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), value);
            return (tmp[0] < tmp[1]) ? tmp[0] : tmp[1];
        }
        OPTINUM_INLINE int64_t hmax() const noexcept {
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), value);
            return (tmp[0] > tmp[1]) ? tmp[0] : tmp[1];
        }
        OPTINUM_INLINE int64_t hprod() const noexcept {
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), value);
            return tmp[0] * tmp[1];
        }

        // Abs
        OPTINUM_INLINE SIMDVec abs() const noexcept {
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), value);
            tmp[0] = (tmp[0] < 0) ? -tmp[0] : tmp[0];
            tmp[1] = (tmp[1] < 0) ? -tmp[1] : tmp[1];
            return SIMDVec(_mm_load_si128(reinterpret_cast<const __m128i *>(tmp)));
        }

        // Min/Max
        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            alignas(16) int64_t ta[2], tb[2], r[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(ta), a.value);
            _mm_store_si128(reinterpret_cast<__m128i *>(tb), b.value);
            r[0] = (ta[0] < tb[0]) ? ta[0] : tb[0];
            r[1] = (ta[1] < tb[1]) ? ta[1] : tb[1];
            return SIMDVec(_mm_load_si128(reinterpret_cast<const __m128i *>(r)));
        }
        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            alignas(16) int64_t ta[2], tb[2], r[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(ta), a.value);
            _mm_store_si128(reinterpret_cast<__m128i *>(tb), b.value);
            r[0] = (ta[0] > tb[0]) ? ta[0] : tb[0];
            r[1] = (ta[1] > tb[1]) ? ta[1] : tb[1];
            return SIMDVec(_mm_load_si128(reinterpret_cast<const __m128i *>(r)));
        }

        // Dot product
        OPTINUM_INLINE int64_t dot(const SIMDVec &other) const noexcept { return (*this * other).hsum(); }
    };

} // namespace optinum::simd

#endif // OPTINUM_HAS_SSE2
