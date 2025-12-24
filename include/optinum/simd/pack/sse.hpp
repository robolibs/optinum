#pragma once

// =============================================================================
// optinum/simd/pack/sse.hpp
// SSE specializations for pack<float,4>, pack<double,2>, pack<int32_t,4>, pack<int64_t,2>
// =============================================================================

#include <optinum/simd/pack/pack.hpp>

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

    // =============================================================================
    // pack<float, 4> - SSE (4 x 32-bit float)
    // =============================================================================

    template <> struct pack<float, 4> {
        using value_type = float;
        using native_type = __m128;
        static constexpr std::size_t width = 4;

        native_type data_;

        OPTINUM_INLINE pack() noexcept : data_(_mm_setzero_ps()) {}
        OPTINUM_INLINE explicit pack(float val) noexcept : data_(_mm_set1_ps(val)) {}
        OPTINUM_INLINE explicit pack(native_type v) noexcept : data_(v) {}
        OPTINUM_INLINE pack(float a, float b, float c, float d) noexcept : data_(_mm_setr_ps(a, b, c, d)) {}

        OPTINUM_INLINE pack &operator=(float val) noexcept {
            data_ = _mm_set1_ps(val);
            return *this;
        }
        OPTINUM_INLINE pack &operator=(native_type v) noexcept {
            data_ = v;
            return *this;
        }

        OPTINUM_INLINE static pack load(const float *ptr) noexcept { return pack(_mm_load_ps(ptr)); }
        OPTINUM_INLINE static pack loadu(const float *ptr) noexcept { return pack(_mm_loadu_ps(ptr)); }
        OPTINUM_INLINE void store(float *ptr) const noexcept { _mm_store_ps(ptr, data_); }
        OPTINUM_INLINE void storeu(float *ptr) const noexcept { _mm_storeu_ps(ptr, data_); }

        OPTINUM_INLINE float operator[](std::size_t i) const noexcept {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, data_);
            return tmp[i];
        }

        OPTINUM_INLINE pack operator+(const pack &rhs) const noexcept { return pack(_mm_add_ps(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator-(const pack &rhs) const noexcept { return pack(_mm_sub_ps(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator*(const pack &rhs) const noexcept { return pack(_mm_mul_ps(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator/(const pack &rhs) const noexcept { return pack(_mm_div_ps(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator-() const noexcept { return pack(detail::neg_ps(data_)); }

        OPTINUM_INLINE pack operator+(float rhs) const noexcept { return pack(_mm_add_ps(data_, _mm_set1_ps(rhs))); }
        OPTINUM_INLINE pack operator-(float rhs) const noexcept { return pack(_mm_sub_ps(data_, _mm_set1_ps(rhs))); }
        OPTINUM_INLINE pack operator*(float rhs) const noexcept { return pack(_mm_mul_ps(data_, _mm_set1_ps(rhs))); }
        OPTINUM_INLINE pack operator/(float rhs) const noexcept { return pack(_mm_div_ps(data_, _mm_set1_ps(rhs))); }

        OPTINUM_INLINE pack &operator+=(const pack &rhs) noexcept {
            data_ = _mm_add_ps(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator-=(const pack &rhs) noexcept {
            data_ = _mm_sub_ps(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator*=(const pack &rhs) noexcept {
            data_ = _mm_mul_ps(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator/=(const pack &rhs) noexcept {
            data_ = _mm_div_ps(data_, rhs.data_);
            return *this;
        }

        OPTINUM_INLINE float hsum() const noexcept { return detail::hsum_ps(data_); }

        OPTINUM_INLINE float hmin() const noexcept {
            __m128 tmp = _mm_min_ps(data_, _mm_shuffle_ps(data_, data_, _MM_SHUFFLE(2, 3, 0, 1)));
            tmp = _mm_min_ps(tmp, _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_cvtss_f32(tmp);
        }

        OPTINUM_INLINE float hmax() const noexcept {
            __m128 tmp = _mm_max_ps(data_, _mm_shuffle_ps(data_, data_, _MM_SHUFFLE(2, 3, 0, 1)));
            tmp = _mm_max_ps(tmp, _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_cvtss_f32(tmp);
        }

        OPTINUM_INLINE float hprod() const noexcept {
            __m128 tmp = _mm_mul_ps(data_, _mm_shuffle_ps(data_, data_, _MM_SHUFFLE(2, 3, 0, 1)));
            tmp = _mm_mul_ps(tmp, _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_cvtss_f32(tmp);
        }

        OPTINUM_INLINE pack sqrt() const noexcept { return pack(_mm_sqrt_ps(data_)); }
        OPTINUM_INLINE pack rsqrt() const noexcept { return pack(_mm_rsqrt_ps(data_)); }
        OPTINUM_INLINE pack abs() const noexcept { return pack(detail::abs_ps(data_)); }
        OPTINUM_INLINE pack rcp() const noexcept { return pack(_mm_rcp_ps(data_)); }

        OPTINUM_INLINE static pack min(const pack &a, const pack &b) noexcept {
            return pack(_mm_min_ps(a.data_, b.data_));
        }

        OPTINUM_INLINE static pack max(const pack &a, const pack &b) noexcept {
            return pack(_mm_max_ps(a.data_, b.data_));
        }

        OPTINUM_INLINE static pack fma(const pack &a, const pack &b, const pack &c) noexcept {
#ifdef OPTINUM_HAS_FMA
            return pack(_mm_fmadd_ps(a.data_, b.data_, c.data_));
#else
            return pack(_mm_add_ps(_mm_mul_ps(a.data_, b.data_), c.data_));
#endif
        }

        OPTINUM_INLINE static pack fms(const pack &a, const pack &b, const pack &c) noexcept {
#ifdef OPTINUM_HAS_FMA
            return pack(_mm_fmsub_ps(a.data_, b.data_, c.data_));
#else
            return pack(_mm_sub_ps(_mm_mul_ps(a.data_, b.data_), c.data_));
#endif
        }

        OPTINUM_INLINE float dot(const pack &other) const noexcept {
#ifdef OPTINUM_HAS_SSE41
            return _mm_cvtss_f32(_mm_dp_ps(data_, other.data_, 0xFF));
#else
            return detail::hsum_ps(_mm_mul_ps(data_, other.data_));
#endif
        }
    };

    // =============================================================================
    // pack<double, 2> - SSE2 (2 x 64-bit double)
    // =============================================================================

    template <> struct pack<double, 2> {
        using value_type = double;
        using native_type = __m128d;
        static constexpr std::size_t width = 2;

        native_type data_;

        OPTINUM_INLINE pack() noexcept : data_(_mm_setzero_pd()) {}
        OPTINUM_INLINE explicit pack(double val) noexcept : data_(_mm_set1_pd(val)) {}
        OPTINUM_INLINE explicit pack(native_type v) noexcept : data_(v) {}
        OPTINUM_INLINE pack(double a, double b) noexcept : data_(_mm_setr_pd(a, b)) {}

        OPTINUM_INLINE pack &operator=(double val) noexcept {
            data_ = _mm_set1_pd(val);
            return *this;
        }
        OPTINUM_INLINE pack &operator=(native_type v) noexcept {
            data_ = v;
            return *this;
        }

        OPTINUM_INLINE static pack load(const double *ptr) noexcept { return pack(_mm_load_pd(ptr)); }
        OPTINUM_INLINE static pack loadu(const double *ptr) noexcept { return pack(_mm_loadu_pd(ptr)); }
        OPTINUM_INLINE void store(double *ptr) const noexcept { _mm_store_pd(ptr, data_); }
        OPTINUM_INLINE void storeu(double *ptr) const noexcept { _mm_storeu_pd(ptr, data_); }

        OPTINUM_INLINE double operator[](std::size_t i) const noexcept {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, data_);
            return tmp[i];
        }

        OPTINUM_INLINE pack operator+(const pack &rhs) const noexcept { return pack(_mm_add_pd(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator-(const pack &rhs) const noexcept { return pack(_mm_sub_pd(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator*(const pack &rhs) const noexcept { return pack(_mm_mul_pd(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator/(const pack &rhs) const noexcept { return pack(_mm_div_pd(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator-() const noexcept { return pack(detail::neg_pd(data_)); }

        OPTINUM_INLINE pack operator+(double rhs) const noexcept { return pack(_mm_add_pd(data_, _mm_set1_pd(rhs))); }
        OPTINUM_INLINE pack operator-(double rhs) const noexcept { return pack(_mm_sub_pd(data_, _mm_set1_pd(rhs))); }
        OPTINUM_INLINE pack operator*(double rhs) const noexcept { return pack(_mm_mul_pd(data_, _mm_set1_pd(rhs))); }
        OPTINUM_INLINE pack operator/(double rhs) const noexcept { return pack(_mm_div_pd(data_, _mm_set1_pd(rhs))); }

        OPTINUM_INLINE pack &operator+=(const pack &rhs) noexcept {
            data_ = _mm_add_pd(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator-=(const pack &rhs) noexcept {
            data_ = _mm_sub_pd(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator*=(const pack &rhs) noexcept {
            data_ = _mm_mul_pd(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator/=(const pack &rhs) noexcept {
            data_ = _mm_div_pd(data_, rhs.data_);
            return *this;
        }

        OPTINUM_INLINE double hsum() const noexcept { return detail::hsum_pd(data_); }

        OPTINUM_INLINE double hmin() const noexcept {
            __m128d tmp = _mm_min_pd(data_, _mm_shuffle_pd(data_, data_, 1));
            return _mm_cvtsd_f64(tmp);
        }

        OPTINUM_INLINE double hmax() const noexcept {
            __m128d tmp = _mm_max_pd(data_, _mm_shuffle_pd(data_, data_, 1));
            return _mm_cvtsd_f64(tmp);
        }

        OPTINUM_INLINE double hprod() const noexcept {
            __m128d tmp = _mm_mul_pd(data_, _mm_shuffle_pd(data_, data_, 1));
            return _mm_cvtsd_f64(tmp);
        }

        OPTINUM_INLINE pack sqrt() const noexcept { return pack(_mm_sqrt_pd(data_)); }

        OPTINUM_INLINE pack rsqrt() const noexcept { return pack(_mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(data_))); }

        OPTINUM_INLINE pack abs() const noexcept { return pack(detail::abs_pd(data_)); }

        OPTINUM_INLINE pack rcp() const noexcept { return pack(_mm_div_pd(_mm_set1_pd(1.0), data_)); }

        OPTINUM_INLINE static pack min(const pack &a, const pack &b) noexcept {
            return pack(_mm_min_pd(a.data_, b.data_));
        }

        OPTINUM_INLINE static pack max(const pack &a, const pack &b) noexcept {
            return pack(_mm_max_pd(a.data_, b.data_));
        }

        OPTINUM_INLINE static pack fma(const pack &a, const pack &b, const pack &c) noexcept {
#ifdef OPTINUM_HAS_FMA
            return pack(_mm_fmadd_pd(a.data_, b.data_, c.data_));
#else
            return pack(_mm_add_pd(_mm_mul_pd(a.data_, b.data_), c.data_));
#endif
        }

        OPTINUM_INLINE static pack fms(const pack &a, const pack &b, const pack &c) noexcept {
#ifdef OPTINUM_HAS_FMA
            return pack(_mm_fmsub_pd(a.data_, b.data_, c.data_));
#else
            return pack(_mm_sub_pd(_mm_mul_pd(a.data_, b.data_), c.data_));
#endif
        }

        OPTINUM_INLINE double dot(const pack &other) const noexcept {
#ifdef OPTINUM_HAS_SSE41
            return _mm_cvtsd_f64(_mm_dp_pd(data_, other.data_, 0xFF));
#else
            return detail::hsum_pd(_mm_mul_pd(data_, other.data_));
#endif
        }
    };

    // =============================================================================
    // pack<int32_t, 4> - SSE2 (4 x 32-bit signed integer)
    // =============================================================================

    template <> struct pack<int32_t, 4> {
        using value_type = int32_t;
        using native_type = __m128i;
        static constexpr std::size_t width = 4;

        native_type data_;

        OPTINUM_INLINE pack() noexcept : data_(_mm_setzero_si128()) {}
        OPTINUM_INLINE explicit pack(int32_t val) noexcept : data_(_mm_set1_epi32(val)) {}
        OPTINUM_INLINE explicit pack(native_type v) noexcept : data_(v) {}
        OPTINUM_INLINE pack(int32_t a, int32_t b, int32_t c, int32_t d) noexcept : data_(_mm_setr_epi32(a, b, c, d)) {}

        OPTINUM_INLINE pack &operator=(int32_t val) noexcept {
            data_ = _mm_set1_epi32(val);
            return *this;
        }
        OPTINUM_INLINE pack &operator=(native_type v) noexcept {
            data_ = v;
            return *this;
        }

        OPTINUM_INLINE static pack load(const int32_t *ptr) noexcept {
            return pack(_mm_load_si128(reinterpret_cast<const __m128i *>(ptr)));
        }
        OPTINUM_INLINE static pack loadu(const int32_t *ptr) noexcept {
            return pack(_mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr)));
        }
        OPTINUM_INLINE void store(int32_t *ptr) const noexcept {
            _mm_store_si128(reinterpret_cast<__m128i *>(ptr), data_);
        }
        OPTINUM_INLINE void storeu(int32_t *ptr) const noexcept {
            _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr), data_);
        }

        OPTINUM_INLINE int32_t operator[](std::size_t i) const noexcept {
            alignas(16) int32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), data_);
            return tmp[i];
        }

        // Arithmetic
        OPTINUM_INLINE pack operator+(const pack &rhs) const noexcept { return pack(_mm_add_epi32(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator-(const pack &rhs) const noexcept { return pack(_mm_sub_epi32(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator*(const pack &rhs) const noexcept {
#ifdef OPTINUM_HAS_SSE41
            return pack(_mm_mullo_epi32(data_, rhs.data_));
#else
            // SSE2 fallback: emulate 32-bit multiply
            __m128i a13 = _mm_shuffle_epi32(data_, _MM_SHUFFLE(2, 3, 0, 1));
            __m128i b13 = _mm_shuffle_epi32(rhs.data_, _MM_SHUFFLE(2, 3, 0, 1));
            __m128i prod02 = _mm_mul_epu32(data_, rhs.data_);
            __m128i prod13 = _mm_mul_epu32(a13, b13);
            __m128i prod01 = _mm_unpacklo_epi32(prod02, prod13);
            __m128i prod23 = _mm_unpackhi_epi32(prod02, prod13);
            return pack(_mm_unpacklo_epi64(prod01, prod23));
#endif
        }
        OPTINUM_INLINE pack operator-() const noexcept { return pack(_mm_sub_epi32(_mm_setzero_si128(), data_)); }

        // Scalar operations
        OPTINUM_INLINE pack operator+(int32_t rhs) const noexcept {
            return pack(_mm_add_epi32(data_, _mm_set1_epi32(rhs)));
        }
        OPTINUM_INLINE pack operator-(int32_t rhs) const noexcept {
            return pack(_mm_sub_epi32(data_, _mm_set1_epi32(rhs)));
        }
        OPTINUM_INLINE pack operator*(int32_t rhs) const noexcept { return *this * pack(rhs); }

        // Compound assignment
        OPTINUM_INLINE pack &operator+=(const pack &rhs) noexcept {
            data_ = _mm_add_epi32(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator-=(const pack &rhs) noexcept {
            data_ = _mm_sub_epi32(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator*=(const pack &rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }

        // Bitwise
        OPTINUM_INLINE pack operator&(const pack &rhs) const noexcept { return pack(_mm_and_si128(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator|(const pack &rhs) const noexcept { return pack(_mm_or_si128(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator^(const pack &rhs) const noexcept { return pack(_mm_xor_si128(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator~() const noexcept { return pack(_mm_xor_si128(data_, _mm_set1_epi32(-1))); }

        // Shifts
        OPTINUM_INLINE pack operator<<(int count) const noexcept { return pack(_mm_slli_epi32(data_, count)); }
        OPTINUM_INLINE pack operator>>(int count) const noexcept {
            return pack(_mm_srai_epi32(data_, count)); // Arithmetic shift
        }
        OPTINUM_INLINE pack shr_logical(int count) const noexcept {
            return pack(_mm_srli_epi32(data_, count)); // Logical shift
        }

        // Reductions
        OPTINUM_INLINE int32_t hsum() const noexcept {
            __m128i sum = _mm_add_epi32(data_, _mm_shuffle_epi32(data_, _MM_SHUFFLE(2, 3, 0, 1)));
            sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_cvtsi128_si32(sum);
        }
        OPTINUM_INLINE int32_t hmin() const noexcept {
            alignas(16) int32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), data_);
            int32_t r = tmp[0];
            for (int i = 1; i < 4; ++i)
                r = (tmp[i] < r) ? tmp[i] : r;
            return r;
        }
        OPTINUM_INLINE int32_t hmax() const noexcept {
            alignas(16) int32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), data_);
            int32_t r = tmp[0];
            for (int i = 1; i < 4; ++i)
                r = (tmp[i] > r) ? tmp[i] : r;
            return r;
        }
        OPTINUM_INLINE int32_t hprod() const noexcept {
            alignas(16) int32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), data_);
            return tmp[0] * tmp[1] * tmp[2] * tmp[3];
        }

        // Abs
        OPTINUM_INLINE pack abs() const noexcept {
#ifdef OPTINUM_HAS_SSSE3
            return pack(_mm_abs_epi32(data_));
#else
            __m128i mask = _mm_cmplt_epi32(data_, _mm_setzero_si128());
            return pack(_mm_sub_epi32(_mm_xor_si128(data_, mask), mask));
#endif
        }

        // Min/Max
        OPTINUM_INLINE static pack min(const pack &a, const pack &b) noexcept {
#ifdef OPTINUM_HAS_SSE41
            return pack(_mm_min_epi32(a.data_, b.data_));
#else
            __m128i cmp = _mm_cmplt_epi32(a.data_, b.data_);
            return pack(_mm_or_si128(_mm_and_si128(cmp, a.data_), _mm_andnot_si128(cmp, b.data_)));
#endif
        }
        OPTINUM_INLINE static pack max(const pack &a, const pack &b) noexcept {
#ifdef OPTINUM_HAS_SSE41
            return pack(_mm_max_epi32(a.data_, b.data_));
#else
            __m128i cmp = _mm_cmpgt_epi32(a.data_, b.data_);
            return pack(_mm_or_si128(_mm_and_si128(cmp, a.data_), _mm_andnot_si128(cmp, b.data_)));
#endif
        }

        // Dot product
        OPTINUM_INLINE int32_t dot(const pack &other) const noexcept { return (*this * other).hsum(); }
    };

    // =============================================================================
    // pack<int64_t, 2> - SSE2 (2 x 64-bit signed integer)
    // =============================================================================

    template <> struct pack<int64_t, 2> {
        using value_type = int64_t;
        using native_type = __m128i;
        static constexpr std::size_t width = 2;

        native_type data_;

        OPTINUM_INLINE pack() noexcept : data_(_mm_setzero_si128()) {}
        OPTINUM_INLINE explicit pack(int64_t val) noexcept : data_(_mm_set1_epi64x(val)) {}
        OPTINUM_INLINE explicit pack(native_type v) noexcept : data_(v) {}
        OPTINUM_INLINE pack(int64_t a, int64_t b) noexcept : data_(_mm_set_epi64x(b, a)) {}

        OPTINUM_INLINE pack &operator=(int64_t val) noexcept {
            data_ = _mm_set1_epi64x(val);
            return *this;
        }
        OPTINUM_INLINE pack &operator=(native_type v) noexcept {
            data_ = v;
            return *this;
        }

        OPTINUM_INLINE static pack load(const int64_t *ptr) noexcept {
            return pack(_mm_load_si128(reinterpret_cast<const __m128i *>(ptr)));
        }
        OPTINUM_INLINE static pack loadu(const int64_t *ptr) noexcept {
            return pack(_mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr)));
        }
        OPTINUM_INLINE void store(int64_t *ptr) const noexcept {
            _mm_store_si128(reinterpret_cast<__m128i *>(ptr), data_);
        }
        OPTINUM_INLINE void storeu(int64_t *ptr) const noexcept {
            _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr), data_);
        }

        OPTINUM_INLINE int64_t operator[](std::size_t i) const noexcept {
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), data_);
            return tmp[i];
        }

        // Arithmetic
        OPTINUM_INLINE pack operator+(const pack &rhs) const noexcept { return pack(_mm_add_epi64(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator-(const pack &rhs) const noexcept { return pack(_mm_sub_epi64(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator*(const pack &rhs) const noexcept {
            // No native 64-bit multiply in SSE2/SSE4.1, emulate
            alignas(16) int64_t a[2], b[2], r[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(a), data_);
            _mm_store_si128(reinterpret_cast<__m128i *>(b), rhs.data_);
            r[0] = a[0] * b[0];
            r[1] = a[1] * b[1];
            return pack(_mm_load_si128(reinterpret_cast<const __m128i *>(r)));
        }
        OPTINUM_INLINE pack operator-() const noexcept { return pack(_mm_sub_epi64(_mm_setzero_si128(), data_)); }

        // Scalar operations
        OPTINUM_INLINE pack operator+(int64_t rhs) const noexcept {
            return pack(_mm_add_epi64(data_, _mm_set1_epi64x(rhs)));
        }
        OPTINUM_INLINE pack operator-(int64_t rhs) const noexcept {
            return pack(_mm_sub_epi64(data_, _mm_set1_epi64x(rhs)));
        }
        OPTINUM_INLINE pack operator*(int64_t rhs) const noexcept { return *this * pack(rhs); }

        // Compound assignment
        OPTINUM_INLINE pack &operator+=(const pack &rhs) noexcept {
            data_ = _mm_add_epi64(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator-=(const pack &rhs) noexcept {
            data_ = _mm_sub_epi64(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator*=(const pack &rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }

        // Bitwise
        OPTINUM_INLINE pack operator&(const pack &rhs) const noexcept { return pack(_mm_and_si128(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator|(const pack &rhs) const noexcept { return pack(_mm_or_si128(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator^(const pack &rhs) const noexcept { return pack(_mm_xor_si128(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator~() const noexcept { return pack(_mm_xor_si128(data_, _mm_set1_epi64x(-1LL))); }

        // Shifts
        OPTINUM_INLINE pack operator<<(int count) const noexcept { return pack(_mm_slli_epi64(data_, count)); }
        OPTINUM_INLINE pack operator>>(int count) const noexcept {
            // SSE2 doesn't have arithmetic shift for 64-bit, emulate
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), data_);
            tmp[0] >>= count;
            tmp[1] >>= count;
            return pack(_mm_load_si128(reinterpret_cast<const __m128i *>(tmp)));
        }
        OPTINUM_INLINE pack shr_logical(int count) const noexcept { return pack(_mm_srli_epi64(data_, count)); }

        // Reductions
        OPTINUM_INLINE int64_t hsum() const noexcept {
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), data_);
            return tmp[0] + tmp[1];
        }
        OPTINUM_INLINE int64_t hmin() const noexcept {
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), data_);
            return (tmp[0] < tmp[1]) ? tmp[0] : tmp[1];
        }
        OPTINUM_INLINE int64_t hmax() const noexcept {
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), data_);
            return (tmp[0] > tmp[1]) ? tmp[0] : tmp[1];
        }
        OPTINUM_INLINE int64_t hprod() const noexcept {
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), data_);
            return tmp[0] * tmp[1];
        }

        // Abs
        OPTINUM_INLINE pack abs() const noexcept {
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(tmp), data_);
            tmp[0] = (tmp[0] < 0) ? -tmp[0] : tmp[0];
            tmp[1] = (tmp[1] < 0) ? -tmp[1] : tmp[1];
            return pack(_mm_load_si128(reinterpret_cast<const __m128i *>(tmp)));
        }

        // Min/Max
        OPTINUM_INLINE static pack min(const pack &a, const pack &b) noexcept {
            alignas(16) int64_t ta[2], tb[2], r[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(ta), a.data_);
            _mm_store_si128(reinterpret_cast<__m128i *>(tb), b.data_);
            r[0] = (ta[0] < tb[0]) ? ta[0] : tb[0];
            r[1] = (ta[1] < tb[1]) ? ta[1] : tb[1];
            return pack(_mm_load_si128(reinterpret_cast<const __m128i *>(r)));
        }
        OPTINUM_INLINE static pack max(const pack &a, const pack &b) noexcept {
            alignas(16) int64_t ta[2], tb[2], r[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(ta), a.data_);
            _mm_store_si128(reinterpret_cast<__m128i *>(tb), b.data_);
            r[0] = (ta[0] > tb[0]) ? ta[0] : tb[0];
            r[1] = (ta[1] > tb[1]) ? ta[1] : tb[1];
            return pack(_mm_load_si128(reinterpret_cast<const __m128i *>(r)));
        }

        // Dot product
        OPTINUM_INLINE int64_t dot(const pack &other) const noexcept { return (*this * other).hsum(); }
    };

} // namespace optinum::simd

#endif // OPTINUM_HAS_SSE2
