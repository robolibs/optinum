#pragma once

// =============================================================================
// optinum/simd/intrinsic/avx.hpp
// AVX specializations for SIMDVec<float,8> and SIMDVec<double,4>
// =============================================================================

#include <optinum/simd/intrinsic/simd_vec.hpp>

#ifdef OPTINUM_HAS_AVX

namespace optinum::simd {

    namespace detail {

        OPTINUM_INLINE float hsum_ps256(__m256 v) noexcept {
            __m128 low = _mm256_castps256_ps128(v);
            __m128 high = _mm256_extractf128_ps(v, 1);
            __m128 sum = _mm_add_ps(low, high);

#ifdef OPTINUM_HAS_SSE3
            __m128 shuf = _mm_movehdup_ps(sum);
            __m128 sums = _mm_add_ps(sum, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
#else
            __m128 shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 sums = _mm_add_ps(sum, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
#endif
        }

        OPTINUM_INLINE double hsum_pd256(__m256d v) noexcept {
            __m128d low = _mm256_castpd256_pd128(v);
            __m128d high = _mm256_extractf128_pd(v, 1);
            __m128d sum = _mm_add_pd(low, high);
            __m128d shuf = _mm_shuffle_pd(sum, sum, 1);
            __m128d sums = _mm_add_pd(sum, shuf);
            return _mm_cvtsd_f64(sums);
        }

        OPTINUM_INLINE __m256 neg_ps256(__m256 v) noexcept { return _mm256_xor_ps(v, _mm256_set1_ps(-0.0f)); }
        OPTINUM_INLINE __m256d neg_pd256(__m256d v) noexcept { return _mm256_xor_pd(v, _mm256_set1_pd(-0.0)); }

        OPTINUM_INLINE __m256 abs_ps256(__m256 v) noexcept { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v); }
        OPTINUM_INLINE __m256d abs_pd256(__m256d v) noexcept { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), v); }

        template <typename T, std::size_t N> OPTINUM_INLINE T hmin_scalar(const T *ptr) noexcept {
            T r = ptr[0];
            for (std::size_t i = 1; i < N; ++i)
                r = (ptr[i] < r) ? ptr[i] : r;
            return r;
        }

        template <typename T, std::size_t N> OPTINUM_INLINE T hmax_scalar(const T *ptr) noexcept {
            T r = ptr[0];
            for (std::size_t i = 1; i < N; ++i)
                r = (ptr[i] > r) ? ptr[i] : r;
            return r;
        }

    } // namespace detail

    template <> struct SIMDVec<float, 8> {
        using value_type = float;
        using native_type = __m256;
        static constexpr std::size_t width = 8;

        native_type value;

        OPTINUM_INLINE SIMDVec() noexcept : value(_mm256_setzero_ps()) {}
        OPTINUM_INLINE explicit SIMDVec(float val) noexcept : value(_mm256_set1_ps(val)) {}
        OPTINUM_INLINE explicit SIMDVec(native_type v) noexcept : value(v) {}

        OPTINUM_INLINE SIMDVec &operator=(float val) noexcept {
            value = _mm256_set1_ps(val);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator=(native_type v) noexcept {
            value = v;
            return *this;
        }

        OPTINUM_INLINE static SIMDVec load(const float *ptr) noexcept { return SIMDVec(_mm256_load_ps(ptr)); }
        OPTINUM_INLINE static SIMDVec loadu(const float *ptr) noexcept { return SIMDVec(_mm256_loadu_ps(ptr)); }
        OPTINUM_INLINE void store(float *ptr) const noexcept { _mm256_store_ps(ptr, value); }
        OPTINUM_INLINE void storeu(float *ptr) const noexcept { _mm256_storeu_ps(ptr, value); }

        OPTINUM_INLINE float operator[](std::size_t i) const noexcept {
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, value);
            return tmp[i];
        }

        OPTINUM_INLINE SIMDVec operator+(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_add_ps(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_sub_ps(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator*(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_mul_ps(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator/(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_div_ps(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-() const noexcept { return SIMDVec(detail::neg_ps256(value)); }

        OPTINUM_INLINE SIMDVec operator+(float rhs) const noexcept {
            return SIMDVec(_mm256_add_ps(value, _mm256_set1_ps(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator-(float rhs) const noexcept {
            return SIMDVec(_mm256_sub_ps(value, _mm256_set1_ps(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator*(float rhs) const noexcept {
            return SIMDVec(_mm256_mul_ps(value, _mm256_set1_ps(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator/(float rhs) const noexcept {
            return SIMDVec(_mm256_div_ps(value, _mm256_set1_ps(rhs)));
        }

        OPTINUM_INLINE SIMDVec &operator+=(const SIMDVec &rhs) noexcept {
            value = _mm256_add_ps(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator-=(const SIMDVec &rhs) noexcept {
            value = _mm256_sub_ps(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator*=(const SIMDVec &rhs) noexcept {
            value = _mm256_mul_ps(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator/=(const SIMDVec &rhs) noexcept {
            value = _mm256_div_ps(value, rhs.value);
            return *this;
        }

        OPTINUM_INLINE float hsum() const noexcept { return detail::hsum_ps256(value); }
        OPTINUM_INLINE float hmin() const noexcept {
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, value);
            return detail::hmin_scalar<float, 8>(tmp);
        }
        OPTINUM_INLINE float hmax() const noexcept {
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, value);
            return detail::hmax_scalar<float, 8>(tmp);
        }

        OPTINUM_INLINE SIMDVec sqrt() const noexcept { return SIMDVec(_mm256_sqrt_ps(value)); }
        OPTINUM_INLINE SIMDVec rsqrt() const noexcept { return SIMDVec(_mm256_rsqrt_ps(value)); }
        OPTINUM_INLINE SIMDVec abs() const noexcept { return SIMDVec(detail::abs_ps256(value)); }
        OPTINUM_INLINE SIMDVec rcp() const noexcept { return SIMDVec(_mm256_rcp_ps(value)); }

        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm256_min_ps(a.value, b.value));
        }
        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm256_max_ps(a.value, b.value));
        }

        OPTINUM_INLINE static SIMDVec fma(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
#ifdef OPTINUM_HAS_FMA
            return SIMDVec(_mm256_fmadd_ps(a.value, b.value, c.value));
#else
            return SIMDVec(_mm256_add_ps(_mm256_mul_ps(a.value, b.value), c.value));
#endif
        }

        OPTINUM_INLINE static SIMDVec fms(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
#ifdef OPTINUM_HAS_FMA
            return SIMDVec(_mm256_fmsub_ps(a.value, b.value, c.value));
#else
            return SIMDVec(_mm256_sub_ps(_mm256_mul_ps(a.value, b.value), c.value));
#endif
        }

        OPTINUM_INLINE float dot(const SIMDVec &other) const noexcept {
            return detail::hsum_ps256(_mm256_mul_ps(value, other.value));
        }
    };

    template <> struct SIMDVec<double, 4> {
        using value_type = double;
        using native_type = __m256d;
        static constexpr std::size_t width = 4;

        native_type value;

        OPTINUM_INLINE SIMDVec() noexcept : value(_mm256_setzero_pd()) {}
        OPTINUM_INLINE explicit SIMDVec(double val) noexcept : value(_mm256_set1_pd(val)) {}
        OPTINUM_INLINE explicit SIMDVec(native_type v) noexcept : value(v) {}

        OPTINUM_INLINE SIMDVec &operator=(double val) noexcept {
            value = _mm256_set1_pd(val);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator=(native_type v) noexcept {
            value = v;
            return *this;
        }

        OPTINUM_INLINE static SIMDVec load(const double *ptr) noexcept { return SIMDVec(_mm256_load_pd(ptr)); }
        OPTINUM_INLINE static SIMDVec loadu(const double *ptr) noexcept { return SIMDVec(_mm256_loadu_pd(ptr)); }
        OPTINUM_INLINE void store(double *ptr) const noexcept { _mm256_store_pd(ptr, value); }
        OPTINUM_INLINE void storeu(double *ptr) const noexcept { _mm256_storeu_pd(ptr, value); }

        OPTINUM_INLINE double operator[](std::size_t i) const noexcept {
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, value);
            return tmp[i];
        }

        OPTINUM_INLINE SIMDVec operator+(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_add_pd(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_sub_pd(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator*(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_mul_pd(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator/(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_div_pd(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-() const noexcept { return SIMDVec(detail::neg_pd256(value)); }

        OPTINUM_INLINE SIMDVec operator+(double rhs) const noexcept {
            return SIMDVec(_mm256_add_pd(value, _mm256_set1_pd(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator-(double rhs) const noexcept {
            return SIMDVec(_mm256_sub_pd(value, _mm256_set1_pd(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator*(double rhs) const noexcept {
            return SIMDVec(_mm256_mul_pd(value, _mm256_set1_pd(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator/(double rhs) const noexcept {
            return SIMDVec(_mm256_div_pd(value, _mm256_set1_pd(rhs)));
        }

        OPTINUM_INLINE SIMDVec &operator+=(const SIMDVec &rhs) noexcept {
            value = _mm256_add_pd(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator-=(const SIMDVec &rhs) noexcept {
            value = _mm256_sub_pd(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator*=(const SIMDVec &rhs) noexcept {
            value = _mm256_mul_pd(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator/=(const SIMDVec &rhs) noexcept {
            value = _mm256_div_pd(value, rhs.value);
            return *this;
        }

        OPTINUM_INLINE double hsum() const noexcept { return detail::hsum_pd256(value); }
        OPTINUM_INLINE double hmin() const noexcept {
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, value);
            return detail::hmin_scalar<double, 4>(tmp);
        }
        OPTINUM_INLINE double hmax() const noexcept {
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, value);
            return detail::hmax_scalar<double, 4>(tmp);
        }

        OPTINUM_INLINE SIMDVec sqrt() const noexcept { return SIMDVec(_mm256_sqrt_pd(value)); }
        OPTINUM_INLINE SIMDVec rsqrt() const noexcept {
            return SIMDVec(_mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(value)));
        }
        OPTINUM_INLINE SIMDVec abs() const noexcept { return SIMDVec(detail::abs_pd256(value)); }
        OPTINUM_INLINE SIMDVec rcp() const noexcept { return SIMDVec(_mm256_div_pd(_mm256_set1_pd(1.0), value)); }

        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm256_min_pd(a.value, b.value));
        }
        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm256_max_pd(a.value, b.value));
        }

        OPTINUM_INLINE static SIMDVec fma(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
#ifdef OPTINUM_HAS_FMA
            return SIMDVec(_mm256_fmadd_pd(a.value, b.value, c.value));
#else
            return SIMDVec(_mm256_add_pd(_mm256_mul_pd(a.value, b.value), c.value));
#endif
        }

        OPTINUM_INLINE static SIMDVec fms(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
#ifdef OPTINUM_HAS_FMA
            return SIMDVec(_mm256_fmsub_pd(a.value, b.value, c.value));
#else
            return SIMDVec(_mm256_sub_pd(_mm256_mul_pd(a.value, b.value), c.value));
#endif
        }

        OPTINUM_INLINE double dot(const SIMDVec &other) const noexcept {
            return detail::hsum_pd256(_mm256_mul_pd(value, other.value));
        }
    };

} // namespace optinum::simd

#endif // OPTINUM_HAS_AVX

// =============================================================================
// AVX2 Integer Types - Requires AVX2
// =============================================================================

#ifdef OPTINUM_HAS_AVX2

namespace optinum::simd {

    // =============================================================================
    // SIMDVec<int32_t, 8> - AVX2 integer (8 x 32-bit signed integers)
    // =============================================================================

    template <> struct SIMDVec<int32_t, 8> {
        using value_type = int32_t;
        using native_type = __m256i;
        static constexpr std::size_t width = 8;

        native_type value;

        OPTINUM_INLINE SIMDVec() noexcept : value(_mm256_setzero_si256()) {}
        OPTINUM_INLINE explicit SIMDVec(int32_t val) noexcept : value(_mm256_set1_epi32(val)) {}
        OPTINUM_INLINE explicit SIMDVec(native_type v) noexcept : value(v) {}

        OPTINUM_INLINE SIMDVec &operator=(int32_t val) noexcept {
            value = _mm256_set1_epi32(val);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator=(native_type v) noexcept {
            value = v;
            return *this;
        }

        OPTINUM_INLINE static SIMDVec load(const int32_t *ptr) noexcept {
            return SIMDVec(_mm256_load_si256(reinterpret_cast<const __m256i *>(ptr)));
        }
        OPTINUM_INLINE static SIMDVec loadu(const int32_t *ptr) noexcept {
            return SIMDVec(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr)));
        }
        OPTINUM_INLINE void store(int32_t *ptr) const noexcept {
            _mm256_store_si256(reinterpret_cast<__m256i *>(ptr), value);
        }
        OPTINUM_INLINE void storeu(int32_t *ptr) const noexcept {
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), value);
        }

        OPTINUM_INLINE int32_t operator[](std::size_t i) const noexcept {
            alignas(32) int32_t tmp[8];
            _mm256_store_si256(reinterpret_cast<__m256i *>(tmp), value);
            return tmp[i];
        }

        // Arithmetic
        OPTINUM_INLINE SIMDVec operator+(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_add_epi32(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_sub_epi32(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator*(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_mullo_epi32(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-() const noexcept {
            return SIMDVec(_mm256_sub_epi32(_mm256_setzero_si256(), value));
        }

        // Scalar operations
        OPTINUM_INLINE SIMDVec operator+(int32_t rhs) const noexcept {
            return SIMDVec(_mm256_add_epi32(value, _mm256_set1_epi32(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator-(int32_t rhs) const noexcept {
            return SIMDVec(_mm256_sub_epi32(value, _mm256_set1_epi32(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator*(int32_t rhs) const noexcept {
            return SIMDVec(_mm256_mullo_epi32(value, _mm256_set1_epi32(rhs)));
        }

        // Compound assignment
        OPTINUM_INLINE SIMDVec &operator+=(const SIMDVec &rhs) noexcept {
            value = _mm256_add_epi32(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator-=(const SIMDVec &rhs) noexcept {
            value = _mm256_sub_epi32(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator*=(const SIMDVec &rhs) noexcept {
            value = _mm256_mullo_epi32(value, rhs.value);
            return *this;
        }

        // Bitwise
        OPTINUM_INLINE SIMDVec operator&(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_and_si256(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator|(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_or_si256(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator^(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_xor_si256(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator~() const noexcept {
            return SIMDVec(_mm256_xor_si256(value, _mm256_set1_epi32(-1)));
        }

        // Shifts
        OPTINUM_INLINE SIMDVec operator<<(int count) const noexcept { return SIMDVec(_mm256_slli_epi32(value, count)); }
        OPTINUM_INLINE SIMDVec operator>>(int count) const noexcept { return SIMDVec(_mm256_srai_epi32(value, count)); }
        OPTINUM_INLINE SIMDVec shr_logical(int count) const noexcept {
            return SIMDVec(_mm256_srli_epi32(value, count));
        }

        // Reductions
        OPTINUM_INLINE int32_t hsum() const noexcept {
            __m128i low = _mm256_castsi256_si128(value);
            __m128i high = _mm256_extracti128_si256(value, 1);
            __m128i sum = _mm_add_epi32(low, high);
            sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
            sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_cvtsi128_si32(sum);
        }
        OPTINUM_INLINE int32_t hmin() const noexcept {
            alignas(32) int32_t tmp[8];
            _mm256_store_si256(reinterpret_cast<__m256i *>(tmp), value);
            return detail::hmin_scalar<int32_t, 8>(tmp);
        }
        OPTINUM_INLINE int32_t hmax() const noexcept {
            alignas(32) int32_t tmp[8];
            _mm256_store_si256(reinterpret_cast<__m256i *>(tmp), value);
            return detail::hmax_scalar<int32_t, 8>(tmp);
        }
        OPTINUM_INLINE int32_t hprod() const noexcept {
            alignas(32) int32_t tmp[8];
            _mm256_store_si256(reinterpret_cast<__m256i *>(tmp), value);
            int32_t r = tmp[0];
            for (int i = 1; i < 8; ++i)
                r *= tmp[i];
            return r;
        }

        // Abs
        OPTINUM_INLINE SIMDVec abs() const noexcept { return SIMDVec(_mm256_abs_epi32(value)); }

        // Min/Max
        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm256_min_epi32(a.value, b.value));
        }
        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm256_max_epi32(a.value, b.value));
        }

        // Dot product
        OPTINUM_INLINE int32_t dot(const SIMDVec &other) const noexcept { return (*this * other).hsum(); }
    };

    // =============================================================================
    // SIMDVec<int64_t, 4> - AVX2 integer (4 x 64-bit signed integers)
    // =============================================================================

    template <> struct SIMDVec<int64_t, 4> {
        using value_type = int64_t;
        using native_type = __m256i;
        static constexpr std::size_t width = 4;

        native_type value;

        OPTINUM_INLINE SIMDVec() noexcept : value(_mm256_setzero_si256()) {}
        OPTINUM_INLINE explicit SIMDVec(int64_t val) noexcept : value(_mm256_set1_epi64x(val)) {}
        OPTINUM_INLINE explicit SIMDVec(native_type v) noexcept : value(v) {}

        OPTINUM_INLINE SIMDVec &operator=(int64_t val) noexcept {
            value = _mm256_set1_epi64x(val);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator=(native_type v) noexcept {
            value = v;
            return *this;
        }

        OPTINUM_INLINE static SIMDVec load(const int64_t *ptr) noexcept {
            return SIMDVec(_mm256_load_si256(reinterpret_cast<const __m256i *>(ptr)));
        }
        OPTINUM_INLINE static SIMDVec loadu(const int64_t *ptr) noexcept {
            return SIMDVec(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr)));
        }
        OPTINUM_INLINE void store(int64_t *ptr) const noexcept {
            _mm256_store_si256(reinterpret_cast<__m256i *>(ptr), value);
        }
        OPTINUM_INLINE void storeu(int64_t *ptr) const noexcept {
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), value);
        }

        OPTINUM_INLINE int64_t operator[](std::size_t i) const noexcept {
            alignas(32) int64_t tmp[4];
            _mm256_store_si256(reinterpret_cast<__m256i *>(tmp), value);
            return tmp[i];
        }

        // Arithmetic
        OPTINUM_INLINE SIMDVec operator+(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_add_epi64(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_sub_epi64(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator*(const SIMDVec &rhs) const noexcept {
            // No native 64-bit multiply in AVX2, emulate
            alignas(32) int64_t a[4], b[4], r[4];
            _mm256_store_si256(reinterpret_cast<__m256i *>(a), value);
            _mm256_store_si256(reinterpret_cast<__m256i *>(b), rhs.value);
            for (int i = 0; i < 4; ++i)
                r[i] = a[i] * b[i];
            return SIMDVec(_mm256_load_si256(reinterpret_cast<const __m256i *>(r)));
        }
        OPTINUM_INLINE SIMDVec operator-() const noexcept {
            return SIMDVec(_mm256_sub_epi64(_mm256_setzero_si256(), value));
        }

        // Scalar operations
        OPTINUM_INLINE SIMDVec operator+(int64_t rhs) const noexcept {
            return SIMDVec(_mm256_add_epi64(value, _mm256_set1_epi64x(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator-(int64_t rhs) const noexcept {
            return SIMDVec(_mm256_sub_epi64(value, _mm256_set1_epi64x(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator*(int64_t rhs) const noexcept { return *this * SIMDVec(rhs); }

        // Compound assignment
        OPTINUM_INLINE SIMDVec &operator+=(const SIMDVec &rhs) noexcept {
            value = _mm256_add_epi64(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator-=(const SIMDVec &rhs) noexcept {
            value = _mm256_sub_epi64(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator*=(const SIMDVec &rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }

        // Bitwise
        OPTINUM_INLINE SIMDVec operator&(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_and_si256(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator|(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_or_si256(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator^(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm256_xor_si256(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator~() const noexcept {
            return SIMDVec(_mm256_xor_si256(value, _mm256_set1_epi64x(-1LL)));
        }

        // Shifts
        OPTINUM_INLINE SIMDVec operator<<(int count) const noexcept { return SIMDVec(_mm256_slli_epi64(value, count)); }
        OPTINUM_INLINE SIMDVec operator>>(int count) const noexcept {
            // AVX2 doesn't have arithmetic shift for 64-bit, emulate
            alignas(32) int64_t tmp[4];
            _mm256_store_si256(reinterpret_cast<__m256i *>(tmp), value);
            for (int i = 0; i < 4; ++i)
                tmp[i] >>= count;
            return SIMDVec(_mm256_load_si256(reinterpret_cast<const __m256i *>(tmp)));
        }
        OPTINUM_INLINE SIMDVec shr_logical(int count) const noexcept {
            return SIMDVec(_mm256_srli_epi64(value, count));
        }

        // Reductions
        OPTINUM_INLINE int64_t hsum() const noexcept {
            alignas(32) int64_t tmp[4];
            _mm256_store_si256(reinterpret_cast<__m256i *>(tmp), value);
            return tmp[0] + tmp[1] + tmp[2] + tmp[3];
        }
        OPTINUM_INLINE int64_t hmin() const noexcept {
            alignas(32) int64_t tmp[4];
            _mm256_store_si256(reinterpret_cast<__m256i *>(tmp), value);
            int64_t r = tmp[0];
            for (int i = 1; i < 4; ++i)
                r = (tmp[i] < r) ? tmp[i] : r;
            return r;
        }
        OPTINUM_INLINE int64_t hmax() const noexcept {
            alignas(32) int64_t tmp[4];
            _mm256_store_si256(reinterpret_cast<__m256i *>(tmp), value);
            int64_t r = tmp[0];
            for (int i = 1; i < 4; ++i)
                r = (tmp[i] > r) ? tmp[i] : r;
            return r;
        }
        OPTINUM_INLINE int64_t hprod() const noexcept {
            alignas(32) int64_t tmp[4];
            _mm256_store_si256(reinterpret_cast<__m256i *>(tmp), value);
            return tmp[0] * tmp[1] * tmp[2] * tmp[3];
        }

        // Abs
        OPTINUM_INLINE SIMDVec abs() const noexcept {
            alignas(32) int64_t tmp[4];
            _mm256_store_si256(reinterpret_cast<__m256i *>(tmp), value);
            for (int i = 0; i < 4; ++i)
                tmp[i] = (tmp[i] < 0) ? -tmp[i] : tmp[i];
            return SIMDVec(_mm256_load_si256(reinterpret_cast<const __m256i *>(tmp)));
        }

        // Min/Max
        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            alignas(32) int64_t ta[4], tb[4], r[4];
            _mm256_store_si256(reinterpret_cast<__m256i *>(ta), a.value);
            _mm256_store_si256(reinterpret_cast<__m256i *>(tb), b.value);
            for (int i = 0; i < 4; ++i)
                r[i] = (ta[i] < tb[i]) ? ta[i] : tb[i];
            return SIMDVec(_mm256_load_si256(reinterpret_cast<const __m256i *>(r)));
        }
        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            alignas(32) int64_t ta[4], tb[4], r[4];
            _mm256_store_si256(reinterpret_cast<__m256i *>(ta), a.value);
            _mm256_store_si256(reinterpret_cast<__m256i *>(tb), b.value);
            for (int i = 0; i < 4; ++i)
                r[i] = (ta[i] > tb[i]) ? ta[i] : tb[i];
            return SIMDVec(_mm256_load_si256(reinterpret_cast<const __m256i *>(r)));
        }

        // Dot product
        OPTINUM_INLINE int64_t dot(const SIMDVec &other) const noexcept { return (*this * other).hsum(); }
    };

} // namespace optinum::simd

#endif // OPTINUM_HAS_AVX2
