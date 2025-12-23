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
