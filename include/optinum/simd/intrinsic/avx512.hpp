#pragma once

// =============================================================================
// optinum/simd/intrinsic/avx512.hpp
// AVX-512 specializations for SIMDVec<float,16> and SIMDVec<double,8>
// =============================================================================

#include <optinum/simd/intrinsic/simd_vec.hpp>

#ifdef OPTINUM_HAS_AVX512F

namespace optinum::simd {

    namespace detail {

        OPTINUM_INLINE __m512 neg_ps512(__m512 v) noexcept { return _mm512_xor_ps(v, _mm512_set1_ps(-0.0f)); }
        OPTINUM_INLINE __m512d neg_pd512(__m512d v) noexcept { return _mm512_xor_pd(v, _mm512_set1_pd(-0.0)); }

        OPTINUM_INLINE __m512 abs_ps512(__m512 v) noexcept { return _mm512_andnot_ps(_mm512_set1_ps(-0.0f), v); }
        OPTINUM_INLINE __m512d abs_pd512(__m512d v) noexcept { return _mm512_andnot_pd(_mm512_set1_pd(-0.0), v); }

        template <typename T, std::size_t N> OPTINUM_INLINE T hsum_scalar(const T *ptr) noexcept {
            T r{};
            for (std::size_t i = 0; i < N; ++i)
                r += ptr[i];
            return r;
        }

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

    template <> struct SIMDVec<float, 16> {
        using value_type = float;
        using native_type = __m512;
        static constexpr std::size_t width = 16;

        native_type value;

        OPTINUM_INLINE SIMDVec() noexcept : value(_mm512_setzero_ps()) {}
        OPTINUM_INLINE explicit SIMDVec(float val) noexcept : value(_mm512_set1_ps(val)) {}
        OPTINUM_INLINE explicit SIMDVec(native_type v) noexcept : value(v) {}

        OPTINUM_INLINE SIMDVec &operator=(float val) noexcept {
            value = _mm512_set1_ps(val);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator=(native_type v) noexcept {
            value = v;
            return *this;
        }

        OPTINUM_INLINE static SIMDVec load(const float *ptr) noexcept { return SIMDVec(_mm512_load_ps(ptr)); }
        OPTINUM_INLINE static SIMDVec loadu(const float *ptr) noexcept { return SIMDVec(_mm512_loadu_ps(ptr)); }
        OPTINUM_INLINE void store(float *ptr) const noexcept { _mm512_store_ps(ptr, value); }
        OPTINUM_INLINE void storeu(float *ptr) const noexcept { _mm512_storeu_ps(ptr, value); }

        OPTINUM_INLINE float operator[](std::size_t i) const noexcept {
            alignas(64) float tmp[16];
            _mm512_store_ps(tmp, value);
            return tmp[i];
        }

        OPTINUM_INLINE SIMDVec operator+(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_add_ps(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_sub_ps(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator*(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_mul_ps(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator/(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_div_ps(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-() const noexcept { return SIMDVec(detail::neg_ps512(value)); }

        OPTINUM_INLINE SIMDVec operator+(float rhs) const noexcept {
            return SIMDVec(_mm512_add_ps(value, _mm512_set1_ps(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator-(float rhs) const noexcept {
            return SIMDVec(_mm512_sub_ps(value, _mm512_set1_ps(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator*(float rhs) const noexcept {
            return SIMDVec(_mm512_mul_ps(value, _mm512_set1_ps(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator/(float rhs) const noexcept {
            return SIMDVec(_mm512_div_ps(value, _mm512_set1_ps(rhs)));
        }

        OPTINUM_INLINE SIMDVec &operator+=(const SIMDVec &rhs) noexcept {
            value = _mm512_add_ps(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator-=(const SIMDVec &rhs) noexcept {
            value = _mm512_sub_ps(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator*=(const SIMDVec &rhs) noexcept {
            value = _mm512_mul_ps(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator/=(const SIMDVec &rhs) noexcept {
            value = _mm512_div_ps(value, rhs.value);
            return *this;
        }

        OPTINUM_INLINE float hsum() const noexcept {
            alignas(64) float tmp[16];
            _mm512_store_ps(tmp, value);
            return detail::hsum_scalar<float, 16>(tmp);
        }
        OPTINUM_INLINE float hmin() const noexcept {
            alignas(64) float tmp[16];
            _mm512_store_ps(tmp, value);
            return detail::hmin_scalar<float, 16>(tmp);
        }
        OPTINUM_INLINE float hmax() const noexcept {
            alignas(64) float tmp[16];
            _mm512_store_ps(tmp, value);
            return detail::hmax_scalar<float, 16>(tmp);
        }

        OPTINUM_INLINE SIMDVec sqrt() const noexcept { return SIMDVec(_mm512_sqrt_ps(value)); }
        OPTINUM_INLINE SIMDVec rsqrt() const noexcept {
            return SIMDVec(_mm512_div_ps(_mm512_set1_ps(1.0f), _mm512_sqrt_ps(value)));
        }
        OPTINUM_INLINE SIMDVec abs() const noexcept { return SIMDVec(detail::abs_ps512(value)); }
        OPTINUM_INLINE SIMDVec rcp() const noexcept { return SIMDVec(_mm512_div_ps(_mm512_set1_ps(1.0f), value)); }

        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm512_min_ps(a.value, b.value));
        }
        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm512_max_ps(a.value, b.value));
        }

        OPTINUM_INLINE static SIMDVec fma(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
#ifdef OPTINUM_HAS_FMA
            return SIMDVec(_mm512_fmadd_ps(a.value, b.value, c.value));
#else
            return SIMDVec(_mm512_add_ps(_mm512_mul_ps(a.value, b.value), c.value));
#endif
        }

        OPTINUM_INLINE static SIMDVec fms(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
#ifdef OPTINUM_HAS_FMA
            return SIMDVec(_mm512_fmsub_ps(a.value, b.value, c.value));
#else
            return SIMDVec(_mm512_sub_ps(_mm512_mul_ps(a.value, b.value), c.value));
#endif
        }

        OPTINUM_INLINE float dot(const SIMDVec &other) const noexcept {
            const __m512 prod = _mm512_mul_ps(value, other.value);
            alignas(64) float tmp[16];
            _mm512_store_ps(tmp, prod);
            return detail::hsum_scalar<float, 16>(tmp);
        }
    };

    template <> struct SIMDVec<double, 8> {
        using value_type = double;
        using native_type = __m512d;
        static constexpr std::size_t width = 8;

        native_type value;

        OPTINUM_INLINE SIMDVec() noexcept : value(_mm512_setzero_pd()) {}
        OPTINUM_INLINE explicit SIMDVec(double val) noexcept : value(_mm512_set1_pd(val)) {}
        OPTINUM_INLINE explicit SIMDVec(native_type v) noexcept : value(v) {}

        OPTINUM_INLINE SIMDVec &operator=(double val) noexcept {
            value = _mm512_set1_pd(val);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator=(native_type v) noexcept {
            value = v;
            return *this;
        }

        OPTINUM_INLINE static SIMDVec load(const double *ptr) noexcept { return SIMDVec(_mm512_load_pd(ptr)); }
        OPTINUM_INLINE static SIMDVec loadu(const double *ptr) noexcept { return SIMDVec(_mm512_loadu_pd(ptr)); }
        OPTINUM_INLINE void store(double *ptr) const noexcept { _mm512_store_pd(ptr, value); }
        OPTINUM_INLINE void storeu(double *ptr) const noexcept { _mm512_storeu_pd(ptr, value); }

        OPTINUM_INLINE double operator[](std::size_t i) const noexcept {
            alignas(64) double tmp[8];
            _mm512_store_pd(tmp, value);
            return tmp[i];
        }

        OPTINUM_INLINE SIMDVec operator+(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_add_pd(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_sub_pd(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator*(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_mul_pd(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator/(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_div_pd(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-() const noexcept { return SIMDVec(detail::neg_pd512(value)); }

        OPTINUM_INLINE SIMDVec operator+(double rhs) const noexcept {
            return SIMDVec(_mm512_add_pd(value, _mm512_set1_pd(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator-(double rhs) const noexcept {
            return SIMDVec(_mm512_sub_pd(value, _mm512_set1_pd(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator*(double rhs) const noexcept {
            return SIMDVec(_mm512_mul_pd(value, _mm512_set1_pd(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator/(double rhs) const noexcept {
            return SIMDVec(_mm512_div_pd(value, _mm512_set1_pd(rhs)));
        }

        OPTINUM_INLINE SIMDVec &operator+=(const SIMDVec &rhs) noexcept {
            value = _mm512_add_pd(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator-=(const SIMDVec &rhs) noexcept {
            value = _mm512_sub_pd(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator*=(const SIMDVec &rhs) noexcept {
            value = _mm512_mul_pd(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator/=(const SIMDVec &rhs) noexcept {
            value = _mm512_div_pd(value, rhs.value);
            return *this;
        }

        OPTINUM_INLINE double hsum() const noexcept {
            alignas(64) double tmp[8];
            _mm512_store_pd(tmp, value);
            return detail::hsum_scalar<double, 8>(tmp);
        }
        OPTINUM_INLINE double hmin() const noexcept {
            alignas(64) double tmp[8];
            _mm512_store_pd(tmp, value);
            return detail::hmin_scalar<double, 8>(tmp);
        }
        OPTINUM_INLINE double hmax() const noexcept {
            alignas(64) double tmp[8];
            _mm512_store_pd(tmp, value);
            return detail::hmax_scalar<double, 8>(tmp);
        }

        OPTINUM_INLINE SIMDVec sqrt() const noexcept { return SIMDVec(_mm512_sqrt_pd(value)); }
        OPTINUM_INLINE SIMDVec rsqrt() const noexcept {
            return SIMDVec(_mm512_div_pd(_mm512_set1_pd(1.0), _mm512_sqrt_pd(value)));
        }
        OPTINUM_INLINE SIMDVec abs() const noexcept { return SIMDVec(detail::abs_pd512(value)); }
        OPTINUM_INLINE SIMDVec rcp() const noexcept { return SIMDVec(_mm512_div_pd(_mm512_set1_pd(1.0), value)); }

        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm512_min_pd(a.value, b.value));
        }
        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm512_max_pd(a.value, b.value));
        }

        OPTINUM_INLINE static SIMDVec fma(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
#ifdef OPTINUM_HAS_FMA
            return SIMDVec(_mm512_fmadd_pd(a.value, b.value, c.value));
#else
            return SIMDVec(_mm512_add_pd(_mm512_mul_pd(a.value, b.value), c.value));
#endif
        }

        OPTINUM_INLINE static SIMDVec fms(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
#ifdef OPTINUM_HAS_FMA
            return SIMDVec(_mm512_fmsub_pd(a.value, b.value, c.value));
#else
            return SIMDVec(_mm512_sub_pd(_mm512_mul_pd(a.value, b.value), c.value));
#endif
        }

        OPTINUM_INLINE double dot(const SIMDVec &other) const noexcept {
            const __m512d prod = _mm512_mul_pd(value, other.value);
            alignas(64) double tmp[8];
            _mm512_store_pd(tmp, prod);
            return detail::hsum_scalar<double, 8>(tmp);
        }
    };

    // =============================================================================
    // SIMDVec<int32_t, 16> - AVX-512 integer (16 x 32-bit signed integers)
    // =============================================================================

    template <> struct SIMDVec<int32_t, 16> {
        using value_type = int32_t;
        using native_type = __m512i;
        static constexpr std::size_t width = 16;

        native_type value;

        OPTINUM_INLINE SIMDVec() noexcept : value(_mm512_setzero_si512()) {}
        OPTINUM_INLINE explicit SIMDVec(int32_t val) noexcept : value(_mm512_set1_epi32(val)) {}
        OPTINUM_INLINE explicit SIMDVec(native_type v) noexcept : value(v) {}

        OPTINUM_INLINE SIMDVec &operator=(int32_t val) noexcept {
            value = _mm512_set1_epi32(val);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator=(native_type v) noexcept {
            value = v;
            return *this;
        }

        OPTINUM_INLINE static SIMDVec load(const int32_t *ptr) noexcept {
            return SIMDVec(_mm512_load_si512(reinterpret_cast<const __m512i *>(ptr)));
        }
        OPTINUM_INLINE static SIMDVec loadu(const int32_t *ptr) noexcept {
            return SIMDVec(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(ptr)));
        }
        OPTINUM_INLINE void store(int32_t *ptr) const noexcept {
            _mm512_store_si512(reinterpret_cast<__m512i *>(ptr), value);
        }
        OPTINUM_INLINE void storeu(int32_t *ptr) const noexcept {
            _mm512_storeu_si512(reinterpret_cast<__m512i *>(ptr), value);
        }

        OPTINUM_INLINE int32_t operator[](std::size_t i) const noexcept {
            alignas(64) int32_t tmp[16];
            _mm512_store_si512(reinterpret_cast<__m512i *>(tmp), value);
            return tmp[i];
        }

        // Arithmetic
        OPTINUM_INLINE SIMDVec operator+(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_add_epi32(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_sub_epi32(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator*(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_mullo_epi32(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-() const noexcept {
            return SIMDVec(_mm512_sub_epi32(_mm512_setzero_si512(), value));
        }

        // Scalar operations
        OPTINUM_INLINE SIMDVec operator+(int32_t rhs) const noexcept {
            return SIMDVec(_mm512_add_epi32(value, _mm512_set1_epi32(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator-(int32_t rhs) const noexcept {
            return SIMDVec(_mm512_sub_epi32(value, _mm512_set1_epi32(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator*(int32_t rhs) const noexcept {
            return SIMDVec(_mm512_mullo_epi32(value, _mm512_set1_epi32(rhs)));
        }

        // Compound assignment
        OPTINUM_INLINE SIMDVec &operator+=(const SIMDVec &rhs) noexcept {
            value = _mm512_add_epi32(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator-=(const SIMDVec &rhs) noexcept {
            value = _mm512_sub_epi32(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator*=(const SIMDVec &rhs) noexcept {
            value = _mm512_mullo_epi32(value, rhs.value);
            return *this;
        }

        // Bitwise
        OPTINUM_INLINE SIMDVec operator&(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_and_si512(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator|(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_or_si512(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator^(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_xor_si512(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator~() const noexcept {
            return SIMDVec(_mm512_xor_si512(value, _mm512_set1_epi32(-1)));
        }

        // Shifts
        OPTINUM_INLINE SIMDVec operator<<(int count) const noexcept { return SIMDVec(_mm512_slli_epi32(value, count)); }
        OPTINUM_INLINE SIMDVec operator>>(int count) const noexcept { return SIMDVec(_mm512_srai_epi32(value, count)); }
        OPTINUM_INLINE SIMDVec shr_logical(int count) const noexcept {
            return SIMDVec(_mm512_srli_epi32(value, count));
        }

        // Reductions
        OPTINUM_INLINE int32_t hsum() const noexcept { return _mm512_reduce_add_epi32(value); }
        OPTINUM_INLINE int32_t hmin() const noexcept { return _mm512_reduce_min_epi32(value); }
        OPTINUM_INLINE int32_t hmax() const noexcept { return _mm512_reduce_max_epi32(value); }
        OPTINUM_INLINE int32_t hprod() const noexcept { return _mm512_reduce_mul_epi32(value); }

        // Abs
        OPTINUM_INLINE SIMDVec abs() const noexcept { return SIMDVec(_mm512_abs_epi32(value)); }

        // Min/Max
        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm512_min_epi32(a.value, b.value));
        }
        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm512_max_epi32(a.value, b.value));
        }

        // Dot product
        OPTINUM_INLINE int32_t dot(const SIMDVec &other) const noexcept { return (*this * other).hsum(); }
    };

    // =============================================================================
    // SIMDVec<int64_t, 8> - AVX-512 integer (8 x 64-bit signed integers)
    // =============================================================================

    template <> struct SIMDVec<int64_t, 8> {
        using value_type = int64_t;
        using native_type = __m512i;
        static constexpr std::size_t width = 8;

        native_type value;

        OPTINUM_INLINE SIMDVec() noexcept : value(_mm512_setzero_si512()) {}
        OPTINUM_INLINE explicit SIMDVec(int64_t val) noexcept : value(_mm512_set1_epi64(val)) {}
        OPTINUM_INLINE explicit SIMDVec(native_type v) noexcept : value(v) {}

        OPTINUM_INLINE SIMDVec &operator=(int64_t val) noexcept {
            value = _mm512_set1_epi64(val);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator=(native_type v) noexcept {
            value = v;
            return *this;
        }

        OPTINUM_INLINE static SIMDVec load(const int64_t *ptr) noexcept {
            return SIMDVec(_mm512_load_si512(reinterpret_cast<const __m512i *>(ptr)));
        }
        OPTINUM_INLINE static SIMDVec loadu(const int64_t *ptr) noexcept {
            return SIMDVec(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(ptr)));
        }
        OPTINUM_INLINE void store(int64_t *ptr) const noexcept {
            _mm512_store_si512(reinterpret_cast<__m512i *>(ptr), value);
        }
        OPTINUM_INLINE void storeu(int64_t *ptr) const noexcept {
            _mm512_storeu_si512(reinterpret_cast<__m512i *>(ptr), value);
        }

        OPTINUM_INLINE int64_t operator[](std::size_t i) const noexcept {
            alignas(64) int64_t tmp[8];
            _mm512_store_si512(reinterpret_cast<__m512i *>(tmp), value);
            return tmp[i];
        }

        // Arithmetic
        OPTINUM_INLINE SIMDVec operator+(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_add_epi64(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator-(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_sub_epi64(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator*(const SIMDVec &rhs) const noexcept {
#ifdef OPTINUM_HAS_AVX512DQ
            return SIMDVec(_mm512_mullo_epi64(value, rhs.value));
#else
            // Fallback for AVX-512F without DQ
            alignas(64) int64_t a[8], b[8], r[8];
            _mm512_store_si512(reinterpret_cast<__m512i *>(a), value);
            _mm512_store_si512(reinterpret_cast<__m512i *>(b), rhs.value);
            for (int i = 0; i < 8; ++i)
                r[i] = a[i] * b[i];
            return SIMDVec(_mm512_load_si512(reinterpret_cast<const __m512i *>(r)));
#endif
        }
        OPTINUM_INLINE SIMDVec operator-() const noexcept {
            return SIMDVec(_mm512_sub_epi64(_mm512_setzero_si512(), value));
        }

        // Scalar operations
        OPTINUM_INLINE SIMDVec operator+(int64_t rhs) const noexcept {
            return SIMDVec(_mm512_add_epi64(value, _mm512_set1_epi64(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator-(int64_t rhs) const noexcept {
            return SIMDVec(_mm512_sub_epi64(value, _mm512_set1_epi64(rhs)));
        }
        OPTINUM_INLINE SIMDVec operator*(int64_t rhs) const noexcept { return *this * SIMDVec(rhs); }

        // Compound assignment
        OPTINUM_INLINE SIMDVec &operator+=(const SIMDVec &rhs) noexcept {
            value = _mm512_add_epi64(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator-=(const SIMDVec &rhs) noexcept {
            value = _mm512_sub_epi64(value, rhs.value);
            return *this;
        }
        OPTINUM_INLINE SIMDVec &operator*=(const SIMDVec &rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }

        // Bitwise
        OPTINUM_INLINE SIMDVec operator&(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_and_si512(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator|(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_or_si512(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator^(const SIMDVec &rhs) const noexcept {
            return SIMDVec(_mm512_xor_si512(value, rhs.value));
        }
        OPTINUM_INLINE SIMDVec operator~() const noexcept {
            return SIMDVec(_mm512_xor_si512(value, _mm512_set1_epi64(-1LL)));
        }

        // Shifts
        OPTINUM_INLINE SIMDVec operator<<(int count) const noexcept { return SIMDVec(_mm512_slli_epi64(value, count)); }
        OPTINUM_INLINE SIMDVec operator>>(int count) const noexcept { return SIMDVec(_mm512_srai_epi64(value, count)); }
        OPTINUM_INLINE SIMDVec shr_logical(int count) const noexcept {
            return SIMDVec(_mm512_srli_epi64(value, count));
        }

        // Reductions
        OPTINUM_INLINE int64_t hsum() const noexcept { return _mm512_reduce_add_epi64(value); }
        OPTINUM_INLINE int64_t hmin() const noexcept { return _mm512_reduce_min_epi64(value); }
        OPTINUM_INLINE int64_t hmax() const noexcept { return _mm512_reduce_max_epi64(value); }
        OPTINUM_INLINE int64_t hprod() const noexcept { return _mm512_reduce_mul_epi64(value); }

        // Abs
        OPTINUM_INLINE SIMDVec abs() const noexcept { return SIMDVec(_mm512_abs_epi64(value)); }

        // Min/Max
        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm512_min_epi64(a.value, b.value));
        }
        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm512_max_epi64(a.value, b.value));
        }

        // Dot product
        OPTINUM_INLINE int64_t dot(const SIMDVec &other) const noexcept { return (*this * other).hsum(); }
    };

} // namespace optinum::simd

#endif // OPTINUM_HAS_AVX512F
