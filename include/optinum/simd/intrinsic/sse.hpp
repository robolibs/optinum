#pragma once

// =============================================================================
// optinum/simd/intrinsic/sse.hpp
// SSE Specializations for SIMDVec<float, sse> and SIMDVec<double, sse>
// =============================================================================

#include <optinum/simd/intrinsic/simd_vec.hpp>

#ifdef OPTINUM_HAS_SSE2

namespace optinum::simd {

    // =============================================================================
    // Helper Functions for SSE Horizontal Operations
    // =============================================================================

    namespace detail {

        // Horizontal sum for __m128 (float x 4)
        OPTINUM_INLINE float hsum_ps(__m128 v) noexcept {
#ifdef OPTINUM_HAS_SSE3
            __m128 shuf = _mm_movehdup_ps(v);  // [1,1,3,3]
            __m128 sums = _mm_add_ps(v, shuf); // [0+1,1+1,2+3,3+3]
            shuf = _mm_movehl_ps(shuf, sums);  // [2+3,3+3,...]
            sums = _mm_add_ss(sums, shuf);     // [0+1+2+3,...]
            return _mm_cvtss_f32(sums);
#else
            __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 sums = _mm_add_ps(v, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
#endif
        }

        // Horizontal sum for __m128d (double x 2)
        OPTINUM_INLINE double hsum_pd(__m128d v) noexcept {
            __m128d shuf = _mm_shuffle_pd(v, v, 1); // [1, 0]
            __m128d sums = _mm_add_pd(v, shuf);     // [0+1, 1+0]
            return _mm_cvtsd_f64(sums);
        }

        // Negate __m128
        OPTINUM_INLINE __m128 neg_ps(__m128 v) noexcept { return _mm_xor_ps(v, _mm_set1_ps(-0.0f)); }

        // Negate __m128d
        OPTINUM_INLINE __m128d neg_pd(__m128d v) noexcept { return _mm_xor_pd(v, _mm_set1_pd(-0.0)); }

        // Absolute value __m128
        OPTINUM_INLINE __m128 abs_ps(__m128 v) noexcept { return _mm_andnot_ps(_mm_set1_ps(-0.0f), v); }

        // Absolute value __m128d
        OPTINUM_INLINE __m128d abs_pd(__m128d v) noexcept { return _mm_andnot_pd(_mm_set1_pd(-0.0), v); }

    } // namespace detail

    // =============================================================================
    // SIMDVec<float, simd_abi::sse> - 4 floats using __m128
    // =============================================================================

    template <> struct SIMDVec<float, simd_abi::sse> {
        using value_type = float;
        using abi_type = simd_abi::sse;
        using native_type = __m128;
        static constexpr std::size_t size = 4;

        native_type value;

        // ==========================================================================
        // Constructors
        // ==========================================================================

        OPTINUM_INLINE SIMDVec() noexcept : value(_mm_setzero_ps()) {}

        OPTINUM_INLINE explicit SIMDVec(float val) noexcept : value(_mm_set1_ps(val)) {}

        OPTINUM_INLINE SIMDVec(native_type v) noexcept : value(v) {}

        OPTINUM_INLINE SIMDVec(float a, float b, float c, float d) noexcept : value(_mm_setr_ps(a, b, c, d)) {}

        OPTINUM_INLINE SIMDVec &operator=(float val) noexcept {
            value = _mm_set1_ps(val);
            return *this;
        }

        OPTINUM_INLINE SIMDVec &operator=(native_type v) noexcept {
            value = v;
            return *this;
        }

        // ==========================================================================
        // Load / Store
        // ==========================================================================

        OPTINUM_INLINE static SIMDVec load(const float *ptr) noexcept { return SIMDVec(_mm_load_ps(ptr)); }

        OPTINUM_INLINE static SIMDVec loadu(const float *ptr) noexcept { return SIMDVec(_mm_loadu_ps(ptr)); }

        OPTINUM_INLINE void store(float *ptr) const noexcept { _mm_store_ps(ptr, value); }

        OPTINUM_INLINE void storeu(float *ptr) const noexcept { _mm_storeu_ps(ptr, value); }

        // ==========================================================================
        // Element Access
        // ==========================================================================

        OPTINUM_INLINE float operator[](std::size_t i) const noexcept {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, value);
            return tmp[i];
        }

        // ==========================================================================
        // Arithmetic Operators
        // ==========================================================================

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

        // Scalar operations
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

        // ==========================================================================
        // Compound Assignment
        // ==========================================================================

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

        // ==========================================================================
        // Reductions
        // ==========================================================================

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

        // ==========================================================================
        // Math Functions
        // ==========================================================================

        OPTINUM_INLINE SIMDVec sqrt() const noexcept { return SIMDVec(_mm_sqrt_ps(value)); }

        OPTINUM_INLINE SIMDVec rsqrt() const noexcept { return SIMDVec(_mm_rsqrt_ps(value)); }

        OPTINUM_INLINE SIMDVec abs() const noexcept { return SIMDVec(detail::abs_ps(value)); }

        OPTINUM_INLINE SIMDVec rcp() const noexcept { return SIMDVec(_mm_rcp_ps(value)); }

        // ==========================================================================
        // Min / Max
        // ==========================================================================

        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm_min_ps(a.value, b.value));
        }

        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm_max_ps(a.value, b.value));
        }

        // ==========================================================================
        // FMA (Fused Multiply-Add) - Falls back to mul+add if FMA not available
        // ==========================================================================

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

        // ==========================================================================
        // Dot Product
        // ==========================================================================

        OPTINUM_INLINE float dot(const SIMDVec &other) const noexcept {
#ifdef OPTINUM_HAS_SSE41
            return _mm_cvtss_f32(_mm_dp_ps(value, other.value, 0xFF));
#else
            return detail::hsum_ps(_mm_mul_ps(value, other.value));
#endif
        }
    };

    // =============================================================================
    // SIMDVec<double, simd_abi::sse> - 2 doubles using __m128d
    // =============================================================================

    template <> struct SIMDVec<double, simd_abi::sse> {
        using value_type = double;
        using abi_type = simd_abi::sse;
        using native_type = __m128d;
        static constexpr std::size_t size = 2;

        native_type value;

        // ==========================================================================
        // Constructors
        // ==========================================================================

        OPTINUM_INLINE SIMDVec() noexcept : value(_mm_setzero_pd()) {}

        OPTINUM_INLINE explicit SIMDVec(double val) noexcept : value(_mm_set1_pd(val)) {}

        OPTINUM_INLINE SIMDVec(native_type v) noexcept : value(v) {}

        OPTINUM_INLINE SIMDVec(double a, double b) noexcept : value(_mm_setr_pd(a, b)) {}

        OPTINUM_INLINE SIMDVec &operator=(double val) noexcept {
            value = _mm_set1_pd(val);
            return *this;
        }

        OPTINUM_INLINE SIMDVec &operator=(native_type v) noexcept {
            value = v;
            return *this;
        }

        // ==========================================================================
        // Load / Store
        // ==========================================================================

        OPTINUM_INLINE static SIMDVec load(const double *ptr) noexcept { return SIMDVec(_mm_load_pd(ptr)); }

        OPTINUM_INLINE static SIMDVec loadu(const double *ptr) noexcept { return SIMDVec(_mm_loadu_pd(ptr)); }

        OPTINUM_INLINE void store(double *ptr) const noexcept { _mm_store_pd(ptr, value); }

        OPTINUM_INLINE void storeu(double *ptr) const noexcept { _mm_storeu_pd(ptr, value); }

        // ==========================================================================
        // Element Access
        // ==========================================================================

        OPTINUM_INLINE double operator[](std::size_t i) const noexcept {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, value);
            return tmp[i];
        }

        // ==========================================================================
        // Arithmetic Operators
        // ==========================================================================

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

        // Scalar operations
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

        // ==========================================================================
        // Compound Assignment
        // ==========================================================================

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

        // ==========================================================================
        // Reductions
        // ==========================================================================

        OPTINUM_INLINE double hsum() const noexcept { return detail::hsum_pd(value); }

        OPTINUM_INLINE double hmin() const noexcept {
            __m128d tmp = _mm_min_pd(value, _mm_shuffle_pd(value, value, 1));
            return _mm_cvtsd_f64(tmp);
        }

        OPTINUM_INLINE double hmax() const noexcept {
            __m128d tmp = _mm_max_pd(value, _mm_shuffle_pd(value, value, 1));
            return _mm_cvtsd_f64(tmp);
        }

        // ==========================================================================
        // Math Functions
        // ==========================================================================

        OPTINUM_INLINE SIMDVec sqrt() const noexcept { return SIMDVec(_mm_sqrt_pd(value)); }

        OPTINUM_INLINE SIMDVec rsqrt() const noexcept {
            // No _mm_rsqrt_pd, compute manually
            return SIMDVec(_mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(value)));
        }

        OPTINUM_INLINE SIMDVec abs() const noexcept { return SIMDVec(detail::abs_pd(value)); }

        OPTINUM_INLINE SIMDVec rcp() const noexcept { return SIMDVec(_mm_div_pd(_mm_set1_pd(1.0), value)); }

        // ==========================================================================
        // Min / Max
        // ==========================================================================

        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm_min_pd(a.value, b.value));
        }

        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            return SIMDVec(_mm_max_pd(a.value, b.value));
        }

        // ==========================================================================
        // FMA
        // ==========================================================================

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

        // ==========================================================================
        // Dot Product
        // ==========================================================================

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
