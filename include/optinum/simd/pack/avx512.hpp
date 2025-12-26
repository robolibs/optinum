#pragma once

// =============================================================================
// optinum/simd/pack/avx512.hpp
// AVX-512 specializations for pack<float,16>, pack<double,8>, pack<int32_t,16>, pack<int64_t,8>
// =============================================================================

#include <optinum/simd/mask.hpp>
#include <optinum/simd/pack/pack.hpp>

#ifdef OPTINUM_HAS_AVX512F

namespace optinum::simd {

    namespace detail {

        // Horizontal sum for __m512
        OPTINUM_INLINE float hsum_ps512(__m512 v) noexcept {
            // Reduce 512 -> 256 -> 128 -> scalar
            __m256 low = _mm512_castps512_ps256(v);
            __m256 high = _mm512_extractf32x8_ps(v, 1);
            __m256 sum256 = _mm256_add_ps(low, high);

            __m128 low128 = _mm256_castps256_ps128(sum256);
            __m128 high128 = _mm256_extractf128_ps(sum256, 1);
            __m128 sum128 = _mm_add_ps(low128, high128);

#ifdef OPTINUM_HAS_SSE3
            __m128 shuf = _mm_movehdup_ps(sum128);
            __m128 sums = _mm_add_ps(sum128, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
#else
            __m128 shuf = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 sums = _mm_add_ps(sum128, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
#endif
        }

        // Horizontal sum for __m512d
        OPTINUM_INLINE double hsum_pd512(__m512d v) noexcept {
            __m256d low = _mm512_castpd512_pd256(v);
            __m256d high = _mm512_extractf64x4_pd(v, 1);
            __m256d sum256 = _mm256_add_pd(low, high);

            __m128d low128 = _mm256_castpd256_pd128(sum256);
            __m128d high128 = _mm256_extractf128_pd(sum256, 1);
            __m128d sum128 = _mm_add_pd(low128, high128);

            __m128d shuf = _mm_shuffle_pd(sum128, sum128, 1);
            __m128d sums = _mm_add_pd(sum128, shuf);
            return _mm_cvtsd_f64(sums);
        }

        // Negate using XOR (flip sign bit)
        OPTINUM_INLINE __m512 neg_ps512(__m512 v) noexcept {
            return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(v), _mm512_set1_epi32(0x80000000)));
        }

        OPTINUM_INLINE __m512d neg_pd512(__m512d v) noexcept {
            return _mm512_castsi512_pd(
                _mm512_xor_si512(_mm512_castpd_si512(v), _mm512_set1_epi64(0x8000000000000000LL)));
        }

        // Absolute value using ANDNOT (clear sign bit)
        OPTINUM_INLINE __m512 abs_ps512(__m512 v) noexcept {
            return _mm512_castsi512_ps(_mm512_andnot_si512(_mm512_set1_epi32(0x80000000), _mm512_castps_si512(v)));
        }

        OPTINUM_INLINE __m512d abs_pd512(__m512d v) noexcept {
            return _mm512_castsi512_pd(
                _mm512_andnot_si512(_mm512_set1_epi64(0x8000000000000000LL), _mm512_castpd_si512(v)));
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

    // =============================================================================
    // pack<float, 16> - AVX-512 (16 x 32-bit float)
    // =============================================================================

    template <> struct pack<float, 16> {
        using value_type = float;
        using native_type = __m512;
        static constexpr std::size_t width = 16;

        native_type data_;

        OPTINUM_INLINE pack() noexcept : data_(_mm512_setzero_ps()) {}
        OPTINUM_INLINE explicit pack(float val) noexcept : data_(_mm512_set1_ps(val)) {}
        OPTINUM_INLINE explicit pack(native_type v) noexcept : data_(v) {}

        OPTINUM_INLINE pack &operator=(float val) noexcept {
            data_ = _mm512_set1_ps(val);
            return *this;
        }
        OPTINUM_INLINE pack &operator=(native_type v) noexcept {
            data_ = v;
            return *this;
        }

        OPTINUM_INLINE static pack load(const float *ptr) noexcept { return pack(_mm512_load_ps(ptr)); }
        OPTINUM_INLINE static pack loadu(const float *ptr) noexcept { return pack(_mm512_loadu_ps(ptr)); }
        OPTINUM_INLINE void store(float *ptr) const noexcept { _mm512_store_ps(ptr, data_); }
        OPTINUM_INLINE void storeu(float *ptr) const noexcept { _mm512_storeu_ps(ptr, data_); }

        OPTINUM_INLINE float operator[](std::size_t i) const noexcept {
            alignas(64) float tmp[16];
            _mm512_store_ps(tmp, data_);
            return tmp[i];
        }

        OPTINUM_INLINE pack operator+(const pack &rhs) const noexcept { return pack(_mm512_add_ps(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator-(const pack &rhs) const noexcept { return pack(_mm512_sub_ps(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator*(const pack &rhs) const noexcept { return pack(_mm512_mul_ps(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator/(const pack &rhs) const noexcept { return pack(_mm512_div_ps(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator-() const noexcept { return pack(detail::neg_ps512(data_)); }

        OPTINUM_INLINE pack operator+(float rhs) const noexcept {
            return pack(_mm512_add_ps(data_, _mm512_set1_ps(rhs)));
        }
        OPTINUM_INLINE pack operator-(float rhs) const noexcept {
            return pack(_mm512_sub_ps(data_, _mm512_set1_ps(rhs)));
        }
        OPTINUM_INLINE pack operator*(float rhs) const noexcept {
            return pack(_mm512_mul_ps(data_, _mm512_set1_ps(rhs)));
        }
        OPTINUM_INLINE pack operator/(float rhs) const noexcept {
            return pack(_mm512_div_ps(data_, _mm512_set1_ps(rhs)));
        }

        OPTINUM_INLINE pack &operator+=(const pack &rhs) noexcept {
            data_ = _mm512_add_ps(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator-=(const pack &rhs) noexcept {
            data_ = _mm512_sub_ps(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator*=(const pack &rhs) noexcept {
            data_ = _mm512_mul_ps(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator/=(const pack &rhs) noexcept {
            data_ = _mm512_div_ps(data_, rhs.data_);
            return *this;
        }

        OPTINUM_INLINE float hsum() const noexcept { return detail::hsum_ps512(data_); }
        OPTINUM_INLINE float hmin() const noexcept { return _mm512_reduce_min_ps(data_); }
        OPTINUM_INLINE float hmax() const noexcept { return _mm512_reduce_max_ps(data_); }
        OPTINUM_INLINE float hprod() const noexcept {
            alignas(64) float tmp[16];
            _mm512_store_ps(tmp, data_);
            float r = tmp[0];
            for (int i = 1; i < 16; ++i)
                r *= tmp[i];
            return r;
        }

        OPTINUM_INLINE pack sqrt() const noexcept { return pack(_mm512_sqrt_ps(data_)); }
        OPTINUM_INLINE pack rsqrt() const noexcept { return pack(_mm512_rsqrt14_ps(data_)); }
        OPTINUM_INLINE pack abs() const noexcept { return pack(detail::abs_ps512(data_)); }
        OPTINUM_INLINE pack rcp() const noexcept { return pack(_mm512_rcp14_ps(data_)); }

        OPTINUM_INLINE static pack min(const pack &a, const pack &b) noexcept {
            return pack(_mm512_min_ps(a.data_, b.data_));
        }

        OPTINUM_INLINE static pack max(const pack &a, const pack &b) noexcept {
            return pack(_mm512_max_ps(a.data_, b.data_));
        }

        OPTINUM_INLINE static pack fma(const pack &a, const pack &b, const pack &c) noexcept {
            return pack(_mm512_fmadd_ps(a.data_, b.data_, c.data_));
        }

        OPTINUM_INLINE static pack fms(const pack &a, const pack &b, const pack &c) noexcept {
            return pack(_mm512_fmsub_ps(a.data_, b.data_, c.data_));
        }

        OPTINUM_INLINE float dot(const pack &other) const noexcept {
            return detail::hsum_ps512(_mm512_mul_ps(data_, other.data_));
        }

        // ==========================================================================
        // Utility Functions
        // ==========================================================================

        // set() - Static factory
        OPTINUM_INLINE static pack set(float a, float b, float c, float d, float e, float f, float g, float h, float i,
                                       float j, float k, float l, float m, float n, float o, float p) noexcept {
            return pack(_mm512_setr_ps(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p));
        }

        // set_sequential() - Fill with sequential values
        OPTINUM_INLINE static pack set_sequential(float start, float step = 1.0f) noexcept {
            return pack(_mm512_setr_ps(start, start + step, start + 2.0f * step, start + 3.0f * step,
                                       start + 4.0f * step, start + 5.0f * step, start + 6.0f * step,
                                       start + 7.0f * step, start + 8.0f * step, start + 9.0f * step,
                                       start + 10.0f * step, start + 11.0f * step, start + 12.0f * step,
                                       start + 13.0f * step, start + 14.0f * step, start + 15.0f * step));
        }

        // reverse() - Reverse lane order using permute
        OPTINUM_INLINE pack reverse() const noexcept {
            __m512i indices = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            return pack(_mm512_permutexvar_ps(indices, data_));
        }

        // rotate<N>() - Rotate lanes by N positions
        template <int N> OPTINUM_INLINE pack rotate() const noexcept {
            constexpr int shift = ((N % 16) + 16) % 16;
            if constexpr (shift == 0)
                return *this;

            __m512i indices =
                _mm512_setr_epi32((shift + 0) % 16, (shift + 1) % 16, (shift + 2) % 16, (shift + 3) % 16,
                                  (shift + 4) % 16, (shift + 5) % 16, (shift + 6) % 16, (shift + 7) % 16,
                                  (shift + 8) % 16, (shift + 9) % 16, (shift + 10) % 16, (shift + 11) % 16,
                                  (shift + 12) % 16, (shift + 13) % 16, (shift + 14) % 16, (shift + 15) % 16);
            return pack(_mm512_permutexvar_ps(indices, data_));
        }

        // shift<N>() - Shift lanes, fill with zero
        template <int N> OPTINUM_INLINE pack shift() const noexcept {
            if constexpr (N == 0)
                return *this;
            else if constexpr (N >= 16 || N <= -16)
                return pack(_mm512_setzero_ps());

            alignas(64) float tmp[16];
            _mm512_store_ps(tmp, data_);
            alignas(64) float result[16] = {0};
            if constexpr (N > 0) {
                for (int i = 0; i < 16 - N; ++i) {
                    result[i] = tmp[i + N];
                }
            } else {
                for (int i = -N; i < 16; ++i) {
                    result[i] = tmp[i + N];
                }
            }
            return pack(_mm512_load_ps(result));
        }

        // cast_to_int() - Convert float to int32
        OPTINUM_INLINE __m512i cast_to_int() const noexcept { return _mm512_cvtps_epi32(data_); }

        // gather() - Load from non-contiguous memory (AVX-512 native)
        OPTINUM_INLINE static pack gather(const float *base, const int32_t *indices) noexcept {
            __m512i idx = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(indices));
            return pack(_mm512_i32gather_ps(idx, base, 4)); // Scale=4 for float
        }

        // scatter() - Store to non-contiguous memory (AVX-512 native)
        OPTINUM_INLINE void scatter(float *base, const int32_t *indices) const noexcept {
            __m512i idx = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(indices));
            _mm512_i32scatter_ps(base, idx, data_, 4); // Scale=4 for float
        }
    };

    // get<I>() - Compile-time lane extraction for pack<float, 16>
    template <std::size_t I> OPTINUM_INLINE float get(const pack<float, 16> &p) noexcept {
        static_assert(I < 16, "Index out of bounds for pack<float, 16>");
        if constexpr (I < 4) {
            __m128 lane = _mm512_castps512_ps128(p.data_);
            if constexpr (I == 0)
                return _mm_cvtss_f32(lane);
            else if constexpr (I == 1)
                return _mm_cvtss_f32(_mm_shuffle_ps(lane, lane, _MM_SHUFFLE(1, 1, 1, 1)));
            else if constexpr (I == 2)
                return _mm_cvtss_f32(_mm_shuffle_ps(lane, lane, _MM_SHUFFLE(2, 2, 2, 2)));
            else
                return _mm_cvtss_f32(_mm_shuffle_ps(lane, lane, _MM_SHUFFLE(3, 3, 3, 3)));
        } else if constexpr (I < 8) {
            __m128 lane = _mm512_extractf32x4_ps(p.data_, 1);
            constexpr int local_i = I - 4;
            if constexpr (local_i == 0)
                return _mm_cvtss_f32(lane);
            else if constexpr (local_i == 1)
                return _mm_cvtss_f32(_mm_shuffle_ps(lane, lane, _MM_SHUFFLE(1, 1, 1, 1)));
            else if constexpr (local_i == 2)
                return _mm_cvtss_f32(_mm_shuffle_ps(lane, lane, _MM_SHUFFLE(2, 2, 2, 2)));
            else
                return _mm_cvtss_f32(_mm_shuffle_ps(lane, lane, _MM_SHUFFLE(3, 3, 3, 3)));
        } else if constexpr (I < 12) {
            __m128 lane = _mm512_extractf32x4_ps(p.data_, 2);
            constexpr int local_i = I - 8;
            if constexpr (local_i == 0)
                return _mm_cvtss_f32(lane);
            else if constexpr (local_i == 1)
                return _mm_cvtss_f32(_mm_shuffle_ps(lane, lane, _MM_SHUFFLE(1, 1, 1, 1)));
            else if constexpr (local_i == 2)
                return _mm_cvtss_f32(_mm_shuffle_ps(lane, lane, _MM_SHUFFLE(2, 2, 2, 2)));
            else
                return _mm_cvtss_f32(_mm_shuffle_ps(lane, lane, _MM_SHUFFLE(3, 3, 3, 3)));
        } else {
            __m128 lane = _mm512_extractf32x4_ps(p.data_, 3);
            constexpr int local_i = I - 12;
            if constexpr (local_i == 0)
                return _mm_cvtss_f32(lane);
            else if constexpr (local_i == 1)
                return _mm_cvtss_f32(_mm_shuffle_ps(lane, lane, _MM_SHUFFLE(1, 1, 1, 1)));
            else if constexpr (local_i == 2)
                return _mm_cvtss_f32(_mm_shuffle_ps(lane, lane, _MM_SHUFFLE(2, 2, 2, 2)));
            else
                return _mm_cvtss_f32(_mm_shuffle_ps(lane, lane, _MM_SHUFFLE(3, 3, 3, 3)));
        }
    }

    // =============================================================================
    // pack<double, 8> - AVX-512 (8 x 64-bit double)
    // =============================================================================

    template <> struct pack<double, 8> {
        using value_type = double;
        using native_type = __m512d;
        static constexpr std::size_t width = 8;

        native_type data_;

        OPTINUM_INLINE pack() noexcept : data_(_mm512_setzero_pd()) {}
        OPTINUM_INLINE explicit pack(double val) noexcept : data_(_mm512_set1_pd(val)) {}
        OPTINUM_INLINE explicit pack(native_type v) noexcept : data_(v) {}

        OPTINUM_INLINE pack &operator=(double val) noexcept {
            data_ = _mm512_set1_pd(val);
            return *this;
        }
        OPTINUM_INLINE pack &operator=(native_type v) noexcept {
            data_ = v;
            return *this;
        }

        OPTINUM_INLINE static pack load(const double *ptr) noexcept { return pack(_mm512_load_pd(ptr)); }
        OPTINUM_INLINE static pack loadu(const double *ptr) noexcept { return pack(_mm512_loadu_pd(ptr)); }
        OPTINUM_INLINE void store(double *ptr) const noexcept { _mm512_store_pd(ptr, data_); }
        OPTINUM_INLINE void storeu(double *ptr) const noexcept { _mm512_storeu_pd(ptr, data_); }

        OPTINUM_INLINE double operator[](std::size_t i) const noexcept {
            alignas(64) double tmp[8];
            _mm512_store_pd(tmp, data_);
            return tmp[i];
        }

        OPTINUM_INLINE pack operator+(const pack &rhs) const noexcept { return pack(_mm512_add_pd(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator-(const pack &rhs) const noexcept { return pack(_mm512_sub_pd(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator*(const pack &rhs) const noexcept { return pack(_mm512_mul_pd(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator/(const pack &rhs) const noexcept { return pack(_mm512_div_pd(data_, rhs.data_)); }
        OPTINUM_INLINE pack operator-() const noexcept { return pack(detail::neg_pd512(data_)); }

        OPTINUM_INLINE pack operator+(double rhs) const noexcept {
            return pack(_mm512_add_pd(data_, _mm512_set1_pd(rhs)));
        }
        OPTINUM_INLINE pack operator-(double rhs) const noexcept {
            return pack(_mm512_sub_pd(data_, _mm512_set1_pd(rhs)));
        }
        OPTINUM_INLINE pack operator*(double rhs) const noexcept {
            return pack(_mm512_mul_pd(data_, _mm512_set1_pd(rhs)));
        }
        OPTINUM_INLINE pack operator/(double rhs) const noexcept {
            return pack(_mm512_div_pd(data_, _mm512_set1_pd(rhs)));
        }

        OPTINUM_INLINE pack &operator+=(const pack &rhs) noexcept {
            data_ = _mm512_add_pd(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator-=(const pack &rhs) noexcept {
            data_ = _mm512_sub_pd(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator*=(const pack &rhs) noexcept {
            data_ = _mm512_mul_pd(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator/=(const pack &rhs) noexcept {
            data_ = _mm512_div_pd(data_, rhs.data_);
            return *this;
        }

        OPTINUM_INLINE double hsum() const noexcept { return detail::hsum_pd512(data_); }
        OPTINUM_INLINE double hmin() const noexcept { return _mm512_reduce_min_pd(data_); }
        OPTINUM_INLINE double hmax() const noexcept { return _mm512_reduce_max_pd(data_); }
        OPTINUM_INLINE double hprod() const noexcept {
            alignas(64) double tmp[8];
            _mm512_store_pd(tmp, data_);
            double r = tmp[0];
            for (int i = 1; i < 8; ++i)
                r *= tmp[i];
            return r;
        }

        OPTINUM_INLINE pack sqrt() const noexcept { return pack(_mm512_sqrt_pd(data_)); }
        OPTINUM_INLINE pack rsqrt() const noexcept { return pack(_mm512_rsqrt14_pd(data_)); }
        OPTINUM_INLINE pack abs() const noexcept { return pack(detail::abs_pd512(data_)); }
        OPTINUM_INLINE pack rcp() const noexcept { return pack(_mm512_rcp14_pd(data_)); }

        OPTINUM_INLINE static pack min(const pack &a, const pack &b) noexcept {
            return pack(_mm512_min_pd(a.data_, b.data_));
        }

        OPTINUM_INLINE static pack max(const pack &a, const pack &b) noexcept {
            return pack(_mm512_max_pd(a.data_, b.data_));
        }

        OPTINUM_INLINE static pack fma(const pack &a, const pack &b, const pack &c) noexcept {
            return pack(_mm512_fmadd_pd(a.data_, b.data_, c.data_));
        }

        OPTINUM_INLINE static pack fms(const pack &a, const pack &b, const pack &c) noexcept {
            return pack(_mm512_fmsub_pd(a.data_, b.data_, c.data_));
        }

        OPTINUM_INLINE double dot(const pack &other) const noexcept {
            return detail::hsum_pd512(_mm512_mul_pd(data_, other.data_));
        }

        // ==========================================================================
        // Utility Functions
        // ==========================================================================

        // set() - Static factory
        OPTINUM_INLINE static pack set(double a, double b, double c, double d, double e, double f, double g,
                                       double h) noexcept {
            return pack(_mm512_setr_pd(a, b, c, d, e, f, g, h));
        }

        // set_sequential() - Fill with sequential values
        OPTINUM_INLINE static pack set_sequential(double start, double step = 1.0) noexcept {
            return pack(_mm512_setr_pd(start, start + step, start + 2.0 * step, start + 3.0 * step, start + 4.0 * step,
                                       start + 5.0 * step, start + 6.0 * step, start + 7.0 * step));
        }

        // reverse() - Reverse lane order using permute
        OPTINUM_INLINE pack reverse() const noexcept {
            __m512i indices = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
            return pack(_mm512_permutexvar_pd(indices, data_));
        }

        // rotate<N>() - Rotate lanes by N positions
        template <int N> OPTINUM_INLINE pack rotate() const noexcept {
            constexpr int shift = ((N % 8) + 8) % 8;
            if constexpr (shift == 0)
                return *this;

            __m512i indices = _mm512_setr_epi64((shift + 0) % 8, (shift + 1) % 8, (shift + 2) % 8, (shift + 3) % 8,
                                                (shift + 4) % 8, (shift + 5) % 8, (shift + 6) % 8, (shift + 7) % 8);
            return pack(_mm512_permutexvar_pd(indices, data_));
        }

        // shift<N>() - Shift lanes, fill with zero
        template <int N> OPTINUM_INLINE pack shift() const noexcept {
            if constexpr (N == 0)
                return *this;
            else if constexpr (N >= 8 || N <= -8)
                return pack(_mm512_setzero_pd());

            alignas(64) double tmp[8];
            _mm512_store_pd(tmp, data_);
            alignas(64) double result[8] = {0};
            if constexpr (N > 0) {
                for (int i = 0; i < 8 - N; ++i) {
                    result[i] = tmp[i + N];
                }
            } else {
                for (int i = -N; i < 8; ++i) {
                    result[i] = tmp[i + N];
                }
            }
            return pack(_mm512_load_pd(result));
        }

        // cast_to_int() - Convert double to int64
        OPTINUM_INLINE __m512i cast_to_int() const noexcept { return _mm512_cvtpd_epi64(data_); }

        // gather() - Load from non-contiguous memory (AVX-512 native)
        OPTINUM_INLINE static pack gather(const double *base, const int64_t *indices) noexcept {
            __m512i idx = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(indices));
            return pack(_mm512_i64gather_pd(idx, base, 8)); // Scale=8 for double
        }

        // scatter() - Store to non-contiguous memory (AVX-512 native)
        OPTINUM_INLINE void scatter(double *base, const int64_t *indices) const noexcept {
            __m512i idx = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(indices));
            _mm512_i64scatter_pd(base, idx, data_, 8); // Scale=8 for double
        }
    };

    // get<I>() - Compile-time lane extraction for pack<double, 8>
    template <std::size_t I> OPTINUM_INLINE double get(const pack<double, 8> &p) noexcept {
        static_assert(I < 8, "Index out of bounds for pack<double, 8>");
        if constexpr (I < 2) {
            __m128d lane = _mm512_castpd512_pd128(p.data_);
            if constexpr (I == 0)
                return _mm_cvtsd_f64(lane);
            else
                return _mm_cvtsd_f64(_mm_shuffle_pd(lane, lane, 1));
        } else if constexpr (I < 4) {
            __m128d lane = _mm512_extractf64x2_pd(p.data_, 1);
            constexpr int local_i = I - 2;
            if constexpr (local_i == 0)
                return _mm_cvtsd_f64(lane);
            else
                return _mm_cvtsd_f64(_mm_shuffle_pd(lane, lane, 1));
        } else if constexpr (I < 6) {
            __m128d lane = _mm512_extractf64x2_pd(p.data_, 2);
            constexpr int local_i = I - 4;
            if constexpr (local_i == 0)
                return _mm_cvtsd_f64(lane);
            else
                return _mm_cvtsd_f64(_mm_shuffle_pd(lane, lane, 1));
        } else {
            __m128d lane = _mm512_extractf64x2_pd(p.data_, 3);
            constexpr int local_i = I - 6;
            if constexpr (local_i == 0)
                return _mm_cvtsd_f64(lane);
            else
                return _mm_cvtsd_f64(_mm_shuffle_pd(lane, lane, 1));
        }
    }

    // =============================================================================
    // pack<int32_t, 16> - AVX-512 (16 x 32-bit signed integer)
    // =============================================================================

    template <> struct pack<int32_t, 16> {
        using value_type = int32_t;
        using native_type = __m512i;
        static constexpr std::size_t width = 16;

        native_type data_;

        OPTINUM_INLINE pack() noexcept : data_(_mm512_setzero_si512()) {}
        OPTINUM_INLINE explicit pack(int32_t val) noexcept : data_(_mm512_set1_epi32(val)) {}
        OPTINUM_INLINE explicit pack(native_type v) noexcept : data_(v) {}

        OPTINUM_INLINE pack &operator=(int32_t val) noexcept {
            data_ = _mm512_set1_epi32(val);
            return *this;
        }
        OPTINUM_INLINE pack &operator=(native_type v) noexcept {
            data_ = v;
            return *this;
        }

        OPTINUM_INLINE static pack load(const int32_t *ptr) noexcept {
            return pack(_mm512_load_si512(reinterpret_cast<const __m512i *>(ptr)));
        }
        OPTINUM_INLINE static pack loadu(const int32_t *ptr) noexcept {
            return pack(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(ptr)));
        }
        OPTINUM_INLINE void store(int32_t *ptr) const noexcept {
            _mm512_store_si512(reinterpret_cast<__m512i *>(ptr), data_);
        }
        OPTINUM_INLINE void storeu(int32_t *ptr) const noexcept {
            _mm512_storeu_si512(reinterpret_cast<__m512i *>(ptr), data_);
        }

        OPTINUM_INLINE int32_t operator[](std::size_t i) const noexcept {
            alignas(64) int32_t tmp[16];
            _mm512_store_si512(reinterpret_cast<__m512i *>(tmp), data_);
            return tmp[i];
        }

        // Arithmetic
        OPTINUM_INLINE pack operator+(const pack &rhs) const noexcept {
            return pack(_mm512_add_epi32(data_, rhs.data_));
        }
        OPTINUM_INLINE pack operator-(const pack &rhs) const noexcept {
            return pack(_mm512_sub_epi32(data_, rhs.data_));
        }
        OPTINUM_INLINE pack operator*(const pack &rhs) const noexcept {
            return pack(_mm512_mullo_epi32(data_, rhs.data_));
        }
        OPTINUM_INLINE pack operator-() const noexcept { return pack(_mm512_sub_epi32(_mm512_setzero_si512(), data_)); }

        // Scalar operations
        OPTINUM_INLINE pack operator+(int32_t rhs) const noexcept {
            return pack(_mm512_add_epi32(data_, _mm512_set1_epi32(rhs)));
        }
        OPTINUM_INLINE pack operator-(int32_t rhs) const noexcept {
            return pack(_mm512_sub_epi32(data_, _mm512_set1_epi32(rhs)));
        }
        OPTINUM_INLINE pack operator*(int32_t rhs) const noexcept {
            return pack(_mm512_mullo_epi32(data_, _mm512_set1_epi32(rhs)));
        }

        // Compound assignment
        OPTINUM_INLINE pack &operator+=(const pack &rhs) noexcept {
            data_ = _mm512_add_epi32(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator-=(const pack &rhs) noexcept {
            data_ = _mm512_sub_epi32(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator*=(const pack &rhs) noexcept {
            data_ = _mm512_mullo_epi32(data_, rhs.data_);
            return *this;
        }

        // Bitwise
        OPTINUM_INLINE pack operator&(const pack &rhs) const noexcept {
            return pack(_mm512_and_si512(data_, rhs.data_));
        }
        OPTINUM_INLINE pack operator|(const pack &rhs) const noexcept {
            return pack(_mm512_or_si512(data_, rhs.data_));
        }
        OPTINUM_INLINE pack operator^(const pack &rhs) const noexcept {
            return pack(_mm512_xor_si512(data_, rhs.data_));
        }
        OPTINUM_INLINE pack operator~() const noexcept { return pack(_mm512_xor_si512(data_, _mm512_set1_epi32(-1))); }

        // Shifts
        OPTINUM_INLINE pack operator<<(int count) const noexcept { return pack(_mm512_slli_epi32(data_, count)); }
        OPTINUM_INLINE pack operator>>(int count) const noexcept { return pack(_mm512_srai_epi32(data_, count)); }
        OPTINUM_INLINE pack shr_logical(int count) const noexcept { return pack(_mm512_srli_epi32(data_, count)); }

        // Reductions
        OPTINUM_INLINE int32_t hsum() const noexcept { return _mm512_reduce_add_epi32(data_); }
        OPTINUM_INLINE int32_t hmin() const noexcept { return _mm512_reduce_min_epi32(data_); }
        OPTINUM_INLINE int32_t hmax() const noexcept { return _mm512_reduce_max_epi32(data_); }
        OPTINUM_INLINE int32_t hprod() const noexcept { return _mm512_reduce_mul_epi32(data_); }

        // Abs
        OPTINUM_INLINE pack abs() const noexcept { return pack(_mm512_abs_epi32(data_)); }

        // Min/Max
        OPTINUM_INLINE static pack min(const pack &a, const pack &b) noexcept {
            return pack(_mm512_min_epi32(a.data_, b.data_));
        }
        OPTINUM_INLINE static pack max(const pack &a, const pack &b) noexcept {
            return pack(_mm512_max_epi32(a.data_, b.data_));
        }

        // Dot product
        OPTINUM_INLINE int32_t dot(const pack &other) const noexcept { return (*this * other).hsum(); }

        // ==========================================================================
        // Utility Functions
        // ==========================================================================

        // set() - Static factory
        OPTINUM_INLINE static pack set(int32_t a, int32_t b, int32_t c, int32_t d, int32_t e, int32_t f, int32_t g,
                                       int32_t h, int32_t i, int32_t j, int32_t k, int32_t l, int32_t m, int32_t n,
                                       int32_t o, int32_t p) noexcept {
            return pack(_mm512_setr_epi32(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p));
        }

        // set_sequential() - Fill with sequential values
        OPTINUM_INLINE static pack set_sequential(int32_t start, int32_t step = 1) noexcept {
            return pack(_mm512_setr_epi32(start, start + step, start + 2 * step, start + 3 * step, start + 4 * step,
                                          start + 5 * step, start + 6 * step, start + 7 * step, start + 8 * step,
                                          start + 9 * step, start + 10 * step, start + 11 * step, start + 12 * step,
                                          start + 13 * step, start + 14 * step, start + 15 * step));
        }

        // reverse() - Reverse lane order
        OPTINUM_INLINE pack reverse() const noexcept {
            __m512i indices = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            return pack(_mm512_permutexvar_epi32(indices, data_));
        }

        // rotate<N>() - Rotate lanes by N positions
        template <int N> OPTINUM_INLINE pack rotate() const noexcept {
            constexpr int shift = ((N % 16) + 16) % 16;
            if constexpr (shift == 0)
                return *this;

            __m512i indices =
                _mm512_setr_epi32((shift + 0) % 16, (shift + 1) % 16, (shift + 2) % 16, (shift + 3) % 16,
                                  (shift + 4) % 16, (shift + 5) % 16, (shift + 6) % 16, (shift + 7) % 16,
                                  (shift + 8) % 16, (shift + 9) % 16, (shift + 10) % 16, (shift + 11) % 16,
                                  (shift + 12) % 16, (shift + 13) % 16, (shift + 14) % 16, (shift + 15) % 16);
            return pack(_mm512_permutexvar_epi32(indices, data_));
        }

        // shift<N>() - Shift lanes, fill with zero
        template <int N> OPTINUM_INLINE pack shift() const noexcept {
            if constexpr (N == 0)
                return *this;
            else if constexpr (N >= 16 || N <= -16)
                return pack(_mm512_setzero_si512());

            alignas(64) int32_t tmp[16];
            _mm512_store_si512(reinterpret_cast<__m512i *>(tmp), data_);
            alignas(64) int32_t result[16] = {0};
            if constexpr (N > 0) {
                for (int i = 0; i < 16 - N; ++i) {
                    result[i] = tmp[i + N];
                }
            } else {
                for (int i = -N; i < 16; ++i) {
                    result[i] = tmp[i + N];
                }
            }
            return pack(_mm512_load_si512(reinterpret_cast<const __m512i *>(result)));
        }

        // gather() - Load from non-contiguous memory
        OPTINUM_INLINE static pack gather(const int32_t *base, const int32_t *indices) noexcept {
            __m512i idx = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(indices));
            return pack(_mm512_i32gather_epi32(idx, base, 4)); // Scale=4 for int32
        }

        // scatter() - Store to non-contiguous memory
        OPTINUM_INLINE void scatter(int32_t *base, const int32_t *indices) const noexcept {
            __m512i idx = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(indices));
            _mm512_i32scatter_epi32(base, idx, data_, 4); // Scale=4 for int32
        }
    };

    // get<I>() - Compile-time lane extraction for pack<int32_t, 16>
    template <std::size_t I> OPTINUM_INLINE int32_t get(const pack<int32_t, 16> &p) noexcept {
        static_assert(I < 16, "Index out of bounds for pack<int32_t, 16>");
        alignas(64) int32_t tmp[16];
        _mm512_store_si512(reinterpret_cast<__m512i *>(tmp), p.data_);
        return tmp[I];
    }

    // =============================================================================
    // pack<int64_t, 8> - AVX-512 (8 x 64-bit signed integer)
    // =============================================================================

    template <> struct pack<int64_t, 8> {
        using value_type = int64_t;
        using native_type = __m512i;
        static constexpr std::size_t width = 8;

        native_type data_;

        OPTINUM_INLINE pack() noexcept : data_(_mm512_setzero_si512()) {}
        OPTINUM_INLINE explicit pack(int64_t val) noexcept : data_(_mm512_set1_epi64(val)) {}
        OPTINUM_INLINE explicit pack(native_type v) noexcept : data_(v) {}

        OPTINUM_INLINE pack &operator=(int64_t val) noexcept {
            data_ = _mm512_set1_epi64(val);
            return *this;
        }
        OPTINUM_INLINE pack &operator=(native_type v) noexcept {
            data_ = v;
            return *this;
        }

        OPTINUM_INLINE static pack load(const int64_t *ptr) noexcept {
            return pack(_mm512_load_si512(reinterpret_cast<const __m512i *>(ptr)));
        }
        OPTINUM_INLINE static pack loadu(const int64_t *ptr) noexcept {
            return pack(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(ptr)));
        }
        OPTINUM_INLINE void store(int64_t *ptr) const noexcept {
            _mm512_store_si512(reinterpret_cast<__m512i *>(ptr), data_);
        }
        OPTINUM_INLINE void storeu(int64_t *ptr) const noexcept {
            _mm512_storeu_si512(reinterpret_cast<__m512i *>(ptr), data_);
        }

        OPTINUM_INLINE int64_t operator[](std::size_t i) const noexcept {
            alignas(64) int64_t tmp[8];
            _mm512_store_si512(reinterpret_cast<__m512i *>(tmp), data_);
            return tmp[i];
        }

        // Arithmetic
        OPTINUM_INLINE pack operator+(const pack &rhs) const noexcept {
            return pack(_mm512_add_epi64(data_, rhs.data_));
        }
        OPTINUM_INLINE pack operator-(const pack &rhs) const noexcept {
            return pack(_mm512_sub_epi64(data_, rhs.data_));
        }
        OPTINUM_INLINE pack operator*(const pack &rhs) const noexcept {
            return pack(_mm512_mullo_epi64(data_, rhs.data_)); // AVX-512DQ has native 64-bit multiply!
        }
        OPTINUM_INLINE pack operator-() const noexcept { return pack(_mm512_sub_epi64(_mm512_setzero_si512(), data_)); }

        // Scalar operations
        OPTINUM_INLINE pack operator+(int64_t rhs) const noexcept {
            return pack(_mm512_add_epi64(data_, _mm512_set1_epi64(rhs)));
        }
        OPTINUM_INLINE pack operator-(int64_t rhs) const noexcept {
            return pack(_mm512_sub_epi64(data_, _mm512_set1_epi64(rhs)));
        }
        OPTINUM_INLINE pack operator*(int64_t rhs) const noexcept {
            return pack(_mm512_mullo_epi64(data_, _mm512_set1_epi64(rhs)));
        }

        // Compound assignment
        OPTINUM_INLINE pack &operator+=(const pack &rhs) noexcept {
            data_ = _mm512_add_epi64(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator-=(const pack &rhs) noexcept {
            data_ = _mm512_sub_epi64(data_, rhs.data_);
            return *this;
        }
        OPTINUM_INLINE pack &operator*=(const pack &rhs) noexcept {
            data_ = _mm512_mullo_epi64(data_, rhs.data_);
            return *this;
        }

        // Bitwise
        OPTINUM_INLINE pack operator&(const pack &rhs) const noexcept {
            return pack(_mm512_and_si512(data_, rhs.data_));
        }
        OPTINUM_INLINE pack operator|(const pack &rhs) const noexcept {
            return pack(_mm512_or_si512(data_, rhs.data_));
        }
        OPTINUM_INLINE pack operator^(const pack &rhs) const noexcept {
            return pack(_mm512_xor_si512(data_, rhs.data_));
        }
        OPTINUM_INLINE pack operator~() const noexcept {
            return pack(_mm512_xor_si512(data_, _mm512_set1_epi64(-1LL)));
        }

        // Shifts
        OPTINUM_INLINE pack operator<<(int count) const noexcept { return pack(_mm512_slli_epi64(data_, count)); }
        OPTINUM_INLINE pack operator>>(int count) const noexcept {
            return pack(_mm512_srai_epi64(data_, count)); // AVX-512F has arithmetic shift for 64-bit!
        }
        OPTINUM_INLINE pack shr_logical(int count) const noexcept { return pack(_mm512_srli_epi64(data_, count)); }

        // Reductions
        OPTINUM_INLINE int64_t hsum() const noexcept { return _mm512_reduce_add_epi64(data_); }
        OPTINUM_INLINE int64_t hmin() const noexcept { return _mm512_reduce_min_epi64(data_); }
        OPTINUM_INLINE int64_t hmax() const noexcept { return _mm512_reduce_max_epi64(data_); }
        OPTINUM_INLINE int64_t hprod() const noexcept { return _mm512_reduce_mul_epi64(data_); }

        // Abs
        OPTINUM_INLINE pack abs() const noexcept { return pack(_mm512_abs_epi64(data_)); }

        // Min/Max
        OPTINUM_INLINE static pack min(const pack &a, const pack &b) noexcept {
            return pack(_mm512_min_epi64(a.data_, b.data_));
        }
        OPTINUM_INLINE static pack max(const pack &a, const pack &b) noexcept {
            return pack(_mm512_max_epi64(a.data_, b.data_));
        }

        // Dot product
        OPTINUM_INLINE int64_t dot(const pack &other) const noexcept { return (*this * other).hsum(); }

        // ==========================================================================
        // Utility Functions
        // ==========================================================================

        // set() - Static factory
        OPTINUM_INLINE static pack set(int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f, int64_t g,
                                       int64_t h) noexcept {
            return pack(_mm512_setr_epi64(a, b, c, d, e, f, g, h));
        }

        // set_sequential() - Fill with sequential values
        OPTINUM_INLINE static pack set_sequential(int64_t start, int64_t step = 1) noexcept {
            return pack(_mm512_setr_epi64(start, start + step, start + 2 * step, start + 3 * step, start + 4 * step,
                                          start + 5 * step, start + 6 * step, start + 7 * step));
        }

        // reverse() - Reverse lane order
        OPTINUM_INLINE pack reverse() const noexcept {
            __m512i indices = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
            return pack(_mm512_permutexvar_epi64(indices, data_));
        }

        // rotate<N>() - Rotate lanes by N positions
        template <int N> OPTINUM_INLINE pack rotate() const noexcept {
            constexpr int shift = ((N % 8) + 8) % 8;
            if constexpr (shift == 0)
                return *this;

            __m512i indices = _mm512_setr_epi64((shift + 0) % 8, (shift + 1) % 8, (shift + 2) % 8, (shift + 3) % 8,
                                                (shift + 4) % 8, (shift + 5) % 8, (shift + 6) % 8, (shift + 7) % 8);
            return pack(_mm512_permutexvar_epi64(indices, data_));
        }

        // shift<N>() - Shift lanes, fill with zero
        template <int N> OPTINUM_INLINE pack shift() const noexcept {
            if constexpr (N == 0)
                return *this;
            else if constexpr (N >= 8 || N <= -8)
                return pack(_mm512_setzero_si512());

            alignas(64) int64_t tmp[8];
            _mm512_store_si512(reinterpret_cast<__m512i *>(tmp), data_);
            alignas(64) int64_t result[8] = {0};
            if constexpr (N > 0) {
                for (int i = 0; i < 8 - N; ++i) {
                    result[i] = tmp[i + N];
                }
            } else {
                for (int i = -N; i < 8; ++i) {
                    result[i] = tmp[i + N];
                }
            }
            return pack(_mm512_load_si512(reinterpret_cast<const __m512i *>(result)));
        }

        // gather() - Load from non-contiguous memory
        OPTINUM_INLINE static pack gather(const int64_t *base, const int64_t *indices) noexcept {
            __m512i idx = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(indices));
            return pack(_mm512_i64gather_epi64(idx, base, 8)); // Scale=8 for int64
        }

        // scatter() - Store to non-contiguous memory
        OPTINUM_INLINE void scatter(int64_t *base, const int64_t *indices) const noexcept {
            __m512i idx = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(indices));
            _mm512_i64scatter_epi64(base, idx, data_, 8); // Scale=8 for int64
        }
    };

    // get<I>() - Compile-time lane extraction for pack<int64_t, 8>
    template <std::size_t I> OPTINUM_INLINE int64_t get(const pack<int64_t, 8> &p) noexcept {
        static_assert(I < 8, "Index out of bounds for pack<int64_t, 8>");
        alignas(64) int64_t tmp[8];
        _mm512_store_si512(reinterpret_cast<__m512i *>(tmp), p.data_);
        return tmp[I];
    }

} // namespace optinum::simd

#endif // OPTINUM_HAS_AVX512F
