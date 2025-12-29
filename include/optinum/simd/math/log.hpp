#pragma once

// =============================================================================
// optinum/simd/math/log.hpp
// Vectorized log() using pack<T,W> with SIMD intrinsics
// New clean API - replaces fast_log.hpp
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/pack/pack.hpp>
#if defined(OPTINUM_HAS_AVX)
#include <optinum/simd/pack/avx.hpp>
#endif
#if defined(OPTINUM_HAS_SSE2)
#include <optinum/simd/pack/sse.hpp>
#endif

#if defined(OPTINUM_HAS_NEON)
#include <optinum/simd/pack/neon.hpp>
#endif

namespace optinum::simd {

    // Forward declaration
    template <typename T, std::size_t W> pack<T, W> log(const pack<T, W> &x) noexcept;

    namespace detail {
        // Constants for log(x) computation
        constexpr float SQRT2_F = 1.41421356237309504880f;
        constexpr float LN2_F = 0.693147180559945309417f;

        constexpr double SQRT2_D = 1.41421356237309504880;
        constexpr double LN2_D = 0.693147180559945309417;
    } // namespace detail

    // =========================================================================
    // pack<float, 4> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE2)

#if defined(OPTINUM_HAS_SSE2)
    template <> inline pack<float, 4> log(const pack<float, 4> &x) noexcept {
        using namespace detail;

        __m128 vx = x.data_;

        __m128 vone = _mm_set1_ps(1.0f);
        __m128 vhalf = _mm_set1_ps(0.5f);
        __m128 vsqrt2 = _mm_set1_ps(SQRT2_F);
        __m128 vln2 = _mm_set1_ps(LN2_F);

        // Extract exponent
        __m128i vi = _mm_castps_si128(vx);
        __m128i vexp = _mm_srli_epi32(vi, 23);
        vexp = _mm_sub_epi32(vexp, _mm_set1_epi32(127));
        __m128 vn = _mm_cvtepi32_ps(vexp);

        // Extract mantissa
        __m128i vmant_mask = _mm_set1_epi32(0x007FFFFF);
        __m128i vexp_zero = _mm_set1_epi32(0x3F800000);
        __m128i vmant = _mm_or_si128(_mm_and_si128(vi, vmant_mask), vexp_zero);
        __m128 vm = _mm_castsi128_ps(vmant);

        // If m > sqrt(2), adjust
        __m128 vmask = _mm_cmpgt_ps(vm, vsqrt2);
        vm = _mm_blendv_ps(vm, _mm_mul_ps(vm, vhalf), vmask);
        vn = _mm_blendv_ps(vn, _mm_add_ps(vn, vone), vmask);

        // f = m - 1
        __m128 vf = _mm_sub_ps(vm, vone);

        // s = f / (2 + f)
        __m128 vs = _mm_div_ps(vf, _mm_add_ps(_mm_set1_ps(2.0f), vf));
        __m128 vs2 = _mm_mul_ps(vs, vs);

        // Polynomial
        __m128 vc9 = _mm_set1_ps(0.11111111111111111f);
        __m128 vc7 = _mm_set1_ps(0.14285714285714285f);
        __m128 vc5 = _mm_set1_ps(0.2f);
        __m128 vc3 = _mm_set1_ps(0.33333333333333333f);

#ifdef OPTINUM_HAS_FMA
        __m128 vp = _mm_fmadd_ps(vc9, vs2, vc7);
        vp = _mm_fmadd_ps(vp, vs2, vc5);
        vp = _mm_fmadd_ps(vp, vs2, vc3);
        vp = _mm_fmadd_ps(vp, vs2, vone);
#else
        __m128 vp = _mm_add_ps(_mm_mul_ps(vc9, vs2), vc7);
        vp = _mm_add_ps(_mm_mul_ps(vp, vs2), vc5);
        vp = _mm_add_ps(_mm_mul_ps(vp, vs2), vc3);
        vp = _mm_add_ps(_mm_mul_ps(vp, vs2), vone);
#endif

        __m128 vlog_m = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(2.0f), vs), vp);

#ifdef OPTINUM_HAS_FMA
        __m128 vresult = _mm_fmadd_ps(vn, vln2, vlog_m);
#else
        __m128 vresult = _mm_add_ps(_mm_mul_ps(vn, vln2), vlog_m);
#endif

        // Handle special cases
        __m128 vzero = _mm_setzero_ps();
        __m128 vneg_inf = _mm_set1_ps(-__builtin_inff());
        __m128 vnan = _mm_set1_ps(__builtin_nanf(""));

        vresult = _mm_blendv_ps(vresult, vneg_inf, _mm_cmpeq_ps(vx, vzero));
        vresult = _mm_blendv_ps(vresult, vnan, _mm_cmplt_ps(vx, vzero));

        return pack<float, 4>(vresult);
    }

#endif // OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

#endif // OPTINUM_HAS_SSE2

#if defined(OPTINUM_HAS_AVX)
    template <> inline pack<float, 8> log(const pack<float, 8> &x) noexcept {
        using namespace detail;

        __m256 vx = x.data_;

        __m256 vone = _mm256_set1_ps(1.0f);
        __m256 vhalf = _mm256_set1_ps(0.5f);
        __m256 vsqrt2 = _mm256_set1_ps(SQRT2_F);
        __m256 vln2 = _mm256_set1_ps(LN2_F);

        // Extract exponent
        __m256i vi = _mm256_castps_si256(vx);
        __m256i vexp = _mm256_srli_epi32(vi, 23);
        vexp = _mm256_sub_epi32(vexp, _mm256_set1_epi32(127));
        __m256 vn = _mm256_cvtepi32_ps(vexp);

        // Extract mantissa
        __m256i vmant_mask = _mm256_set1_epi32(0x007FFFFF);
        __m256i vexp_zero = _mm256_set1_epi32(0x3F800000);
        __m256i vmant = _mm256_or_si256(_mm256_and_si256(vi, vmant_mask), vexp_zero);
        __m256 vm = _mm256_castsi256_ps(vmant);

        // If m > sqrt(2), use m/2 and n+1
        __m256 vmask = _mm256_cmp_ps(vm, vsqrt2, _CMP_GT_OQ);
        vm = _mm256_blendv_ps(vm, _mm256_mul_ps(vm, vhalf), vmask);
        vn = _mm256_blendv_ps(vn, _mm256_add_ps(vn, vone), vmask);

        // f = m - 1
        __m256 vf = _mm256_sub_ps(vm, vone);

        // s = f / (2 + f)
        __m256 vs = _mm256_div_ps(vf, _mm256_add_ps(_mm256_set1_ps(2.0f), vf));
        __m256 vs2 = _mm256_mul_ps(vs, vs);

        // Polynomial
        __m256 vc9 = _mm256_set1_ps(0.11111111111111111f);
        __m256 vc7 = _mm256_set1_ps(0.14285714285714285f);
        __m256 vc5 = _mm256_set1_ps(0.2f);
        __m256 vc3 = _mm256_set1_ps(0.33333333333333333f);

#ifdef OPTINUM_HAS_FMA
        __m256 vp = _mm256_fmadd_ps(vc9, vs2, vc7);
        vp = _mm256_fmadd_ps(vp, vs2, vc5);
        vp = _mm256_fmadd_ps(vp, vs2, vc3);
        vp = _mm256_fmadd_ps(vp, vs2, vone);
#else
        __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc9, vs2), vc7);
        vp = _mm256_add_ps(_mm256_mul_ps(vp, vs2), vc5);
        vp = _mm256_add_ps(_mm256_mul_ps(vp, vs2), vc3);
        vp = _mm256_add_ps(_mm256_mul_ps(vp, vs2), vone);
#endif

        __m256 vlog_m = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), vs), vp);

#ifdef OPTINUM_HAS_FMA
        __m256 vresult = _mm256_fmadd_ps(vn, vln2, vlog_m);
#else
        __m256 vresult = _mm256_add_ps(_mm256_mul_ps(vn, vln2), vlog_m);
#endif

        // Handle special cases
        __m256 vzero = _mm256_setzero_ps();
        __m256 vneg_inf = _mm256_set1_ps(-__builtin_inff());
        __m256 vnan = _mm256_set1_ps(__builtin_nanf(""));

        vresult = _mm256_blendv_ps(vresult, vneg_inf, _mm256_cmp_ps(vx, vzero, _CMP_EQ_OQ));
        vresult = _mm256_blendv_ps(vresult, vnan, _mm256_cmp_ps(vx, vzero, _CMP_LT_OQ));

        return pack<float, 8>(vresult);
    }

#endif // OPTINUM_HAS_AVX

    // =========================================================================
#endif // OPTINUM_HAS_AVX

    // pack<double, 2> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE41)

    template <> inline pack<double, 2> log(const pack<double, 2> &x) noexcept {
        using namespace detail;

        __m128d vx = x.data_;

        __m128d vone = _mm_set1_pd(1.0);
        __m128d vhalf = _mm_set1_pd(0.5);
        __m128d vsqrt2 = _mm_set1_pd(SQRT2_D);
        __m128d vln2 = _mm_set1_pd(LN2_D);

        // Extract exponent (double has 11-bit exponent at bits 52-62)
        __m128i vi = _mm_castpd_si128(vx);
        __m128i vexp = _mm_srli_epi64(vi, 52);
        vexp = _mm_sub_epi64(vexp, _mm_set1_epi64x(1023));

        // Convert exponent to double (need to extract lower 32 bits and convert)
        __m128i vexp32 = _mm_shuffle_epi32(vexp, _MM_SHUFFLE(2, 0, 2, 0)); // Extract low 32 bits
        __m128d vn = _mm_cvtepi32_pd(vexp32);

        // Extract mantissa (52-bit mantissa)
        __m128i vmant_mask = _mm_set1_epi64x(0x000FFFFFFFFFFFFF);
        __m128i vexp_zero = _mm_set1_epi64x(0x3FF0000000000000); // 1.0 in double format
        __m128i vmant = _mm_or_si128(_mm_and_si128(vi, vmant_mask), vexp_zero);
        __m128d vm = _mm_castsi128_pd(vmant);

        // If m > sqrt(2), adjust
        __m128d vmask = _mm_cmpgt_pd(vm, vsqrt2);
        vm = _mm_blendv_pd(vm, _mm_mul_pd(vm, vhalf), vmask);
        vn = _mm_blendv_pd(vn, _mm_add_pd(vn, vone), vmask);

        // f = m - 1
        __m128d vf = _mm_sub_pd(vm, vone);

        // s = f / (2 + f)
        __m128d vs = _mm_div_pd(vf, _mm_add_pd(_mm_set1_pd(2.0), vf));
        __m128d vs2 = _mm_mul_pd(vs, vs);

        // Higher-order polynomial for double precision
        __m128d vc11 = _mm_set1_pd(0.090909090909090909); // 1/11
        __m128d vc9 = _mm_set1_pd(0.11111111111111111);   // 1/9
        __m128d vc7 = _mm_set1_pd(0.14285714285714285);   // 1/7
        __m128d vc5 = _mm_set1_pd(0.2);                   // 1/5
        __m128d vc3 = _mm_set1_pd(0.33333333333333333);   // 1/3

#ifdef OPTINUM_HAS_FMA
        __m128d vp = _mm_fmadd_pd(vc11, vs2, vc9);
        vp = _mm_fmadd_pd(vp, vs2, vc7);
        vp = _mm_fmadd_pd(vp, vs2, vc5);
        vp = _mm_fmadd_pd(vp, vs2, vc3);
        vp = _mm_fmadd_pd(vp, vs2, vone);
#else
        __m128d vp = _mm_add_pd(_mm_mul_pd(vc11, vs2), vc9);
        vp = _mm_add_pd(_mm_mul_pd(vp, vs2), vc7);
        vp = _mm_add_pd(_mm_mul_pd(vp, vs2), vc5);
        vp = _mm_add_pd(_mm_mul_pd(vp, vs2), vc3);
        vp = _mm_add_pd(_mm_mul_pd(vp, vs2), vone);
#endif

        __m128d vlog_m = _mm_mul_pd(_mm_mul_pd(_mm_set1_pd(2.0), vs), vp);

#ifdef OPTINUM_HAS_FMA
        __m128d vresult = _mm_fmadd_pd(vn, vln2, vlog_m);
#else
        __m128d vresult = _mm_add_pd(_mm_mul_pd(vn, vln2), vlog_m);
#endif

        // Handle special cases
        __m128d vzero = _mm_setzero_pd();
        __m128d vneg_inf = _mm_set1_pd(-__builtin_inf());
        __m128d vnan = _mm_set1_pd(__builtin_nan(""));

        vresult = _mm_blendv_pd(vresult, vneg_inf, _mm_cmpeq_pd(vx, vzero));
        vresult = _mm_blendv_pd(vresult, vnan, _mm_cmplt_pd(vx, vzero));

        return pack<double, 2>(vresult);
    }

#endif // OPTINUM_HAS_SSE41

    // =========================================================================
    // pack<double, 4> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<double, 4> log(const pack<double, 4> &x) noexcept {
        using namespace detail;

        __m256d vx = x.data_;

        __m256d vone = _mm256_set1_pd(1.0);
        __m256d vhalf = _mm256_set1_pd(0.5);
        __m256d vsqrt2 = _mm256_set1_pd(SQRT2_D);
        __m256d vln2 = _mm256_set1_pd(LN2_D);

        // Extract exponent
        __m256i vi = _mm256_castpd_si256(vx);
        __m256i vexp = _mm256_srli_epi64(vi, 52);
        vexp = _mm256_sub_epi64(vexp, _mm256_set1_epi64x(1023));

        // Convert exponent to double (extract low 32 bits from each 64-bit lane)
        __m128i vexp_low = _mm256_castsi256_si128(vexp);
        __m128i vexp_high = _mm256_extractf128_si256(vexp, 1);
        __m128i vexp32_low = _mm_shuffle_epi32(vexp_low, _MM_SHUFFLE(2, 0, 2, 0));
        __m128i vexp32_high = _mm_shuffle_epi32(vexp_high, _MM_SHUFFLE(2, 0, 2, 0));
        __m128i vexp32 = _mm_unpacklo_epi64(vexp32_low, vexp32_high);
        __m256d vn = _mm256_cvtepi32_pd(vexp32);

        // Extract mantissa
        __m256i vmant_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFF);
        __m256i vexp_zero = _mm256_set1_epi64x(0x3FF0000000000000);
        __m256i vmant = _mm256_or_si256(_mm256_and_si256(vi, vmant_mask), vexp_zero);
        __m256d vm = _mm256_castsi256_pd(vmant);

        // If m > sqrt(2), adjust
        __m256d vmask = _mm256_cmp_pd(vm, vsqrt2, _CMP_GT_OQ);
        vm = _mm256_blendv_pd(vm, _mm256_mul_pd(vm, vhalf), vmask);
        vn = _mm256_blendv_pd(vn, _mm256_add_pd(vn, vone), vmask);

        // f = m - 1
        __m256d vf = _mm256_sub_pd(vm, vone);

        // s = f / (2 + f)
        __m256d vs = _mm256_div_pd(vf, _mm256_add_pd(_mm256_set1_pd(2.0), vf));
        __m256d vs2 = _mm256_mul_pd(vs, vs);

        // Higher-order polynomial for double precision
        __m256d vc11 = _mm256_set1_pd(0.090909090909090909);
        __m256d vc9 = _mm256_set1_pd(0.11111111111111111);
        __m256d vc7 = _mm256_set1_pd(0.14285714285714285);
        __m256d vc5 = _mm256_set1_pd(0.2);
        __m256d vc3 = _mm256_set1_pd(0.33333333333333333);

#ifdef OPTINUM_HAS_FMA
        __m256d vp = _mm256_fmadd_pd(vc11, vs2, vc9);
        vp = _mm256_fmadd_pd(vp, vs2, vc7);
        vp = _mm256_fmadd_pd(vp, vs2, vc5);
        vp = _mm256_fmadd_pd(vp, vs2, vc3);
        vp = _mm256_fmadd_pd(vp, vs2, vone);
#else
        __m256d vp = _mm256_add_pd(_mm256_mul_pd(vc11, vs2), vc9);
        vp = _mm256_add_pd(_mm256_mul_pd(vp, vs2), vc7);
        vp = _mm256_add_pd(_mm256_mul_pd(vp, vs2), vc5);
        vp = _mm256_add_pd(_mm256_mul_pd(vp, vs2), vc3);
        vp = _mm256_add_pd(_mm256_mul_pd(vp, vs2), vone);
#endif

        __m256d vlog_m = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(2.0), vs), vp);

#ifdef OPTINUM_HAS_FMA
        __m256d vresult = _mm256_fmadd_pd(vn, vln2, vlog_m);
#else
        __m256d vresult = _mm256_add_pd(_mm256_mul_pd(vn, vln2), vlog_m);
#endif

        // Handle special cases
        __m256d vzero = _mm256_setzero_pd();
        __m256d vneg_inf = _mm256_set1_pd(-__builtin_inf());
        __m256d vnan = _mm256_set1_pd(__builtin_nan(""));

        vresult = _mm256_blendv_pd(vresult, vneg_inf, _mm256_cmp_pd(vx, vzero, _CMP_EQ_OQ));
        vresult = _mm256_blendv_pd(vresult, vnan, _mm256_cmp_pd(vx, vzero, _CMP_LT_OQ));

        return pack<double, 4>(vresult);
    }

#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
