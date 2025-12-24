#pragma once

#ifdef OPTINUM_USE_SLEEF

// SLEEF is an external library (often built via FetchContent / xmake package).
// When enabled, provide non-template overloads for the common SIMD widths so
// `optinum::simd::{exp,log,sin,...}` uses SLEEF instead of lane-wise std:: calls.

#include <sleef.h>

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/intrinsic/simd_vec.hpp>

namespace optinum::simd {

    // exp family
#if defined(OPTINUM_HAS_SSE2) || defined(OPTINUM_HAS_NEON)
    OPTINUM_INLINE SIMDVec<float, 4> exp(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_expf4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 4> exp2(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_exp2f4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 4> expm1(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_expm1f4_u10(a.value));
    }
#endif

#if defined(OPTINUM_HAS_SSE2) || (defined(OPTINUM_HAS_NEON) && defined(__aarch64__))
    OPTINUM_INLINE SIMDVec<double, 2> exp(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_expd2_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> exp2(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_exp2d2_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> expm1(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_expm1d2_u10(a.value));
    }
#endif

#if defined(OPTINUM_HAS_AVX)
    OPTINUM_INLINE SIMDVec<float, 8> exp(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_expf8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> exp(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_expd4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 8> exp2(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_exp2f8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> exp2(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_exp2d4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 8> expm1(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_expm1f8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> expm1(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_expm1d4_u10(a.value));
    }
#endif

#if defined(OPTINUM_HAS_AVX512F)
    OPTINUM_INLINE SIMDVec<float, 16> exp(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_expf16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> exp(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_expd8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 16> exp2(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_exp2f16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> exp2(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_exp2d8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 16> expm1(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_expm1f16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> expm1(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_expm1d8_u10(a.value));
    }
#endif

    // log family
#if defined(OPTINUM_HAS_SSE2) || defined(OPTINUM_HAS_NEON)
    OPTINUM_INLINE SIMDVec<float, 4> log(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_logf4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 4> log2(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_log2f4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 4> log10(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_log10f4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 4> log1p(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_log1pf4_u10(a.value));
    }
#endif

#if defined(OPTINUM_HAS_SSE2) || (defined(OPTINUM_HAS_NEON) && defined(__aarch64__))
    OPTINUM_INLINE SIMDVec<double, 2> log(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_logd2_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> log2(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_log2d2_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> log10(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_log10d2_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> log1p(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_log1pd2_u10(a.value));
    }
#endif

#if defined(OPTINUM_HAS_AVX)
    OPTINUM_INLINE SIMDVec<float, 8> log(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_logf8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> log(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_logd4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 8> log2(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_log2f8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> log2(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_log2d4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 8> log10(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_log10f8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> log10(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_log10d4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 8> log1p(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_log1pf8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> log1p(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_log1pd4_u10(a.value));
    }
#endif

#if defined(OPTINUM_HAS_AVX512F)
    OPTINUM_INLINE SIMDVec<float, 16> log(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_logf16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> log(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_logd8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 16> log2(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_log2f16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> log2(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_log2d8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 16> log10(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_log10f16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> log10(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_log10d8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 16> log1p(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_log1pf16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> log1p(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_log1pd8_u10(a.value));
    }
#endif

    // pow/cbrt
#if defined(OPTINUM_HAS_SSE2) || defined(OPTINUM_HAS_NEON)
    OPTINUM_INLINE SIMDVec<float, 4> pow(const SIMDVec<float, 4> &a, const SIMDVec<float, 4> &b) {
        return SIMDVec<float, 4>(Sleef_powf4_u10(a.value, b.value));
    }
    OPTINUM_INLINE SIMDVec<float, 4> cbrt(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_cbrtf4_u10(a.value));
    }
#endif

#if defined(OPTINUM_HAS_SSE2) || (defined(OPTINUM_HAS_NEON) && defined(__aarch64__))
    OPTINUM_INLINE SIMDVec<double, 2> pow(const SIMDVec<double, 2> &a, const SIMDVec<double, 2> &b) {
        return SIMDVec<double, 2>(Sleef_powd2_u10(a.value, b.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> cbrt(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_cbrtd2_u10(a.value));
    }
#endif

#if defined(OPTINUM_HAS_AVX)
    OPTINUM_INLINE SIMDVec<float, 8> pow(const SIMDVec<float, 8> &a, const SIMDVec<float, 8> &b) {
        return SIMDVec<float, 8>(Sleef_powf8_u10(a.value, b.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> pow(const SIMDVec<double, 4> &a, const SIMDVec<double, 4> &b) {
        return SIMDVec<double, 4>(Sleef_powd4_u10(a.value, b.value));
    }
    OPTINUM_INLINE SIMDVec<float, 8> cbrt(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_cbrtf8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> cbrt(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_cbrtd4_u10(a.value));
    }
#endif

#if defined(OPTINUM_HAS_AVX512F)
    OPTINUM_INLINE SIMDVec<float, 16> pow(const SIMDVec<float, 16> &a, const SIMDVec<float, 16> &b) {
        return SIMDVec<float, 16>(Sleef_powf16_u10(a.value, b.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> pow(const SIMDVec<double, 8> &a, const SIMDVec<double, 8> &b) {
        return SIMDVec<double, 8>(Sleef_powd8_u10(a.value, b.value));
    }
    OPTINUM_INLINE SIMDVec<float, 16> cbrt(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_cbrtf16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> cbrt(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_cbrtd8_u10(a.value));
    }
#endif

    // trig
#if defined(OPTINUM_HAS_SSE2) || defined(OPTINUM_HAS_NEON)
    OPTINUM_INLINE SIMDVec<float, 4> sin(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_sinf4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 4> cos(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_cosf4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 4> tan(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_tanf4_u10(a.value));
    }

    OPTINUM_INLINE SIMDVec<float, 4> asin(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_asinf4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 4> acos(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_acosf4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 4> atan(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_atanf4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 4> atan2(const SIMDVec<float, 4> &y, const SIMDVec<float, 4> &x) {
        return SIMDVec<float, 4>(Sleef_atan2f4_u10(y.value, x.value));
    }
#endif

#if defined(OPTINUM_HAS_SSE2) || (defined(OPTINUM_HAS_NEON) && defined(__aarch64__))
    OPTINUM_INLINE SIMDVec<double, 2> sin(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_sind2_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> cos(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_cosd2_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> tan(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_tand2_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> asin(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_asind2_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> acos(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_acosd2_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> atan(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_atand2_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> atan2(const SIMDVec<double, 2> &y, const SIMDVec<double, 2> &x) {
        return SIMDVec<double, 2>(Sleef_atan2d2_u10(y.value, x.value));
    }
#endif

#if defined(OPTINUM_HAS_AVX)
    OPTINUM_INLINE SIMDVec<float, 8> sin(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_sinf8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> sin(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_sind4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 8> cos(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_cosf8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> cos(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_cosd4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 8> tan(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_tanf8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> tan(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_tand4_u10(a.value));
    }

    OPTINUM_INLINE SIMDVec<float, 8> asin(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_asinf8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> asin(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_asind4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 8> acos(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_acosf8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> acos(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_acosd4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 8> atan(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_atanf8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> atan(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_atand4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 8> atan2(const SIMDVec<float, 8> &y, const SIMDVec<float, 8> &x) {
        return SIMDVec<float, 8>(Sleef_atan2f8_u10(y.value, x.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> atan2(const SIMDVec<double, 4> &y, const SIMDVec<double, 4> &x) {
        return SIMDVec<double, 4>(Sleef_atan2d4_u10(y.value, x.value));
    }
#endif

#if defined(OPTINUM_HAS_AVX512F)
    OPTINUM_INLINE SIMDVec<float, 16> sin(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_sinf16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> sin(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_sind8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 16> cos(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_cosf16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> cos(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_cosd8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 16> tan(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_tanf16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> tan(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_tand8_u10(a.value));
    }

    OPTINUM_INLINE SIMDVec<float, 16> asin(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_asinf16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> asin(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_asind8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 16> acos(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_acosf16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> acos(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_acosd8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 16> atan(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_atanf16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> atan(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_atand8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 16> atan2(const SIMDVec<float, 16> &y, const SIMDVec<float, 16> &x) {
        return SIMDVec<float, 16>(Sleef_atan2f16_u10(y.value, x.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> atan2(const SIMDVec<double, 8> &y, const SIMDVec<double, 8> &x) {
        return SIMDVec<double, 8>(Sleef_atan2d8_u10(y.value, x.value));
    }
#endif

    // hyperbolic
#if defined(OPTINUM_HAS_SSE2) || defined(OPTINUM_HAS_NEON)
    OPTINUM_INLINE SIMDVec<float, 4> sinh(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_sinhf4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 4> cosh(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_coshf4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 4> tanh(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_tanhf4_u10(a.value));
    }

    OPTINUM_INLINE SIMDVec<float, 4> asinh(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_asinhf4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 4> acosh(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_acoshf4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 4> atanh(const SIMDVec<float, 4> &a) {
        return SIMDVec<float, 4>(Sleef_atanhf4_u10(a.value));
    }
#endif

#if defined(OPTINUM_HAS_SSE2) || (defined(OPTINUM_HAS_NEON) && defined(__aarch64__))
    OPTINUM_INLINE SIMDVec<double, 2> sinh(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_sinhd2_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> cosh(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_coshd2_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> tanh(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_tanhd2_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> asinh(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_asinhd2_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> acosh(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_acoshd2_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 2> atanh(const SIMDVec<double, 2> &a) {
        return SIMDVec<double, 2>(Sleef_atanhd2_u10(a.value));
    }
#endif

#if defined(OPTINUM_HAS_AVX)
    OPTINUM_INLINE SIMDVec<float, 8> sinh(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_sinhf8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> sinh(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_sinhd4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 8> cosh(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_coshf8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> cosh(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_coshd4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 8> tanh(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_tanhf8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> tanh(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_tanhd4_u10(a.value));
    }

    OPTINUM_INLINE SIMDVec<float, 8> asinh(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_asinhf8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> asinh(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_asinhd4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 8> acosh(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_acoshf8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> acosh(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_acoshd4_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 8> atanh(const SIMDVec<float, 8> &a) {
        return SIMDVec<float, 8>(Sleef_atanhf8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 4> atanh(const SIMDVec<double, 4> &a) {
        return SIMDVec<double, 4>(Sleef_atanhd4_u10(a.value));
    }
#endif

#if defined(OPTINUM_HAS_AVX512F)
    OPTINUM_INLINE SIMDVec<float, 16> sinh(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_sinhf16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> sinh(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_sinhd8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 16> cosh(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_coshf16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> cosh(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_coshd8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 16> tanh(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_tanhf16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> tanh(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_tanhd8_u10(a.value));
    }

    OPTINUM_INLINE SIMDVec<float, 16> asinh(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_asinhf16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> asinh(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_asinhd8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 16> acosh(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_acoshf16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> acosh(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_acoshd8_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<float, 16> atanh(const SIMDVec<float, 16> &a) {
        return SIMDVec<float, 16>(Sleef_atanhf16_u10(a.value));
    }
    OPTINUM_INLINE SIMDVec<double, 8> atanh(const SIMDVec<double, 8> &a) {
        return SIMDVec<double, 8>(Sleef_atanhd8_u10(a.value));
    }
#endif

} // namespace optinum::simd

#endif
