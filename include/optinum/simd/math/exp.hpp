#pragma once

// =============================================================================
// optinum/simd/math/exp.hpp
// Vectorized exp() using pack<T,W> with SIMD intrinsics
// New clean API - replaces fast_exp.hpp
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/pack/avx.hpp>
#include <optinum/simd/pack/sse.hpp>

namespace optinum::simd {

    // Forward declaration
    template <typename T, std::size_t W> pack<T, W> exp(const pack<T, W> &x) noexcept;

    namespace detail {
        // Constants for exp(x) computation
        constexpr float LN2_HI_F = 0.693145751953125f;
        constexpr float LN2_LO_F = 1.42860682030941723212e-6f;
        constexpr float LOG2E_F = 1.44269504088896341f;

        constexpr double LN2_HI_D = 0.69314718055994528623;
        constexpr double LN2_LO_D = 2.3190468138462996e-17;
        constexpr double LOG2E_D = 1.4426950408889634074;

        // Polynomial coefficients
        constexpr float EXP_C1_F = 1.0f;
        constexpr float EXP_C2_F = 0.5f;
        constexpr float EXP_C3_F = 0.16666666666666666f;
        constexpr float EXP_C4_F = 0.041666666666666664f;
        constexpr float EXP_C5_F = 0.008333333333333333f;

        constexpr double EXP_C1_D = 1.0;
        constexpr double EXP_C2_D = 0.5;
        constexpr double EXP_C3_D = 0.16666666666666666;
        constexpr double EXP_C4_D = 0.041666666666666664;
        constexpr double EXP_C5_D = 0.008333333333333333;
        constexpr double EXP_C6_D = 0.001388888888888889;
        constexpr double EXP_C7_D = 0.000198412698412698;
    } // namespace detail

// =============================================================================
// SSE Implementation for float (W=4)
// =============================================================================
#if defined(OPTINUM_HAS_SSE41)

    template <> inline pack<float, 4> exp(const pack<float, 4> &x) noexcept {
        using namespace detail;

        __m128 vx = x.data_;

        // Clamp
        vx = _mm_min_ps(vx, _mm_set1_ps(88.0f));
        vx = _mm_max_ps(vx, _mm_set1_ps(-88.0f));

        // n = round(x / ln2)
        __m128 vn = _mm_round_ps(_mm_mul_ps(vx, _mm_set1_ps(LOG2E_F)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // r = x - n * ln2 (range reduction)
        __m128 vr = _mm_sub_ps(vx, _mm_mul_ps(vn, _mm_set1_ps(LN2_HI_F)));
        vr = _mm_sub_ps(vr, _mm_mul_ps(vn, _mm_set1_ps(LN2_LO_F)));

        // Polynomial using FMA
        __m128 vpoly;
#ifdef OPTINUM_HAS_FMA
        vpoly = _mm_fmadd_ps(_mm_set1_ps(EXP_C5_F), vr, _mm_set1_ps(EXP_C4_F));
        vpoly = _mm_fmadd_ps(vpoly, vr, _mm_set1_ps(EXP_C3_F));
        vpoly = _mm_fmadd_ps(vpoly, vr, _mm_set1_ps(EXP_C2_F));
        vpoly = _mm_fmadd_ps(vpoly, vr, _mm_set1_ps(EXP_C1_F));
        vpoly = _mm_fmadd_ps(vpoly, vr, _mm_set1_ps(1.0f));
#else
        vpoly = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(EXP_C5_F), vr), _mm_set1_ps(EXP_C4_F));
        vpoly = _mm_add_ps(_mm_mul_ps(vpoly, vr), _mm_set1_ps(EXP_C3_F));
        vpoly = _mm_add_ps(_mm_mul_ps(vpoly, vr), _mm_set1_ps(EXP_C2_F));
        vpoly = _mm_add_ps(_mm_mul_ps(vpoly, vr), _mm_set1_ps(EXP_C1_F));
        vpoly = _mm_add_ps(_mm_mul_ps(vpoly, vr), _mm_set1_ps(1.0f));
#endif

        // Scale by 2^n: 2^n = reinterpret((n + 127) << 23)
        __m128i vni = _mm_cvtps_epi32(vn);
        vni = _mm_add_epi32(vni, _mm_set1_epi32(127));
        vni = _mm_slli_epi32(vni, 23);
        __m128 vscale = _mm_castsi128_ps(vni);

        return pack<float, 4>(_mm_mul_ps(vpoly, vscale));
    }

#endif // OPTINUM_HAS_SSE41

// =============================================================================
// AVX Implementation for float (W=8)
// =============================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<float, 8> exp(const pack<float, 8> &x) noexcept {
        using namespace detail;

        __m256 vx = x.data_;

        // Clamp
        vx = _mm256_min_ps(vx, _mm256_set1_ps(88.0f));
        vx = _mm256_max_ps(vx, _mm256_set1_ps(-88.0f));

        // n = round(x / ln2)
        __m256 vn =
            _mm256_round_ps(_mm256_mul_ps(vx, _mm256_set1_ps(LOG2E_F)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // r = x - n * ln2
        __m256 vr = _mm256_sub_ps(vx, _mm256_mul_ps(vn, _mm256_set1_ps(LN2_HI_F)));
        vr = _mm256_sub_ps(vr, _mm256_mul_ps(vn, _mm256_set1_ps(LN2_LO_F)));

        // Polynomial using FMA
        __m256 vpoly;
#ifdef OPTINUM_HAS_FMA
        vpoly = _mm256_fmadd_ps(_mm256_set1_ps(EXP_C5_F), vr, _mm256_set1_ps(EXP_C4_F));
        vpoly = _mm256_fmadd_ps(vpoly, vr, _mm256_set1_ps(EXP_C3_F));
        vpoly = _mm256_fmadd_ps(vpoly, vr, _mm256_set1_ps(EXP_C2_F));
        vpoly = _mm256_fmadd_ps(vpoly, vr, _mm256_set1_ps(EXP_C1_F));
        vpoly = _mm256_fmadd_ps(vpoly, vr, _mm256_set1_ps(1.0f));
#else
        vpoly = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(EXP_C5_F), vr), _mm256_set1_ps(EXP_C4_F));
        vpoly = _mm256_add_ps(_mm256_mul_ps(vpoly, vr), _mm256_set1_ps(EXP_C3_F));
        vpoly = _mm256_add_ps(_mm256_mul_ps(vpoly, vr), _mm256_set1_ps(EXP_C2_F));
        vpoly = _mm256_add_ps(_mm256_mul_ps(vpoly, vr), _mm256_set1_ps(EXP_C1_F));
        vpoly = _mm256_add_ps(_mm256_mul_ps(vpoly, vr), _mm256_set1_ps(1.0f));
#endif

        // Scale by 2^n
        __m256i vni = _mm256_cvtps_epi32(vn);
        vni = _mm256_add_epi32(vni, _mm256_set1_epi32(127));
        vni = _mm256_slli_epi32(vni, 23);
        __m256 vscale = _mm256_castsi256_ps(vni);

        return pack<float, 8>(_mm256_mul_ps(vpoly, vscale));
    }

#endif // OPTINUM_HAS_AVX

// =============================================================================
// SSE Implementation for double (W=2)
// =============================================================================
#if defined(OPTINUM_HAS_SSE41)

    template <> inline pack<double, 2> exp(const pack<double, 2> &x) noexcept {
        using namespace detail;

        __m128d vx = x.data_;

        // Clamp
        vx = _mm_min_pd(vx, _mm_set1_pd(708.0));
        vx = _mm_max_pd(vx, _mm_set1_pd(-708.0));

        // n = round(x / ln2)
        __m128d vn = _mm_round_pd(_mm_mul_pd(vx, _mm_set1_pd(LOG2E_D)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // r = x - n * ln2
        __m128d vr = _mm_sub_pd(vx, _mm_mul_pd(vn, _mm_set1_pd(LN2_HI_D)));
        vr = _mm_sub_pd(vr, _mm_mul_pd(vn, _mm_set1_pd(LN2_LO_D)));

        // Higher-order polynomial for double
        __m128d vpoly;
#ifdef OPTINUM_HAS_FMA
        vpoly = _mm_fmadd_pd(_mm_set1_pd(EXP_C7_D), vr, _mm_set1_pd(EXP_C6_D));
        vpoly = _mm_fmadd_pd(vpoly, vr, _mm_set1_pd(EXP_C5_D));
        vpoly = _mm_fmadd_pd(vpoly, vr, _mm_set1_pd(EXP_C4_D));
        vpoly = _mm_fmadd_pd(vpoly, vr, _mm_set1_pd(EXP_C3_D));
        vpoly = _mm_fmadd_pd(vpoly, vr, _mm_set1_pd(EXP_C2_D));
        vpoly = _mm_fmadd_pd(vpoly, vr, _mm_set1_pd(EXP_C1_D));
        vpoly = _mm_fmadd_pd(vpoly, vr, _mm_set1_pd(1.0));
#else
        vpoly = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(EXP_C7_D), vr), _mm_set1_pd(EXP_C6_D));
        vpoly = _mm_add_pd(_mm_mul_pd(vpoly, vr), _mm_set1_pd(EXP_C5_D));
        vpoly = _mm_add_pd(_mm_mul_pd(vpoly, vr), _mm_set1_pd(EXP_C4_D));
        vpoly = _mm_add_pd(_mm_mul_pd(vpoly, vr), _mm_set1_pd(EXP_C3_D));
        vpoly = _mm_add_pd(_mm_mul_pd(vpoly, vr), _mm_set1_pd(EXP_C2_D));
        vpoly = _mm_add_pd(_mm_mul_pd(vpoly, vr), _mm_set1_pd(EXP_C1_D));
        vpoly = _mm_add_pd(_mm_mul_pd(vpoly, vr), _mm_set1_pd(1.0));
#endif

        // Scale by 2^n
        __m128i vni = _mm_cvtpd_epi32(vn); // Converts 2 doubles to 2 int32s (lower half)
        vni = _mm_add_epi32(vni, _mm_set1_epi32(1023));
        __m128i vni64 = _mm_cvtepi32_epi64(vni); // Extend to 64-bit
        vni64 = _mm_slli_epi64(vni64, 52);
        __m128d vscale = _mm_castsi128_pd(vni64);

        return pack<double, 2>(_mm_mul_pd(vpoly, vscale));
    }

#endif // OPTINUM_HAS_SSE41

// =============================================================================
// AVX Implementation for double (W=4)
// =============================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<double, 4> exp(const pack<double, 4> &x) noexcept {
        using namespace detail;

        __m256d vx = x.data_;

        // Clamp
        vx = _mm256_min_pd(vx, _mm256_set1_pd(708.0));
        vx = _mm256_max_pd(vx, _mm256_set1_pd(-708.0));

        // n = round(x / ln2)
        __m256d vn =
            _mm256_round_pd(_mm256_mul_pd(vx, _mm256_set1_pd(LOG2E_D)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // r = x - n * ln2
        __m256d vr = _mm256_sub_pd(vx, _mm256_mul_pd(vn, _mm256_set1_pd(LN2_HI_D)));
        vr = _mm256_sub_pd(vr, _mm256_mul_pd(vn, _mm256_set1_pd(LN2_LO_D)));

        // Polynomial
        __m256d vpoly;
#ifdef OPTINUM_HAS_FMA
        vpoly = _mm256_fmadd_pd(_mm256_set1_pd(EXP_C7_D), vr, _mm256_set1_pd(EXP_C6_D));
        vpoly = _mm256_fmadd_pd(vpoly, vr, _mm256_set1_pd(EXP_C5_D));
        vpoly = _mm256_fmadd_pd(vpoly, vr, _mm256_set1_pd(EXP_C4_D));
        vpoly = _mm256_fmadd_pd(vpoly, vr, _mm256_set1_pd(EXP_C3_D));
        vpoly = _mm256_fmadd_pd(vpoly, vr, _mm256_set1_pd(EXP_C2_D));
        vpoly = _mm256_fmadd_pd(vpoly, vr, _mm256_set1_pd(EXP_C1_D));
        vpoly = _mm256_fmadd_pd(vpoly, vr, _mm256_set1_pd(1.0));
#else
        vpoly = _mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(EXP_C7_D), vr), _mm256_set1_pd(EXP_C6_D));
        vpoly = _mm256_add_pd(_mm256_mul_pd(vpoly, vr), _mm256_set1_pd(EXP_C5_D));
        vpoly = _mm256_add_pd(_mm256_mul_pd(vpoly, vr), _mm256_set1_pd(EXP_C4_D));
        vpoly = _mm256_add_pd(_mm256_mul_pd(vpoly, vr), _mm256_set1_pd(EXP_C3_D));
        vpoly = _mm256_add_pd(_mm256_mul_pd(vpoly, vr), _mm256_set1_pd(EXP_C2_D));
        vpoly = _mm256_add_pd(_mm256_mul_pd(vpoly, vr), _mm256_set1_pd(EXP_C1_D));
        vpoly = _mm256_add_pd(_mm256_mul_pd(vpoly, vr), _mm256_set1_pd(1.0));
#endif

        // Scale by 2^n
        __m128i vni = _mm256_cvtpd_epi32(vn); // 4 doubles -> 4 int32s (lower 128 bits)
        vni = _mm_add_epi32(vni, _mm_set1_epi32(1023));

        // Convert 4x int32 to 4x int64, then shift
        __m256i vni64 = _mm256_cvtepi32_epi64(vni);
        vni64 = _mm256_slli_epi64(vni64, 52);
        __m256d vscale = _mm256_castsi256_pd(vni64);

        return pack<double, 4>(_mm256_mul_pd(vpoly, vscale));
    }

#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
