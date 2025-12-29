#pragma once

// =============================================================================
// optinum/simd/math/erf.hpp
// Vectorized erf() (error function) using pack<T,W> with SIMD intrinsics
//
// erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
//
// Uses Abramowitz and Stegun approximation 7.1.26:
//   erf(x) ≈ 1 - (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵)e^(-x²)
//   where t = 1/(1 + p|x|)
//
// This provides accuracy ~1.5e-7 (maximum error)
//
// Properties:
//   - erf(-x) = -erf(x)  (odd function)
//   - erf(0) = 0
//   - lim_{x→∞} erf(x) = 1
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/math/exp.hpp>
#include <optinum/simd/pack/pack.hpp>

#if defined(OPTINUM_HAS_SSE2)
#include <optinum/simd/pack/sse.hpp>
#endif

#if defined(OPTINUM_HAS_AVX)
#include <optinum/simd/pack/avx.hpp>
#endif

#if defined(OPTINUM_HAS_NEON)
#include <optinum/simd/pack/neon.hpp>
#endif

namespace optinum::simd {

    // Forward declaration
    template <typename T, std::size_t W> pack<T, W> erf(const pack<T, W> &x) noexcept;

    namespace detail {
        // Abramowitz and Stegun 7.1.26 coefficients
        constexpr float ERF_P_F = 0.3275911f;
        constexpr float ERF_A1_F = 0.254829592f;
        constexpr float ERF_A2_F = -0.284496736f;
        constexpr float ERF_A3_F = 1.421413741f;
        constexpr float ERF_A4_F = -1.453152027f;
        constexpr float ERF_A5_F = 1.061405429f;

        constexpr double ERF_P_D = 0.3275911;
        constexpr double ERF_A1_D = 0.254829592;
        constexpr double ERF_A2_D = -0.284496736;
        constexpr double ERF_A3_D = 1.421413741;
        constexpr double ERF_A4_D = -1.453152027;
        constexpr double ERF_A5_D = 1.061405429;
    } // namespace detail

// =============================================================================
// SSE Implementation for float (W=4)
// =============================================================================
#if defined(OPTINUM_HAS_SSE41)
    template <> inline pack<float, 4> erf(const pack<float, 4> &x) noexcept {
        using namespace detail;

        __m128 vx = x.data_;

        // Save sign
        __m128 vsign = _mm_and_ps(vx, _mm_set1_ps(-0.0f));

        // |x|
        __m128 vabs_x = _mm_andnot_ps(_mm_set1_ps(-0.0f), vx);

        // t = 1 / (1 + p*|x|)
        __m128 vt =
            _mm_div_ps(_mm_set1_ps(1.0f), _mm_add_ps(_mm_set1_ps(1.0f), _mm_mul_ps(_mm_set1_ps(ERF_P_F), vabs_x)));

        // Polynomial: a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵
        __m128 vpoly;
#ifdef OPTINUM_HAS_FMA
        vpoly = _mm_fmadd_ps(_mm_set1_ps(ERF_A5_F), vt, _mm_set1_ps(ERF_A4_F));
        vpoly = _mm_fmadd_ps(vpoly, vt, _mm_set1_ps(ERF_A3_F));
        vpoly = _mm_fmadd_ps(vpoly, vt, _mm_set1_ps(ERF_A2_F));
        vpoly = _mm_fmadd_ps(vpoly, vt, _mm_set1_ps(ERF_A1_F));
        vpoly = _mm_mul_ps(vpoly, vt);
#else
        vpoly = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(ERF_A5_F), vt), _mm_set1_ps(ERF_A4_F));
        vpoly = _mm_add_ps(_mm_mul_ps(vpoly, vt), _mm_set1_ps(ERF_A3_F));
        vpoly = _mm_add_ps(_mm_mul_ps(vpoly, vt), _mm_set1_ps(ERF_A2_F));
        vpoly = _mm_add_ps(_mm_mul_ps(vpoly, vt), _mm_set1_ps(ERF_A1_F));
        vpoly = _mm_mul_ps(vpoly, vt);
#endif

        // e^(-x²) using our accurate exp() function
        __m128 vneg_x2 = _mm_xor_ps(_mm_mul_ps(vabs_x, vabs_x), _mm_set1_ps(-0.0f));
        __m128 vexp = exp(pack<float, 4>(vneg_x2)).data_;

        // result = 1 - poly * exp(-x²)
        __m128 vresult = _mm_sub_ps(_mm_set1_ps(1.0f), _mm_mul_ps(vpoly, vexp));

        // Restore sign
        vresult = _mm_or_ps(vresult, vsign);

        return pack<float, 4>(vresult);
    }
#endif // OPTINUM_HAS_SSE41

// =============================================================================
// AVX Implementation for float (W=8)
// =============================================================================
#if defined(OPTINUM_HAS_AVX)
    template <> inline pack<float, 8> erf(const pack<float, 8> &x) noexcept {
        using namespace detail;

        __m256 vx = x.data_;

        // Save sign
        __m256 vsign = _mm256_and_ps(vx, _mm256_set1_ps(-0.0f));

        // |x|
        __m256 vabs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vx);

        // t = 1 / (1 + p*|x|)
        __m256 vt = _mm256_div_ps(_mm256_set1_ps(1.0f),
                                  _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(_mm256_set1_ps(ERF_P_F), vabs_x)));

        // Polynomial
        __m256 vpoly;
#ifdef OPTINUM_HAS_FMA
        vpoly = _mm256_fmadd_ps(_mm256_set1_ps(ERF_A5_F), vt, _mm256_set1_ps(ERF_A4_F));
        vpoly = _mm256_fmadd_ps(vpoly, vt, _mm256_set1_ps(ERF_A3_F));
        vpoly = _mm256_fmadd_ps(vpoly, vt, _mm256_set1_ps(ERF_A2_F));
        vpoly = _mm256_fmadd_ps(vpoly, vt, _mm256_set1_ps(ERF_A1_F));
        vpoly = _mm256_mul_ps(vpoly, vt);
#else
        vpoly = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(ERF_A5_F), vt), _mm256_set1_ps(ERF_A4_F));
        vpoly = _mm256_add_ps(_mm256_mul_ps(vpoly, vt), _mm256_set1_ps(ERF_A3_F));
        vpoly = _mm256_add_ps(_mm256_mul_ps(vpoly, vt), _mm256_set1_ps(ERF_A2_F));
        vpoly = _mm256_add_ps(_mm256_mul_ps(vpoly, vt), _mm256_set1_ps(ERF_A1_F));
        vpoly = _mm256_mul_ps(vpoly, vt);
#endif

        // e^(-x²)
        __m256 vneg_x2 = _mm256_xor_ps(_mm256_mul_ps(vabs_x, vabs_x), _mm256_set1_ps(-0.0f));
        __m256 vexp = exp(pack<float, 8>(vneg_x2)).data_;

        // result = 1 - poly * exp(-x²)
        __m256 vresult = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(vpoly, vexp));

        // Restore sign
        vresult = _mm256_or_ps(vresult, vsign);

        return pack<float, 8>(vresult);
    }
#endif // OPTINUM_HAS_AVX

// =============================================================================
// SSE Implementation for double (W=2)
// =============================================================================
#if defined(OPTINUM_HAS_SSE41)
    template <> inline pack<double, 2> erf(const pack<double, 2> &x) noexcept {
        using namespace detail;

        __m128d vx = x.data_;

        // Save sign
        __m128d vsign = _mm_and_pd(vx, _mm_set1_pd(-0.0));

        // |x|
        __m128d vabs_x = _mm_andnot_pd(_mm_set1_pd(-0.0), vx);

        // t = 1 / (1 + p*|x|)
        __m128d vt =
            _mm_div_pd(_mm_set1_pd(1.0), _mm_add_pd(_mm_set1_pd(1.0), _mm_mul_pd(_mm_set1_pd(ERF_P_D), vabs_x)));

        // Polynomial
        __m128d vpoly;
#ifdef OPTINUM_HAS_FMA
        vpoly = _mm_fmadd_pd(_mm_set1_pd(ERF_A5_D), vt, _mm_set1_pd(ERF_A4_D));
        vpoly = _mm_fmadd_pd(vpoly, vt, _mm_set1_pd(ERF_A3_D));
        vpoly = _mm_fmadd_pd(vpoly, vt, _mm_set1_pd(ERF_A2_D));
        vpoly = _mm_fmadd_pd(vpoly, vt, _mm_set1_pd(ERF_A1_D));
        vpoly = _mm_mul_pd(vpoly, vt);
#else
        vpoly = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(ERF_A5_D), vt), _mm_set1_pd(ERF_A4_D));
        vpoly = _mm_add_pd(_mm_mul_pd(vpoly, vt), _mm_set1_pd(ERF_A3_D));
        vpoly = _mm_add_pd(_mm_mul_pd(vpoly, vt), _mm_set1_pd(ERF_A2_D));
        vpoly = _mm_add_pd(_mm_mul_pd(vpoly, vt), _mm_set1_pd(ERF_A1_D));
        vpoly = _mm_mul_pd(vpoly, vt);
#endif

        // e^(-x²)
        __m128d vneg_x2 = _mm_xor_pd(_mm_mul_pd(vabs_x, vabs_x), _mm_set1_pd(-0.0));
        __m128d vexp = exp(pack<double, 2>(vneg_x2)).data_;

        // result = 1 - poly * exp(-x²)
        __m128d vresult = _mm_sub_pd(_mm_set1_pd(1.0), _mm_mul_pd(vpoly, vexp));

        // Restore sign
        vresult = _mm_or_pd(vresult, vsign);

        return pack<double, 2>(vresult);
    }
#endif // OPTINUM_HAS_SSE41

// =============================================================================
// AVX Implementation for double (W=4)
// =============================================================================
#if defined(OPTINUM_HAS_AVX)
    template <> inline pack<double, 4> erf(const pack<double, 4> &x) noexcept {
        using namespace detail;

        __m256d vx = x.data_;

        // Save sign
        __m256d vsign = _mm256_and_pd(vx, _mm256_set1_pd(-0.0));

        // |x|
        __m256d vabs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), vx);

        // t = 1 / (1 + p*|x|)
        __m256d vt = _mm256_div_pd(_mm256_set1_pd(1.0),
                                   _mm256_add_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(_mm256_set1_pd(ERF_P_D), vabs_x)));

        // Polynomial
        __m256d vpoly;
#ifdef OPTINUM_HAS_FMA
        vpoly = _mm256_fmadd_pd(_mm256_set1_pd(ERF_A5_D), vt, _mm256_set1_pd(ERF_A4_D));
        vpoly = _mm256_fmadd_pd(vpoly, vt, _mm256_set1_pd(ERF_A3_D));
        vpoly = _mm256_fmadd_pd(vpoly, vt, _mm256_set1_pd(ERF_A2_D));
        vpoly = _mm256_fmadd_pd(vpoly, vt, _mm256_set1_pd(ERF_A1_D));
        vpoly = _mm256_mul_pd(vpoly, vt);
#else
        vpoly = _mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(ERF_A5_D), vt), _mm256_set1_pd(ERF_A4_D));
        vpoly = _mm256_add_pd(_mm256_mul_pd(vpoly, vt), _mm256_set1_pd(ERF_A3_D));
        vpoly = _mm256_add_pd(_mm256_mul_pd(vpoly, vt), _mm256_set1_pd(ERF_A2_D));
        vpoly = _mm256_add_pd(_mm256_mul_pd(vpoly, vt), _mm256_set1_pd(ERF_A1_D));
        vpoly = _mm256_mul_pd(vpoly, vt);
#endif

        // e^(-x²)
        __m256d vneg_x2 = _mm256_xor_pd(_mm256_mul_pd(vabs_x, vabs_x), _mm256_set1_pd(-0.0));
        __m256d vexp = exp(pack<double, 4>(vneg_x2)).data_;

        // result = 1 - poly * exp(-x²)
        __m256d vresult = _mm256_sub_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(vpoly, vexp));

        // Restore sign
        vresult = _mm256_or_pd(vresult, vsign);

        return pack<double, 4>(vresult);
    }
#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
