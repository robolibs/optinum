#pragma once

// =============================================================================
// optinum/simd/math/lgamma.hpp
// Vectorized lgamma() (log gamma function) using pack<T,W> with SIMD
//
// lgamma(x) = log(Γ(x)) = log(|Γ(x)|)
//
// Uses Lanczos approximation:
//   log Γ(x+1) = 0.5*log(2π) + (x+0.5)*log(x+g+0.5) - (x+g+0.5) + log(Aᵍ(x))
//
// More stable than log(tgamma(x)) for large x
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/math/log.hpp>
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
    template <typename T, std::size_t W> pack<T, W> lgamma(const pack<T, W> &x) noexcept;

    namespace detail {
        // Lanczos coefficients for g = 7
        constexpr float LGAMMA_G_F = 7.0f;
        constexpr float LGAMMA_C0_F = 0.99999999999980993f;
        constexpr float LGAMMA_C1_F = 676.5203681218851f;
        constexpr float LGAMMA_C2_F = -1259.1392167224028f;
        constexpr float LGAMMA_C3_F = 771.32342877765313f;
        constexpr float LGAMMA_C4_F = -176.61502916214059f;
        constexpr float LGAMMA_C5_F = 12.507343278686905f;
        constexpr float LGAMMA_C6_F = -0.13857109526572012f;
        constexpr float LGAMMA_C7_F = 9.9843695780195716e-6f;
        constexpr float LGAMMA_C8_F = 1.5056327351493116e-7f;

        constexpr double LGAMMA_G_D = 7.0;
        constexpr double LGAMMA_C0_D = 0.99999999999980993;
        constexpr double LGAMMA_C1_D = 676.5203681218851;
        constexpr double LGAMMA_C2_D = -1259.1392167224028;
        constexpr double LGAMMA_C3_D = 771.32342877765313;
        constexpr double LGAMMA_C4_D = -176.61502916214059;
        constexpr double LGAMMA_C5_D = 12.507343278686905;
        constexpr double LGAMMA_C6_D = -0.13857109526572012;
        constexpr double LGAMMA_C7_D = 9.9843695780195716e-6;
        constexpr double LGAMMA_C8_D = 1.5056327351493116e-7;

        constexpr float LOG_SQRT_2PI_F = 0.91893853320467274f; // log(√(2π))
        constexpr double LOG_SQRT_2PI_D = 0.91893853320467274;
    } // namespace detail

// =============================================================================
// SSE Implementation for float (W=4)
// =============================================================================
#if defined(OPTINUM_HAS_SSE41)

    template <> inline pack<float, 4> lgamma(const pack<float, 4> &x) noexcept {
        using namespace detail;

        __m128 vx = x.data_;

        // Lanczos computes log Γ(z+1), so use z = x - 1 to get log Γ(x)
        __m128 vz = _mm_sub_ps(vx, _mm_set1_ps(1.0f));

        // Compute Lanczos sum with z
        __m128 vsum = _mm_set1_ps(LGAMMA_C0_F);

        __m128 vxp1 = _mm_add_ps(vz, _mm_set1_ps(1.0f));
#ifdef OPTINUM_HAS_FMA
        vsum = _mm_fmadd_ps(_mm_div_ps(_mm_set1_ps(LGAMMA_C1_F), vxp1), _mm_set1_ps(1.0f), vsum);
        __m128 vxp2 = _mm_add_ps(vz, _mm_set1_ps(2.0f));
        vsum = _mm_fmadd_ps(_mm_div_ps(_mm_set1_ps(LGAMMA_C2_F), vxp2), _mm_set1_ps(1.0f), vsum);
        __m128 vxp3 = _mm_add_ps(vz, _mm_set1_ps(3.0f));
        vsum = _mm_fmadd_ps(_mm_div_ps(_mm_set1_ps(LGAMMA_C3_F), vxp3), _mm_set1_ps(1.0f), vsum);
        __m128 vxp4 = _mm_add_ps(vz, _mm_set1_ps(4.0f));
        vsum = _mm_fmadd_ps(_mm_div_ps(_mm_set1_ps(LGAMMA_C4_F), vxp4), _mm_set1_ps(1.0f), vsum);
        __m128 vxp5 = _mm_add_ps(vz, _mm_set1_ps(5.0f));
        vsum = _mm_fmadd_ps(_mm_div_ps(_mm_set1_ps(LGAMMA_C5_F), vxp5), _mm_set1_ps(1.0f), vsum);
        __m128 vxp6 = _mm_add_ps(vz, _mm_set1_ps(6.0f));
        vsum = _mm_fmadd_ps(_mm_div_ps(_mm_set1_ps(LGAMMA_C6_F), vxp6), _mm_set1_ps(1.0f), vsum);
        __m128 vxp7 = _mm_add_ps(vz, _mm_set1_ps(7.0f));
        vsum = _mm_fmadd_ps(_mm_div_ps(_mm_set1_ps(LGAMMA_C7_F), vxp7), _mm_set1_ps(1.0f), vsum);
        __m128 vxp8 = _mm_add_ps(vz, _mm_set1_ps(8.0f));
        vsum = _mm_fmadd_ps(_mm_div_ps(_mm_set1_ps(LGAMMA_C8_F), vxp8), _mm_set1_ps(1.0f), vsum);
#else
        vsum = _mm_add_ps(vsum, _mm_div_ps(_mm_set1_ps(LGAMMA_C1_F), vxp1));
        __m128 vxp2 = _mm_add_ps(vz, _mm_set1_ps(2.0f));
        vsum = _mm_add_ps(vsum, _mm_div_ps(_mm_set1_ps(LGAMMA_C2_F), vxp2));
        __m128 vxp3 = _mm_add_ps(vz, _mm_set1_ps(3.0f));
        vsum = _mm_add_ps(vsum, _mm_div_ps(_mm_set1_ps(LGAMMA_C3_F), vxp3));
        __m128 vxp4 = _mm_add_ps(vz, _mm_set1_ps(4.0f));
        vsum = _mm_add_ps(vsum, _mm_div_ps(_mm_set1_ps(LGAMMA_C4_F), vxp4));
        __m128 vxp5 = _mm_add_ps(vz, _mm_set1_ps(5.0f));
        vsum = _mm_add_ps(vsum, _mm_div_ps(_mm_set1_ps(LGAMMA_C5_F), vxp5));
        __m128 vxp6 = _mm_add_ps(vz, _mm_set1_ps(6.0f));
        vsum = _mm_add_ps(vsum, _mm_div_ps(_mm_set1_ps(LGAMMA_C6_F), vxp6));
        __m128 vxp7 = _mm_add_ps(vz, _mm_set1_ps(7.0f));
        vsum = _mm_add_ps(vsum, _mm_div_ps(_mm_set1_ps(LGAMMA_C7_F), vxp7));
        __m128 vxp8 = _mm_add_ps(vz, _mm_set1_ps(8.0f));
        vsum = _mm_add_ps(vsum, _mm_div_ps(_mm_set1_ps(LGAMMA_C8_F), vxp8));
#endif

        // t = z + g + 0.5
        __m128 vt = _mm_add_ps(vz, _mm_set1_ps(LGAMMA_G_F + 0.5f));

        // (z + 0.5)
        __m128 vxp05 = _mm_add_ps(vz, _mm_set1_ps(0.5f));

        // Use our accurate log() function
        __m128 vlog_t = log(pack<float, 4>(vt)).data_;
        __m128 vlog_sum = log(pack<float, 4>(vsum)).data_;

        // Result: log(√(2π)) + (x + 0.5) * log(t) - t + log(sum)
        __m128 vresult = _mm_set1_ps(LOG_SQRT_2PI_F);
        vresult = _mm_add_ps(vresult, _mm_mul_ps(vxp05, vlog_t));
        vresult = _mm_sub_ps(vresult, vt);
        vresult = _mm_add_ps(vresult, vlog_sum);

        return pack<float, 4>(vresult);
    }

#endif // OPTINUM_HAS_SSE41

// =============================================================================
// AVX Implementation for float (W=8)
// =============================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<float, 8> lgamma(const pack<float, 8> &x) noexcept {
        using namespace detail;

        __m256 vx = x.data_;

        // Lanczos computes log Γ(z+1), so use z = x - 1
        __m256 vz = _mm256_sub_ps(vx, _mm256_set1_ps(1.0f));

        // Lanczos sum
        __m256 vsum = _mm256_set1_ps(LGAMMA_C0_F);

        __m256 vxp1 = _mm256_add_ps(vz, _mm256_set1_ps(1.0f));
#ifdef OPTINUM_HAS_FMA
        vsum = _mm256_fmadd_ps(_mm256_div_ps(_mm256_set1_ps(LGAMMA_C1_F), vxp1), _mm256_set1_ps(1.0f), vsum);
        __m256 vxp2 = _mm256_add_ps(vz, _mm256_set1_ps(2.0f));
        vsum = _mm256_fmadd_ps(_mm256_div_ps(_mm256_set1_ps(LGAMMA_C2_F), vxp2), _mm256_set1_ps(1.0f), vsum);
        __m256 vxp3 = _mm256_add_ps(vz, _mm256_set1_ps(3.0f));
        vsum = _mm256_fmadd_ps(_mm256_div_ps(_mm256_set1_ps(LGAMMA_C3_F), vxp3), _mm256_set1_ps(1.0f), vsum);
        __m256 vxp4 = _mm256_add_ps(vz, _mm256_set1_ps(4.0f));
        vsum = _mm256_fmadd_ps(_mm256_div_ps(_mm256_set1_ps(LGAMMA_C4_F), vxp4), _mm256_set1_ps(1.0f), vsum);
        __m256 vxp5 = _mm256_add_ps(vz, _mm256_set1_ps(5.0f));
        vsum = _mm256_fmadd_ps(_mm256_div_ps(_mm256_set1_ps(LGAMMA_C5_F), vxp5), _mm256_set1_ps(1.0f), vsum);
        __m256 vxp6 = _mm256_add_ps(vz, _mm256_set1_ps(6.0f));
        vsum = _mm256_fmadd_ps(_mm256_div_ps(_mm256_set1_ps(LGAMMA_C6_F), vxp6), _mm256_set1_ps(1.0f), vsum);
        __m256 vxp7 = _mm256_add_ps(vz, _mm256_set1_ps(7.0f));
        vsum = _mm256_fmadd_ps(_mm256_div_ps(_mm256_set1_ps(LGAMMA_C7_F), vxp7), _mm256_set1_ps(1.0f), vsum);
        __m256 vxp8 = _mm256_add_ps(vz, _mm256_set1_ps(8.0f));
        vsum = _mm256_fmadd_ps(_mm256_div_ps(_mm256_set1_ps(LGAMMA_C8_F), vxp8), _mm256_set1_ps(1.0f), vsum);
#else
        vsum = _mm256_add_ps(vsum, _mm256_div_ps(_mm256_set1_ps(LGAMMA_C1_F), vxp1));
        __m256 vxp2 = _mm256_add_ps(vz, _mm256_set1_ps(2.0f));
        vsum = _mm256_add_ps(vsum, _mm256_div_ps(_mm256_set1_ps(LGAMMA_C2_F), vxp2));
        __m256 vxp3 = _mm256_add_ps(vz, _mm256_set1_ps(3.0f));
        vsum = _mm256_add_ps(vsum, _mm256_div_ps(_mm256_set1_ps(LGAMMA_C3_F), vxp3));
        __m256 vxp4 = _mm256_add_ps(vz, _mm256_set1_ps(4.0f));
        vsum = _mm256_add_ps(vsum, _mm256_div_ps(_mm256_set1_ps(LGAMMA_C4_F), vxp4));
        __m256 vxp5 = _mm256_add_ps(vz, _mm256_set1_ps(5.0f));
        vsum = _mm256_add_ps(vsum, _mm256_div_ps(_mm256_set1_ps(LGAMMA_C5_F), vxp5));
        __m256 vxp6 = _mm256_add_ps(vz, _mm256_set1_ps(6.0f));
        vsum = _mm256_add_ps(vsum, _mm256_div_ps(_mm256_set1_ps(LGAMMA_C6_F), vxp6));
        __m256 vxp7 = _mm256_add_ps(vz, _mm256_set1_ps(7.0f));
        vsum = _mm256_add_ps(vsum, _mm256_div_ps(_mm256_set1_ps(LGAMMA_C7_F), vxp7));
        __m256 vxp8 = _mm256_add_ps(vz, _mm256_set1_ps(8.0f));
        vsum = _mm256_add_ps(vsum, _mm256_div_ps(_mm256_set1_ps(LGAMMA_C8_F), vxp8));
#endif

        // t = x + g + 0.5
        __m256 vt = _mm256_add_ps(vz, _mm256_set1_ps(LGAMMA_G_F + 0.5f));
        __m256 vxp05 = _mm256_add_ps(vz, _mm256_set1_ps(0.5f));

        // Use accurate log() function
        __m256 vlog_t = log(pack<float, 8>(vt)).data_;
        __m256 vlog_sum = log(pack<float, 8>(vsum)).data_;

        // Result: log(√(2π)) + (x + 0.5) * log(t) - t + log(sum)
        __m256 vresult = _mm256_set1_ps(LOG_SQRT_2PI_F);
        vresult = _mm256_add_ps(vresult, _mm256_mul_ps(vxp05, vlog_t));
        vresult = _mm256_sub_ps(vresult, vt);
        vresult = _mm256_add_ps(vresult, vlog_sum);

        return pack<float, 8>(vresult);
    }

#endif // OPTINUM_HAS_AVX

// =============================================================================
// SSE Implementation for double (W=2)
// =============================================================================
#if defined(OPTINUM_HAS_SSE41)

    template <> inline pack<double, 2> lgamma(const pack<double, 2> &x) noexcept {
        using namespace detail;

        __m128d vx = x.data_;

        // Lanczos computes log Γ(z+1), so use z = x - 1
        __m128d vz = _mm_sub_pd(vx, _mm_set1_pd(1.0));

        // Lanczos sum
        __m128d vsum = _mm_set1_pd(LGAMMA_C0_D);

        __m128d vxp1 = _mm_add_pd(vz, _mm_set1_pd(1.0));
#ifdef OPTINUM_HAS_FMA
        vsum = _mm_fmadd_pd(_mm_div_pd(_mm_set1_pd(LGAMMA_C1_D), vxp1), _mm_set1_pd(1.0), vsum);
        __m128d vxp2 = _mm_add_pd(vz, _mm_set1_pd(2.0));
        vsum = _mm_fmadd_pd(_mm_div_pd(_mm_set1_pd(LGAMMA_C2_D), vxp2), _mm_set1_pd(1.0), vsum);
        __m128d vxp3 = _mm_add_pd(vz, _mm_set1_pd(3.0));
        vsum = _mm_fmadd_pd(_mm_div_pd(_mm_set1_pd(LGAMMA_C3_D), vxp3), _mm_set1_pd(1.0), vsum);
        __m128d vxp4 = _mm_add_pd(vz, _mm_set1_pd(4.0));
        vsum = _mm_fmadd_pd(_mm_div_pd(_mm_set1_pd(LGAMMA_C4_D), vxp4), _mm_set1_pd(1.0), vsum);
        __m128d vxp5 = _mm_add_pd(vz, _mm_set1_pd(5.0));
        vsum = _mm_fmadd_pd(_mm_div_pd(_mm_set1_pd(LGAMMA_C5_D), vxp5), _mm_set1_pd(1.0), vsum);
        __m128d vxp6 = _mm_add_pd(vz, _mm_set1_pd(6.0));
        vsum = _mm_fmadd_pd(_mm_div_pd(_mm_set1_pd(LGAMMA_C6_D), vxp6), _mm_set1_pd(1.0), vsum);
        __m128d vxp7 = _mm_add_pd(vz, _mm_set1_pd(7.0));
        vsum = _mm_fmadd_pd(_mm_div_pd(_mm_set1_pd(LGAMMA_C7_D), vxp7), _mm_set1_pd(1.0), vsum);
        __m128d vxp8 = _mm_add_pd(vz, _mm_set1_pd(8.0));
        vsum = _mm_fmadd_pd(_mm_div_pd(_mm_set1_pd(LGAMMA_C8_D), vxp8), _mm_set1_pd(1.0), vsum);
#else
        vsum = _mm_add_pd(vsum, _mm_div_pd(_mm_set1_pd(LGAMMA_C1_D), vxp1));
        __m128d vxp2 = _mm_add_pd(vz, _mm_set1_pd(2.0));
        vsum = _mm_add_pd(vsum, _mm_div_pd(_mm_set1_pd(LGAMMA_C2_D), vxp2));
        __m128d vxp3 = _mm_add_pd(vz, _mm_set1_pd(3.0));
        vsum = _mm_add_pd(vsum, _mm_div_pd(_mm_set1_pd(LGAMMA_C3_D), vxp3));
        __m128d vxp4 = _mm_add_pd(vz, _mm_set1_pd(4.0));
        vsum = _mm_add_pd(vsum, _mm_div_pd(_mm_set1_pd(LGAMMA_C4_D), vxp4));
        __m128d vxp5 = _mm_add_pd(vz, _mm_set1_pd(5.0));
        vsum = _mm_add_pd(vsum, _mm_div_pd(_mm_set1_pd(LGAMMA_C5_D), vxp5));
        __m128d vxp6 = _mm_add_pd(vz, _mm_set1_pd(6.0));
        vsum = _mm_add_pd(vsum, _mm_div_pd(_mm_set1_pd(LGAMMA_C6_D), vxp6));
        __m128d vxp7 = _mm_add_pd(vz, _mm_set1_pd(7.0));
        vsum = _mm_add_pd(vsum, _mm_div_pd(_mm_set1_pd(LGAMMA_C7_D), vxp7));
        __m128d vxp8 = _mm_add_pd(vz, _mm_set1_pd(8.0));
        vsum = _mm_add_pd(vsum, _mm_div_pd(_mm_set1_pd(LGAMMA_C8_D), vxp8));
#endif

        __m128d vt = _mm_add_pd(vz, _mm_set1_pd(LGAMMA_G_D + 0.5));
        __m128d vxp05 = _mm_add_pd(vz, _mm_set1_pd(0.5));

        __m128d vlog_t = log(pack<double, 2>(vt)).data_;
        __m128d vlog_sum = log(pack<double, 2>(vsum)).data_;

        __m128d vresult = _mm_set1_pd(LOG_SQRT_2PI_D);
        vresult = _mm_add_pd(vresult, _mm_mul_pd(vxp05, vlog_t));
        vresult = _mm_sub_pd(vresult, vt);
        vresult = _mm_add_pd(vresult, vlog_sum);

        return pack<double, 2>(vresult);
    }

#endif // OPTINUM_HAS_SSE41

// =============================================================================
// AVX Implementation for double (W=4)
// =============================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<double, 4> lgamma(const pack<double, 4> &x) noexcept {
        using namespace detail;

        __m256d vx = x.data_;

        // Lanczos computes log Γ(z+1), so use z = x - 1
        __m256d vz = _mm256_sub_pd(vx, _mm256_set1_pd(1.0));

        __m256d vsum = _mm256_set1_pd(LGAMMA_C0_D);

        __m256d vxp1 = _mm256_add_pd(vz, _mm256_set1_pd(1.0));
#ifdef OPTINUM_HAS_FMA
        vsum = _mm256_fmadd_pd(_mm256_div_pd(_mm256_set1_pd(LGAMMA_C1_D), vxp1), _mm256_set1_pd(1.0), vsum);
        __m256d vxp2 = _mm256_add_pd(vz, _mm256_set1_pd(2.0));
        vsum = _mm256_fmadd_pd(_mm256_div_pd(_mm256_set1_pd(LGAMMA_C2_D), vxp2), _mm256_set1_pd(1.0), vsum);
        __m256d vxp3 = _mm256_add_pd(vz, _mm256_set1_pd(3.0));
        vsum = _mm256_fmadd_pd(_mm256_div_pd(_mm256_set1_pd(LGAMMA_C3_D), vxp3), _mm256_set1_pd(1.0), vsum);
        __m256d vxp4 = _mm256_add_pd(vz, _mm256_set1_pd(4.0));
        vsum = _mm256_fmadd_pd(_mm256_div_pd(_mm256_set1_pd(LGAMMA_C4_D), vxp4), _mm256_set1_pd(1.0), vsum);
        __m256d vxp5 = _mm256_add_pd(vz, _mm256_set1_pd(5.0));
        vsum = _mm256_fmadd_pd(_mm256_div_pd(_mm256_set1_pd(LGAMMA_C5_D), vxp5), _mm256_set1_pd(1.0), vsum);
        __m256d vxp6 = _mm256_add_pd(vz, _mm256_set1_pd(6.0));
        vsum = _mm256_fmadd_pd(_mm256_div_pd(_mm256_set1_pd(LGAMMA_C6_D), vxp6), _mm256_set1_pd(1.0), vsum);
        __m256d vxp7 = _mm256_add_pd(vz, _mm256_set1_pd(7.0));
        vsum = _mm256_fmadd_pd(_mm256_div_pd(_mm256_set1_pd(LGAMMA_C7_D), vxp7), _mm256_set1_pd(1.0), vsum);
        __m256d vxp8 = _mm256_add_pd(vz, _mm256_set1_pd(8.0));
        vsum = _mm256_fmadd_pd(_mm256_div_pd(_mm256_set1_pd(LGAMMA_C8_D), vxp8), _mm256_set1_pd(1.0), vsum);
#else
        vsum = _mm256_add_pd(vsum, _mm256_div_pd(_mm256_set1_pd(LGAMMA_C1_D), vxp1));
        __m256d vxp2 = _mm256_add_pd(vz, _mm256_set1_pd(2.0));
        vsum = _mm256_add_pd(vsum, _mm256_div_pd(_mm256_set1_pd(LGAMMA_C2_D), vxp2));
        __m256d vxp3 = _mm256_add_pd(vz, _mm256_set1_pd(3.0));
        vsum = _mm256_add_pd(vsum, _mm256_div_pd(_mm256_set1_pd(LGAMMA_C3_D), vxp3));
        __m256d vxp4 = _mm256_add_pd(vz, _mm256_set1_pd(4.0));
        vsum = _mm256_add_pd(vsum, _mm256_div_pd(_mm256_set1_pd(LGAMMA_C4_D), vxp4));
        __m256d vxp5 = _mm256_add_pd(vz, _mm256_set1_pd(5.0));
        vsum = _mm256_add_pd(vsum, _mm256_div_pd(_mm256_set1_pd(LGAMMA_C5_D), vxp5));
        __m256d vxp6 = _mm256_add_pd(vx, _mm256_set1_pd(6.0));
        vsum = _mm256_add_pd(vsum, _mm256_div_pd(_mm256_set1_pd(LGAMMA_C6_D), vxp6));
        __m256d vxp7 = _mm256_add_pd(vx, _mm256_set1_pd(7.0));
        vsum = _mm256_add_pd(vsum, _mm256_div_pd(_mm256_set1_pd(LGAMMA_C7_D), vxp7));
        __m256d vxp8 = _mm256_add_pd(vx, _mm256_set1_pd(8.0));
        vsum = _mm256_add_pd(vsum, _mm256_div_pd(_mm256_set1_pd(LGAMMA_C8_D), vxp8));
#endif

        __m256d vt = _mm256_add_pd(vz, _mm256_set1_pd(LGAMMA_G_D + 0.5));
        __m256d vxp05 = _mm256_add_pd(vz, _mm256_set1_pd(0.5));

        __m256d vlog_t = log(pack<double, 4>(vt)).data_;
        __m256d vlog_sum = log(pack<double, 4>(vsum)).data_;

        __m256d vresult = _mm256_set1_pd(LOG_SQRT_2PI_D);
        vresult = _mm256_add_pd(vresult, _mm256_mul_pd(vxp05, vlog_t));
        vresult = _mm256_sub_pd(vresult, vt);
        vresult = _mm256_add_pd(vresult, vlog_sum);

        return pack<double, 4>(vresult);
    }

#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
