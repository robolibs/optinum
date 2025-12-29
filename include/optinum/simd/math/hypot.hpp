#pragma once

// =============================================================================
// optinum/simd/math/hypot.hpp
// Vectorized hypot() using pack<T,W> with SIMD intrinsics
// Hypotenuse: hypot(x, y) = sqrt(x² + y²)
// Avoids overflow/underflow by scaling
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/math/abs.hpp>
#include <optinum/simd/math/sqrt.hpp>
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
    template <typename T, std::size_t W> pack<T, W> hypot(const pack<T, W> &x, const pack<T, W> &y) noexcept;

    // =========================================================================
    // pack<float, 4> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE2)

#if defined(OPTINUM_HAS_SSE2)
    template <> inline pack<float, 4> hypot(const pack<float, 4> &x, const pack<float, 4> &y) noexcept {
        __m128 vx = x.data_;
        __m128 vy = y.data_;

        // Get absolute values
        __m128i abs_mask = _mm_set1_epi32(0x7FFFFFFF);
        __m128 vax = _mm_and_ps(vx, _mm_castsi128_ps(abs_mask));
        __m128 vay = _mm_and_ps(vy, _mm_castsi128_ps(abs_mask));

        // Scale by max to avoid overflow
        __m128 vmax = _mm_max_ps(vax, vay);
        __m128 vmin = _mm_min_ps(vax, vay);

        // Compute ratio = min / max
        __m128 vratio = _mm_div_ps(vmin, vmax);

        // result = max * sqrt(1 + ratio²)
        __m128 vratio2 = _mm_mul_ps(vratio, vratio);
        __m128 vone = _mm_set1_ps(1.0f);
        __m128 vsum = _mm_add_ps(vone, vratio2);
        __m128 vsqrt = _mm_sqrt_ps(vsum);
        __m128 vresult = _mm_mul_ps(vmax, vsqrt);

        // Handle special cases: if max == 0, result = 0
        __m128 vzero = _mm_setzero_ps();
        __m128 vmask_zero = _mm_cmpeq_ps(vmax, vzero);
        vresult = _mm_blendv_ps(vresult, vzero, vmask_zero);

        return pack<float, 4>(vresult);
    }

#endif // OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<float, 8> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

#endif // OPTINUM_HAS_SSE2

#if defined(OPTINUM_HAS_AVX)
    template <> inline pack<float, 8> hypot(const pack<float, 8> &x, const pack<float, 8> &y) noexcept {
        __m256 vx = x.data_;
        __m256 vy = y.data_;

        // Get absolute values
        __m256i abs_mask = _mm256_set1_epi32(0x7FFFFFFF);
        __m256 vax = _mm256_and_ps(vx, _mm256_castsi256_ps(abs_mask));
        __m256 vay = _mm256_and_ps(vy, _mm256_castsi256_ps(abs_mask));

        // Scale by max to avoid overflow
        __m256 vmax = _mm256_max_ps(vax, vay);
        __m256 vmin = _mm256_min_ps(vax, vay);

        // Compute ratio = min / max
        __m256 vratio = _mm256_div_ps(vmin, vmax);

        // result = max * sqrt(1 + ratio²)
        __m256 vratio2 = _mm256_mul_ps(vratio, vratio);
        __m256 vone = _mm256_set1_ps(1.0f);
        __m256 vsum = _mm256_add_ps(vone, vratio2);
        __m256 vsqrt = _mm256_sqrt_ps(vsum);
        __m256 vresult = _mm256_mul_ps(vmax, vsqrt);

        // Handle special cases: if max == 0, result = 0
        __m256 vzero = _mm256_setzero_ps();
        __m256 vmask_zero = _mm256_cmp_ps(vmax, vzero, _CMP_EQ_OQ);
        vresult = _mm256_blendv_ps(vresult, vzero, vmask_zero);

        return pack<float, 8>(vresult);
    }

#endif // OPTINUM_HAS_AVX

    // =========================================================================
#endif // OPTINUM_HAS_AVX

    // pack<double, 2> - SSE implementation
    // =========================================================================
#if defined(OPTINUM_HAS_SSE2)

    template <> inline pack<double, 2> hypot(const pack<double, 2> &x, const pack<double, 2> &y) noexcept {
        __m128d vx = x.data_;
        __m128d vy = y.data_;

        // Get absolute values
        __m128i abs_mask = _mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL);
        __m128d vax = _mm_and_pd(vx, _mm_castsi128_pd(abs_mask));
        __m128d vay = _mm_and_pd(vy, _mm_castsi128_pd(abs_mask));

        // Scale by max to avoid overflow
        __m128d vmax = _mm_max_pd(vax, vay);
        __m128d vmin = _mm_min_pd(vax, vay);

        // Compute ratio = min / max
        __m128d vratio = _mm_div_pd(vmin, vmax);

        // result = max * sqrt(1 + ratio²)
        __m128d vratio2 = _mm_mul_pd(vratio, vratio);
        __m128d vone = _mm_set1_pd(1.0);
        __m128d vsum = _mm_add_pd(vone, vratio2);
        __m128d vsqrt = _mm_sqrt_pd(vsum);
        __m128d vresult = _mm_mul_pd(vmax, vsqrt);

        // Handle special cases: if max == 0, result = 0
        __m128d vzero = _mm_setzero_pd();
        __m128d vmask_zero = _mm_cmpeq_pd(vmax, vzero);
        vresult = _mm_blendv_pd(vresult, vzero, vmask_zero);

        return pack<double, 2>(vresult);
    }

#endif // OPTINUM_HAS_SSE2

    // =========================================================================
    // pack<double, 4> - AVX implementation
    // =========================================================================
#if defined(OPTINUM_HAS_AVX)

    template <> inline pack<double, 4> hypot(const pack<double, 4> &x, const pack<double, 4> &y) noexcept {
        __m256d vx = x.data_;
        __m256d vy = y.data_;

        // Get absolute values
        __m256i abs_mask = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL);
        __m256d vax = _mm256_and_pd(vx, _mm256_castsi256_pd(abs_mask));
        __m256d vay = _mm256_and_pd(vy, _mm256_castsi256_pd(abs_mask));

        // Scale by max to avoid overflow
        __m256d vmax = _mm256_max_pd(vax, vay);
        __m256d vmin = _mm256_min_pd(vax, vay);

        // Compute ratio = min / max
        __m256d vratio = _mm256_div_pd(vmin, vmax);

        // result = max * sqrt(1 + ratio²)
        __m256d vratio2 = _mm256_mul_pd(vratio, vratio);
        __m256d vone = _mm256_set1_pd(1.0);
        __m256d vsum = _mm256_add_pd(vone, vratio2);
        __m256d vsqrt = _mm256_sqrt_pd(vsum);
        __m256d vresult = _mm256_mul_pd(vmax, vsqrt);

        // Handle special cases: if max == 0, result = 0
        __m256d vzero = _mm256_setzero_pd();
        __m256d vmask_zero = _mm256_cmp_pd(vmax, vzero, _CMP_EQ_OQ);
        vresult = _mm256_blendv_pd(vresult, vzero, vmask_zero);

        return pack<double, 4>(vresult);
    }

#endif // OPTINUM_HAS_AVX

} // namespace optinum::simd
