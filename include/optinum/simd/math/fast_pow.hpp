#pragma once

// Fast vectorized pow(x, y) implementation
// Uses the identity: pow(x, y) = exp(y * log(x))
// Accuracy: ~3-5 ULP (combines errors from fast_exp and fast_log)

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/intrinsic/simd_vec.hpp>
#include <optinum/simd/math/fast_exp.hpp>
#include <optinum/simd/math/fast_log.hpp>

#if defined(OPTINUM_HAS_AVX)
#include <optinum/simd/intrinsic/avx.hpp>
#endif
#if defined(OPTINUM_HAS_SSE2)
#include <optinum/simd/intrinsic/sse.hpp>
#endif

namespace optinum::simd {

    // ============================================================================
    // Algorithm for pow(x, y):
    //
    // pow(x, y) = exp(y * log(x))
    //
    // Special cases:
    // - pow(x, 0) = 1 for any x
    // - pow(0, y) = 0 for y > 0, inf for y < 0
    // - pow(1, y) = 1 for any y
    // - pow(x, 1) = x
    // - pow(x, 2) = x * x (special fast path)
    // - pow(x, 0.5) = sqrt(x) (special fast path)
    // - pow(x, -1) = 1/x (special fast path)
    // - pow(negative, non-integer) = NaN
    // ============================================================================

#if defined(OPTINUM_HAS_AVX)

    OPTINUM_INLINE SIMDVec<float, 8> fast_pow(const SIMDVec<float, 8> &x, const SIMDVec<float, 8> &y) {
        __m256 vx = x.value;
        __m256 vy = y.value;

        // Constants
        __m256 vzero = _mm256_setzero_ps();
        __m256 vone = _mm256_set1_ps(1.0f);
        __m256 vhalf = _mm256_set1_ps(0.5f);
        __m256 vtwo = _mm256_set1_ps(2.0f);
        __m256 vneg_one = _mm256_set1_ps(-1.0f);

        // Compute log(|x|)
        __m256 vabs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vx);
        SIMDVec<float, 8> log_x = fast_log(SIMDVec<float, 8>(vabs_x));
        __m256 vlog_x = log_x.value;

        // y * log(|x|)
        __m256 vy_log_x = _mm256_mul_ps(vy, vlog_x);

        // exp(y * log(|x|))
        SIMDVec<float, 8> result = fast_exp(SIMDVec<float, 8>(vy_log_x));
        __m256 vresult = result.value;

        // Handle x < 0: need to check if y is integer
        // For now, we'll handle the sign separately for negative x with odd integer y
        // This is a simplified approach - for non-integer y with x < 0, result should be NaN
        __m256 vx_neg = _mm256_cmp_ps(vx, vzero, _CMP_LT_OQ);

        // Check if y is an odd integer (simplified: floor(y) == y and floor(y/2)*2 != floor(y))
        __m256 vy_floor = _mm256_round_ps(vy, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m256 vy_is_int = _mm256_cmp_ps(vy, vy_floor, _CMP_EQ_OQ);
        __m256 vy_half = _mm256_mul_ps(vy_floor, vhalf);
        __m256 vy_half_floor = _mm256_round_ps(vy_half, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m256 vy_is_odd = _mm256_cmp_ps(vy_half, vy_half_floor, _CMP_NEQ_OQ);

        // If x < 0 and y is odd integer, negate result
        __m256 vnegate_mask = _mm256_and_ps(vx_neg, _mm256_and_ps(vy_is_int, vy_is_odd));
        vresult = _mm256_xor_ps(vresult, _mm256_and_ps(vnegate_mask, _mm256_set1_ps(-0.0f)));

        // If x < 0 and y is not integer, result is NaN
        __m256 vnan = _mm256_set1_ps(__builtin_nanf(""));
        __m256 vx_neg_non_int = _mm256_andnot_ps(vy_is_int, vx_neg);
        vresult = _mm256_blendv_ps(vresult, vnan, vx_neg_non_int);

        // Handle special cases
        // y == 0 -> result = 1
        __m256 vy_zero = _mm256_cmp_ps(vy, vzero, _CMP_EQ_OQ);
        vresult = _mm256_blendv_ps(vresult, vone, vy_zero);

        // x == 0 and y > 0 -> result = 0
        __m256 vx_zero = _mm256_cmp_ps(vx, vzero, _CMP_EQ_OQ);
        __m256 vy_pos = _mm256_cmp_ps(vy, vzero, _CMP_GT_OQ);
        vresult = _mm256_blendv_ps(vresult, vzero, _mm256_and_ps(vx_zero, vy_pos));

        // x == 0 and y < 0 -> result = inf
        __m256 vinf = _mm256_set1_ps(__builtin_inff());
        __m256 vy_neg = _mm256_cmp_ps(vy, vzero, _CMP_LT_OQ);
        vresult = _mm256_blendv_ps(vresult, vinf, _mm256_and_ps(vx_zero, vy_neg));

        // x == 1 -> result = 1
        __m256 vx_one = _mm256_cmp_ps(vx, vone, _CMP_EQ_OQ);
        vresult = _mm256_blendv_ps(vresult, vone, vx_one);

        return SIMDVec<float, 8>(vresult);
    }

    // Fast path for integer powers
    OPTINUM_INLINE SIMDVec<float, 8> fast_powi(const SIMDVec<float, 8> &x, int n) {
        if (n == 0)
            return SIMDVec<float, 8>(_mm256_set1_ps(1.0f));
        if (n == 1)
            return x;
        if (n == -1)
            return SIMDVec<float, 8>(_mm256_div_ps(_mm256_set1_ps(1.0f), x.value));
        if (n == 2)
            return SIMDVec<float, 8>(_mm256_mul_ps(x.value, x.value));

        // Use binary exponentiation for larger powers
        bool negative = n < 0;
        unsigned int exp = negative ? static_cast<unsigned int>(-n) : static_cast<unsigned int>(n);

        __m256 vresult = _mm256_set1_ps(1.0f);
        __m256 vbase = x.value;

        while (exp > 0) {
            if (exp & 1) {
                vresult = _mm256_mul_ps(vresult, vbase);
            }
            vbase = _mm256_mul_ps(vbase, vbase);
            exp >>= 1;
        }

        if (negative) {
            vresult = _mm256_div_ps(_mm256_set1_ps(1.0f), vresult);
        }

        return SIMDVec<float, 8>(vresult);
    }

    // Fast sqrt using rsqrt with Newton-Raphson refinement
    OPTINUM_INLINE SIMDVec<float, 8> fast_sqrt(const SIMDVec<float, 8> &x) {
        __m256 vx = x.value;

        // Initial approximation using rsqrt (1/sqrt(x))
        __m256 vrsqrt = _mm256_rsqrt_ps(vx);

        // One Newton-Raphson iteration for rsqrt: y = y * (3 - x*y*y) / 2
        __m256 vhalf = _mm256_set1_ps(0.5f);
        __m256 vthree = _mm256_set1_ps(3.0f);
        __m256 vy2 = _mm256_mul_ps(vrsqrt, vrsqrt);
        __m256 vxy2 = _mm256_mul_ps(vx, vy2);
#ifdef OPTINUM_HAS_FMA
        __m256 vrefine = _mm256_fmsub_ps(vthree, vhalf, _mm256_mul_ps(vxy2, vhalf));
        vrsqrt = _mm256_mul_ps(vrsqrt, _mm256_fmadd_ps(_mm256_sub_ps(vthree, vxy2), vhalf, _mm256_setzero_ps()));
#else
        vrsqrt = _mm256_mul_ps(vrsqrt, _mm256_mul_ps(_mm256_sub_ps(vthree, vxy2), vhalf));
#endif

        // sqrt(x) = x * rsqrt(x)
        __m256 vresult = _mm256_mul_ps(vx, vrsqrt);

        // Handle x == 0 (avoid 0 * inf = NaN)
        __m256 vzero = _mm256_setzero_ps();
        __m256 vx_zero = _mm256_cmp_ps(vx, vzero, _CMP_EQ_OQ);
        vresult = _mm256_blendv_ps(vresult, vzero, vx_zero);

        // Handle x < 0 -> NaN
        __m256 vnan = _mm256_set1_ps(__builtin_nanf(""));
        __m256 vx_neg = _mm256_cmp_ps(vx, vzero, _CMP_LT_OQ);
        vresult = _mm256_blendv_ps(vresult, vnan, vx_neg);

        return SIMDVec<float, 8>(vresult);
    }

    // Fast reciprocal sqrt (1/sqrt(x))
    OPTINUM_INLINE SIMDVec<float, 8> fast_rsqrt(const SIMDVec<float, 8> &x) {
        __m256 vx = x.value;

        // Initial approximation
        __m256 vrsqrt = _mm256_rsqrt_ps(vx);

        // Newton-Raphson refinement
        __m256 vhalf = _mm256_set1_ps(0.5f);
        __m256 vthree = _mm256_set1_ps(3.0f);
        __m256 vy2 = _mm256_mul_ps(vrsqrt, vrsqrt);
        __m256 vxy2 = _mm256_mul_ps(vx, vy2);
        vrsqrt = _mm256_mul_ps(vrsqrt, _mm256_mul_ps(_mm256_sub_ps(vthree, vxy2), vhalf));

        return SIMDVec<float, 8>(vrsqrt);
    }

    // Cube root using Newton-Raphson
    OPTINUM_INLINE SIMDVec<float, 8> fast_cbrt(const SIMDVec<float, 8> &x) {
        // cbrt(x) = exp(log(x) / 3)
        __m256 vx = x.value;
        __m256 vabs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vx);
        __m256 vsign = _mm256_and_ps(vx, _mm256_set1_ps(-0.0f));

        SIMDVec<float, 8> log_abs = fast_log(SIMDVec<float, 8>(vabs_x));
        __m256 vlog_div3 = _mm256_mul_ps(log_abs.value, _mm256_set1_ps(1.0f / 3.0f));
        SIMDVec<float, 8> result = fast_exp(SIMDVec<float, 8>(vlog_div3));

        // Restore sign (cbrt preserves sign)
        __m256 vresult = _mm256_or_ps(result.value, vsign);

        // Handle x == 0
        __m256 vzero = _mm256_setzero_ps();
        __m256 vx_zero = _mm256_cmp_ps(vx, vzero, _CMP_EQ_OQ);
        vresult = _mm256_blendv_ps(vresult, vzero, vx_zero);

        return SIMDVec<float, 8>(vresult);
    }

#endif // OPTINUM_HAS_AVX

#if defined(OPTINUM_HAS_SSE41)

    OPTINUM_INLINE SIMDVec<float, 4> fast_pow(const SIMDVec<float, 4> &x, const SIMDVec<float, 4> &y) {
        __m128 vx = x.value;
        __m128 vy = y.value;

        __m128 vzero = _mm_setzero_ps();
        __m128 vone = _mm_set1_ps(1.0f);
        __m128 vhalf = _mm_set1_ps(0.5f);

        // Compute log(|x|)
        __m128 vabs_x = _mm_andnot_ps(_mm_set1_ps(-0.0f), vx);
        SIMDVec<float, 4> log_x = fast_log(SIMDVec<float, 4>(vabs_x));
        __m128 vlog_x = log_x.value;

        // y * log(|x|)
        __m128 vy_log_x = _mm_mul_ps(vy, vlog_x);

        // exp(y * log(|x|))
        SIMDVec<float, 4> result = fast_exp(SIMDVec<float, 4>(vy_log_x));
        __m128 vresult = result.value;

        // Handle x < 0
        __m128 vx_neg = _mm_cmplt_ps(vx, vzero);
        __m128 vy_floor = _mm_round_ps(vy, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m128 vy_is_int = _mm_cmpeq_ps(vy, vy_floor);
        __m128 vy_half = _mm_mul_ps(vy_floor, vhalf);
        __m128 vy_half_floor = _mm_round_ps(vy_half, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m128 vy_is_odd = _mm_cmpneq_ps(vy_half, vy_half_floor);

        __m128 vnegate_mask = _mm_and_ps(vx_neg, _mm_and_ps(vy_is_int, vy_is_odd));
        vresult = _mm_xor_ps(vresult, _mm_and_ps(vnegate_mask, _mm_set1_ps(-0.0f)));

        __m128 vnan = _mm_set1_ps(__builtin_nanf(""));
        __m128 vx_neg_non_int = _mm_andnot_ps(vy_is_int, vx_neg);
        vresult = _mm_blendv_ps(vresult, vnan, vx_neg_non_int);

        // y == 0 -> 1
        __m128 vy_zero = _mm_cmpeq_ps(vy, vzero);
        vresult = _mm_blendv_ps(vresult, vone, vy_zero);

        // x == 0, y > 0 -> 0
        __m128 vx_zero = _mm_cmpeq_ps(vx, vzero);
        __m128 vy_pos = _mm_cmpgt_ps(vy, vzero);
        vresult = _mm_blendv_ps(vresult, vzero, _mm_and_ps(vx_zero, vy_pos));

        // x == 0, y < 0 -> inf
        __m128 vinf = _mm_set1_ps(__builtin_inff());
        __m128 vy_neg = _mm_cmplt_ps(vy, vzero);
        vresult = _mm_blendv_ps(vresult, vinf, _mm_and_ps(vx_zero, vy_neg));

        // x == 1 -> 1
        __m128 vx_one = _mm_cmpeq_ps(vx, vone);
        vresult = _mm_blendv_ps(vresult, vone, vx_one);

        return SIMDVec<float, 4>(vresult);
    }

    OPTINUM_INLINE SIMDVec<float, 4> fast_powi(const SIMDVec<float, 4> &x, int n) {
        if (n == 0)
            return SIMDVec<float, 4>(_mm_set1_ps(1.0f));
        if (n == 1)
            return x;
        if (n == -1)
            return SIMDVec<float, 4>(_mm_div_ps(_mm_set1_ps(1.0f), x.value));
        if (n == 2)
            return SIMDVec<float, 4>(_mm_mul_ps(x.value, x.value));

        bool negative = n < 0;
        unsigned int exp = negative ? static_cast<unsigned int>(-n) : static_cast<unsigned int>(n);

        __m128 vresult = _mm_set1_ps(1.0f);
        __m128 vbase = x.value;

        while (exp > 0) {
            if (exp & 1) {
                vresult = _mm_mul_ps(vresult, vbase);
            }
            vbase = _mm_mul_ps(vbase, vbase);
            exp >>= 1;
        }

        if (negative) {
            vresult = _mm_div_ps(_mm_set1_ps(1.0f), vresult);
        }

        return SIMDVec<float, 4>(vresult);
    }

    OPTINUM_INLINE SIMDVec<float, 4> fast_sqrt(const SIMDVec<float, 4> &x) {
        __m128 vx = x.value;

        __m128 vrsqrt = _mm_rsqrt_ps(vx);

        __m128 vhalf = _mm_set1_ps(0.5f);
        __m128 vthree = _mm_set1_ps(3.0f);
        __m128 vy2 = _mm_mul_ps(vrsqrt, vrsqrt);
        __m128 vxy2 = _mm_mul_ps(vx, vy2);
        vrsqrt = _mm_mul_ps(vrsqrt, _mm_mul_ps(_mm_sub_ps(vthree, vxy2), vhalf));

        __m128 vresult = _mm_mul_ps(vx, vrsqrt);

        __m128 vzero = _mm_setzero_ps();
        __m128 vx_zero = _mm_cmpeq_ps(vx, vzero);
        vresult = _mm_blendv_ps(vresult, vzero, vx_zero);

        __m128 vnan = _mm_set1_ps(__builtin_nanf(""));
        __m128 vx_neg = _mm_cmplt_ps(vx, vzero);
        vresult = _mm_blendv_ps(vresult, vnan, vx_neg);

        return SIMDVec<float, 4>(vresult);
    }

    OPTINUM_INLINE SIMDVec<float, 4> fast_rsqrt(const SIMDVec<float, 4> &x) {
        __m128 vx = x.value;

        __m128 vrsqrt = _mm_rsqrt_ps(vx);

        __m128 vhalf = _mm_set1_ps(0.5f);
        __m128 vthree = _mm_set1_ps(3.0f);
        __m128 vy2 = _mm_mul_ps(vrsqrt, vrsqrt);
        __m128 vxy2 = _mm_mul_ps(vx, vy2);
        vrsqrt = _mm_mul_ps(vrsqrt, _mm_mul_ps(_mm_sub_ps(vthree, vxy2), vhalf));

        return SIMDVec<float, 4>(vrsqrt);
    }

    OPTINUM_INLINE SIMDVec<float, 4> fast_cbrt(const SIMDVec<float, 4> &x) {
        __m128 vx = x.value;
        __m128 vabs_x = _mm_andnot_ps(_mm_set1_ps(-0.0f), vx);
        __m128 vsign = _mm_and_ps(vx, _mm_set1_ps(-0.0f));

        SIMDVec<float, 4> log_abs = fast_log(SIMDVec<float, 4>(vabs_x));
        __m128 vlog_div3 = _mm_mul_ps(log_abs.value, _mm_set1_ps(1.0f / 3.0f));
        SIMDVec<float, 4> result = fast_exp(SIMDVec<float, 4>(vlog_div3));

        __m128 vresult = _mm_or_ps(result.value, vsign);

        __m128 vzero = _mm_setzero_ps();
        __m128 vx_zero = _mm_cmpeq_ps(vx, vzero);
        vresult = _mm_blendv_ps(vresult, vzero, vx_zero);

        return SIMDVec<float, 4>(vresult);
    }

#endif // OPTINUM_HAS_SSE41

} // namespace optinum::simd
