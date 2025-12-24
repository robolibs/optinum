#pragma once

// Mathematical constants for SIMD math implementations
// All constants are provided as both float and double

namespace optinum::simd::math_constants {

    // ============================================================================
    // Basic Mathematical Constants
    // ============================================================================

    // Pi and related
    constexpr float PI_F = 3.14159265358979323846f;
    constexpr double PI_D = 3.14159265358979323846;

    constexpr float PI_2_F = 1.57079632679489661923f; // pi/2
    constexpr double PI_2_D = 1.57079632679489661923;

    constexpr float PI_4_F = 0.78539816339744830962f; // pi/4
    constexpr double PI_4_D = 0.78539816339744830962;

    constexpr float TWO_PI_F = 6.28318530717958647693f; // 2*pi
    constexpr double TWO_PI_D = 6.28318530717958647693;

    constexpr float INV_PI_F = 0.31830988618379067154f; // 1/pi
    constexpr double INV_PI_D = 0.31830988618379067154;

    constexpr float TWO_INV_PI_F = 0.63661977236758134308f; // 2/pi
    constexpr double TWO_INV_PI_D = 0.63661977236758134308;

    constexpr float FOUR_INV_PI_F = 1.27323954473516268615f; // 4/pi
    constexpr double FOUR_INV_PI_D = 1.27323954473516268615;

    // ============================================================================
    // Euler's Number and Logarithms
    // ============================================================================

    constexpr float E_F = 2.71828182845904523536f;
    constexpr double E_D = 2.71828182845904523536;

    constexpr float LN2_F = 0.69314718055994530942f;
    constexpr double LN2_D = 0.69314718055994530942;

    constexpr float LN10_F = 2.30258509299404568402f;
    constexpr double LN10_D = 2.30258509299404568402;

    constexpr float LOG2E_F = 1.44269504088896340736f; // 1/ln(2)
    constexpr double LOG2E_D = 1.44269504088896340736;

    constexpr float LOG10E_F = 0.43429448190325182765f; // 1/ln(10)
    constexpr double LOG10E_D = 0.43429448190325182765;

    // High/Low parts of ln(2) for accurate range reduction
    constexpr float LN2_HI_F = 0.693145751953125f;
    constexpr float LN2_LO_F = 1.42860682030941723212e-6f;

    constexpr double LN2_HI_D = 0.6931471803691238164901733;
    constexpr double LN2_LO_D = 1.9082149292705877e-10;

    // ============================================================================
    // Polynomial Coefficients for exp(x)
    // Minimax polynomial on [-ln2/2, ln2/2]
    // exp(x) = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5!
    // ============================================================================

    constexpr float EXP_C1_F = 1.0f;
    constexpr float EXP_C2_F = 0.5f;
    constexpr float EXP_C3_F = 0.16666666666666666f;  // 1/6
    constexpr float EXP_C4_F = 0.041666666666666664f; // 1/24
    constexpr float EXP_C5_F = 0.008333333333333333f; // 1/120
    constexpr float EXP_C6_F = 0.001388888888888889f; // 1/720

    // ============================================================================
    // Polynomial Coefficients for log(x)
    // Using log(1+x) = x - x^2/2 + x^3/3 - x^4/4 + ...
    // Or Remez-optimized coefficients for better accuracy
    // ============================================================================

    // Coefficients for log(1+x) approximation on [sqrt(2)/2 - 1, sqrt(2) - 1]
    constexpr float LOG_C1_F = -0.5f;
    constexpr float LOG_C2_F = 0.33333333333333333f;  // 1/3
    constexpr float LOG_C3_F = -0.25f;                // 1/4
    constexpr float LOG_C4_F = 0.2f;                  // 1/5
    constexpr float LOG_C5_F = -0.16666666666666666f; // 1/6

    // Alternative: Remez-optimized coefficients for log(x) on [1, 2]
    // P(m) where m = (x-1)/(x+1), log(x) = 2*P(m)
    constexpr float LOG_P1_F = 2.0f;
    constexpr float LOG_P3_F = 0.6666666666666666f; // 2/3
    constexpr float LOG_P5_F = 0.4f;                // 2/5
    constexpr float LOG_P7_F = 0.2857142857142857f; // 2/7

    // ============================================================================
    // Polynomial Coefficients for sin(x)
    // sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
    // Valid for x in [-pi/4, pi/4]
    // ============================================================================

    constexpr float SIN_C3_F = -0.16666666666666666f;   // -1/6
    constexpr float SIN_C5_F = 0.008333333333333333f;   // 1/120
    constexpr float SIN_C7_F = -0.0001984126984126984f; // -1/5040
    constexpr float SIN_C9_F = 2.7557319223985893e-6f;  // 1/362880

    // Remez-optimized for better accuracy
    constexpr float SIN_P3_F = -0.16666666641626524f;
    constexpr float SIN_P5_F = 0.008333329385889463f;
    constexpr float SIN_P7_F = -0.00019840874255936685f;
    constexpr float SIN_P9_F = 2.718311493989822e-6f;

    // ============================================================================
    // Polynomial Coefficients for cos(x)
    // cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + ...
    // Valid for x in [-pi/4, pi/4]
    // ============================================================================

    constexpr float COS_C2_F = -0.5f;                  // -1/2
    constexpr float COS_C4_F = 0.041666666666666664f;  // 1/24
    constexpr float COS_C6_F = -0.001388888888888889f; // -1/720
    constexpr float COS_C8_F = 2.48015873015873e-5f;   // 1/40320

    // Remez-optimized
    constexpr float COS_P2_F = -0.49999999997024012f;
    constexpr float COS_P4_F = 0.041666666473384543f;
    constexpr float COS_P6_F = -0.001388888839238262f;
    constexpr float COS_P8_F = 2.4801428034205937e-5f;

    // ============================================================================
    // Polynomial Coefficients for tanh(x)
    // tanh(x) = x - x^3/3 + 2x^5/15 - 17x^7/315 + ...
    // Valid for |x| < 0.625
    // ============================================================================

    constexpr float TANH_C3_F = -0.33333333333333333f; // -1/3
    constexpr float TANH_C5_F = 0.13333333333333333f;  // 2/15
    constexpr float TANH_C7_F = -0.05396825396825397f; // -17/315
    constexpr float TANH_C9_F = 0.021869488536155203f; // 62/2835

    // ============================================================================
    // Range Reduction Constants
    // ============================================================================

    // For trig functions: reduce to [-pi, pi] then to [-pi/4, pi/4]
    constexpr float INV_TWO_PI_F = 0.15915494309189535f; // 1/(2*pi)
    constexpr double INV_TWO_PI_D = 0.15915494309189535;

    // Cody-Waite constants for pi reduction (high precision)
    constexpr float PI_HI_F = 3.140625f;
    constexpr float PI_LO_F = 9.67653589793e-4f;

    // ============================================================================
    // Overflow/Underflow Thresholds
    // ============================================================================

    constexpr float EXP_OVERFLOW_F = 88.7228391f;   // ln(FLT_MAX)
    constexpr float EXP_UNDERFLOW_F = -87.3365447f; // ln(FLT_MIN) approx

    constexpr double EXP_OVERFLOW_D = 709.782712893384;   // ln(DBL_MAX)
    constexpr double EXP_UNDERFLOW_D = -708.396418532264; // ln(DBL_MIN) approx

    // ============================================================================
    // Special Values
    // ============================================================================

    constexpr float SQRT2_F = 1.41421356237309504880f;
    constexpr double SQRT2_D = 1.41421356237309504880;

    constexpr float SQRT2_2_F = 0.70710678118654752440f; // sqrt(2)/2
    constexpr double SQRT2_2_D = 0.70710678118654752440;

} // namespace optinum::simd::math_constants
