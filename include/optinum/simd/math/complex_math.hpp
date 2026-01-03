#pragma once

// =============================================================================
// optinum/simd/math/complex_math.hpp
// Complex math functions for pack<complex<T>, W>
// =============================================================================

#include <datapod/matrix/math/complex.hpp>
#include <optinum/simd/math/acos.hpp>
#include <optinum/simd/math/acosh.hpp>
#include <optinum/simd/math/asin.hpp>
#include <optinum/simd/math/asinh.hpp>
#include <optinum/simd/math/atan.hpp>
#include <optinum/simd/math/atan2.hpp>
#include <optinum/simd/math/atanh.hpp>
#include <optinum/simd/math/cos.hpp>
#include <optinum/simd/math/cosh.hpp>
#include <optinum/simd/math/exp.hpp>
#include <optinum/simd/math/log.hpp>
#include <optinum/simd/math/sin.hpp>
#include <optinum/simd/math/sinh.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/math/tan.hpp>
#include <optinum/simd/math/tanh.hpp>
#include <optinum/simd/pack/complex.hpp>

namespace optinum::simd {

    namespace dp = ::datapod;

    // ===== Exponential functions =====

    /**
     * @brief Complex exponential: exp(a+bi) = e^a * (cos(b) + i*sin(b))
     */
    template <typename T, std::size_t W>
    inline pack<dp::mat::Complex<T>, W> exp(const pack<dp::mat::Complex<T>, W> &z) noexcept {
        auto exp_real = exp(z.real());
        auto cos_imag = cos(z.imag());
        auto sin_imag = sin(z.imag());
        return pack<dp::mat::Complex<T>, W>(exp_real * cos_imag, exp_real * sin_imag);
    }

    /**
     * @brief Complex natural logarithm: log(a+bi) = log(|z|) + i*arg(z)
     */
    template <typename T, std::size_t W>
    inline pack<dp::mat::Complex<T>, W> log(const pack<dp::mat::Complex<T>, W> &z) noexcept {
        auto mag = z.magnitude();
        auto phase = z.phase();
        return pack<dp::mat::Complex<T>, W>(log(mag), phase);
    }

    /**
     * @brief Complex square root: sqrt(a+bi) = sqrt(|z|) * (cos(arg/2) + i*sin(arg/2))
     */
    template <typename T, std::size_t W>
    inline pack<dp::mat::Complex<T>, W> sqrt(const pack<dp::mat::Complex<T>, W> &z) noexcept {
        auto mag = z.magnitude();
        auto phase = z.phase();
        auto sqrt_mag = sqrt(mag);
        auto half_phase = phase * pack<T, W>(T{0.5});
        return pack<dp::mat::Complex<T>, W>(sqrt_mag * cos(half_phase), sqrt_mag * sin(half_phase));
    }

    /**
     * @brief Complex power: z^w = exp(w * log(z))
     */
    template <typename T, std::size_t W>
    inline pack<dp::mat::Complex<T>, W> pow(const pack<dp::mat::Complex<T>, W> &z,
                                            const pack<dp::mat::Complex<T>, W> &w) noexcept {
        return exp(w * log(z));
    }

    // ===== Trigonometric functions =====

    /**
     * @brief Complex sine: sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
     */
    template <typename T, std::size_t W>
    inline pack<dp::mat::Complex<T>, W> sin(const pack<dp::mat::Complex<T>, W> &z) noexcept {
        auto sin_real = sin(z.real());
        auto cos_real = cos(z.real());
        auto sinh_imag = sinh(z.imag());
        auto cosh_imag = cosh(z.imag());
        return pack<dp::mat::Complex<T>, W>(sin_real * cosh_imag, cos_real * sinh_imag);
    }

    /**
     * @brief Complex cosine: cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
     */
    template <typename T, std::size_t W>
    inline pack<dp::mat::Complex<T>, W> cos(const pack<dp::mat::Complex<T>, W> &z) noexcept {
        auto sin_real = sin(z.real());
        auto cos_real = cos(z.real());
        auto sinh_imag = sinh(z.imag());
        auto cosh_imag = cosh(z.imag());
        return pack<dp::mat::Complex<T>, W>(cos_real * cosh_imag, -sin_real * sinh_imag);
    }

    /**
     * @brief Complex tangent: tan(z) = sin(z) / cos(z)
     */
    template <typename T, std::size_t W>
    inline pack<dp::mat::Complex<T>, W> tan(const pack<dp::mat::Complex<T>, W> &z) noexcept {
        return sin(z) / cos(z);
    }

    // ===== Hyperbolic functions =====

    /**
     * @brief Complex hyperbolic sine: sinh(a+bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b)
     */
    template <typename T, std::size_t W>
    inline pack<dp::mat::Complex<T>, W> sinh(const pack<dp::mat::Complex<T>, W> &z) noexcept {
        auto sinh_real = sinh(z.real());
        auto cosh_real = cosh(z.real());
        auto sin_imag = sin(z.imag());
        auto cos_imag = cos(z.imag());
        return pack<dp::mat::Complex<T>, W>(sinh_real * cos_imag, cosh_real * sin_imag);
    }

    /**
     * @brief Complex hyperbolic cosine: cosh(a+bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b)
     */
    template <typename T, std::size_t W>
    inline pack<dp::mat::Complex<T>, W> cosh(const pack<dp::mat::Complex<T>, W> &z) noexcept {
        auto sinh_real = sinh(z.real());
        auto cosh_real = cosh(z.real());
        auto sin_imag = sin(z.imag());
        auto cos_imag = cos(z.imag());
        return pack<dp::mat::Complex<T>, W>(cosh_real * cos_imag, sinh_real * sin_imag);
    }

    /**
     * @brief Complex hyperbolic tangent: tanh(z) = sinh(z) / cosh(z)
     */
    template <typename T, std::size_t W>
    inline pack<dp::mat::Complex<T>, W> tanh(const pack<dp::mat::Complex<T>, W> &z) noexcept {
        return sinh(z) / cosh(z);
    }

    // ===== Inverse trigonometric functions =====

    /**
     * @brief Complex arc sine: asin(z) = -i * log(iz + sqrt(1 - z²))
     */
    template <typename T, std::size_t W>
    inline pack<dp::mat::Complex<T>, W> asin(const pack<dp::mat::Complex<T>, W> &z) noexcept {
        using cpack = pack<dp::mat::Complex<T>, W>;
        auto one = cpack(pack<T, W>(T{1}), pack<T, W>(T{}));
        auto i = cpack(pack<T, W>(T{}), pack<T, W>(T{1}));
        auto z_sq = z * z;
        auto inner = i * z + sqrt(one - z_sq);
        return -i * log(inner);
    }

    /**
     * @brief Complex arc cosine: acos(z) = -i * log(z + i*sqrt(1 - z²))
     */
    template <typename T, std::size_t W>
    inline pack<dp::mat::Complex<T>, W> acos(const pack<dp::mat::Complex<T>, W> &z) noexcept {
        using cpack = pack<dp::mat::Complex<T>, W>;
        auto one = cpack(pack<T, W>(T{1}), pack<T, W>(T{}));
        auto i = cpack(pack<T, W>(T{}), pack<T, W>(T{1}));
        auto z_sq = z * z;
        auto inner = z + i * sqrt(one - z_sq);
        return -i * log(inner);
    }

    /**
     * @brief Complex arc tangent: atan(z) = (i/2) * log((i+z)/(i-z))
     */
    template <typename T, std::size_t W>
    inline pack<dp::mat::Complex<T>, W> atan(const pack<dp::mat::Complex<T>, W> &z) noexcept {
        using cpack = pack<dp::mat::Complex<T>, W>;
        auto i = cpack(pack<T, W>(T{}), pack<T, W>(T{1}));
        auto half_i = i * pack<T, W>(T{0.5});
        return half_i * log((i + z) / (i - z));
    }

    // ===== Inverse hyperbolic functions =====

    /**
     * @brief Complex inverse hyperbolic sine: asinh(z) = log(z + sqrt(z² + 1))
     */
    template <typename T, std::size_t W>
    inline pack<dp::mat::Complex<T>, W> asinh(const pack<dp::mat::Complex<T>, W> &z) noexcept {
        using cpack = pack<dp::mat::Complex<T>, W>;
        auto one = cpack(pack<T, W>(T{1}), pack<T, W>(T{}));
        auto z_sq = z * z;
        return log(z + sqrt(z_sq + one));
    }

    /**
     * @brief Complex inverse hyperbolic cosine: acosh(z) = log(z + sqrt(z² - 1))
     */
    template <typename T, std::size_t W>
    inline pack<dp::mat::Complex<T>, W> acosh(const pack<dp::mat::Complex<T>, W> &z) noexcept {
        using cpack = pack<dp::mat::Complex<T>, W>;
        auto one = cpack(pack<T, W>(T{1}), pack<T, W>(T{}));
        auto z_sq = z * z;
        return log(z + sqrt(z_sq - one));
    }

    /**
     * @brief Complex inverse hyperbolic tangent: atanh(z) = 0.5 * log((1+z)/(1-z))
     */
    template <typename T, std::size_t W>
    inline pack<dp::mat::Complex<T>, W> atanh(const pack<dp::mat::Complex<T>, W> &z) noexcept {
        using cpack = pack<dp::mat::Complex<T>, W>;
        auto one = cpack(pack<T, W>(T{1}), pack<T, W>(T{}));
        auto half = pack<T, W>(T{0.5});
        return log((one + z) / (one - z)) * half;
    }

} // namespace optinum::simd
