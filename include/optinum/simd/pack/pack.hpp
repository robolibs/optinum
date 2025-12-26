#pragma once

// =============================================================================
// optinum/simd/pack/pack.hpp
// SIMD pack Abstraction - Primary template and scalar fallback
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/arch/macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace optinum::simd {

    // =============================================================================
    // pack - Primary Template (Scalar Fallback)
    // =============================================================================

    template <typename T, std::size_t W> struct pack {
        static_assert(W > 0, "pack<T,W> requires W > 0");
        static_assert(std::is_arithmetic_v<T>, "pack<T,W> requires arithmetic T");

        using value_type = T;
        static constexpr std::size_t width = W;

        // Storage
        alignas(W * sizeof(T)) T data_[width];

        // ==========================================================================
        // Constructors
        // ==========================================================================

        OPTINUM_INLINE pack() noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] = T{};
        }

        OPTINUM_INLINE explicit pack(T val) noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] = val;
        }

        OPTINUM_INLINE pack(const pack &other) noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] = other.data_[i];
        }

        OPTINUM_INLINE pack &operator=(const pack &other) noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] = other.data_[i];
            return *this;
        }

        OPTINUM_INLINE pack &operator=(T val) noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] = val;
            return *this;
        }

        // ==========================================================================
        // Load / Store
        // ==========================================================================

        OPTINUM_INLINE static pack load(const T *ptr) noexcept {
            pack v;
            for (std::size_t i = 0; i < width; ++i)
                v.data_[i] = ptr[i];
            return v;
        }

        OPTINUM_INLINE static pack loadu(const T *ptr) noexcept {
            return load(ptr); // Same for scalar fallback
        }

        OPTINUM_INLINE void store(T *ptr) const noexcept {
            for (std::size_t i = 0; i < width; ++i)
                ptr[i] = data_[i];
        }

        OPTINUM_INLINE void storeu(T *ptr) const noexcept {
            store(ptr); // Same for scalar fallback
        }

        // ==========================================================================
        // Element Access
        // ==========================================================================

        OPTINUM_INLINE T operator[](std::size_t i) const noexcept { return data_[i]; }

        // ==========================================================================
        // Arithmetic Operators
        // ==========================================================================

        OPTINUM_INLINE pack operator+(const pack &rhs) const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] + rhs.data_[i];
            return result;
        }

        OPTINUM_INLINE pack operator-(const pack &rhs) const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] - rhs.data_[i];
            return result;
        }

        OPTINUM_INLINE pack operator*(const pack &rhs) const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] * rhs.data_[i];
            return result;
        }

        OPTINUM_INLINE pack operator/(const pack &rhs) const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] / rhs.data_[i];
            return result;
        }

        OPTINUM_INLINE pack operator-() const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = -data_[i];
            return result;
        }

        // Scalar operations
        OPTINUM_INLINE pack operator+(T rhs) const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] + rhs;
            return result;
        }

        OPTINUM_INLINE pack operator-(T rhs) const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] - rhs;
            return result;
        }

        OPTINUM_INLINE pack operator*(T rhs) const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] * rhs;
            return result;
        }

        OPTINUM_INLINE pack operator/(T rhs) const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] / rhs;
            return result;
        }

        // ==========================================================================
        // Compound Assignment
        // ==========================================================================

        OPTINUM_INLINE pack &operator+=(const pack &rhs) noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] += rhs.data_[i];
            return *this;
        }

        OPTINUM_INLINE pack &operator-=(const pack &rhs) noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] -= rhs.data_[i];
            return *this;
        }

        OPTINUM_INLINE pack &operator*=(const pack &rhs) noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] *= rhs.data_[i];
            return *this;
        }

        OPTINUM_INLINE pack &operator/=(const pack &rhs) noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] /= rhs.data_[i];
            return *this;
        }

        // ==========================================================================
        // Bitwise (for integer types)
        // ==========================================================================

        template <typename U = T, typename = std::enable_if_t<std::is_integral_v<U>>>
        OPTINUM_INLINE pack operator&(const pack &rhs) const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] & rhs.data_[i];
            return result;
        }

        template <typename U = T, typename = std::enable_if_t<std::is_integral_v<U>>>
        OPTINUM_INLINE pack operator|(const pack &rhs) const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] | rhs.data_[i];
            return result;
        }

        template <typename U = T, typename = std::enable_if_t<std::is_integral_v<U>>>
        OPTINUM_INLINE pack operator^(const pack &rhs) const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] ^ rhs.data_[i];
            return result;
        }

        template <typename U = T, typename = std::enable_if_t<std::is_integral_v<U>>>
        OPTINUM_INLINE pack operator~() const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = ~data_[i];
            return result;
        }

        template <typename U = T, typename = std::enable_if_t<std::is_integral_v<U>>>
        OPTINUM_INLINE pack operator<<(int count) const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] << count;
            return result;
        }

        template <typename U = T, typename = std::enable_if_t<std::is_integral_v<U>>>
        OPTINUM_INLINE pack operator>>(int count) const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] >> count;
            return result;
        }

        template <typename U = T, typename = std::enable_if_t<std::is_integral_v<U>>>
        OPTINUM_INLINE pack shr_logical(int count) const noexcept {
            pack result;
            using unsigned_type = std::make_unsigned_t<T>;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = static_cast<T>(static_cast<unsigned_type>(data_[i]) >> count);
            return result;
        }

        // ==========================================================================
        // Reductions
        // ==========================================================================

        OPTINUM_INLINE T hsum() const noexcept {
            T result{};
            for (std::size_t i = 0; i < width; ++i)
                result += data_[i];
            return result;
        }

        OPTINUM_INLINE T hmin() const noexcept {
            T result = data_[0];
            for (std::size_t i = 1; i < width; ++i)
                result = (data_[i] < result) ? data_[i] : result;
            return result;
        }

        OPTINUM_INLINE T hmax() const noexcept {
            T result = data_[0];
            for (std::size_t i = 1; i < width; ++i)
                result = (data_[i] > result) ? data_[i] : result;
            return result;
        }

        OPTINUM_INLINE T hprod() const noexcept {
            T result = data_[0];
            for (std::size_t i = 1; i < width; ++i)
                result *= data_[i];
            return result;
        }

        // ==========================================================================
        // Math Functions
        // ==========================================================================

        OPTINUM_INLINE pack sqrt() const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = std::sqrt(data_[i]);
            return result;
        }

        OPTINUM_INLINE pack rsqrt() const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = T{1} / std::sqrt(data_[i]);
            return result;
        }

        OPTINUM_INLINE pack abs() const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = std::abs(data_[i]);
            return result;
        }

        OPTINUM_INLINE pack rcp() const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = T{1} / data_[i];
            return result;
        }

        // ==========================================================================
        // Min / Max
        // ==========================================================================

        OPTINUM_INLINE static pack min(const pack &a, const pack &b) noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = (a.data_[i] < b.data_[i]) ? a.data_[i] : b.data_[i];
            return result;
        }

        OPTINUM_INLINE static pack max(const pack &a, const pack &b) noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = (a.data_[i] > b.data_[i]) ? a.data_[i] : b.data_[i];
            return result;
        }

        // ==========================================================================
        // FMA (Fused Multiply-Add)
        // ==========================================================================

        OPTINUM_INLINE static pack fma(const pack &a, const pack &b, const pack &c) noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = a.data_[i] * b.data_[i] + c.data_[i];
            return result;
        }

        OPTINUM_INLINE static pack fms(const pack &a, const pack &b, const pack &c) noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = a.data_[i] * b.data_[i] - c.data_[i];
            return result;
        }

        // ==========================================================================
        // Dot Product
        // ==========================================================================

        OPTINUM_INLINE T dot(const pack &other) const noexcept {
            T result{};
            for (std::size_t i = 0; i < width; ++i)
                result += data_[i] * other.data_[i];
            return result;
        }

        // ==========================================================================
        // Utility - Reverse lane order
        // ==========================================================================

        OPTINUM_INLINE pack reverse() const noexcept {
            pack result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[width - 1 - i];
            return result;
        }
    };

    // =============================================================================
    // Free Function Operators (scalar on left side)
    // =============================================================================

    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> operator+(T lhs, const pack<T, W> &rhs) noexcept {
        return rhs + lhs;
    }

    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> operator-(T lhs, const pack<T, W> &rhs) noexcept {
        return pack<T, W>(lhs) - rhs;
    }

    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> operator*(T lhs, const pack<T, W> &rhs) noexcept {
        return rhs * lhs;
    }

    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> operator/(T lhs, const pack<T, W> &rhs) noexcept {
        return pack<T, W>(lhs) / rhs;
    }

    // =============================================================================
    // Free Functions for Math Operations
    // =============================================================================

    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> sqrt(const pack<T, W> &v) noexcept {
        return v.sqrt();
    }

    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> rsqrt(const pack<T, W> &v) noexcept {
        return v.rsqrt();
    }

    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> abs(const pack<T, W> &v) noexcept { return v.abs(); }

    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> rcp(const pack<T, W> &v) noexcept { return v.rcp(); }

    template <typename T, std::size_t W>
    OPTINUM_INLINE pack<T, W> min(const pack<T, W> &a, const pack<T, W> &b) noexcept {
        return pack<T, W>::min(a, b);
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE pack<T, W> max(const pack<T, W> &a, const pack<T, W> &b) noexcept {
        return pack<T, W>::max(a, b);
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE pack<T, W> fma(const pack<T, W> &a, const pack<T, W> &b, const pack<T, W> &c) noexcept {
        return pack<T, W>::fma(a, b, c);
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE pack<T, W> fms(const pack<T, W> &a, const pack<T, W> &b, const pack<T, W> &c) noexcept {
        return pack<T, W>::fms(a, b, c);
    }

} // namespace optinum::simd
