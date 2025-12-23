#pragma once

// =============================================================================
// optinum/simd/intrinsic/simd_vec.hpp
// SIMD Vector Abstraction - Width-based primary template and scalar fallback
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/arch/macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace optinum::simd {

    // =============================================================================
    // SIMDVec - Primary Template (Scalar Fallback)
    // =============================================================================

    template <typename T, std::size_t Width> struct SIMDVec {
        static_assert(Width > 0, "SIMDVec<T,Width> requires Width > 0");
        static_assert(std::is_arithmetic_v<T>, "SIMDVec<T,Width> requires arithmetic T");

        using value_type = T;
        static constexpr std::size_t width = Width;

        // Storage
        alignas(OPTINUM_SIMD_ALIGNMENT) T data_[width];

        // ==========================================================================
        // Constructors
        // ==========================================================================

        OPTINUM_INLINE SIMDVec() noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] = T{};
        }

        OPTINUM_INLINE explicit SIMDVec(T val) noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] = val;
        }

        OPTINUM_INLINE SIMDVec(const SIMDVec &other) noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] = other.data_[i];
        }

        OPTINUM_INLINE SIMDVec &operator=(const SIMDVec &other) noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] = other.data_[i];
            return *this;
        }

        OPTINUM_INLINE SIMDVec &operator=(T val) noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] = val;
            return *this;
        }

        // ==========================================================================
        // Load / Store
        // ==========================================================================

        OPTINUM_INLINE static SIMDVec load(const T *ptr) noexcept {
            SIMDVec v;
            for (std::size_t i = 0; i < width; ++i)
                v.data_[i] = ptr[i];
            return v;
        }

        OPTINUM_INLINE static SIMDVec loadu(const T *ptr) noexcept {
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

        OPTINUM_INLINE SIMDVec operator+(const SIMDVec &rhs) const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] + rhs.data_[i];
            return result;
        }

        OPTINUM_INLINE SIMDVec operator-(const SIMDVec &rhs) const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] - rhs.data_[i];
            return result;
        }

        OPTINUM_INLINE SIMDVec operator*(const SIMDVec &rhs) const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] * rhs.data_[i];
            return result;
        }

        OPTINUM_INLINE SIMDVec operator/(const SIMDVec &rhs) const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] / rhs.data_[i];
            return result;
        }

        OPTINUM_INLINE SIMDVec operator-() const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = -data_[i];
            return result;
        }

        // Scalar operations
        OPTINUM_INLINE SIMDVec operator+(T rhs) const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] + rhs;
            return result;
        }

        OPTINUM_INLINE SIMDVec operator-(T rhs) const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] - rhs;
            return result;
        }

        OPTINUM_INLINE SIMDVec operator*(T rhs) const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] * rhs;
            return result;
        }

        OPTINUM_INLINE SIMDVec operator/(T rhs) const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = data_[i] / rhs;
            return result;
        }

        // ==========================================================================
        // Compound Assignment
        // ==========================================================================

        OPTINUM_INLINE SIMDVec &operator+=(const SIMDVec &rhs) noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] += rhs.data_[i];
            return *this;
        }

        OPTINUM_INLINE SIMDVec &operator-=(const SIMDVec &rhs) noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] -= rhs.data_[i];
            return *this;
        }

        OPTINUM_INLINE SIMDVec &operator*=(const SIMDVec &rhs) noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] *= rhs.data_[i];
            return *this;
        }

        OPTINUM_INLINE SIMDVec &operator/=(const SIMDVec &rhs) noexcept {
            for (std::size_t i = 0; i < width; ++i)
                data_[i] /= rhs.data_[i];
            return *this;
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

        // ==========================================================================
        // Math Functions
        // ==========================================================================

        OPTINUM_INLINE SIMDVec sqrt() const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = std::sqrt(data_[i]);
            return result;
        }

        OPTINUM_INLINE SIMDVec rsqrt() const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = T{1} / std::sqrt(data_[i]);
            return result;
        }

        OPTINUM_INLINE SIMDVec abs() const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = std::abs(data_[i]);
            return result;
        }

        OPTINUM_INLINE SIMDVec rcp() const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = T{1} / data_[i];
            return result;
        }

        // ==========================================================================
        // Min / Max
        // ==========================================================================

        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = (a.data_[i] < b.data_[i]) ? a.data_[i] : b.data_[i];
            return result;
        }

        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = (a.data_[i] > b.data_[i]) ? a.data_[i] : b.data_[i];
            return result;
        }

        // ==========================================================================
        // FMA (Fused Multiply-Add)
        // ==========================================================================

        OPTINUM_INLINE static SIMDVec fma(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = a.data_[i] * b.data_[i] + c.data_[i];
            return result;
        }

        OPTINUM_INLINE static SIMDVec fms(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < width; ++i)
                result.data_[i] = a.data_[i] * b.data_[i] - c.data_[i];
            return result;
        }

        // ==========================================================================
        // Dot Product
        // ==========================================================================

        OPTINUM_INLINE T dot(const SIMDVec &other) const noexcept {
            T result{};
            for (std::size_t i = 0; i < width; ++i)
                result += data_[i] * other.data_[i];
            return result;
        }
    };

    // =============================================================================
    // Free Function Operators (scalar on left side)
    // =============================================================================

    template <typename T, std::size_t W>
    OPTINUM_INLINE SIMDVec<T, W> operator+(T lhs, const SIMDVec<T, W> &rhs) noexcept {
        return rhs + lhs;
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE SIMDVec<T, W> operator-(T lhs, const SIMDVec<T, W> &rhs) noexcept {
        return SIMDVec<T, W>(lhs) - rhs;
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE SIMDVec<T, W> operator*(T lhs, const SIMDVec<T, W> &rhs) noexcept {
        return rhs * lhs;
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE SIMDVec<T, W> operator/(T lhs, const SIMDVec<T, W> &rhs) noexcept {
        return SIMDVec<T, W>(lhs) / rhs;
    }

    // =============================================================================
    // Free Functions for Math Operations
    // =============================================================================

    template <typename T, std::size_t W> OPTINUM_INLINE SIMDVec<T, W> sqrt(const SIMDVec<T, W> &v) noexcept {
        return v.sqrt();
    }

    template <typename T, std::size_t W> OPTINUM_INLINE SIMDVec<T, W> rsqrt(const SIMDVec<T, W> &v) noexcept {
        return v.rsqrt();
    }

    template <typename T, std::size_t W> OPTINUM_INLINE SIMDVec<T, W> abs(const SIMDVec<T, W> &v) noexcept {
        return v.abs();
    }

    template <typename T, std::size_t W> OPTINUM_INLINE SIMDVec<T, W> rcp(const SIMDVec<T, W> &v) noexcept {
        return v.rcp();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE SIMDVec<T, W> min(const SIMDVec<T, W> &a, const SIMDVec<T, W> &b) noexcept {
        return SIMDVec<T, W>::min(a, b);
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE SIMDVec<T, W> max(const SIMDVec<T, W> &a, const SIMDVec<T, W> &b) noexcept {
        return SIMDVec<T, W>::max(a, b);
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE SIMDVec<T, W> fma(const SIMDVec<T, W> &a, const SIMDVec<T, W> &b, const SIMDVec<T, W> &c) noexcept {
        return SIMDVec<T, W>::fma(a, b, c);
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE SIMDVec<T, W> fms(const SIMDVec<T, W> &a, const SIMDVec<T, W> &b, const SIMDVec<T, W> &c) noexcept {
        return SIMDVec<T, W>::fms(a, b, c);
    }

} // namespace optinum::simd
