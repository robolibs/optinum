#pragma once

// =============================================================================
// optinum/simd/intrinsic/simd_vec.hpp
// SIMD Vector Abstraction - Base Template and Scalar Fallback
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/arch/macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace optinum::simd {

    // =============================================================================
    // SIMD ABI Tags
    // =============================================================================

    namespace simd_abi {

        struct scalar {};
        struct sse {};
        struct avx {};
        struct avx512 {};
        struct neon {};

// Native ABI - best available for this platform
#if defined(OPTINUM_HAS_AVX512F)
        using native = avx512;
#elif defined(OPTINUM_HAS_AVX)
        using native = avx;
#elif defined(OPTINUM_HAS_SSE2)
        using native = sse;
#elif defined(OPTINUM_HAS_NEON)
        using native = neon;
#else
        using native = scalar;
#endif

    } // namespace simd_abi

    // =============================================================================
    // Helper: Get SIMD vector size for a given type and ABI
    // =============================================================================

    namespace detail {

        template <typename T, typename ABI> struct simd_size {
            static constexpr std::size_t value = 1; // Default scalar
        };

        template <typename T> struct simd_size<T, simd_abi::sse> {
            static constexpr std::size_t value = 16 / sizeof(T);
        };

        template <typename T> struct simd_size<T, simd_abi::avx> {
            static constexpr std::size_t value = 32 / sizeof(T);
        };

        template <typename T> struct simd_size<T, simd_abi::avx512> {
            static constexpr std::size_t value = 64 / sizeof(T);
        };

        template <typename T> struct simd_size<T, simd_abi::neon> {
            static constexpr std::size_t value = 16 / sizeof(T);
        };

        template <typename T, typename ABI> inline constexpr std::size_t simd_size_v = simd_size<T, ABI>::value;

    } // namespace detail

    // =============================================================================
    // SIMDVec - Primary Template (Scalar Fallback)
    // =============================================================================

    template <typename T, typename ABI = simd_abi::native> struct SIMDVec {
        using value_type = T;
        using abi_type = ABI;
        static constexpr std::size_t size = detail::simd_size_v<T, ABI>;

        // Storage
        alignas(sizeof(T) * size) T data_[size];

        // ==========================================================================
        // Constructors
        // ==========================================================================

        OPTINUM_INLINE SIMDVec() noexcept {
            for (std::size_t i = 0; i < size; ++i)
                data_[i] = T{};
        }

        OPTINUM_INLINE explicit SIMDVec(T val) noexcept {
            for (std::size_t i = 0; i < size; ++i)
                data_[i] = val;
        }

        OPTINUM_INLINE SIMDVec(const SIMDVec &other) noexcept {
            for (std::size_t i = 0; i < size; ++i)
                data_[i] = other.data_[i];
        }

        OPTINUM_INLINE SIMDVec &operator=(const SIMDVec &other) noexcept {
            for (std::size_t i = 0; i < size; ++i)
                data_[i] = other.data_[i];
            return *this;
        }

        OPTINUM_INLINE SIMDVec &operator=(T val) noexcept {
            for (std::size_t i = 0; i < size; ++i)
                data_[i] = val;
            return *this;
        }

        // ==========================================================================
        // Load / Store
        // ==========================================================================

        OPTINUM_INLINE static SIMDVec load(const T *ptr) noexcept {
            SIMDVec v;
            for (std::size_t i = 0; i < size; ++i)
                v.data_[i] = ptr[i];
            return v;
        }

        OPTINUM_INLINE static SIMDVec loadu(const T *ptr) noexcept {
            return load(ptr); // Same for scalar fallback
        }

        OPTINUM_INLINE void store(T *ptr) const noexcept {
            for (std::size_t i = 0; i < size; ++i)
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
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = data_[i] + rhs.data_[i];
            return result;
        }

        OPTINUM_INLINE SIMDVec operator-(const SIMDVec &rhs) const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = data_[i] - rhs.data_[i];
            return result;
        }

        OPTINUM_INLINE SIMDVec operator*(const SIMDVec &rhs) const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = data_[i] * rhs.data_[i];
            return result;
        }

        OPTINUM_INLINE SIMDVec operator/(const SIMDVec &rhs) const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = data_[i] / rhs.data_[i];
            return result;
        }

        OPTINUM_INLINE SIMDVec operator-() const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = -data_[i];
            return result;
        }

        // Scalar operations
        OPTINUM_INLINE SIMDVec operator+(T rhs) const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = data_[i] + rhs;
            return result;
        }

        OPTINUM_INLINE SIMDVec operator-(T rhs) const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = data_[i] - rhs;
            return result;
        }

        OPTINUM_INLINE SIMDVec operator*(T rhs) const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = data_[i] * rhs;
            return result;
        }

        OPTINUM_INLINE SIMDVec operator/(T rhs) const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = data_[i] / rhs;
            return result;
        }

        // ==========================================================================
        // Compound Assignment
        // ==========================================================================

        OPTINUM_INLINE SIMDVec &operator+=(const SIMDVec &rhs) noexcept {
            for (std::size_t i = 0; i < size; ++i)
                data_[i] += rhs.data_[i];
            return *this;
        }

        OPTINUM_INLINE SIMDVec &operator-=(const SIMDVec &rhs) noexcept {
            for (std::size_t i = 0; i < size; ++i)
                data_[i] -= rhs.data_[i];
            return *this;
        }

        OPTINUM_INLINE SIMDVec &operator*=(const SIMDVec &rhs) noexcept {
            for (std::size_t i = 0; i < size; ++i)
                data_[i] *= rhs.data_[i];
            return *this;
        }

        OPTINUM_INLINE SIMDVec &operator/=(const SIMDVec &rhs) noexcept {
            for (std::size_t i = 0; i < size; ++i)
                data_[i] /= rhs.data_[i];
            return *this;
        }

        // ==========================================================================
        // Reductions
        // ==========================================================================

        OPTINUM_INLINE T hsum() const noexcept {
            T result{};
            for (std::size_t i = 0; i < size; ++i)
                result += data_[i];
            return result;
        }

        OPTINUM_INLINE T hprod() const noexcept {
            T result{1};
            for (std::size_t i = 0; i < size; ++i)
                result *= data_[i];
            return result;
        }

        OPTINUM_INLINE T hmin() const noexcept {
            T result = data_[0];
            for (std::size_t i = 1; i < size; ++i)
                result = (data_[i] < result) ? data_[i] : result;
            return result;
        }

        OPTINUM_INLINE T hmax() const noexcept {
            T result = data_[0];
            for (std::size_t i = 1; i < size; ++i)
                result = (data_[i] > result) ? data_[i] : result;
            return result;
        }

        // ==========================================================================
        // Math Functions
        // ==========================================================================

        OPTINUM_INLINE SIMDVec sqrt() const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = std::sqrt(data_[i]);
            return result;
        }

        OPTINUM_INLINE SIMDVec rsqrt() const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = T{1} / std::sqrt(data_[i]);
            return result;
        }

        OPTINUM_INLINE SIMDVec abs() const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = std::abs(data_[i]);
            return result;
        }

        OPTINUM_INLINE SIMDVec rcp() const noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = T{1} / data_[i];
            return result;
        }

        // ==========================================================================
        // Min / Max
        // ==========================================================================

        OPTINUM_INLINE static SIMDVec min(const SIMDVec &a, const SIMDVec &b) noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = (a.data_[i] < b.data_[i]) ? a.data_[i] : b.data_[i];
            return result;
        }

        OPTINUM_INLINE static SIMDVec max(const SIMDVec &a, const SIMDVec &b) noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = (a.data_[i] > b.data_[i]) ? a.data_[i] : b.data_[i];
            return result;
        }

        // ==========================================================================
        // FMA (Fused Multiply-Add)
        // ==========================================================================

        OPTINUM_INLINE static SIMDVec fma(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = a.data_[i] * b.data_[i] + c.data_[i];
            return result;
        }

        OPTINUM_INLINE static SIMDVec fms(const SIMDVec &a, const SIMDVec &b, const SIMDVec &c) noexcept {
            SIMDVec result;
            for (std::size_t i = 0; i < size; ++i)
                result.data_[i] = a.data_[i] * b.data_[i] - c.data_[i];
            return result;
        }

        // ==========================================================================
        // Dot Product
        // ==========================================================================

        OPTINUM_INLINE T dot(const SIMDVec &other) const noexcept {
            T result{};
            for (std::size_t i = 0; i < size; ++i)
                result += data_[i] * other.data_[i];
            return result;
        }
    };

    // =============================================================================
    // Free Function Operators (scalar on left side)
    // =============================================================================

    template <typename T, typename ABI>
    OPTINUM_INLINE SIMDVec<T, ABI> operator+(T lhs, const SIMDVec<T, ABI> &rhs) noexcept {
        return rhs + lhs;
    }

    template <typename T, typename ABI>
    OPTINUM_INLINE SIMDVec<T, ABI> operator-(T lhs, const SIMDVec<T, ABI> &rhs) noexcept {
        SIMDVec<T, ABI> result;
        for (std::size_t i = 0; i < SIMDVec<T, ABI>::size; ++i)
            result.data_[i] = lhs - rhs.data_[i];
        return result;
    }

    template <typename T, typename ABI>
    OPTINUM_INLINE SIMDVec<T, ABI> operator*(T lhs, const SIMDVec<T, ABI> &rhs) noexcept {
        return rhs * lhs;
    }

    template <typename T, typename ABI>
    OPTINUM_INLINE SIMDVec<T, ABI> operator/(T lhs, const SIMDVec<T, ABI> &rhs) noexcept {
        SIMDVec<T, ABI> result;
        for (std::size_t i = 0; i < SIMDVec<T, ABI>::size; ++i)
            result.data_[i] = lhs / rhs.data_[i];
        return result;
    }

    // =============================================================================
    // Free Functions for Math Operations
    // =============================================================================

    template <typename T, typename ABI> OPTINUM_INLINE SIMDVec<T, ABI> sqrt(const SIMDVec<T, ABI> &v) noexcept {
        return v.sqrt();
    }

    template <typename T, typename ABI> OPTINUM_INLINE SIMDVec<T, ABI> rsqrt(const SIMDVec<T, ABI> &v) noexcept {
        return v.rsqrt();
    }

    template <typename T, typename ABI> OPTINUM_INLINE SIMDVec<T, ABI> abs(const SIMDVec<T, ABI> &v) noexcept {
        return v.abs();
    }

    template <typename T, typename ABI> OPTINUM_INLINE SIMDVec<T, ABI> rcp(const SIMDVec<T, ABI> &v) noexcept {
        return v.rcp();
    }

    template <typename T, typename ABI>
    OPTINUM_INLINE SIMDVec<T, ABI> min(const SIMDVec<T, ABI> &a, const SIMDVec<T, ABI> &b) noexcept {
        return SIMDVec<T, ABI>::min(a, b);
    }

    template <typename T, typename ABI>
    OPTINUM_INLINE SIMDVec<T, ABI> max(const SIMDVec<T, ABI> &a, const SIMDVec<T, ABI> &b) noexcept {
        return SIMDVec<T, ABI>::max(a, b);
    }

    template <typename T, typename ABI>
    OPTINUM_INLINE SIMDVec<T, ABI> fma(const SIMDVec<T, ABI> &a, const SIMDVec<T, ABI> &b,
                                       const SIMDVec<T, ABI> &c) noexcept {
        return SIMDVec<T, ABI>::fma(a, b, c);
    }

    template <typename T, typename ABI>
    OPTINUM_INLINE SIMDVec<T, ABI> fms(const SIMDVec<T, ABI> &a, const SIMDVec<T, ABI> &b,
                                       const SIMDVec<T, ABI> &c) noexcept {
        return SIMDVec<T, ABI>::fms(a, b, c);
    }

    // =============================================================================
    // Type Aliases for Native SIMD
    // =============================================================================

    template <typename T> using NativeSIMD = SIMDVec<T, simd_abi::native>;

    using float4 = SIMDVec<float, simd_abi::sse>;
    using float8 = SIMDVec<float, simd_abi::avx>;
    using float16 = SIMDVec<float, simd_abi::avx512>;

    using double2 = SIMDVec<double, simd_abi::sse>;
    using double4 = SIMDVec<double, simd_abi::avx>;
    using double8 = SIMDVec<double, simd_abi::avx512>;

} // namespace optinum::simd
