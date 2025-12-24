#pragma once

// =============================================================================
// optinum/simd/mask.hpp
// SIMD mask type for comparisons and masked operations
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/arch/macros.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace optinum::simd {

    // Forward declaration
    template <typename T, std::size_t W> struct pack;

    // =============================================================================
    // mask - Primary Template (Scalar Fallback using std::array<bool>)
    // =============================================================================

    template <typename T, std::size_t W> struct mask {
        static_assert(W > 0, "mask<T,W> requires W > 0");
        static_assert(std::is_arithmetic_v<T>, "mask<T,W> requires arithmetic T");

        using value_type = T;
        static constexpr std::size_t width = W;

        // Storage: array of bools (scalar fallback)
        std::array<bool, W> data_;

        // ==========================================================================
        // Constructors
        // ==========================================================================

        OPTINUM_INLINE mask() noexcept : data_{} {}

        OPTINUM_INLINE explicit mask(bool val) noexcept {
            for (std::size_t i = 0; i < W; ++i)
                data_[i] = val;
        }

        OPTINUM_INLINE mask(const std::array<bool, W> &arr) noexcept : data_(arr) {}

        // ==========================================================================
        // Factory Functions
        // ==========================================================================

        OPTINUM_INLINE static mask all_true() noexcept {
            mask m;
            for (std::size_t i = 0; i < W; ++i)
                m.data_[i] = true;
            return m;
        }

        OPTINUM_INLINE static mask all_false() noexcept {
            mask m;
            for (std::size_t i = 0; i < W; ++i)
                m.data_[i] = false;
            return m;
        }

        OPTINUM_INLINE static mask first_n(std::size_t n) noexcept {
            mask m;
            for (std::size_t i = 0; i < W; ++i)
                m.data_[i] = (i < n);
            return m;
        }

        // ==========================================================================
        // Boolean Operators
        // ==========================================================================

        OPTINUM_INLINE mask operator&(const mask &rhs) const noexcept {
            mask result;
            for (std::size_t i = 0; i < W; ++i)
                result.data_[i] = data_[i] && rhs.data_[i];
            return result;
        }

        OPTINUM_INLINE mask operator|(const mask &rhs) const noexcept {
            mask result;
            for (std::size_t i = 0; i < W; ++i)
                result.data_[i] = data_[i] || rhs.data_[i];
            return result;
        }

        OPTINUM_INLINE mask operator^(const mask &rhs) const noexcept {
            mask result;
            for (std::size_t i = 0; i < W; ++i)
                result.data_[i] = data_[i] != rhs.data_[i];
            return result;
        }

        OPTINUM_INLINE mask operator!() const noexcept {
            mask result;
            for (std::size_t i = 0; i < W; ++i)
                result.data_[i] = !data_[i];
            return result;
        }

        // ==========================================================================
        // Query Operations
        // ==========================================================================

        OPTINUM_INLINE bool all() const noexcept {
            for (std::size_t i = 0; i < W; ++i)
                if (!data_[i])
                    return false;
            return true;
        }

        OPTINUM_INLINE bool any() const noexcept {
            for (std::size_t i = 0; i < W; ++i)
                if (data_[i])
                    return true;
            return false;
        }

        OPTINUM_INLINE bool none() const noexcept { return !any(); }

        OPTINUM_INLINE int popcount() const noexcept {
            int count = 0;
            for (std::size_t i = 0; i < W; ++i)
                if (data_[i])
                    ++count;
            return count;
        }

        // ==========================================================================
        // Element Access
        // ==========================================================================

        OPTINUM_INLINE bool operator[](std::size_t i) const noexcept { return data_[i]; }
    };

    // =============================================================================
    // Comparison Functions (return mask)
    // =============================================================================

    template <typename T, std::size_t W>
    OPTINUM_INLINE mask<T, W> cmp_eq(const pack<T, W> &a, const pack<T, W> &b) noexcept {
        mask<T, W> result;
        for (std::size_t i = 0; i < W; ++i)
            result.data_[i] = (a[i] == b[i]);
        return result;
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE mask<T, W> cmp_ne(const pack<T, W> &a, const pack<T, W> &b) noexcept {
        mask<T, W> result;
        for (std::size_t i = 0; i < W; ++i)
            result.data_[i] = (a[i] != b[i]);
        return result;
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE mask<T, W> cmp_lt(const pack<T, W> &a, const pack<T, W> &b) noexcept {
        mask<T, W> result;
        for (std::size_t i = 0; i < W; ++i)
            result.data_[i] = (a[i] < b[i]);
        return result;
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE mask<T, W> cmp_le(const pack<T, W> &a, const pack<T, W> &b) noexcept {
        mask<T, W> result;
        for (std::size_t i = 0; i < W; ++i)
            result.data_[i] = (a[i] <= b[i]);
        return result;
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE mask<T, W> cmp_gt(const pack<T, W> &a, const pack<T, W> &b) noexcept {
        mask<T, W> result;
        for (std::size_t i = 0; i < W; ++i)
            result.data_[i] = (a[i] > b[i]);
        return result;
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE mask<T, W> cmp_ge(const pack<T, W> &a, const pack<T, W> &b) noexcept {
        mask<T, W> result;
        for (std::size_t i = 0; i < W; ++i)
            result.data_[i] = (a[i] >= b[i]);
        return result;
    }

    // =============================================================================
    // Masked Operations
    // =============================================================================

    // blend: m ? b : a  (select b where mask is true, else a)
    template <typename T, std::size_t W>
    OPTINUM_INLINE pack<T, W> blend(const pack<T, W> &a, const pack<T, W> &b, const mask<T, W> &m) noexcept {
        pack<T, W> result;
        for (std::size_t i = 0; i < W; ++i)
            result.data_[i] = m[i] ? b[i] : a[i];
        return result;
    }

    // maskload: load from ptr where mask is true, else zero
    template <typename T, std::size_t W>
    OPTINUM_INLINE pack<T, W> maskload(const T *ptr, const mask<T, W> &m) noexcept {
        pack<T, W> result;
        for (std::size_t i = 0; i < W; ++i)
            result.data_[i] = m[i] ? ptr[i] : T{0};
        return result;
    }

    // maskstore: store v to ptr where mask is true
    template <typename T, std::size_t W>
    OPTINUM_INLINE void maskstore(T *ptr, const pack<T, W> &v, const mask<T, W> &m) noexcept {
        for (std::size_t i = 0; i < W; ++i)
            if (m[i])
                ptr[i] = v[i];
    }

} // namespace optinum::simd
