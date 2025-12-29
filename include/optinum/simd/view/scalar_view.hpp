#pragma once

// =============================================================================
// optinum/simd/view/scalar_view.hpp
// scalar_view<T,W> - Non-owning view over a single scalar value
// =============================================================================

#include <optinum/simd/kernel.hpp>

namespace optinum::simd {

    // =============================================================================
    // scalar_view<T, W> - Rank-0 view (single element)
    // =============================================================================

    template <typename T, std::size_t W> struct scalar_view {
        using value_type = std::remove_const_t<T>;
        using kernel_type = Kernel<T, W, 0>;
        static constexpr std::size_t width = W;
        static constexpr std::size_t rank = 0;

        kernel_type kernel_;

        // ==========================================================================
        // Constructors
        // ==========================================================================

        OPTINUM_INLINE constexpr scalar_view() noexcept = default;

        OPTINUM_INLINE constexpr explicit scalar_view(T *ptr) noexcept : kernel_(ptr) {}

        OPTINUM_INLINE constexpr explicit scalar_view(const kernel_type &k) noexcept : kernel_(k) {}

        // ==========================================================================
        // Access
        // ==========================================================================

        OPTINUM_INLINE value_type &get() const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot get non-const reference from const view");
            return kernel_.get();
        }

        OPTINUM_INLINE const value_type &get_const() const noexcept { return kernel_.get_const(); }

        OPTINUM_INLINE operator value_type &() const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot convert const view to non-const reference");
            return get();
        }

        OPTINUM_INLINE operator const value_type &() const noexcept { return get_const(); }

        // ==========================================================================
        // Assignment
        // ==========================================================================

        OPTINUM_INLINE scalar_view &operator=(value_type val) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot assign to const view");
            get() = val;
            return const_cast<scalar_view &>(*this);
        }

        // ==========================================================================
        // Compound assignment operators
        // ==========================================================================

        OPTINUM_INLINE scalar_view &operator+=(value_type val) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            get() += val;
            return const_cast<scalar_view &>(*this);
        }

        OPTINUM_INLINE scalar_view &operator-=(value_type val) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            get() -= val;
            return const_cast<scalar_view &>(*this);
        }

        OPTINUM_INLINE scalar_view &operator*=(value_type val) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            get() *= val;
            return const_cast<scalar_view &>(*this);
        }

        OPTINUM_INLINE scalar_view &operator/=(value_type val) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            get() /= val;
            return const_cast<scalar_view &>(*this);
        }

        // Compound assignment with another scalar_view
        OPTINUM_INLINE scalar_view &operator+=(const scalar_view &rhs) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            get() += rhs.get_const();
            return const_cast<scalar_view &>(*this);
        }

        OPTINUM_INLINE scalar_view &operator-=(const scalar_view &rhs) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            get() -= rhs.get_const();
            return const_cast<scalar_view &>(*this);
        }

        OPTINUM_INLINE scalar_view &operator*=(const scalar_view &rhs) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            get() *= rhs.get_const();
            return const_cast<scalar_view &>(*this);
        }

        OPTINUM_INLINE scalar_view &operator/=(const scalar_view &rhs) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            get() /= rhs.get_const();
            return const_cast<scalar_view &>(*this);
        }

        // ==========================================================================
        // Data access
        // ==========================================================================

        OPTINUM_INLINE T *data() const noexcept { return kernel_.data(); }

        OPTINUM_INLINE const value_type *data_const() const noexcept { return kernel_.data_const(); }
    };

    // =============================================================================
    // Free function operators for scalar_view
    // =============================================================================

    // Binary arithmetic operators (return value_type, not view)
    template <typename T, std::size_t W>
    OPTINUM_INLINE typename scalar_view<T, W>::value_type operator+(const scalar_view<T, W> &lhs,
                                                                    const scalar_view<T, W> &rhs) noexcept {
        return lhs.get_const() + rhs.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE typename scalar_view<T, W>::value_type operator-(const scalar_view<T, W> &lhs,
                                                                    const scalar_view<T, W> &rhs) noexcept {
        return lhs.get_const() - rhs.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE typename scalar_view<T, W>::value_type operator*(const scalar_view<T, W> &lhs,
                                                                    const scalar_view<T, W> &rhs) noexcept {
        return lhs.get_const() * rhs.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE typename scalar_view<T, W>::value_type operator/(const scalar_view<T, W> &lhs,
                                                                    const scalar_view<T, W> &rhs) noexcept {
        return lhs.get_const() / rhs.get_const();
    }

    // Binary operators with raw value_type on right
    template <typename T, std::size_t W>
    OPTINUM_INLINE typename scalar_view<T, W>::value_type
    operator+(const scalar_view<T, W> &lhs, typename scalar_view<T, W>::value_type rhs) noexcept {
        return lhs.get_const() + rhs;
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE typename scalar_view<T, W>::value_type
    operator-(const scalar_view<T, W> &lhs, typename scalar_view<T, W>::value_type rhs) noexcept {
        return lhs.get_const() - rhs;
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE typename scalar_view<T, W>::value_type
    operator*(const scalar_view<T, W> &lhs, typename scalar_view<T, W>::value_type rhs) noexcept {
        return lhs.get_const() * rhs;
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE typename scalar_view<T, W>::value_type
    operator/(const scalar_view<T, W> &lhs, typename scalar_view<T, W>::value_type rhs) noexcept {
        return lhs.get_const() / rhs;
    }

    // Binary operators with raw value_type on left
    template <typename T, std::size_t W>
    OPTINUM_INLINE typename scalar_view<T, W>::value_type operator+(typename scalar_view<T, W>::value_type lhs,
                                                                    const scalar_view<T, W> &rhs) noexcept {
        return lhs + rhs.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE typename scalar_view<T, W>::value_type operator-(typename scalar_view<T, W>::value_type lhs,
                                                                    const scalar_view<T, W> &rhs) noexcept {
        return lhs - rhs.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE typename scalar_view<T, W>::value_type operator*(typename scalar_view<T, W>::value_type lhs,
                                                                    const scalar_view<T, W> &rhs) noexcept {
        return lhs * rhs.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE typename scalar_view<T, W>::value_type operator/(typename scalar_view<T, W>::value_type lhs,
                                                                    const scalar_view<T, W> &rhs) noexcept {
        return lhs / rhs.get_const();
    }

    // Unary operators
    template <typename T, std::size_t W>
    OPTINUM_INLINE typename scalar_view<T, W>::value_type operator-(const scalar_view<T, W> &v) noexcept {
        return -v.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE typename scalar_view<T, W>::value_type operator+(const scalar_view<T, W> &v) noexcept {
        return v.get_const();
    }

    // Comparison operators
    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator==(const scalar_view<T, W> &lhs, const scalar_view<T, W> &rhs) noexcept {
        return lhs.get_const() == rhs.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator!=(const scalar_view<T, W> &lhs, const scalar_view<T, W> &rhs) noexcept {
        return lhs.get_const() != rhs.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator<(const scalar_view<T, W> &lhs, const scalar_view<T, W> &rhs) noexcept {
        return lhs.get_const() < rhs.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator<=(const scalar_view<T, W> &lhs, const scalar_view<T, W> &rhs) noexcept {
        return lhs.get_const() <= rhs.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator>(const scalar_view<T, W> &lhs, const scalar_view<T, W> &rhs) noexcept {
        return lhs.get_const() > rhs.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator>=(const scalar_view<T, W> &lhs, const scalar_view<T, W> &rhs) noexcept {
        return lhs.get_const() >= rhs.get_const();
    }

    // Comparison with raw value_type
    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator==(const scalar_view<T, W> &lhs, typename scalar_view<T, W>::value_type rhs) noexcept {
        return lhs.get_const() == rhs;
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator!=(const scalar_view<T, W> &lhs, typename scalar_view<T, W>::value_type rhs) noexcept {
        return lhs.get_const() != rhs;
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator<(const scalar_view<T, W> &lhs, typename scalar_view<T, W>::value_type rhs) noexcept {
        return lhs.get_const() < rhs;
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator<=(const scalar_view<T, W> &lhs, typename scalar_view<T, W>::value_type rhs) noexcept {
        return lhs.get_const() <= rhs;
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator>(const scalar_view<T, W> &lhs, typename scalar_view<T, W>::value_type rhs) noexcept {
        return lhs.get_const() > rhs;
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator>=(const scalar_view<T, W> &lhs, typename scalar_view<T, W>::value_type rhs) noexcept {
        return lhs.get_const() >= rhs;
    }

    // Comparison with raw value_type on left
    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator==(typename scalar_view<T, W>::value_type lhs, const scalar_view<T, W> &rhs) noexcept {
        return lhs == rhs.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator!=(typename scalar_view<T, W>::value_type lhs, const scalar_view<T, W> &rhs) noexcept {
        return lhs != rhs.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator<(typename scalar_view<T, W>::value_type lhs, const scalar_view<T, W> &rhs) noexcept {
        return lhs < rhs.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator<=(typename scalar_view<T, W>::value_type lhs, const scalar_view<T, W> &rhs) noexcept {
        return lhs <= rhs.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator>(typename scalar_view<T, W>::value_type lhs, const scalar_view<T, W> &rhs) noexcept {
        return lhs > rhs.get_const();
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE bool operator>=(typename scalar_view<T, W>::value_type lhs, const scalar_view<T, W> &rhs) noexcept {
        return lhs >= rhs.get_const();
    }

} // namespace optinum::simd
