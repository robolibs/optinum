#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <optinum/simd/math/detail/map.hpp>

namespace optinum::simd {

    template <typename T, std::size_t W>
    OPTINUM_INLINE SIMDVec<T, W> clamp(const SIMDVec<T, W> &x, const SIMDVec<T, W> &lo,
                                       const SIMDVec<T, W> &hi) noexcept {
        return min(max(x, lo), hi);
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE SIMDVec<T, W> clamp(const SIMDVec<T, W> &x, T lo, T hi) noexcept {
        return clamp(x, SIMDVec<T, W>(lo), SIMDVec<T, W>(hi));
    }

    template <typename T, std::size_t W> OPTINUM_INLINE SIMDVec<T, W> sign(const SIMDVec<T, W> &x) {
        if constexpr (!std::is_floating_point_v<T>) {
            return math_detail::map_unary<T, W>(x, [](T v) -> T { return (v > 0) ? T{1} : (v < 0) ? T{-1} : T{0}; });
        } else {
            return math_detail::map_unary<T, W>(x, [](T v) -> T {
                if (std::isnan(v))
                    return std::numeric_limits<T>::quiet_NaN();
                if (v == T{0})
                    return T{0};
                return std::signbit(v) ? T{-1} : T{1};
            });
        }
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> floor(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::floor(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> ceil(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::ceil(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> round(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::round(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> trunc(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::trunc(v); });
    }

} // namespace optinum::simd
