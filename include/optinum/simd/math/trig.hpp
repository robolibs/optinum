#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <optinum/simd/math/detail/map.hpp>

namespace optinum::simd {

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> sin(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::sin(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> cos(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::cos(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> tan(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::tan(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> asin(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::asin(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> acos(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::acos(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> atan(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::atan(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> atan2(const SIMDVec<T, W> &y, const SIMDVec<T, W> &x) {
        return math_detail::map_binary<T, W>(y, x, [](T a, T b) -> T { return std::atan2(a, b); });
    }

} // namespace optinum::simd
