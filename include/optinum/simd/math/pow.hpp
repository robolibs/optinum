#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <optinum/simd/math/detail/map.hpp>

namespace optinum::simd {

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> pow(const SIMDVec<T, W> &a, const SIMDVec<T, W> &b) {
        return math_detail::map_binary<T, W>(a, b, [](T x, T y) -> T { return std::pow(x, y); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> pow(const SIMDVec<T, W> &a, T b) {
        return pow(a, SIMDVec<T, W>(b));
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> pow(T a, const SIMDVec<T, W> &b) {
        return pow(SIMDVec<T, W>(a), b);
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> cbrt(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::cbrt(v); });
    }

} // namespace optinum::simd
