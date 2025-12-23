#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <optinum/simd/math/detail/map.hpp>

namespace optinum::simd {

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> sinh(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::sinh(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> cosh(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::cosh(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> tanh(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::tanh(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> asinh(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::asinh(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> acosh(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::acosh(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> atanh(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::atanh(v); });
    }

} // namespace optinum::simd
