#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <optinum/simd/math/detail/map.hpp>

namespace optinum::simd {

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> exp(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::exp(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> exp2(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::exp2(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> expm1(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::expm1(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> log(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::log(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> log2(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::log2(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> log10(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::log10(v); });
    }

    template <typename T, std::size_t W>
    requires std::is_floating_point_v<T>
    OPTINUM_INLINE SIMDVec<T, W> log1p(const SIMDVec<T, W> &x) {
        return math_detail::map_unary<T, W>(x, [](T v) -> T { return std::log1p(v); });
    }

} // namespace optinum::simd
