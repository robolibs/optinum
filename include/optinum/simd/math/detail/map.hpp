#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include <optinum/simd/intrinsic/simd_vec.hpp>

namespace optinum::simd::math_detail {

    template <typename T, std::size_t W, typename F>
    OPTINUM_INLINE SIMDVec<T, W> map_unary(const SIMDVec<T, W> &v, F &&f) {
        static_assert(W > 0);
        static_assert(std::is_arithmetic_v<T>);

        std::array<T, W> tmp{};
        v.storeu(tmp.data());
        for (std::size_t i = 0; i < W; ++i) {
            tmp[i] = static_cast<T>(std::invoke(std::forward<F>(f), tmp[i]));
        }
        return SIMDVec<T, W>::loadu(tmp.data());
    }

    template <typename T, std::size_t W, typename F>
    OPTINUM_INLINE SIMDVec<T, W> map_binary(const SIMDVec<T, W> &a, const SIMDVec<T, W> &b, F &&f) {
        static_assert(W > 0);
        static_assert(std::is_arithmetic_v<T>);

        std::array<T, W> ta{};
        std::array<T, W> tb{};
        a.storeu(ta.data());
        b.storeu(tb.data());
        for (std::size_t i = 0; i < W; ++i) {
            ta[i] = static_cast<T>(std::invoke(std::forward<F>(f), ta[i], tb[i]));
        }
        return SIMDVec<T, W>::loadu(ta.data());
    }

} // namespace optinum::simd::math_detail
