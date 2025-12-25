#pragma once

// =============================================================================
// optinum/simd/algo/transform.hpp
// Elementwise mathematical transformations on views (vector, matrix, tensor)
// =============================================================================

#include <cstddef>
#include <optinum/simd/algo/traits.hpp>
#include <optinum/simd/math/ceil.hpp>
#include <optinum/simd/math/cos.hpp>
#include <optinum/simd/math/exp.hpp>
#include <optinum/simd/math/floor.hpp>
#include <optinum/simd/math/log.hpp>
#include <optinum/simd/math/pow.hpp>
#include <optinum/simd/math/round.hpp>
#include <optinum/simd/math/sin.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/math/tan.hpp>
#include <optinum/simd/math/tanh.hpp>
#include <optinum/simd/math/trunc.hpp>

namespace optinum::simd {

    // =============================================================================
    // Generic transform helpers
    // =============================================================================

    namespace detail {

        // Two-argument transform: y = f(x)
        template <typename SrcView, typename DstView, typename Func>
        OPTINUM_INLINE void transform_impl(const SrcView &x, const DstView &y, Func f) noexcept {
            const std::size_t num_packs = x.num_packs();

            // Process full packs
            for (std::size_t i = 0; i < num_packs - 1; ++i) {
                y.store_pack(i, f(x.load_pack(i)));
            }

            // Handle tail
            if (num_packs > 0) {
                const std::size_t last_idx = num_packs - 1;
                y.store_pack_tail(last_idx, f(x.load_pack_tail(last_idx)));
            }
        }

        // In-place transform: x = f(x)
        template <typename View, typename Func>
        OPTINUM_INLINE void transform_inplace_impl(const View &x, Func f) noexcept {
            const std::size_t num_packs = x.num_packs();

            for (std::size_t i = 0; i < num_packs - 1; ++i) {
                x.store_pack(i, f(x.load_pack(i)));
            }

            if (num_packs > 0) {
                const std::size_t last_idx = num_packs - 1;
                x.store_pack_tail(last_idx, f(x.load_pack_tail(last_idx)));
            }
        }

    } // namespace detail

    // =============================================================================
    // exp: y = exp(x)
    // Elementwise exponential - works with vector_view, matrix_view, tensor_view
    // =============================================================================

    // y = exp(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void exp(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return exp(p); });
    }

    // In-place: x = exp(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void exp(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return exp(p); });
    }

    // =============================================================================
    // log: y = log(x)
    // Elementwise natural logarithm
    // =============================================================================

    // y = log(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void log(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return log(p); });
    }

    // In-place: x = log(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void log(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return log(p); });
    }

    // =============================================================================
    // sin: y = sin(x)
    // Elementwise sine
    // =============================================================================

    // y = sin(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void sin(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return sin(p); });
    }

    // In-place: x = sin(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void sin(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return sin(p); });
    }

    // =============================================================================
    // cos: y = cos(x)
    // Elementwise cosine
    // =============================================================================

    // y = cos(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void cos(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return cos(p); });
    }

    // In-place: x = cos(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void cos(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return cos(p); });
    }

    // =============================================================================
    // tan: y = tan(x)
    // Elementwise tangent
    // =============================================================================

    // y = tan(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void tan(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return tan(p); });
    }

    // In-place: x = tan(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void tan(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return tan(p); });
    }

    // =============================================================================
    // tanh: y = tanh(x)
    // Elementwise hyperbolic tangent
    // =============================================================================

    // y = tanh(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void tanh(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return tanh(p); });
    }

    // In-place: x = tanh(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void tanh(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return tanh(p); });
    }

    // =============================================================================
    // sqrt: y = sqrt(x)
    // Elementwise square root
    // =============================================================================

    // y = sqrt(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void sqrt(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return sqrt(p); });
    }

    // In-place: x = sqrt(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void sqrt(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return sqrt(p); });
    }

    // =============================================================================
    // ceil: y = ceil(x)
    // Elementwise ceiling (round toward +infinity)
    // =============================================================================

    // y = ceil(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void ceil(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return ceil(p); });
    }

    // In-place: x = ceil(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void ceil(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return ceil(p); });
    }

    // =============================================================================
    // floor: y = floor(x)
    // Elementwise floor (round toward -infinity)
    // =============================================================================

    // y = floor(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void floor(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return floor(p); });
    }

    // In-place: x = floor(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void floor(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return floor(p); });
    }

    // =============================================================================
    // round: y = round(x)
    // Elementwise round to nearest integer
    // =============================================================================

    // y = round(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void round(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return round(p); });
    }

    // In-place: x = round(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void round(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return round(p); });
    }

    // =============================================================================
    // trunc: y = trunc(x)
    // Elementwise truncate (round toward zero)
    // =============================================================================

    // y = trunc(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void trunc(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return trunc(p); });
    }

    // In-place: x = trunc(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void trunc(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return trunc(p); });
    }

    // =============================================================================
    // pow: z = pow(x, y)
    // Elementwise power function
    // =============================================================================

    // z = pow(x, y)
    template <typename SrcView1, typename SrcView2, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView1> && detail::is_packable_view_v<SrcView2> &&
                                   detail::is_packable_view_v<DstView>,
                               int> = 0>
    OPTINUM_INLINE void pow(const SrcView1 &x, const SrcView2 &y, const DstView &z) noexcept {
        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            z.store_pack(i, pow(x.load_pack(i), y.load_pack(i)));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            z.store_pack_tail(last_idx, pow(x.load_pack_tail(last_idx), y.load_pack_tail(last_idx)));
        }
    }

} // namespace optinum::simd
