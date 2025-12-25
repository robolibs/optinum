#pragma once

// =============================================================================
// optinum/simd/algo/transform.hpp
// Elementwise mathematical transformations on views (vector, matrix, tensor)
// =============================================================================

#include <cstddef>
#include <optinum/simd/algo/traits.hpp>
#include <optinum/simd/math/cos.hpp>
#include <optinum/simd/math/exp.hpp>
#include <optinum/simd/math/log.hpp>
#include <optinum/simd/math/sin.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/math/tan.hpp>
#include <optinum/simd/math/tanh.hpp>

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

} // namespace optinum::simd
