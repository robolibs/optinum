#pragma once

// =============================================================================
// optinum/simd/algo/transform.hpp
// Elementwise mathematical transformations on views (vector, matrix, tensor)
// =============================================================================

#include <cstddef>
#include <optinum/simd/algo/traits.hpp>
#include <optinum/simd/math/acos.hpp>
#include <optinum/simd/math/acosh.hpp>
#include <optinum/simd/math/asin.hpp>
#include <optinum/simd/math/asinh.hpp>
#include <optinum/simd/math/atan.hpp>
#include <optinum/simd/math/atan2.hpp>
#include <optinum/simd/math/atanh.hpp>
#include <optinum/simd/math/ceil.hpp>
#include <optinum/simd/math/cos.hpp>
#include <optinum/simd/math/cosh.hpp>
#include <optinum/simd/math/exp.hpp>
#include <optinum/simd/math/exp2.hpp>
#include <optinum/simd/math/expm1.hpp>
#include <optinum/simd/math/floor.hpp>
#include <optinum/simd/math/log.hpp>
#include <optinum/simd/math/log10.hpp>
#include <optinum/simd/math/log1p.hpp>
#include <optinum/simd/math/log2.hpp>
#include <optinum/simd/math/pow.hpp>
#include <optinum/simd/math/round.hpp>
#include <optinum/simd/math/sin.hpp>
#include <optinum/simd/math/sinh.hpp>
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

    // =============================================================================
    // sinh: y = sinh(x)
    // Elementwise hyperbolic sine
    // =============================================================================

    // y = sinh(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void sinh(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return sinh(p); });
    }

    // In-place: x = sinh(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void sinh(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return sinh(p); });
    }

    // =============================================================================
    // cosh: y = cosh(x)
    // Elementwise hyperbolic cosine
    // =============================================================================

    // y = cosh(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void cosh(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return cosh(p); });
    }

    // In-place: x = cosh(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void cosh(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return cosh(p); });
    }

    // =============================================================================
    // exp2: y = exp2(x)
    // Elementwise base-2 exponential (2^x)
    // =============================================================================

    // y = exp2(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void exp2(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return exp2(p); });
    }

    // In-place: x = exp2(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void exp2(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return exp2(p); });
    }

    // =============================================================================
    // log2: y = log2(x)
    // Elementwise base-2 logarithm
    // =============================================================================

    // y = log2(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void log2(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return log2(p); });
    }

    // In-place: x = log2(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void log2(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return log2(p); });
    }

    // =============================================================================
    // log10: y = log10(x)
    // Elementwise base-10 logarithm
    // =============================================================================

    // y = log10(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void log10(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return log10(p); });
    }

    // In-place: x = log10(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void log10(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return log10(p); });
    }

    // =============================================================================
    // atan: y = atan(x)
    // Elementwise arc tangent
    // =============================================================================

    // y = atan(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void atan(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return atan(p); });
    }

    // In-place: x = atan(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void atan(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return atan(p); });
    }

    // =============================================================================
    // atan2: z = atan2(y, x)
    // Elementwise two-argument arc tangent (y/x with correct quadrant)
    // =============================================================================

    // z = atan2(y, x)
    template <typename SrcView1, typename SrcView2, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView1> && detail::is_packable_view_v<SrcView2> &&
                                   detail::is_packable_view_v<DstView>,
                               int> = 0>
    OPTINUM_INLINE void atan2(const SrcView1 &y, const SrcView2 &x, const DstView &z) noexcept {
        const std::size_t num_packs = y.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            z.store_pack(i, atan2(y.load_pack(i), x.load_pack(i)));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            z.store_pack_tail(last_idx, atan2(y.load_pack_tail(last_idx), x.load_pack_tail(last_idx)));
        }
    }

    // =============================================================================
    // asin: y = asin(x)
    // Elementwise arc sine
    // =============================================================================

    // y = asin(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void asin(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return asin(p); });
    }

    // In-place: x = asin(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void asin(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return asin(p); });
    }

    // =============================================================================
    // acos: y = acos(x)
    // Elementwise arc cosine
    // =============================================================================

    // y = acos(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void acos(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return acos(p); });
    }

    // In-place: x = acos(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void acos(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return acos(p); });
    }

    // =============================================================================
    // asinh: y = asinh(x)
    // Elementwise inverse hyperbolic sine
    // =============================================================================

    // y = asinh(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void asinh(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return asinh(p); });
    }

    // In-place: x = asinh(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void asinh(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return asinh(p); });
    }

    // =============================================================================
    // acosh: y = acosh(x)
    // Elementwise inverse hyperbolic cosine
    // =============================================================================

    // y = acosh(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void acosh(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return acosh(p); });
    }

    // In-place: x = acosh(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void acosh(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return acosh(p); });
    }

    // =============================================================================
    // atanh: y = atanh(x)
    // Elementwise inverse hyperbolic tangent
    // =============================================================================

    // y = atanh(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void atanh(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return atanh(p); });
    }

    // In-place: x = atanh(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void atanh(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return atanh(p); });
    }

    // =============================================================================
    // expm1: y = expm1(x)
    // Elementwise exp(x) - 1 (accurate for small x)
    // =============================================================================

    // y = expm1(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void expm1(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return expm1(p); });
    }

    // In-place: x = expm1(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void expm1(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return expm1(p); });
    }

    // =============================================================================
    // log1p: y = log1p(x)
    // Elementwise log(1 + x) (accurate for small x)
    // =============================================================================

    // y = log1p(x)
    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView>, int> = 0>
    OPTINUM_INLINE void log1p(const SrcView &x, const DstView &y) noexcept {
        detail::transform_impl(x, y, [](auto p) { return log1p(p); });
    }

    // In-place: x = log1p(x)
    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void log1p(const View &x) noexcept {
        detail::transform_inplace_impl(x, [](auto p) { return log1p(p); });
    }

} // namespace optinum::simd
