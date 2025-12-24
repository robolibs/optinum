#pragma once

// =============================================================================
// optinum/simd/algo/elementwise.hpp
// Elementwise SIMD algorithms operating on views (vector, matrix, tensor)
// =============================================================================

#include <cstddef>
#include <optinum/simd/algo/traits.hpp>
#include <optinum/simd/pack/pack.hpp>

namespace optinum::simd {

    // =============================================================================
    // axpy: y = alpha * x + y
    // Classic BLAS operation with fused multiply-add
    // Works with any packable view (vector, matrix, tensor)
    // =============================================================================

    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView> &&
                                   !detail::is_const_view_v<DstView>,
                               int> = 0>
    OPTINUM_INLINE void axpy(detail::view_value_t<SrcView> alpha, const SrcView &x, const DstView &y) noexcept {
        using T = detail::view_value_t<SrcView>;
        constexpr std::size_t W = detail::view_width_v<SrcView>;

        const std::size_t num_packs = x.num_packs();
        const pack<T, W> alpha_vec(alpha);

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            auto x_pack = x.load_pack(i);
            auto y_pack = y.load_pack(i);
            y.store_pack(i, fma(alpha_vec, x_pack, y_pack));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            auto x_pack = x.load_pack_tail(last_idx);
            auto y_pack = y.load_pack_tail(last_idx);
            y.store_pack_tail(last_idx, fma(alpha_vec, x_pack, y_pack));
        }
    }

    // =============================================================================
    // scale: x = alpha * x
    // =============================================================================

    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void scale(detail::view_value_t<View> alpha, const View &x) noexcept {
        using T = detail::view_value_t<View>;
        constexpr std::size_t W = detail::view_width_v<View>;

        const std::size_t num_packs = x.num_packs();
        const pack<T, W> alpha_vec(alpha);

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            x.store_pack(i, alpha_vec * x.load_pack(i));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            x.store_pack_tail(last_idx, alpha_vec * x.load_pack_tail(last_idx));
        }
    }

    // =============================================================================
    // add: z = x + y OR x = x + y
    // =============================================================================

    // z = x + y
    template <typename SrcView1, typename SrcView2, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView1> && detail::is_packable_view_v<SrcView2> &&
                                   detail::is_packable_view_v<DstView> && !detail::is_const_view_v<DstView>,
                               int> = 0>
    OPTINUM_INLINE void add(const SrcView1 &x, const SrcView2 &y, const DstView &z) noexcept {
        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            z.store_pack(i, x.load_pack(i) + y.load_pack(i));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            z.store_pack_tail(last_idx, x.load_pack_tail(last_idx) + y.load_pack_tail(last_idx));
        }
    }

    // x = x + y (in-place)
    template <typename View1, typename View2,
              std::enable_if_t<detail::is_packable_view_v<View1> && detail::is_packable_view_v<View2> &&
                                   !detail::is_const_view_v<View1>,
                               int> = 0>
    OPTINUM_INLINE void add(const View1 &x, const View2 &y) noexcept {
        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            x.store_pack(i, x.load_pack(i) + y.load_pack(i));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            x.store_pack_tail(last_idx, x.load_pack_tail(last_idx) + y.load_pack_tail(last_idx));
        }
    }

    // =============================================================================
    // sub: z = x - y OR x = x - y
    // =============================================================================

    // z = x - y
    template <typename SrcView1, typename SrcView2, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView1> && detail::is_packable_view_v<SrcView2> &&
                                   detail::is_packable_view_v<DstView> && !detail::is_const_view_v<DstView>,
                               int> = 0>
    OPTINUM_INLINE void sub(const SrcView1 &x, const SrcView2 &y, const DstView &z) noexcept {
        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            z.store_pack(i, x.load_pack(i) - y.load_pack(i));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            z.store_pack_tail(last_idx, x.load_pack_tail(last_idx) - y.load_pack_tail(last_idx));
        }
    }

    // x = x - y (in-place)
    template <typename View1, typename View2,
              std::enable_if_t<detail::is_packable_view_v<View1> && detail::is_packable_view_v<View2> &&
                                   !detail::is_const_view_v<View1>,
                               int> = 0>
    OPTINUM_INLINE void sub(const View1 &x, const View2 &y) noexcept {
        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            x.store_pack(i, x.load_pack(i) - y.load_pack(i));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            x.store_pack_tail(last_idx, x.load_pack_tail(last_idx) - y.load_pack_tail(last_idx));
        }
    }

    // =============================================================================
    // mul: z = x * y OR x = x * y (Hadamard product)
    // =============================================================================

    // z = x * y
    template <typename SrcView1, typename SrcView2, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView1> && detail::is_packable_view_v<SrcView2> &&
                                   detail::is_packable_view_v<DstView> && !detail::is_const_view_v<DstView>,
                               int> = 0>
    OPTINUM_INLINE void mul(const SrcView1 &x, const SrcView2 &y, const DstView &z) noexcept {
        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            z.store_pack(i, x.load_pack(i) * y.load_pack(i));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            z.store_pack_tail(last_idx, x.load_pack_tail(last_idx) * y.load_pack_tail(last_idx));
        }
    }

    // x = x * y (in-place)
    template <typename View1, typename View2,
              std::enable_if_t<detail::is_packable_view_v<View1> && detail::is_packable_view_v<View2> &&
                                   !detail::is_const_view_v<View1>,
                               int> = 0>
    OPTINUM_INLINE void mul(const View1 &x, const View2 &y) noexcept {
        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            x.store_pack(i, x.load_pack(i) * y.load_pack(i));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            x.store_pack_tail(last_idx, x.load_pack_tail(last_idx) * y.load_pack_tail(last_idx));
        }
    }

    // =============================================================================
    // div: z = x / y OR x = x / y
    // =============================================================================

    // z = x / y
    template <typename SrcView1, typename SrcView2, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView1> && detail::is_packable_view_v<SrcView2> &&
                                   detail::is_packable_view_v<DstView> && !detail::is_const_view_v<DstView>,
                               int> = 0>
    OPTINUM_INLINE void div(const SrcView1 &x, const SrcView2 &y, const DstView &z) noexcept {
        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            z.store_pack(i, x.load_pack(i) / y.load_pack(i));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            z.store_pack_tail(last_idx, x.load_pack_tail(last_idx) / y.load_pack_tail(last_idx));
        }
    }

    // x = x / y (in-place)
    template <typename View1, typename View2,
              std::enable_if_t<detail::is_packable_view_v<View1> && detail::is_packable_view_v<View2> &&
                                   !detail::is_const_view_v<View1>,
                               int> = 0>
    OPTINUM_INLINE void div(const View1 &x, const View2 &y) noexcept {
        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            x.store_pack(i, x.load_pack(i) / y.load_pack(i));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            x.store_pack_tail(last_idx, x.load_pack_tail(last_idx) / y.load_pack_tail(last_idx));
        }
    }

    // =============================================================================
    // fill: x = alpha
    // =============================================================================

    template <typename View,
              std::enable_if_t<detail::is_packable_view_v<View> && !detail::is_const_view_v<View>, int> = 0>
    OPTINUM_INLINE void fill(const View &x, detail::view_value_t<View> alpha) noexcept {
        using T = detail::view_value_t<View>;
        constexpr std::size_t W = detail::view_width_v<View>;

        const std::size_t num_packs = x.num_packs();
        const pack<T, W> alpha_vec(alpha);

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            x.store_pack(i, alpha_vec);
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            x.store_pack_tail(last_idx, alpha_vec);
        }
    }

    // =============================================================================
    // copy: y = x
    // =============================================================================

    template <typename SrcView, typename DstView,
              std::enable_if_t<detail::is_packable_view_v<SrcView> && detail::is_packable_view_v<DstView> &&
                                   !detail::is_const_view_v<DstView>,
                               int> = 0>
    OPTINUM_INLINE void copy(const SrcView &x, const DstView &y) noexcept {
        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            y.store_pack(i, x.load_pack(i));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            y.store_pack_tail(last_idx, x.load_pack_tail(last_idx));
        }
    }

} // namespace optinum::simd
