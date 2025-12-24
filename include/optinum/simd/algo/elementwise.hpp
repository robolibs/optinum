#pragma once

// =============================================================================
// optinum/simd/algo/elementwise.hpp
// Elementwise SIMD algorithms operating on views
// =============================================================================

#include <cstddef>
#include <optinum/simd/pack/pack.hpp>
#include <optinum/simd/view/vector_view.hpp>

namespace optinum::simd {

    // =============================================================================
    // axpy: y = alpha * x + y
    // Classic BLAS operation with fused multiply-add
    // =============================================================================

    template <typename T, std::size_t W>
    OPTINUM_INLINE void axpy(T alpha, const vector_view<const T, W> &x, const vector_view<T, W> &y) noexcept {
        const std::size_t n = x.size();
        if (n != y.size())
            return;

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

    template <typename T, std::size_t W>
    OPTINUM_INLINE void axpy(T alpha, const vector_view<T, W> &x, const vector_view<T, W> &y) noexcept {
        axpy(alpha, vector_view<const T, W>(x.data(), x.size()), y);
    }

    // =============================================================================
    // scale: x = alpha * x
    // =============================================================================

    template <typename T, std::size_t W> OPTINUM_INLINE void scale(T alpha, const vector_view<T, W> &x) noexcept {
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

    template <typename T, std::size_t W>
    OPTINUM_INLINE void add(const vector_view<const T, W> &x, const vector_view<const T, W> &y,
                            const vector_view<T, W> &z) noexcept {
        const std::size_t n = x.size();
        if (n != y.size() || n != z.size())
            return;

        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            z.store_pack(i, x.load_pack(i) + y.load_pack(i));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            z.store_pack_tail(last_idx, x.load_pack_tail(last_idx) + y.load_pack_tail(last_idx));
        }
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void add(const vector_view<T, W> &x, const vector_view<T, W> &y,
                            const vector_view<T, W> &z) noexcept {
        add(vector_view<const T, W>(x.data(), x.size()), vector_view<const T, W>(y.data(), y.size()), z);
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void add(const vector_view<T, W> &x, const vector_view<T, W> &y) noexcept {
        add(vector_view<const T, W>(x.data(), x.size()), vector_view<const T, W>(y.data(), y.size()), x);
    }

    // =============================================================================
    // sub: z = x - y OR x = x - y
    // =============================================================================

    template <typename T, std::size_t W>
    OPTINUM_INLINE void sub(const vector_view<const T, W> &x, const vector_view<const T, W> &y,
                            const vector_view<T, W> &z) noexcept {
        const std::size_t n = x.size();
        if (n != y.size() || n != z.size())
            return;

        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            z.store_pack(i, x.load_pack(i) - y.load_pack(i));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            z.store_pack_tail(last_idx, x.load_pack_tail(last_idx) - y.load_pack_tail(last_idx));
        }
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void sub(const vector_view<T, W> &x, const vector_view<T, W> &y,
                            const vector_view<T, W> &z) noexcept {
        sub(vector_view<const T, W>(x.data(), x.size()), vector_view<const T, W>(y.data(), y.size()), z);
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void sub(const vector_view<T, W> &x, const vector_view<T, W> &y) noexcept {
        sub(vector_view<const T, W>(x.data(), x.size()), vector_view<const T, W>(y.data(), y.size()), x);
    }

    // =============================================================================
    // mul: z = x * y OR x = x * y (Hadamard product)
    // =============================================================================

    template <typename T, std::size_t W>
    OPTINUM_INLINE void mul(const vector_view<const T, W> &x, const vector_view<const T, W> &y,
                            const vector_view<T, W> &z) noexcept {
        const std::size_t n = x.size();
        if (n != y.size() || n != z.size())
            return;

        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            z.store_pack(i, x.load_pack(i) * y.load_pack(i));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            z.store_pack_tail(last_idx, x.load_pack_tail(last_idx) * y.load_pack_tail(last_idx));
        }
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void mul(const vector_view<T, W> &x, const vector_view<T, W> &y,
                            const vector_view<T, W> &z) noexcept {
        mul(vector_view<const T, W>(x.data(), x.size()), vector_view<const T, W>(y.data(), y.size()), z);
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void mul(const vector_view<T, W> &x, const vector_view<T, W> &y) noexcept {
        mul(vector_view<const T, W>(x.data(), x.size()), vector_view<const T, W>(y.data(), y.size()), x);
    }

    // =============================================================================
    // div: z = x / y OR x = x / y
    // =============================================================================

    template <typename T, std::size_t W>
    OPTINUM_INLINE void div(const vector_view<const T, W> &x, const vector_view<const T, W> &y,
                            const vector_view<T, W> &z) noexcept {
        const std::size_t n = x.size();
        if (n != y.size() || n != z.size())
            return;

        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            z.store_pack(i, x.load_pack(i) / y.load_pack(i));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            z.store_pack_tail(last_idx, x.load_pack_tail(last_idx) / y.load_pack_tail(last_idx));
        }
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void div(const vector_view<T, W> &x, const vector_view<T, W> &y,
                            const vector_view<T, W> &z) noexcept {
        div(vector_view<const T, W>(x.data(), x.size()), vector_view<const T, W>(y.data(), y.size()), z);
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void div(const vector_view<T, W> &x, const vector_view<T, W> &y) noexcept {
        div(vector_view<const T, W>(x.data(), x.size()), vector_view<const T, W>(y.data(), y.size()), x);
    }

    // =============================================================================
    // fill: x = alpha
    // =============================================================================

    template <typename T, std::size_t W> OPTINUM_INLINE void fill(const vector_view<T, W> &x, T alpha) noexcept {
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

    template <typename T, std::size_t W>
    OPTINUM_INLINE void copy(const vector_view<const T, W> &x, const vector_view<T, W> &y) noexcept {
        const std::size_t n = x.size();
        if (n != y.size())
            return;

        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            y.store_pack(i, x.load_pack(i));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            y.store_pack_tail(last_idx, x.load_pack_tail(last_idx));
        }
    }

    template <typename T, std::size_t W>
    OPTINUM_INLINE void copy(const vector_view<T, W> &x, const vector_view<T, W> &y) noexcept {
        copy(vector_view<const T, W>(x.data(), x.size()), y);
    }

} // namespace optinum::simd
