#pragma once

// =============================================================================
// optinum/simd/algo/transform.hpp
// Elementwise mathematical transformations on views
// =============================================================================

#include <cstddef>
#include <optinum/simd/math/cos.hpp>
#include <optinum/simd/math/exp.hpp>
#include <optinum/simd/math/log.hpp>
#include <optinum/simd/math/sin.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/math/tanh.hpp>
#include <optinum/simd/view/vector_view.hpp>

namespace optinum::simd {

    // =============================================================================
    // exp: y = exp(x)
    // Elementwise exponential
    // =============================================================================

    template <typename T, std::size_t W>
    OPTINUM_INLINE void exp(const vector_view<const T, W> &x, const vector_view<T, W> &y) noexcept {
        const std::size_t n = x.size();
        if (n != y.size())
            return;

        const std::size_t num_packs = x.num_packs();

        // Process full packs
        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            y.store_pack(i, exp(x.load_pack(i)));
        }

        // Handle tail
        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            y.store_pack_tail(last_idx, exp(x.load_pack_tail(last_idx)));
        }
    }

    // Overload accepting non-const views
    template <typename T, std::size_t W>
    OPTINUM_INLINE void exp(const vector_view<T, W> &x, const vector_view<T, W> &y) noexcept {
        exp(vector_view<const T, W>(x.data(), x.size()), y);
    }

    // In-place variant: x = exp(x)
    template <typename T, std::size_t W> OPTINUM_INLINE void exp(const vector_view<T, W> &x) noexcept {
        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            x.store_pack(i, exp(x.load_pack(i)));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            x.store_pack_tail(last_idx, exp(x.load_pack_tail(last_idx)));
        }
    }

    // =============================================================================
    // log: y = log(x)
    // Elementwise natural logarithm
    // =============================================================================

    template <typename T, std::size_t W>
    OPTINUM_INLINE void log(const vector_view<const T, W> &x, const vector_view<T, W> &y) noexcept {
        const std::size_t n = x.size();
        if (n != y.size())
            return;

        const std::size_t num_packs = x.num_packs();

        // Process full packs
        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            y.store_pack(i, log(x.load_pack(i)));
        }

        // Handle tail
        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            y.store_pack_tail(last_idx, log(x.load_pack_tail(last_idx)));
        }
    }

    // Overload accepting non-const views
    template <typename T, std::size_t W>
    OPTINUM_INLINE void log(const vector_view<T, W> &x, const vector_view<T, W> &y) noexcept {
        log(vector_view<const T, W>(x.data(), x.size()), y);
    }

    // In-place variant: x = log(x)
    template <typename T, std::size_t W> OPTINUM_INLINE void log(const vector_view<T, W> &x) noexcept {
        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            x.store_pack(i, log(x.load_pack(i)));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            x.store_pack_tail(last_idx, log(x.load_pack_tail(last_idx)));
        }
    }

    // =============================================================================
    // sin: y = sin(x)
    // Elementwise sine
    // =============================================================================

    template <typename T, std::size_t W>
    OPTINUM_INLINE void sin(const vector_view<const T, W> &x, const vector_view<T, W> &y) noexcept {
        const std::size_t n = x.size();
        if (n != y.size())
            return;

        const std::size_t num_packs = x.num_packs();

        // Process full packs
        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            y.store_pack(i, sin(x.load_pack(i)));
        }

        // Handle tail
        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            y.store_pack_tail(last_idx, sin(x.load_pack_tail(last_idx)));
        }
    }

    // Overload accepting non-const views
    template <typename T, std::size_t W>
    OPTINUM_INLINE void sin(const vector_view<T, W> &x, const vector_view<T, W> &y) noexcept {
        sin(vector_view<const T, W>(x.data(), x.size()), y);
    }

    // In-place variant: x = sin(x)
    template <typename T, std::size_t W> OPTINUM_INLINE void sin(const vector_view<T, W> &x) noexcept {
        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            x.store_pack(i, sin(x.load_pack(i)));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            x.store_pack_tail(last_idx, sin(x.load_pack_tail(last_idx)));
        }
    }

    // =============================================================================
    // cos: y = cos(x)
    // Elementwise cosine
    // =============================================================================

    template <typename T, std::size_t W>
    OPTINUM_INLINE void cos(const vector_view<const T, W> &x, const vector_view<T, W> &y) noexcept {
        const std::size_t n = x.size();
        if (n != y.size())
            return;

        const std::size_t num_packs = x.num_packs();

        // Process full packs
        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            y.store_pack(i, cos(x.load_pack(i)));
        }

        // Handle tail
        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            y.store_pack_tail(last_idx, cos(x.load_pack_tail(last_idx)));
        }
    }

    // Overload accepting non-const views
    template <typename T, std::size_t W>
    OPTINUM_INLINE void cos(const vector_view<T, W> &x, const vector_view<T, W> &y) noexcept {
        cos(vector_view<const T, W>(x.data(), x.size()), y);
    }

    // In-place variant: x = cos(x)
    template <typename T, std::size_t W> OPTINUM_INLINE void cos(const vector_view<T, W> &x) noexcept {
        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            x.store_pack(i, cos(x.load_pack(i)));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            x.store_pack_tail(last_idx, cos(x.load_pack_tail(last_idx)));
        }
    }

    // =============================================================================
    // tanh: y = tanh(x)
    // Elementwise hyperbolic tangent
    // =============================================================================

    template <typename T, std::size_t W>
    OPTINUM_INLINE void tanh(const vector_view<const T, W> &x, const vector_view<T, W> &y) noexcept {
        const std::size_t n = x.size();
        if (n != y.size())
            return;

        const std::size_t num_packs = x.num_packs();

        // Process full packs
        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            y.store_pack(i, tanh(x.load_pack(i)));
        }

        // Handle tail
        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            y.store_pack_tail(last_idx, tanh(x.load_pack_tail(last_idx)));
        }
    }

    // Overload accepting non-const views
    template <typename T, std::size_t W>
    OPTINUM_INLINE void tanh(const vector_view<T, W> &x, const vector_view<T, W> &y) noexcept {
        tanh(vector_view<const T, W>(x.data(), x.size()), y);
    }

    // In-place variant: x = tanh(x)
    template <typename T, std::size_t W> OPTINUM_INLINE void tanh(const vector_view<T, W> &x) noexcept {
        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            x.store_pack(i, tanh(x.load_pack(i)));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            x.store_pack_tail(last_idx, tanh(x.load_pack_tail(last_idx)));
        }
    }

    // =============================================================================
    // sqrt: y = sqrt(x)
    // Elementwise square root
    // =============================================================================

    template <typename T, std::size_t W>
    OPTINUM_INLINE void sqrt(const vector_view<const T, W> &x, const vector_view<T, W> &y) noexcept {
        const std::size_t n = x.size();
        if (n != y.size())
            return;

        const std::size_t num_packs = x.num_packs();

        // Process full packs
        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            y.store_pack(i, sqrt(x.load_pack(i)));
        }

        // Handle tail
        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            y.store_pack_tail(last_idx, sqrt(x.load_pack_tail(last_idx)));
        }
    }

    // Overload accepting non-const views
    template <typename T, std::size_t W>
    OPTINUM_INLINE void sqrt(const vector_view<T, W> &x, const vector_view<T, W> &y) noexcept {
        sqrt(vector_view<const T, W>(x.data(), x.size()), y);
    }

    // In-place variant: x = sqrt(x)
    template <typename T, std::size_t W> OPTINUM_INLINE void sqrt(const vector_view<T, W> &x) noexcept {
        const std::size_t num_packs = x.num_packs();

        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            x.store_pack(i, sqrt(x.load_pack(i)));
        }

        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            x.store_pack_tail(last_idx, sqrt(x.load_pack_tail(last_idx)));
        }
    }

} // namespace optinum::simd
