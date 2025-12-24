#pragma once

// =============================================================================
// optinum/simd/bridge.hpp
// Bridge between datapod types and SIMD views
// =============================================================================

#include <cstdint>
#include <datapod/matrix.hpp>
#include <optinum/simd/view/matrix_view.hpp>
#include <optinum/simd/view/scalar_view.hpp>
#include <optinum/simd/view/tensor_view.hpp>
#include <optinum/simd/view/vector_view.hpp>
#include <type_traits>

namespace optinum::simd {

    // =============================================================================
    // Width detection
    // =============================================================================

    namespace detail {
        template <typename T> constexpr std::size_t default_width() {
            if constexpr (std::is_same_v<T, float>) {
#if defined(__AVX__)
                return 8; // AVX for float
#elif defined(__SSE__)
                return 4; // SSE for float
#else
                return 1; // Scalar fallback
#endif
            } else if constexpr (std::is_same_v<T, double>) {
#if defined(__AVX__)
                return 4; // AVX for double
#elif defined(__SSE2__)
                return 2; // SSE2 for double
#else
                return 1; // Scalar fallback
#endif
            } else if constexpr (std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t>) {
#if defined(__AVX2__)
                return 8; // AVX2 for int32
#elif defined(__SSE2__)
                return 4; // SSE2 for int32
#else
                return 1; // Scalar fallback
#endif
            } else if constexpr (std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t>) {
#if defined(__AVX2__)
                return 4; // AVX2 for int64
#elif defined(__SSE2__)
                return 2; // SSE2 for int64
#else
                return 1; // Scalar fallback
#endif
            } else {
                return 1; // Scalar fallback for unknown types
            }
        }
    } // namespace detail

    // =============================================================================
    // view() - Factory functions for datapod -> SIMD views
    // =============================================================================

    // -------------------------------------------------------------------------
    // Scalar view (from datapod::mat::scalar<T>)
    // -------------------------------------------------------------------------

    template <std::size_t W, typename T> OPTINUM_INLINE scalar_view<T, W> view(datapod::mat::scalar<T> &s) noexcept {
        return scalar_view<T, W>(&s.value);
    }

    template <std::size_t W, typename T>
    OPTINUM_INLINE scalar_view<const T, W> view(const datapod::mat::scalar<T> &s) noexcept {
        return scalar_view<const T, W>(&s.value);
    }

    // Auto-detect width for scalar
    template <typename T> OPTINUM_INLINE auto view(datapod::mat::scalar<T> &s) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return scalar_view<T, W>(&s.value);
    }

    template <typename T> OPTINUM_INLINE auto view(const datapod::mat::scalar<T> &s) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return scalar_view<const T, W>(&s.value);
    }

    // -------------------------------------------------------------------------
    // Vector view (from datapod::mat::vector<T, N>)
    // -------------------------------------------------------------------------

    template <std::size_t W, typename T, std::size_t N>
    OPTINUM_INLINE vector_view<T, W> view(datapod::mat::vector<T, N> &v) noexcept {
        return vector_view<T, W>(v.data(), N);
    }

    template <std::size_t W, typename T, std::size_t N>
    OPTINUM_INLINE vector_view<const T, W> view(const datapod::mat::vector<T, N> &v) noexcept {
        return vector_view<const T, W>(v.data(), N);
    }

    // Auto-detect width for vector
    template <typename T, std::size_t N> OPTINUM_INLINE auto view(datapod::mat::vector<T, N> &v) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return vector_view<T, W>(v.data(), N);
    }

    template <typename T, std::size_t N> OPTINUM_INLINE auto view(const datapod::mat::vector<T, N> &v) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return vector_view<const T, W>(v.data(), N);
    }

    // -------------------------------------------------------------------------
    // Matrix view (from datapod::mat::matrix<T, R, C>)
    // -------------------------------------------------------------------------

    template <std::size_t W, typename T, std::size_t R, std::size_t C>
    OPTINUM_INLINE matrix_view<T, W> view(datapod::mat::matrix<T, R, C> &m) noexcept {
        return matrix_view<T, W>(m.data(), R, C);
    }

    template <std::size_t W, typename T, std::size_t R, std::size_t C>
    OPTINUM_INLINE matrix_view<const T, W> view(const datapod::mat::matrix<T, R, C> &m) noexcept {
        return matrix_view<const T, W>(m.data(), R, C);
    }

    // Auto-detect width for matrix
    template <typename T, std::size_t R, std::size_t C>
    OPTINUM_INLINE auto view(datapod::mat::matrix<T, R, C> &m) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return matrix_view<T, W>(m.data(), R, C);
    }

    template <typename T, std::size_t R, std::size_t C>
    OPTINUM_INLINE auto view(const datapod::mat::matrix<T, R, C> &m) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return matrix_view<const T, W>(m.data(), R, C);
    }

    // -------------------------------------------------------------------------
    // Tensor view (from datapod::mat::tensor<T, Dims...>)
    // -------------------------------------------------------------------------

    namespace detail {
        // Helper to compute column-major strides for tensor
        template <std::size_t Rank>
        constexpr std::array<std::size_t, Rank> compute_strides(const std::array<std::size_t, Rank> &dims) {
            std::array<std::size_t, Rank> strides{};
            strides[0] = 1;
            for (std::size_t i = 1; i < Rank; ++i) {
                strides[i] = strides[i - 1] * dims[i - 1];
            }
            return strides;
        }
    } // namespace detail

    template <std::size_t W, typename T, std::size_t... Dims>
    OPTINUM_INLINE auto view(datapod::mat::tensor<T, Dims...> &t) noexcept {
        constexpr std::size_t Rank = sizeof...(Dims);
        constexpr std::array<std::size_t, Rank> extents = {Dims...};
        constexpr std::array<std::size_t, Rank> strides = detail::compute_strides(extents);
        return tensor_view<T, W, Rank>(t.data(), extents, strides);
    }

    template <std::size_t W, typename T, std::size_t... Dims>
    OPTINUM_INLINE auto view(const datapod::mat::tensor<T, Dims...> &t) noexcept {
        constexpr std::size_t Rank = sizeof...(Dims);
        constexpr std::array<std::size_t, Rank> extents = {Dims...};
        constexpr std::array<std::size_t, Rank> strides = detail::compute_strides(extents);
        return tensor_view<const T, W, Rank>(t.data(), extents, strides);
    }

    // Auto-detect width for tensor
    template <typename T, std::size_t... Dims> OPTINUM_INLINE auto view(datapod::mat::tensor<T, Dims...> &t) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        constexpr std::size_t Rank = sizeof...(Dims);
        constexpr std::array<std::size_t, Rank> extents = {Dims...};
        constexpr std::array<std::size_t, Rank> strides = detail::compute_strides(extents);
        return tensor_view<T, W, Rank>(t.data(), extents, strides);
    }

    template <typename T, std::size_t... Dims>
    OPTINUM_INLINE auto view(const datapod::mat::tensor<T, Dims...> &t) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        constexpr std::size_t Rank = sizeof...(Dims);
        constexpr std::array<std::size_t, Rank> extents = {Dims...};
        constexpr std::array<std::size_t, Rank> strides = detail::compute_strides(extents);
        return tensor_view<const T, W, Rank>(t.data(), extents, strides);
    }

} // namespace optinum::simd
