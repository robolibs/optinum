#pragma once

// =============================================================================
// optinum/simd/bridge.hpp
// Bridge between datapod types and SIMD views
// =============================================================================

#include <cstdint>
#include <datapod/datapod.hpp>
#include <optinum/simd/view/complex_view.hpp>
#include <optinum/simd/view/matrix_view.hpp>
#include <optinum/simd/view/quaternion_view.hpp>
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
    // Scalar view (from datapod::mat::Scalar<T>)
    // -------------------------------------------------------------------------

    template <std::size_t W, typename T> OPTINUM_INLINE scalar_view<T, W> view(datapod::mat::Scalar<T> &s) noexcept {
        return scalar_view<T, W>(&s.value);
    }

    template <std::size_t W, typename T>
    OPTINUM_INLINE scalar_view<const T, W> view(const datapod::mat::Scalar<T> &s) noexcept {
        return scalar_view<const T, W>(&s.value);
    }

    // Auto-detect width for scalar
    template <typename T> OPTINUM_INLINE auto view(datapod::mat::Scalar<T> &s) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return scalar_view<T, W>(&s.value);
    }

    template <typename T> OPTINUM_INLINE auto view(const datapod::mat::Scalar<T> &s) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return scalar_view<const T, W>(&s.value);
    }

    // -------------------------------------------------------------------------
    // Vector view (from datapod::mat::Vector<T, N>)
    // -------------------------------------------------------------------------

    template <std::size_t W, typename T, std::size_t N>
    OPTINUM_INLINE vector_view<T, W> view(datapod::mat::Vector<T, N> &v) noexcept {
        // Use v.size() for runtime size (handles both fixed and Dynamic)
        return vector_view<T, W>(v.data(), v.size());
    }

    template <std::size_t W, typename T, std::size_t N>
    OPTINUM_INLINE vector_view<const T, W> view(const datapod::mat::Vector<T, N> &v) noexcept {
        return vector_view<const T, W>(v.data(), v.size());
    }

    // Auto-detect width for vector
    template <typename T, std::size_t N> OPTINUM_INLINE auto view(datapod::mat::Vector<T, N> &v) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return vector_view<T, W>(v.data(), v.size());
    }

    template <typename T, std::size_t N> OPTINUM_INLINE auto view(const datapod::mat::Vector<T, N> &v) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return vector_view<const T, W>(v.data(), v.size());
    }

    // -------------------------------------------------------------------------
    // Matrix view (from datapod::mat::Matrix<T, R, C>)
    // -------------------------------------------------------------------------

    template <std::size_t W, typename T, std::size_t R, std::size_t C>
    OPTINUM_INLINE matrix_view<T, W> view(datapod::mat::Matrix<T, R, C> &m) noexcept {
        // Use m.rows()/m.cols() for runtime size (handles both fixed and Dynamic)
        return matrix_view<T, W>(m.data(), m.rows(), m.cols());
    }

    template <std::size_t W, typename T, std::size_t R, std::size_t C>
    OPTINUM_INLINE matrix_view<const T, W> view(const datapod::mat::Matrix<T, R, C> &m) noexcept {
        return matrix_view<const T, W>(m.data(), m.rows(), m.cols());
    }

    // Auto-detect width for matrix
    template <typename T, std::size_t R, std::size_t C>
    OPTINUM_INLINE auto view(datapod::mat::Matrix<T, R, C> &m) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return matrix_view<T, W>(m.data(), m.rows(), m.cols());
    }

    template <typename T, std::size_t R, std::size_t C>
    OPTINUM_INLINE auto view(const datapod::mat::Matrix<T, R, C> &m) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return matrix_view<const T, W>(m.data(), m.rows(), m.cols());
    }

    // -------------------------------------------------------------------------
    // Tensor view (from datapod::mat::Tensor<T, Dims...>)
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
    OPTINUM_INLINE auto view(datapod::mat::Tensor<T, Dims...> &t) noexcept {
        constexpr std::size_t Rank = sizeof...(Dims);
        constexpr std::array<std::size_t, Rank> extents = {Dims...};
        constexpr std::array<std::size_t, Rank> strides = detail::compute_strides(extents);
        return tensor_view<T, W, Rank>(t.data(), extents, strides);
    }

    template <std::size_t W, typename T, std::size_t... Dims>
    OPTINUM_INLINE auto view(const datapod::mat::Tensor<T, Dims...> &t) noexcept {
        constexpr std::size_t Rank = sizeof...(Dims);
        constexpr std::array<std::size_t, Rank> extents = {Dims...};
        constexpr std::array<std::size_t, Rank> strides = detail::compute_strides(extents);
        return tensor_view<const T, W, Rank>(t.data(), extents, strides);
    }

    // Auto-detect width for tensor
    template <typename T, std::size_t... Dims> OPTINUM_INLINE auto view(datapod::mat::Tensor<T, Dims...> &t) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        constexpr std::size_t Rank = sizeof...(Dims);
        constexpr std::array<std::size_t, Rank> extents = {Dims...};
        constexpr std::array<std::size_t, Rank> strides = detail::compute_strides(extents);
        return tensor_view<T, W, Rank>(t.data(), extents, strides);
    }

    template <typename T, std::size_t... Dims>
    OPTINUM_INLINE auto view(const datapod::mat::Tensor<T, Dims...> &t) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        constexpr std::size_t Rank = sizeof...(Dims);
        constexpr std::array<std::size_t, Rank> extents = {Dims...};
        constexpr std::array<std::size_t, Rank> strides = detail::compute_strides(extents);
        return tensor_view<const T, W, Rank>(t.data(), extents, strides);
    }

    // -------------------------------------------------------------------------
    // Complex view (from dp::mat::Complex<T> arrays)
    // -------------------------------------------------------------------------

    // From raw pointer + size
    template <std::size_t W, typename T>
    OPTINUM_INLINE complex_view<T, W> view(datapod::mat::Complex<T> *ptr, std::size_t n) noexcept {
        return complex_view<T, W>(ptr, n);
    }

    template <std::size_t W, typename T>
    OPTINUM_INLINE complex_view<T, W> view(const datapod::mat::Complex<T> *ptr, std::size_t n) noexcept {
        return complex_view<T, W>(const_cast<datapod::mat::Complex<T> *>(ptr), n);
    }

    // Auto-detect width from raw pointer
    template <typename T> OPTINUM_INLINE auto view(datapod::mat::Complex<T> *ptr, std::size_t n) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return complex_view<T, W>(ptr, n);
    }

    template <typename T> OPTINUM_INLINE auto view(const datapod::mat::Complex<T> *ptr, std::size_t n) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return complex_view<T, W>(const_cast<datapod::mat::Complex<T> *>(ptr), n);
    }

    // From C-style array
    template <std::size_t W, typename T, std::size_t N>
    OPTINUM_INLINE complex_view<T, W> view(datapod::mat::Complex<T> (&arr)[N]) noexcept {
        return complex_view<T, W>(arr, N);
    }

    template <std::size_t W, typename T, std::size_t N>
    OPTINUM_INLINE complex_view<T, W> view(const datapod::mat::Complex<T> (&arr)[N]) noexcept {
        return complex_view<T, W>(const_cast<datapod::mat::Complex<T> *>(arr), N);
    }

    // Auto-detect width from C-style array
    template <typename T, std::size_t N> OPTINUM_INLINE auto view(datapod::mat::Complex<T> (&arr)[N]) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return complex_view<T, W>(arr, N);
    }

    template <typename T, std::size_t N> OPTINUM_INLINE auto view(const datapod::mat::Complex<T> (&arr)[N]) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return complex_view<T, W>(const_cast<datapod::mat::Complex<T> *>(arr), N);
    }

    // From datapod::mat::Vector<complex<T>, N>
    template <std::size_t W, typename T, std::size_t N>
    OPTINUM_INLINE complex_view<T, W> view(datapod::mat::Vector<datapod::mat::Complex<T>, N> &v) noexcept {
        return complex_view<T, W>(v.data(), N);
    }

    template <std::size_t W, typename T, std::size_t N>
    OPTINUM_INLINE complex_view<T, W> view(const datapod::mat::Vector<datapod::mat::Complex<T>, N> &v) noexcept {
        return complex_view<T, W>(const_cast<datapod::mat::Complex<T> *>(v.data()), N);
    }

    // Auto-detect width from datapod::mat::Vector<complex<T>, N>
    template <typename T, std::size_t N>
    OPTINUM_INLINE auto view(datapod::mat::Vector<datapod::mat::Complex<T>, N> &v) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return complex_view<T, W>(v.data(), N);
    }

    template <typename T, std::size_t N>
    OPTINUM_INLINE auto view(const datapod::mat::Vector<datapod::mat::Complex<T>, N> &v) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return complex_view<T, W>(const_cast<datapod::mat::Complex<T> *>(v.data()), N);
    }

    // -------------------------------------------------------------------------
    // Quaternion view (from dp::mat::Quaternion<T> arrays)
    // -------------------------------------------------------------------------

    // From raw pointer + size
    template <std::size_t W, typename T>
    OPTINUM_INLINE quaternion_view<T, W> view(datapod::mat::Quaternion<T> *ptr, std::size_t n) noexcept {
        return quaternion_view<T, W>(ptr, n);
    }

    template <std::size_t W, typename T>
    OPTINUM_INLINE quaternion_view<T, W> view(const datapod::mat::Quaternion<T> *ptr, std::size_t n) noexcept {
        return quaternion_view<T, W>(const_cast<datapod::mat::Quaternion<T> *>(ptr), n);
    }

    // Auto-detect width from raw pointer
    template <typename T> OPTINUM_INLINE auto view(datapod::mat::Quaternion<T> *ptr, std::size_t n) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return quaternion_view<T, W>(ptr, n);
    }

    template <typename T> OPTINUM_INLINE auto view(const datapod::mat::Quaternion<T> *ptr, std::size_t n) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return quaternion_view<T, W>(const_cast<datapod::mat::Quaternion<T> *>(ptr), n);
    }

    // From C-style array
    template <std::size_t W, typename T, std::size_t N>
    OPTINUM_INLINE quaternion_view<T, W> view(datapod::mat::Quaternion<T> (&arr)[N]) noexcept {
        return quaternion_view<T, W>(arr, N);
    }

    template <std::size_t W, typename T, std::size_t N>
    OPTINUM_INLINE quaternion_view<T, W> view(const datapod::mat::Quaternion<T> (&arr)[N]) noexcept {
        return quaternion_view<T, W>(const_cast<datapod::mat::Quaternion<T> *>(arr), N);
    }

    // Auto-detect width from C-style array
    template <typename T, std::size_t N> OPTINUM_INLINE auto view(datapod::mat::Quaternion<T> (&arr)[N]) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return quaternion_view<T, W>(arr, N);
    }

    template <typename T, std::size_t N>
    OPTINUM_INLINE auto view(const datapod::mat::Quaternion<T> (&arr)[N]) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return quaternion_view<T, W>(const_cast<datapod::mat::Quaternion<T> *>(arr), N);
    }

    // From datapod::mat::Vector<quaternion<T>, N>
    template <std::size_t W, typename T, std::size_t N>
    OPTINUM_INLINE quaternion_view<T, W> view(datapod::mat::Vector<datapod::mat::Quaternion<T>, N> &v) noexcept {
        return quaternion_view<T, W>(v.data(), N);
    }

    template <std::size_t W, typename T, std::size_t N>
    OPTINUM_INLINE quaternion_view<T, W> view(const datapod::mat::Vector<datapod::mat::Quaternion<T>, N> &v) noexcept {
        return quaternion_view<T, W>(const_cast<datapod::mat::Quaternion<T> *>(v.data()), N);
    }

    // Auto-detect width from datapod::mat::Vector<quaternion<T>, N>
    template <typename T, std::size_t N>
    OPTINUM_INLINE auto view(datapod::mat::Vector<datapod::mat::Quaternion<T>, N> &v) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return quaternion_view<T, W>(v.data(), N);
    }

    template <typename T, std::size_t N>
    OPTINUM_INLINE auto view(const datapod::mat::Vector<datapod::mat::Quaternion<T>, N> &v) noexcept {
        constexpr std::size_t W = detail::default_width<T>();
        return quaternion_view<T, W>(const_cast<datapod::mat::Quaternion<T> *>(v.data()), N);
    }

    // -------------------------------------------------------------------------
    // Spatial Quaternion support (dp::Quaternion -> dp::mat::Quaternion<double>)
    // These work because dp::Quaternion inherits from dp::mat::Quaternion<double>
    // -------------------------------------------------------------------------

    // From raw pointer + size (spatial Quaternion)
    template <std::size_t W>
    OPTINUM_INLINE quaternion_view<double, W> view(datapod::Quaternion *ptr, std::size_t n) noexcept {
        // dp::Quaternion inherits from dp::mat::Quaternion<double>, so this cast is safe
        return quaternion_view<double, W>(reinterpret_cast<datapod::mat::Quaternion<double> *>(ptr), n);
    }

    template <std::size_t W>
    OPTINUM_INLINE quaternion_view<double, W> view(const datapod::Quaternion *ptr, std::size_t n) noexcept {
        return quaternion_view<double, W>(const_cast<datapod::mat::Quaternion<double> *>(
                                              reinterpret_cast<const datapod::mat::Quaternion<double> *>(ptr)),
                                          n);
    }

    // Auto-detect width for spatial Quaternion
    OPTINUM_INLINE auto view(datapod::Quaternion *ptr, std::size_t n) noexcept {
        constexpr std::size_t W = detail::default_width<double>();
        return quaternion_view<double, W>(reinterpret_cast<datapod::mat::Quaternion<double> *>(ptr), n);
    }

    OPTINUM_INLINE auto view(const datapod::Quaternion *ptr, std::size_t n) noexcept {
        constexpr std::size_t W = detail::default_width<double>();
        return quaternion_view<double, W>(const_cast<datapod::mat::Quaternion<double> *>(
                                              reinterpret_cast<const datapod::mat::Quaternion<double> *>(ptr)),
                                          n);
    }

    // From C-style array of spatial Quaternion
    template <std::size_t W, std::size_t N>
    OPTINUM_INLINE quaternion_view<double, W> view(datapod::Quaternion (&arr)[N]) noexcept {
        return quaternion_view<double, W>(reinterpret_cast<datapod::mat::Quaternion<double> *>(arr), N);
    }

    template <std::size_t N> OPTINUM_INLINE auto view(datapod::Quaternion (&arr)[N]) noexcept {
        constexpr std::size_t W = detail::default_width<double>();
        return quaternion_view<double, W>(reinterpret_cast<datapod::mat::Quaternion<double> *>(arr), N);
    }

} // namespace optinum::simd
