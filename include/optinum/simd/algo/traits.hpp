#pragma once

// =============================================================================
// optinum/simd/algo/traits.hpp
// Type traits for detecting packable views (vector, matrix, tensor)
// =============================================================================

#include <cstddef>
#include <optinum/simd/view/matrix_view.hpp>
#include <optinum/simd/view/tensor_view.hpp>
#include <optinum/simd/view/vector_view.hpp>
#include <type_traits>

namespace optinum::simd::detail {

    // =============================================================================
    // is_packable_view - detect if a type has the pack interface
    // =============================================================================

    template <typename T, typename = void> struct is_packable_view : std::false_type {};

    template <typename T>
    struct is_packable_view<
        T, std::void_t<decltype(std::declval<T>().num_packs()), decltype(std::declval<T>().load_pack(0)),
                       decltype(std::declval<T>().store_pack(0, std::declval<T>().load_pack(0))),
                       decltype(std::declval<T>().load_pack_tail(0)),
                       decltype(std::declval<T>().store_pack_tail(0, std::declval<T>().load_pack(0)))>>
        : std::true_type {};

    template <typename T> inline constexpr bool is_packable_view_v = is_packable_view<T>::value;

    // =============================================================================
    // is_const_view - check if view is read-only (const T)
    // =============================================================================

    template <typename View> struct is_const_view : std::false_type {};

    template <typename T, std::size_t W> struct is_const_view<vector_view<const T, W>> : std::true_type {};

    template <typename T, std::size_t W> struct is_const_view<matrix_view<const T, W>> : std::true_type {};

    template <typename T, std::size_t W, std::size_t R>
    struct is_const_view<tensor_view<const T, W, R>> : std::true_type {};

    template <typename T> inline constexpr bool is_const_view_v = is_const_view<T>::value;

    // =============================================================================
    // view_value_t - extract value_type from a view
    // =============================================================================

    template <typename View> using view_value_t = typename View::value_type;

    // =============================================================================
    // view_width_v - extract width from a view
    // =============================================================================

    template <typename View> inline constexpr std::size_t view_width_v = View::width;

} // namespace optinum::simd::detail
