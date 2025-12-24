#pragma once

// =============================================================================
// optinum/lina/algebra/einsum.hpp
// Minimal compile-time einsum dispatcher for rank-1/2 operands
// =============================================================================

#include <optinum/lina/basic/matmul.hpp>
#include <optinum/lina/basic/transpose.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cstddef>
#include <string_view>
#include <type_traits>

namespace optinum::lina {

    template <std::size_t N> struct fixed_string {
        char value[N];
        constexpr fixed_string(const char (&s)[N]) noexcept {
            for (std::size_t i = 0; i < N; ++i)
                value[i] = s[i];
        }
        [[nodiscard]] constexpr operator std::string_view() const noexcept { return std::string_view(value, N - 1); }
    };

    namespace einsum_detail {
        template <typename> struct is_tensor : std::false_type {};
        template <typename T, std::size_t N> struct is_tensor<simd::Vector<T, N>> : std::true_type {};
        template <typename> struct is_matrix : std::false_type {};
        template <typename T, std::size_t R, std::size_t C> struct is_matrix<simd::Matrix<T, R, C>> : std::true_type {};

        template <typename T> inline constexpr bool is_tensor_v = is_tensor<T>::value;
        template <typename T> inline constexpr bool is_matrix_v = is_matrix<T>::value;
    } // namespace einsum_detail

    // -------------------------------------------------------------------------
    // einsum with 1 operand
    // Supported:
    // - "ij->ji" (transpose)
    // - "ij->ij" (identity)
    // - "i->i"   (identity)
    // -------------------------------------------------------------------------
    template <fixed_string Spec, typename A> [[nodiscard]] constexpr auto einsum(const A &a) noexcept {
        constexpr std::string_view s = Spec;

        if constexpr (einsum_detail::is_matrix_v<A>) {
            if constexpr (s == "ij->ji") {
                return simd::transpose(a);
            } else if constexpr (s == "ij->ij") {
                return a;
            } else {
                static_assert(s == "ij->ji" || s == "ij->ij", "unsupported einsum spec for Matrix");
            }
        } else if constexpr (einsum_detail::is_tensor_v<A>) {
            if constexpr (s == "i->i") {
                return a;
            } else {
                static_assert(s == "i->i", "unsupported einsum spec for Tensor");
            }
        } else {
            static_assert(einsum_detail::is_matrix_v<A> || einsum_detail::is_tensor_v<A>,
                          "einsum expects simd::Matrix or simd::Vector");
        }
    }

    // -------------------------------------------------------------------------
    // einsum with 2 operands
    // Supported:
    // - "ij,jk->ik" (matmul)
    // - "ij,j->i"   (matvec)
    // - "i,i->"     (dot)
    // - "i,j->ij"   (outer)
    // -------------------------------------------------------------------------
    template <fixed_string Spec, typename A, typename B>
    [[nodiscard]] constexpr auto einsum(const A &a, const B &b) noexcept {
        constexpr std::string_view s = Spec;

        if constexpr (s == "ij,jk->ik") {
            static_assert(einsum_detail::is_matrix_v<A> && einsum_detail::is_matrix_v<B>,
                          "ij,jk->ik expects (Matrix,Matrix)");
            return matmul(a, b);
        } else if constexpr (s == "ij,j->i") {
            static_assert(einsum_detail::is_matrix_v<A> && einsum_detail::is_tensor_v<B>,
                          "ij,j->i expects (Matrix,Tensor)");
            return matmul(a, b);
        } else if constexpr (s == "i,i->") {
            static_assert(einsum_detail::is_tensor_v<A> && einsum_detail::is_tensor_v<B>,
                          "i,i-> expects (Tensor,Tensor)");
            return simd::dot(a, b);
        } else if constexpr (s == "i,j->ij") {
            static_assert(einsum_detail::is_tensor_v<A> && einsum_detail::is_tensor_v<B>,
                          "i,j->ij expects (Tensor,Tensor)");
            using T = typename A::value_type;
            constexpr std::size_t M = A::extent;
            constexpr std::size_t N = B::extent;
            simd::Matrix<T, M, N> out;
            for (std::size_t j = 0; j < N; ++j) {
                for (std::size_t i = 0; i < M; ++i) {
                    out(i, j) = a[i] * b[j];
                }
            }
            return out;
        } else {
            static_assert(s == "ij,jk->ik" || s == "ij,j->i" || s == "i,i->" || s == "i,j->ij",
                          "unsupported einsum spec");
        }
    }

} // namespace optinum::lina
