#pragma once

// =============================================================================
// optinum/lina/expr/expr.hpp
// Minimal expression templates for rank-1/2 elementwise + scalar ops
// =============================================================================

#include <datapod/matrix.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cstddef>
#include <type_traits>

namespace optinum::lina::expr {

    namespace dp = ::datapod;

    // -------------------------------------------------------------------------
    // Vector expressions
    // -------------------------------------------------------------------------

    template <typename Derived, typename T, std::size_t N> struct VecExpr {
        using value_type = T;
        static constexpr std::size_t size = N;
        [[nodiscard]] constexpr T eval(std::size_t i) const noexcept {
            return static_cast<const Derived &>(*this).eval(i);
        }
    };

    template <typename T, std::size_t N> struct VecRef : VecExpr<VecRef<T, N>, T, N> {
        const simd::Vector<T, N> *ptr;
        constexpr explicit VecRef(const simd::Vector<T, N> &v) noexcept : ptr(&v) {}
        [[nodiscard]] constexpr T eval(std::size_t i) const noexcept { return (*ptr)[i]; }
    };

    template <typename L, typename R> struct VecAdd : VecExpr<VecAdd<L, R>, typename L::value_type, L::size> {
        L l;
        R r;
        constexpr VecAdd(const L &lhs, const R &rhs) noexcept : l(lhs), r(rhs) {}
        [[nodiscard]] constexpr typename L::value_type eval(std::size_t i) const noexcept {
            return l.eval(i) + r.eval(i);
        }
    };

    template <typename E> struct VecScale : VecExpr<VecScale<E>, typename E::value_type, E::size> {
        E e;
        typename E::value_type s;
        constexpr VecScale(const E &expr, typename E::value_type scalar) noexcept : e(expr), s(scalar) {}
        [[nodiscard]] constexpr typename E::value_type eval(std::size_t i) const noexcept { return e.eval(i) * s; }
    };

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr VecRef<T, N> ref(const simd::Vector<T, N> &v) noexcept {
        return VecRef<T, N>(v);
    }

    template <typename L, typename R>
    requires requires {
        L::size;
        R::size;
    }
    [[nodiscard]] constexpr VecAdd<L, R> add(const L &l, const R &r) noexcept {
        static_assert(L::size == R::size, "vector sizes must match");
        return VecAdd<L, R>(l, r);
    }

    template <typename E>
    requires requires { E::size; }
    [[nodiscard]] constexpr VecScale<E> scale(const E &e, typename E::value_type s) noexcept {
        return VecScale<E>(e, s);
    }

    // Specialized eval() for VecAdd - use SIMD backend (returns owning type)
    template <typename L, typename R>
    [[nodiscard]] constexpr dp::mat::vector<typename L::value_type, L::size> eval(const VecAdd<L, R> &e) noexcept {
        auto lhs = eval(e.l);
        auto rhs = eval(e.r);
        dp::mat::vector<typename L::value_type, L::size> out;

        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < L::size; ++i) {
                out[i] = lhs[i] + rhs[i];
            }
        } else {
            // Use SIMD backend for addition
            simd::backend::add<typename L::value_type, L::size>(out.data(), lhs.data(), rhs.data());
        }
        return out;
    }

    // Specialized eval() for VecScale - use SIMD backend (returns owning type)
    template <typename E>
    [[nodiscard]] constexpr dp::mat::vector<typename E::value_type, E::size> eval(const VecScale<E> &e) noexcept {
        auto src = eval(e.e);
        dp::mat::vector<typename E::value_type, E::size> out;

        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < E::size; ++i) {
                out[i] = src[i] * e.s;
            }
        } else {
            // Use SIMD backend for scalar multiplication
            simd::backend::mul_scalar<typename E::value_type, E::size>(out.data(), src.data(), e.s);
        }
        return out;
    }

    // Specialized eval() for VecRef - copy to owning type
    template <typename T, std::size_t N>
    [[nodiscard]] constexpr dp::mat::vector<T, N> eval(const VecRef<T, N> &e) noexcept {
        dp::mat::vector<T, N> out;
        for (std::size_t i = 0; i < N; ++i) {
            out[i] = (*e.ptr)[i];
        }
        return out;
    }

    // Generic fallback for other expression types - scalar loop (returns owning type)
    template <typename E>
    requires requires { E::size; }
    [[nodiscard]] constexpr dp::mat::vector<typename E::value_type, E::size> eval(const E &e) noexcept {
        dp::mat::vector<typename E::value_type, E::size> out;
        for (std::size_t i = 0; i < E::size; ++i) {
            out[i] = e.eval(i);
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // Matrix expressions
    // -------------------------------------------------------------------------

    template <typename Derived, typename T, std::size_t R, std::size_t C> struct MatExpr {
        using value_type = T;
        static constexpr std::size_t rows = R;
        static constexpr std::size_t cols = C;
        [[nodiscard]] constexpr T eval(std::size_t r, std::size_t c) const noexcept {
            return static_cast<const Derived &>(*this).eval(r, c);
        }
    };

    template <typename T, std::size_t R, std::size_t C> struct MatRef : MatExpr<MatRef<T, R, C>, T, R, C> {
        const simd::Matrix<T, R, C> *ptr;
        constexpr explicit MatRef(const simd::Matrix<T, R, C> &m) noexcept : ptr(&m) {}
        [[nodiscard]] constexpr T eval(std::size_t r, std::size_t c) const noexcept { return (*ptr)(r, c); }
    };

    template <typename L, typename R> struct MatAdd : MatExpr<MatAdd<L, R>, typename L::value_type, L::rows, L::cols> {
        L l;
        R r;
        constexpr MatAdd(const L &lhs, const R &rhs) noexcept : l(lhs), r(rhs) {}
        [[nodiscard]] constexpr typename L::value_type eval(std::size_t i, std::size_t j) const noexcept {
            return l.eval(i, j) + r.eval(i, j);
        }
    };

    template <typename E> struct MatScale : MatExpr<MatScale<E>, typename E::value_type, E::rows, E::cols> {
        E e;
        typename E::value_type s;
        constexpr MatScale(const E &expr, typename E::value_type scalar) noexcept : e(expr), s(scalar) {}
        [[nodiscard]] constexpr typename E::value_type eval(std::size_t i, std::size_t j) const noexcept {
            return e.eval(i, j) * s;
        }
    };

    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr MatRef<T, R, C> ref(const simd::Matrix<T, R, C> &m) noexcept {
        return MatRef<T, R, C>(m);
    }

    template <typename L, typename R>
    requires requires {
        L::rows;
        L::cols;
        R::rows;
        R::cols;
    }
    [[nodiscard]] constexpr MatAdd<L, R> add(const L &l, const R &r) noexcept {
        static_assert(L::rows == R::rows && L::cols == R::cols, "matrix shapes must match");
        return MatAdd<L, R>(l, r);
    }

    template <typename E>
    requires requires {
        E::rows;
        E::cols;
    }
    [[nodiscard]] constexpr MatScale<E> scale(const E &e, typename E::value_type s) noexcept {
        return MatScale<E>(e, s);
    }

    // Specialized eval() for MatAdd - use SIMD backend (returns owning type)
    template <typename L, typename R>
    [[nodiscard]] constexpr dp::mat::matrix<typename L::value_type, L::rows, L::cols>
    eval(const MatAdd<L, R> &e) noexcept {
        auto lhs = eval(e.l);
        auto rhs = eval(e.r);
        dp::mat::matrix<typename L::value_type, L::rows, L::cols> out;

        constexpr std::size_t N = L::rows * L::cols;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = lhs[i] + rhs[i];
            }
        } else {
            // Use SIMD backend for addition (operate on flattened matrix)
            simd::backend::add<typename L::value_type, N>(out.data(), lhs.data(), rhs.data());
        }
        return out;
    }

    // Specialized eval() for MatScale - use SIMD backend (returns owning type)
    template <typename E>
    [[nodiscard]] constexpr dp::mat::matrix<typename E::value_type, E::rows, E::cols>
    eval(const MatScale<E> &e) noexcept {
        auto src = eval(e.e);
        dp::mat::matrix<typename E::value_type, E::rows, E::cols> out;

        constexpr std::size_t N = E::rows * E::cols;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = src[i] * e.s;
            }
        } else {
            // Use SIMD backend for scalar multiplication
            simd::backend::mul_scalar<typename E::value_type, N>(out.data(), src.data(), e.s);
        }
        return out;
    }

    // Specialized eval() for MatRef - copy to owning type
    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr dp::mat::matrix<T, R, C> eval(const MatRef<T, R, C> &e) noexcept {
        dp::mat::matrix<T, R, C> out;
        for (std::size_t i = 0; i < R * C; ++i) {
            out[i] = e.ptr->data()[i];
        }
        return out;
    }

    // Generic fallback for other expression types - scalar loop (returns owning type)
    template <typename E>
    requires requires {
        E::rows;
        E::cols;
    }
    [[nodiscard]] constexpr dp::mat::matrix<typename E::value_type, E::rows, E::cols> eval(const E &e) noexcept {
        dp::mat::matrix<typename E::value_type, E::rows, E::cols> out;
        for (std::size_t j = 0; j < E::cols; ++j) {
            for (std::size_t i = 0; i < E::rows; ++i) {
                out(i, j) = e.eval(i, j);
            }
        }
        return out;
    }

} // namespace optinum::lina::expr
