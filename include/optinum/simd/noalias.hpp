#pragma once

// =============================================================================
// optinum/simd/noalias.hpp
// No-alias optimization hints for compilers
// =============================================================================

#include <optinum/simd/matrix.hpp>
#include <optinum/simd/tensor.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::simd {

    /**
     * @brief Wrapper to hint that no aliasing occurs
     *
     * Tells the compiler that the destination does not alias with sources,
     * allowing better optimization (especially for expression templates).
     *
     * Example:
     *   auto result = noalias(c) = a + b;
     *
     * This hints that 'c' doesn't overlap with 'a' or 'b', allowing
     * the compiler to optimize without aliasing checks.
     */
    template <typename T> class NoAliasWrapper {
      private:
        T &ref_;

      public:
        constexpr explicit NoAliasWrapper(T &ref) noexcept : ref_(ref) {}

        // Assignment from same type
        constexpr NoAliasWrapper &operator=(const T &other) noexcept {
// Use __restrict__ hint if available
#if defined(__GNUC__) || defined(__clang__)
            T *__restrict__ dst = &ref_;
            const T *__restrict__ src = &other;
            *dst = *src;
#else
            ref_ = other;
#endif
            return *this;
        }

        // Assignment from different type (expression templates, etc.)
        template <typename U> constexpr NoAliasWrapper &operator=(const U &other) noexcept {
            ref_ = other;
            return *this;
        }

        // Get the underlying reference
        [[nodiscard]] constexpr T &get() noexcept { return ref_; }
        [[nodiscard]] constexpr const T &get() const noexcept { return ref_; }
    };

    // Factory function for Vector
    template <typename T, std::size_t N>
    [[nodiscard]] constexpr NoAliasWrapper<Vector<T, N>> noalias(Vector<T, N> &v) noexcept {
        return NoAliasWrapper<Vector<T, N>>(v);
    }

    // Factory function for Matrix
    template <typename T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr NoAliasWrapper<Matrix<T, R, C>> noalias(Matrix<T, R, C> &m) noexcept {
        return NoAliasWrapper<Matrix<T, R, C>>(m);
    }

    // Factory function for Tensor
    template <typename T, std::size_t... Dims>
    [[nodiscard]] constexpr NoAliasWrapper<Tensor<T, Dims...>> noalias(Tensor<T, Dims...> &t) noexcept {
        return NoAliasWrapper<Tensor<T, Dims...>>(t);
    }

} // namespace optinum::simd
