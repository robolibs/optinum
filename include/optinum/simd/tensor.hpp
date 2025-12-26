#pragma once

#include <datapod/matrix/tensor.hpp>
#include <iostream>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/backend/reduce.hpp>

#include <type_traits>

namespace optinum::simd {

    namespace dp = ::datapod;

    // Tensor: N-dimensional fixed-size array (rank >= 3) with SIMD-accelerated operations
    // Wraps datapod::mat::tensor<T, Dims...>
    template <typename T, std::size_t... Dims> class Tensor {
        static_assert(sizeof...(Dims) >= 3, "Tensor requires at least 3 dimensions (use Vector for 1D, Matrix for 2D)");
        static_assert(((Dims > 0) && ...), "All tensor dimensions must be > 0");
        static_assert(std::is_arithmetic_v<T>, "Tensor requires arithmetic type");

      public:
        using value_type = T;
        using pod_type = dp::mat::tensor<T, Dims...>;
        using size_type = std::size_t;
        using reference = T &;
        using const_reference = const T &;
        using pointer = T *;
        using const_pointer = const T *;
        using iterator = T *;
        using const_iterator = const T *;

        static constexpr size_type rank = sizeof...(Dims);
        static constexpr std::array<size_type, rank> dims = {Dims...};
        static constexpr size_type total_size = (Dims * ...);

        constexpr Tensor() noexcept = default;
        constexpr explicit Tensor(const pod_type &pod) noexcept : pod_(pod) {}
        constexpr explicit Tensor(pod_type &&pod) noexcept : pod_(static_cast<pod_type &&>(pod)) {}

        constexpr pod_type &pod() noexcept { return pod_; }
        constexpr const pod_type &pod() const noexcept { return pod_; }

        constexpr pointer data() noexcept { return pod_.data(); }
        constexpr const_pointer data() const noexcept { return pod_.data(); }

        // Multi-dimensional indexing
        template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == rank &&
                                                                   (std::is_convertible_v<Indices, size_type> && ...)>>
        constexpr reference operator()(Indices... indices) noexcept {
            return pod_(indices...);
        }

        template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == rank &&
                                                                   (std::is_convertible_v<Indices, size_type> && ...)>>
        constexpr const_reference operator()(Indices... indices) const noexcept {
            return pod_(indices...);
        }

        // Checked multi-dimensional access
        template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == rank &&
                                                                   (std::is_convertible_v<Indices, size_type> && ...)>>
        constexpr reference at(Indices... indices) {
            return pod_.at(indices...);
        }

        template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == rank &&
                                                                   (std::is_convertible_v<Indices, size_type> && ...)>>
        constexpr const_reference at(Indices... indices) const {
            return pod_.at(indices...);
        }

        // Linear indexing
        constexpr reference operator[](size_type i) noexcept { return pod_[i]; }
        constexpr const_reference operator[](size_type i) const noexcept { return pod_[i]; }

        static constexpr size_type size() noexcept { return total_size; }
        static constexpr bool empty() noexcept { return false; }
        static constexpr std::array<size_type, rank> shape() noexcept { return dims; }
        static constexpr size_type dim(size_type i) noexcept { return dims[i]; }

        constexpr iterator begin() noexcept { return pod_.begin(); }
        constexpr const_iterator begin() const noexcept { return pod_.begin(); }
        constexpr const_iterator cbegin() const noexcept { return pod_.cbegin(); }

        constexpr iterator end() noexcept { return pod_.end(); }
        constexpr const_iterator end() const noexcept { return pod_.end(); }
        constexpr const_iterator cend() const noexcept { return pod_.cend(); }

        constexpr Tensor &fill(const T &value) noexcept {
            pod_.fill(value);
            return *this;
        }

        // Compound assignment (element-wise)
        constexpr Tensor &operator+=(const Tensor &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < total_size; ++i)
                    pod_[i] += rhs.pod_[i];
            } else {
                backend::add<T, total_size>(data(), data(), rhs.data());
            }
            return *this;
        }

        constexpr Tensor &operator-=(const Tensor &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < total_size; ++i)
                    pod_[i] -= rhs.pod_[i];
            } else {
                backend::sub<T, total_size>(data(), data(), rhs.data());
            }
            return *this;
        }

        constexpr Tensor &operator*=(const Tensor &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < total_size; ++i)
                    pod_[i] *= rhs.pod_[i];
            } else {
                backend::mul<T, total_size>(data(), data(), rhs.data());
            }
            return *this;
        }

        constexpr Tensor &operator/=(const Tensor &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < total_size; ++i)
                    pod_[i] /= rhs.pod_[i];
            } else {
                backend::div<T, total_size>(data(), data(), rhs.data());
            }
            return *this;
        }

        constexpr Tensor &operator*=(T scalar) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < total_size; ++i)
                    pod_[i] *= scalar;
            } else {
                backend::mul_scalar<T, total_size>(data(), data(), scalar);
            }
            return *this;
        }

        constexpr Tensor &operator/=(T scalar) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < total_size; ++i)
                    pod_[i] /= scalar;
            } else {
                backend::div_scalar<T, total_size>(data(), data(), scalar);
            }
            return *this;
        }

        // Unary
        constexpr Tensor operator-() const noexcept {
            Tensor result;
            for (size_type i = 0; i < total_size; ++i)
                result.pod_[i] = -pod_[i];
            return result;
        }

        constexpr Tensor operator+() const noexcept { return *this; }

        // ==========================================================================
        // Shape Manipulation
        // ==========================================================================

        // Reshape to new dimensions (must have same total size)
        // Example: Tensor<float, 2, 3, 4>.reshape<3, 8>() -> Tensor<float, 3, 8>
        template <std::size_t... NewDims> constexpr Tensor<T, NewDims...> reshape() const noexcept {
            static_assert((NewDims * ...) == total_size,
                          "reshape(): new dimensions must have same total size as original");
            static_assert(sizeof...(NewDims) >= 3, "reshape() for Tensor must result in rank >= 3");

            Tensor<T, NewDims...> result;
            // Copy data linearly (tensors are stored in row-major order)
            for (size_type i = 0; i < total_size; ++i) {
                result[i] = pod_[i];
            }
            return result;
        }

      private:
        pod_type pod_{};
    };

    // Binary ops (element-wise)
    template <typename T, std::size_t... Dims>
    constexpr Tensor<T, Dims...> operator+(const Tensor<T, Dims...> &lhs, const Tensor<T, Dims...> &rhs) noexcept {
        Tensor<T, Dims...> result;
        constexpr auto N = Tensor<T, Dims...>::total_size;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                result[i] = lhs[i] + rhs[i];
        } else {
            backend::add<T, N>(result.data(), lhs.data(), rhs.data());
        }
        return result;
    }

    template <typename T, std::size_t... Dims>
    constexpr Tensor<T, Dims...> operator-(const Tensor<T, Dims...> &lhs, const Tensor<T, Dims...> &rhs) noexcept {
        Tensor<T, Dims...> result;
        constexpr auto N = Tensor<T, Dims...>::total_size;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                result[i] = lhs[i] - rhs[i];
        } else {
            backend::sub<T, N>(result.data(), lhs.data(), rhs.data());
        }
        return result;
    }

    template <typename T, std::size_t... Dims>
    constexpr Tensor<T, Dims...> operator*(const Tensor<T, Dims...> &lhs, const Tensor<T, Dims...> &rhs) noexcept {
        Tensor<T, Dims...> result;
        constexpr auto N = Tensor<T, Dims...>::total_size;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                result[i] = lhs[i] * rhs[i];
        } else {
            backend::mul<T, N>(result.data(), lhs.data(), rhs.data());
        }
        return result;
    }

    template <typename T, std::size_t... Dims>
    constexpr Tensor<T, Dims...> operator/(const Tensor<T, Dims...> &lhs, const Tensor<T, Dims...> &rhs) noexcept {
        Tensor<T, Dims...> result;
        constexpr auto N = Tensor<T, Dims...>::total_size;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                result[i] = lhs[i] / rhs[i];
        } else {
            backend::div<T, N>(result.data(), lhs.data(), rhs.data());
        }
        return result;
    }

    // Scalar ops
    template <typename T, std::size_t... Dims>
    constexpr Tensor<T, Dims...> operator*(const Tensor<T, Dims...> &lhs, T scalar) noexcept {
        Tensor<T, Dims...> result;
        constexpr auto N = Tensor<T, Dims...>::total_size;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                result[i] = lhs[i] * scalar;
        } else {
            backend::mul_scalar<T, N>(result.data(), lhs.data(), scalar);
        }
        return result;
    }

    template <typename T, std::size_t... Dims>
    constexpr Tensor<T, Dims...> operator*(T scalar, const Tensor<T, Dims...> &rhs) noexcept {
        return rhs * scalar;
    }

    template <typename T, std::size_t... Dims>
    constexpr Tensor<T, Dims...> operator/(const Tensor<T, Dims...> &lhs, T scalar) noexcept {
        Tensor<T, Dims...> result;
        constexpr auto N = Tensor<T, Dims...>::total_size;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                result[i] = lhs[i] / scalar;
        } else {
            backend::div_scalar<T, N>(result.data(), lhs.data(), scalar);
        }
        return result;
    }

    // Comparisons
    template <typename T, std::size_t... Dims>
    constexpr bool operator==(const Tensor<T, Dims...> &lhs, const Tensor<T, Dims...> &rhs) noexcept {
        constexpr auto N = Tensor<T, Dims...>::total_size;
        for (std::size_t i = 0; i < N; ++i) {
            if (lhs[i] != rhs[i])
                return false;
        }
        return true;
    }

    template <typename T, std::size_t... Dims>
    constexpr bool operator!=(const Tensor<T, Dims...> &lhs, const Tensor<T, Dims...> &rhs) noexcept {
        return !(lhs == rhs);
    }

    // Common operations
    template <typename T, std::size_t... Dims> constexpr T sum(const Tensor<T, Dims...> &t) noexcept {
        constexpr auto N = Tensor<T, Dims...>::total_size;
        if (std::is_constant_evaluated()) {
            T result{};
            for (std::size_t i = 0; i < N; ++i)
                result += t[i];
            return result;
        }
        return backend::reduce_sum<T, N>(t.data());
    }

    // =============================================================================
    // I/O - Stream output operator
    // =============================================================================

    template <typename T, std::size_t... Dims> std::ostream &operator<<(std::ostream &os, const Tensor<T, Dims...> &t) {
        constexpr auto N = Tensor<T, Dims...>::total_size;
        constexpr auto rank = Tensor<T, Dims...>::rank;
        auto shape = Tensor<T, Dims...>::shape();

        os << "Tensor<";
        for (std::size_t i = 0; i < rank; ++i) {
            os << shape[i];
            if (i < rank - 1)
                os << "x";
        }
        os << ">[";

        for (std::size_t i = 0; i < N; ++i) {
            os << t[i];
            if (i < N - 1)
                os << ", ";
        }
        os << "]";
        return os;
    }

    // =============================================================================
    // Type conversion - cast<U>()
    // =============================================================================

    template <typename U, typename T, std::size_t... Dims>
    Tensor<U, Dims...> cast(const Tensor<T, Dims...> &t) noexcept {
        Tensor<U, Dims...> result;
        constexpr auto N = Tensor<T, Dims...>::total_size;
        for (std::size_t i = 0; i < N; ++i) {
            result[i] = static_cast<U>(t[i]);
        }
        return result;
    }

    // =============================================================================
    // squeeze() - Remove dimensions of size 1
    // =============================================================================

    // squeeze() removes all dimensions of size 1
    // Note: If result would have rank < 3, this function won't compile (use reshape() instead)
    // Common patterns:
    //   Tensor<T, 1, N, M> -> Tensor<T, N, M, 1> (add dummy dimension to keep rank >= 3)
    //   Tensor<T, N, 1, M> -> Tensor<T, N, M, 1> (add dummy dimension to keep rank >= 3)
    //   Tensor<T, N, M, 1> -> Tensor<T, N, M, 1> (already rank 3, keep as is or remove)

    // Specific squeeze implementations for common cases
    // squeeze<1, N, M>() -> Would need rank 2, so pad with 1: <N, M, 1>
    template <typename T, std::size_t N, std::size_t M>
    constexpr Tensor<T, N, M, 1> squeeze(const Tensor<T, 1, N, M> &t) noexcept {
        return t.template reshape<N, M, 1>();
    }

    // squeeze<N, 1, M>() -> Would need rank 2, so pad with 1: <N, M, 1>
    template <typename T, std::size_t N, std::size_t M>
    constexpr Tensor<T, N, M, 1> squeeze(const Tensor<T, N, 1, M> &t) noexcept {
        return t.template reshape<N, M, 1>();
    }

    // squeeze<N, M, 1>() -> Would need rank 2, keep as rank 3: <N, M, 1> (no-op)
    template <typename T, std::size_t N, std::size_t M>
    constexpr Tensor<T, N, M, 1> squeeze(const Tensor<T, N, M, 1> &t) noexcept {
        return t;
    }

    // squeeze<1, 1, N, M>() -> Tensor<N, M, 1>
    template <typename T, std::size_t N, std::size_t M>
    constexpr Tensor<T, N, M, 1> squeeze(const Tensor<T, 1, 1, N, M> &t) noexcept {
        return t.template reshape<N, M, 1>();
    }

    // squeeze<N, 1, M, 1>() -> Tensor<N, M, 1>
    template <typename T, std::size_t N, std::size_t M>
    constexpr Tensor<T, N, M, 1> squeeze(const Tensor<T, N, 1, M, 1> &t) noexcept {
        return t.template reshape<N, M, 1>();
    }

    // squeeze<N, M, P> with all > 1 -> no-op (no dimensions to squeeze)
    template <typename T, std::size_t N, std::size_t M, std::size_t P>
    constexpr Tensor<T, N, M, P> squeeze(const Tensor<T, N, M, P> &t) noexcept
    requires(N > 1 && M > 1 && P > 1)
    {
        return t;
    }

    // squeeze<1, N, M, P>() -> Tensor<N, M, P>
    template <typename T, std::size_t N, std::size_t M, std::size_t P>
    constexpr Tensor<T, N, M, P> squeeze(const Tensor<T, 1, N, M, P> &t) noexcept {
        return t.template reshape<N, M, P>();
    }

    // squeeze<N, 1, M, P>() -> Tensor<N, M, P>
    template <typename T, std::size_t N, std::size_t M, std::size_t P>
    constexpr Tensor<T, N, M, P> squeeze(const Tensor<T, N, 1, M, P> &t) noexcept {
        return t.template reshape<N, M, P>();
    }

    // squeeze<N, M, 1, P>() -> Tensor<N, M, P>
    template <typename T, std::size_t N, std::size_t M, std::size_t P>
    constexpr Tensor<T, N, M, P> squeeze(const Tensor<T, N, M, 1, P> &t) noexcept {
        return t.template reshape<N, M, P>();
    }

    // squeeze<N, M, P, 1>() -> Tensor<N, M, P>
    template <typename T, std::size_t N, std::size_t M, std::size_t P>
    constexpr Tensor<T, N, M, P> squeeze(const Tensor<T, N, M, P, 1> &t) noexcept {
        return t.template reshape<N, M, P>();
    }

    // Common type aliases
    template <typename T> using Tensor3D_2x2x2 = Tensor<T, 2, 2, 2>;
    template <typename T> using Tensor3D_3x3x3 = Tensor<T, 3, 3, 3>;
    template <typename T> using Tensor3D_4x4x4 = Tensor<T, 4, 4, 4>;

    using Tensor3D_2x2x2f = Tensor<float, 2, 2, 2>;
    using Tensor3D_2x2x2d = Tensor<double, 2, 2, 2>;
    using Tensor3D_3x3x3f = Tensor<float, 3, 3, 3>;
    using Tensor3D_3x3x3d = Tensor<double, 3, 3, 3>;
    using Tensor3D_4x4x4f = Tensor<float, 4, 4, 4>;
    using Tensor3D_4x4x4d = Tensor<double, 4, 4, 4>;

} // namespace optinum::simd
