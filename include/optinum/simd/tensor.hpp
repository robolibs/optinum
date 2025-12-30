#pragma once

#include <datapod/matrix/tensor.hpp>
#include <iostream>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/backend/reduce.hpp>

#include <type_traits>

namespace optinum::simd {

    namespace dp = ::datapod;

    // Tensor: N-dimensional fixed-size non-owning view (rank >= 3) with SIMD-accelerated operations
    // Non-owning view over datapod::mat::tensor<T, Dims...> or raw T* data
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

        // Default constructor (null view)
        constexpr Tensor() noexcept : ptr_(nullptr) {}

        // Constructor from raw pointer (non-owning view)
        constexpr explicit Tensor(T *ptr) noexcept : ptr_(ptr) {}

        // Constructor from pod_type reference (non-owning view)
        constexpr Tensor(pod_type &pod) noexcept : ptr_(pod.data()) {}
        constexpr Tensor(const pod_type &pod) noexcept : ptr_(const_cast<T *>(pod.data())) {}

        // Access to underlying pointer
        constexpr pointer data() noexcept { return ptr_; }
        constexpr const_pointer data() const noexcept { return ptr_; }

        // Check if view is valid (non-null)
        constexpr bool valid() const noexcept { return ptr_ != nullptr; }
        constexpr explicit operator bool() const noexcept { return valid(); }

        // Multi-dimensional indexing
        template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == rank &&
                                                                   (std::is_convertible_v<Indices, size_type> && ...)>>
        constexpr reference operator()(Indices... indices) noexcept {
            return ptr_[linear_index(indices...)];
        }

        template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == rank &&
                                                                   (std::is_convertible_v<Indices, size_type> && ...)>>
        constexpr const_reference operator()(Indices... indices) const noexcept {
            return ptr_[linear_index(indices...)];
        }

        // Checked multi-dimensional access
        template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == rank &&
                                                                   (std::is_convertible_v<Indices, size_type> && ...)>>
        constexpr reference at(Indices... indices) {
            check_bounds(indices...);
            return ptr_[linear_index(indices...)];
        }

        template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == rank &&
                                                                   (std::is_convertible_v<Indices, size_type> && ...)>>
        constexpr const_reference at(Indices... indices) const {
            check_bounds(indices...);
            return ptr_[linear_index(indices...)];
        }

        // Linear indexing
        constexpr reference operator[](size_type i) noexcept { return ptr_[i]; }
        constexpr const_reference operator[](size_type i) const noexcept { return ptr_[i]; }

        static constexpr size_type size() noexcept { return total_size; }
        static constexpr bool empty() noexcept { return false; }
        static constexpr std::array<size_type, rank> shape() noexcept { return dims; }
        static constexpr size_type dim(size_type i) noexcept { return dims[i]; }

        constexpr iterator begin() noexcept { return ptr_; }
        constexpr const_iterator begin() const noexcept { return ptr_; }
        constexpr const_iterator cbegin() const noexcept { return ptr_; }

        constexpr iterator end() noexcept { return ptr_ + total_size; }
        constexpr const_iterator end() const noexcept { return ptr_ + total_size; }
        constexpr const_iterator cend() const noexcept { return ptr_ + total_size; }

        constexpr Tensor &fill(const T &value) noexcept {
            for (size_type i = 0; i < total_size; ++i) {
                ptr_[i] = value;
            }
            return *this;
        }

        // Fill with sequential values: 0, 1, 2, ...
        constexpr Tensor &iota() noexcept {
            const size_type n = size();
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < n; ++i)
                    ptr_[i] = static_cast<T>(i);
            } else {
                backend::iota<T, total_size>(ptr_, T{0}, T{1});
            }
            return *this;
        }

        // Fill with sequential values starting from 'start'
        constexpr Tensor &iota(T start) noexcept {
            const size_type n = size();
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < n; ++i)
                    ptr_[i] = start + static_cast<T>(i);
            } else {
                backend::iota<T, total_size>(ptr_, start, T{1});
            }
            return *this;
        }

        // Fill with sequential values with custom start and step
        constexpr Tensor &iota(T start, T step) noexcept {
            const size_type n = size();
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < n; ++i)
                    ptr_[i] = start + static_cast<T>(i) * step;
            } else {
                backend::iota<T, total_size>(ptr_, start, step);
            }
            return *this;
        }

        // Reverse elements in-place
        constexpr Tensor &reverse() noexcept {
            const size_type n = size();
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < n / 2; ++i)
                    std::swap(ptr_[i], ptr_[n - 1 - i]);
            } else {
                backend::reverse<T, total_size>(ptr_);
            }
            return *this;
        }

        // Compound assignment (element-wise)
        constexpr Tensor &operator+=(const Tensor &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < total_size; ++i)
                    ptr_[i] += rhs.ptr_[i];
            } else {
                backend::add<T, total_size>(data(), data(), rhs.data());
            }
            return *this;
        }

        constexpr Tensor &operator-=(const Tensor &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < total_size; ++i)
                    ptr_[i] -= rhs.ptr_[i];
            } else {
                backend::sub<T, total_size>(data(), data(), rhs.data());
            }
            return *this;
        }

        constexpr Tensor &operator*=(const Tensor &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < total_size; ++i)
                    ptr_[i] *= rhs.ptr_[i];
            } else {
                backend::mul<T, total_size>(data(), data(), rhs.data());
            }
            return *this;
        }

        constexpr Tensor &operator/=(const Tensor &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < total_size; ++i)
                    ptr_[i] /= rhs.ptr_[i];
            } else {
                backend::div<T, total_size>(data(), data(), rhs.data());
            }
            return *this;
        }

        constexpr Tensor &operator*=(T scalar) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < total_size; ++i)
                    ptr_[i] *= scalar;
            } else {
                backend::mul_scalar<T, total_size>(data(), data(), scalar);
            }
            return *this;
        }

        constexpr Tensor &operator/=(T scalar) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < total_size; ++i)
                    ptr_[i] /= scalar;
            } else {
                backend::div_scalar<T, total_size>(data(), data(), scalar);
            }
            return *this;
        }

        // Unary negation - writes negated values to output pointer
        constexpr void negate_to(T *out) const noexcept {
            for (size_type i = 0; i < total_size; ++i)
                out[i] = -ptr_[i];
        }

        // ==========================================================================
        // Shape Manipulation
        // ==========================================================================

        // Reshape to new dimensions (must have same total size)
        // Returns a new view over the same data with different indexing
        // Example: Tensor<float, 2, 3, 4>.reshape<3, 8>() -> Tensor<float, 3, 8>
        template <std::size_t... NewDims> constexpr Tensor<T, NewDims...> reshape() const noexcept {
            static_assert((NewDims * ...) == total_size,
                          "reshape(): new dimensions must have same total size as original");
            static_assert(sizeof...(NewDims) >= 3, "reshape() for Tensor must result in rank >= 3");

            return Tensor<T, NewDims...>(ptr_);
        }

      private:
        T *ptr_;

        // Helper to compute linear index from multi-dimensional indices (row-major order)
        template <typename... Indices> static constexpr size_type linear_index(Indices... indices) noexcept {
            std::array<size_type, rank> idx = {static_cast<size_type>(indices)...};
            size_type linear = 0;
            size_type stride = 1;
            for (size_type i = rank; i > 0; --i) {
                linear += idx[i - 1] * stride;
                stride *= dims[i - 1];
            }
            return linear;
        }

        // Bounds checking helper
        template <typename... Indices> static constexpr void check_bounds(Indices... indices) {
            std::array<size_type, rank> idx = {static_cast<size_type>(indices)...};
            for (size_type i = 0; i < rank; ++i) {
                if (idx[i] >= dims[i]) {
                    throw std::out_of_range("Tensor index out of bounds");
                }
            }
        }
    };

    // Binary ops (element-wise) - write results to output pointer
    template <typename T, std::size_t... Dims>
    constexpr void add(T *out, const Tensor<T, Dims...> &lhs, const Tensor<T, Dims...> &rhs) noexcept {
        constexpr auto N = Tensor<T, Dims...>::total_size;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                out[i] = lhs[i] + rhs[i];
        } else {
            backend::add<T, N>(out, lhs.data(), rhs.data());
        }
    }

    template <typename T, std::size_t... Dims>
    constexpr void sub(T *out, const Tensor<T, Dims...> &lhs, const Tensor<T, Dims...> &rhs) noexcept {
        constexpr auto N = Tensor<T, Dims...>::total_size;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                out[i] = lhs[i] - rhs[i];
        } else {
            backend::sub<T, N>(out, lhs.data(), rhs.data());
        }
    }

    template <typename T, std::size_t... Dims>
    constexpr void mul(T *out, const Tensor<T, Dims...> &lhs, const Tensor<T, Dims...> &rhs) noexcept {
        constexpr auto N = Tensor<T, Dims...>::total_size;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                out[i] = lhs[i] * rhs[i];
        } else {
            backend::mul<T, N>(out, lhs.data(), rhs.data());
        }
    }

    template <typename T, std::size_t... Dims>
    constexpr void div(T *out, const Tensor<T, Dims...> &lhs, const Tensor<T, Dims...> &rhs) noexcept {
        constexpr auto N = Tensor<T, Dims...>::total_size;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                out[i] = lhs[i] / rhs[i];
        } else {
            backend::div<T, N>(out, lhs.data(), rhs.data());
        }
    }

    // Scalar ops - write results to output pointer
    template <typename T, std::size_t... Dims>
    constexpr void mul_scalar(T *out, const Tensor<T, Dims...> &lhs, T scalar) noexcept {
        constexpr auto N = Tensor<T, Dims...>::total_size;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                out[i] = lhs[i] * scalar;
        } else {
            backend::mul_scalar<T, N>(out, lhs.data(), scalar);
        }
    }

    template <typename T, std::size_t... Dims>
    constexpr void div_scalar(T *out, const Tensor<T, Dims...> &lhs, T scalar) noexcept {
        constexpr auto N = Tensor<T, Dims...>::total_size;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                out[i] = lhs[i] / scalar;
        } else {
            backend::div_scalar<T, N>(out, lhs.data(), scalar);
        }
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
        constexpr auto r = Tensor<T, Dims...>::rank;
        auto shape = Tensor<T, Dims...>::shape();

        os << "Tensor<";
        for (std::size_t i = 0; i < r; ++i) {
            os << shape[i];
            if (i < r - 1)
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
    // Type conversion - cast_to() - writes to output pointer
    // =============================================================================

    template <typename U, typename T, std::size_t... Dims> void cast_to(U *out, const Tensor<T, Dims...> &t) noexcept {
        constexpr auto N = Tensor<T, Dims...>::total_size;
        for (std::size_t i = 0; i < N; ++i) {
            out[i] = static_cast<U>(t[i]);
        }
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
