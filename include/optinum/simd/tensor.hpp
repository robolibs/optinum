#pragma once

#include <datapod/matrix/tensor.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/backend/norm.hpp>
#include <optinum/simd/backend/reduce.hpp>

#include <type_traits>

namespace optinum::simd {

    namespace dp = ::datapod;

    template <typename T, std::size_t N> class Tensor {
        static_assert(N > 0, "Tensor size must be > 0");
        static_assert(std::is_arithmetic_v<T>, "Tensor<T, N> requires arithmetic type");

      public:
        using value_type = T;
        using pod_type = dp::tensor<T, N>;
        using size_type = std::size_t;
        using reference = T &;
        using const_reference = const T &;
        using pointer = T *;
        using const_pointer = const T *;
        using iterator = T *;
        using const_iterator = const T *;

        static constexpr size_type extent = N;

        constexpr Tensor() noexcept = default;
        constexpr explicit Tensor(const pod_type &pod) noexcept : pod_(pod) {}
        constexpr explicit Tensor(pod_type &&pod) noexcept : pod_(static_cast<pod_type &&>(pod)) {}

        constexpr pod_type &pod() noexcept { return pod_; }
        constexpr const pod_type &pod() const noexcept { return pod_; }

        constexpr pointer data() noexcept { return pod_.data(); }
        constexpr const_pointer data() const noexcept { return pod_.data(); }

        constexpr reference operator[](size_type i) noexcept { return pod_[i]; }
        constexpr const_reference operator[](size_type i) const noexcept { return pod_[i]; }

        constexpr reference at(size_type i) { return pod_.at(i); }
        constexpr const_reference at(size_type i) const { return pod_.at(i); }

        constexpr reference front() noexcept { return pod_.front(); }
        constexpr const_reference front() const noexcept { return pod_.front(); }

        constexpr reference back() noexcept { return pod_.back(); }
        constexpr const_reference back() const noexcept { return pod_.back(); }

        static constexpr size_type size() noexcept { return N; }
        static constexpr bool empty() noexcept { return false; }

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

        // Compound assignment
        constexpr Tensor &operator+=(const Tensor &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < N; ++i)
                    pod_[i] += rhs.pod_[i];
            } else {
                backend::add<T, N>(data(), data(), rhs.data());
            }
            return *this;
        }

        constexpr Tensor &operator-=(const Tensor &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < N; ++i)
                    pod_[i] -= rhs.pod_[i];
            } else {
                backend::sub<T, N>(data(), data(), rhs.data());
            }
            return *this;
        }

        constexpr Tensor &operator*=(const Tensor &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < N; ++i)
                    pod_[i] *= rhs.pod_[i];
            } else {
                backend::mul<T, N>(data(), data(), rhs.data());
            }
            return *this;
        }

        constexpr Tensor &operator/=(const Tensor &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < N; ++i)
                    pod_[i] /= rhs.pod_[i];
            } else {
                backend::div<T, N>(data(), data(), rhs.data());
            }
            return *this;
        }

        constexpr Tensor &operator*=(T scalar) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < N; ++i)
                    pod_[i] *= scalar;
            } else {
                backend::mul_scalar<T, N>(data(), data(), scalar);
            }
            return *this;
        }

        constexpr Tensor &operator/=(T scalar) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < N; ++i)
                    pod_[i] /= scalar;
            } else {
                backend::div_scalar<T, N>(data(), data(), scalar);
            }
            return *this;
        }

        // Unary
        constexpr Tensor operator-() const noexcept {
            Tensor result;
            for (size_type i = 0; i < N; ++i)
                result.pod_[i] = -pod_[i];
            return result;
        }

        constexpr Tensor operator+() const noexcept { return *this; }

      private:
        pod_type pod_{};
    };

    // Binary ops (element-wise)
    template <typename T, std::size_t N>
    constexpr Tensor<T, N> operator+(const Tensor<T, N> &lhs, const Tensor<T, N> &rhs) noexcept {
        Tensor<T, N> result;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                result[i] = lhs[i] + rhs[i];
        } else {
            backend::add<T, N>(result.data(), lhs.data(), rhs.data());
        }
        return result;
    }

    template <typename T, std::size_t N>
    constexpr Tensor<T, N> operator-(const Tensor<T, N> &lhs, const Tensor<T, N> &rhs) noexcept {
        Tensor<T, N> result;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                result[i] = lhs[i] - rhs[i];
        } else {
            backend::sub<T, N>(result.data(), lhs.data(), rhs.data());
        }
        return result;
    }

    template <typename T, std::size_t N>
    constexpr Tensor<T, N> operator*(const Tensor<T, N> &lhs, const Tensor<T, N> &rhs) noexcept {
        Tensor<T, N> result;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                result[i] = lhs[i] * rhs[i];
        } else {
            backend::mul<T, N>(result.data(), lhs.data(), rhs.data());
        }
        return result;
    }

    template <typename T, std::size_t N>
    constexpr Tensor<T, N> operator/(const Tensor<T, N> &lhs, const Tensor<T, N> &rhs) noexcept {
        Tensor<T, N> result;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                result[i] = lhs[i] / rhs[i];
        } else {
            backend::div<T, N>(result.data(), lhs.data(), rhs.data());
        }
        return result;
    }

    // Scalar ops
    template <typename T, std::size_t N> constexpr Tensor<T, N> operator*(const Tensor<T, N> &lhs, T scalar) noexcept {
        Tensor<T, N> result;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                result[i] = lhs[i] * scalar;
        } else {
            backend::mul_scalar<T, N>(result.data(), lhs.data(), scalar);
        }
        return result;
    }

    template <typename T, std::size_t N> constexpr Tensor<T, N> operator*(T scalar, const Tensor<T, N> &rhs) noexcept {
        return rhs * scalar;
    }

    template <typename T, std::size_t N> constexpr Tensor<T, N> operator/(const Tensor<T, N> &lhs, T scalar) noexcept {
        Tensor<T, N> result;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                result[i] = lhs[i] / scalar;
        } else {
            backend::div_scalar<T, N>(result.data(), lhs.data(), scalar);
        }
        return result;
    }

    // Comparisons
    template <typename T, std::size_t N>
    constexpr bool operator==(const Tensor<T, N> &lhs, const Tensor<T, N> &rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) {
            if (lhs[i] != rhs[i])
                return false;
        }
        return true;
    }

    template <typename T, std::size_t N>
    constexpr bool operator!=(const Tensor<T, N> &lhs, const Tensor<T, N> &rhs) noexcept {
        return !(lhs == rhs);
    }

    // Common operations
    template <typename T, std::size_t N> constexpr T dot(const Tensor<T, N> &lhs, const Tensor<T, N> &rhs) noexcept {
        if (std::is_constant_evaluated()) {
            T result{};
            for (std::size_t i = 0; i < N; ++i)
                result += lhs[i] * rhs[i];
            return result;
        }
        return backend::dot<T, N>(lhs.data(), rhs.data());
    }

    template <typename T, std::size_t N> constexpr T sum(const Tensor<T, N> &t) noexcept {
        if (std::is_constant_evaluated()) {
            T result{};
            for (std::size_t i = 0; i < N; ++i)
                result += t[i];
            return result;
        }
        return backend::reduce_sum<T, N>(t.data());
    }

    template <typename T, std::size_t N> T norm(const Tensor<T, N> &t) noexcept {
        return backend::norm_l2<T, N>(t.data());
    }

    template <typename T, std::size_t N> Tensor<T, N> normalized(const Tensor<T, N> &t) noexcept {
        Tensor<T, N> result;
        backend::normalize<T, N>(result.data(), t.data());
        return result;
    }

    // Type aliases
    template <typename T> using Tensor1 = Tensor<T, 1>;
    template <typename T> using Tensor2 = Tensor<T, 2>;
    template <typename T> using Tensor3 = Tensor<T, 3>;
    template <typename T> using Tensor4 = Tensor<T, 4>;
    template <typename T> using Tensor6 = Tensor<T, 6>;

    using Tensor3f = Tensor<float, 3>;
    using Tensor3d = Tensor<double, 3>;
    using Tensor4f = Tensor<float, 4>;
    using Tensor4d = Tensor<double, 4>;
    using Tensor6f = Tensor<float, 6>;
    using Tensor6d = Tensor<double, 6>;

} // namespace optinum::simd
