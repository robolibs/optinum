#pragma once

#include <datapod/matrix/tensor.hpp>

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
            for (size_type i = 0; i < N; ++i)
                pod_[i] += rhs.pod_[i];
            return *this;
        }

        constexpr Tensor &operator-=(const Tensor &rhs) noexcept {
            for (size_type i = 0; i < N; ++i)
                pod_[i] -= rhs.pod_[i];
            return *this;
        }

        constexpr Tensor &operator*=(const Tensor &rhs) noexcept {
            for (size_type i = 0; i < N; ++i)
                pod_[i] *= rhs.pod_[i];
            return *this;
        }

        constexpr Tensor &operator/=(const Tensor &rhs) noexcept {
            for (size_type i = 0; i < N; ++i)
                pod_[i] /= rhs.pod_[i];
            return *this;
        }

        constexpr Tensor &operator*=(T scalar) noexcept {
            for (size_type i = 0; i < N; ++i)
                pod_[i] *= scalar;
            return *this;
        }

        constexpr Tensor &operator/=(T scalar) noexcept {
            for (size_type i = 0; i < N; ++i)
                pod_[i] /= scalar;
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
        for (std::size_t i = 0; i < N; ++i)
            result[i] = lhs[i] + rhs[i];
        return result;
    }

    template <typename T, std::size_t N>
    constexpr Tensor<T, N> operator-(const Tensor<T, N> &lhs, const Tensor<T, N> &rhs) noexcept {
        Tensor<T, N> result;
        for (std::size_t i = 0; i < N; ++i)
            result[i] = lhs[i] - rhs[i];
        return result;
    }

    template <typename T, std::size_t N>
    constexpr Tensor<T, N> operator*(const Tensor<T, N> &lhs, const Tensor<T, N> &rhs) noexcept {
        Tensor<T, N> result;
        for (std::size_t i = 0; i < N; ++i)
            result[i] = lhs[i] * rhs[i];
        return result;
    }

    template <typename T, std::size_t N>
    constexpr Tensor<T, N> operator/(const Tensor<T, N> &lhs, const Tensor<T, N> &rhs) noexcept {
        Tensor<T, N> result;
        for (std::size_t i = 0; i < N; ++i)
            result[i] = lhs[i] / rhs[i];
        return result;
    }

    // Scalar ops
    template <typename T, std::size_t N> constexpr Tensor<T, N> operator*(const Tensor<T, N> &lhs, T scalar) noexcept {
        Tensor<T, N> result;
        for (std::size_t i = 0; i < N; ++i)
            result[i] = lhs[i] * scalar;
        return result;
    }

    template <typename T, std::size_t N> constexpr Tensor<T, N> operator*(T scalar, const Tensor<T, N> &rhs) noexcept {
        return rhs * scalar;
    }

    template <typename T, std::size_t N> constexpr Tensor<T, N> operator/(const Tensor<T, N> &lhs, T scalar) noexcept {
        Tensor<T, N> result;
        for (std::size_t i = 0; i < N; ++i)
            result[i] = lhs[i] / scalar;
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
        T result{};
        for (std::size_t i = 0; i < N; ++i)
            result += lhs[i] * rhs[i];
        return result;
    }

    template <typename T, std::size_t N> constexpr T sum(const Tensor<T, N> &t) noexcept {
        T result{};
        for (std::size_t i = 0; i < N; ++i)
            result += t[i];
        return result;
    }

    template <typename T, std::size_t N> T norm(const Tensor<T, N> &t) noexcept { return std::sqrt(dot(t, t)); }

    template <typename T, std::size_t N> Tensor<T, N> normalized(const Tensor<T, N> &t) noexcept {
        T n = norm(t);
        return (n > T{}) ? t / n : t;
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
