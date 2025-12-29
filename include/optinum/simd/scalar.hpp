#pragma once

#include <datapod/matrix/scalar.hpp>

namespace optinum::simd {

    namespace dp = ::datapod;

    template <typename T> class Scalar {
        static_assert(std::is_arithmetic_v<T>, "Scalar<T> requires arithmetic type");

      public:
        using value_type = T;
        using pod_type = dp::mat::scalar<T>;

        constexpr Scalar() noexcept = default;

        // POD constructors (implicit to allow dp::mat::scalar -> simd::Scalar conversion)
        constexpr Scalar(const pod_type &pod) noexcept : pod_(pod) {}
        constexpr Scalar(pod_type &&pod) noexcept : pod_(static_cast<pod_type &&>(pod)) {}
        constexpr Scalar(T val) noexcept : pod_(val) {}

        constexpr pod_type &pod() noexcept { return pod_; }
        constexpr const pod_type &pod() const noexcept { return pod_; }

        // Implicit conversion to pod_type (allows simd::Scalar -> dp::mat::scalar)
        constexpr operator pod_type &() noexcept { return pod_; }
        constexpr operator const pod_type &() const noexcept { return pod_; }

        constexpr T &get() noexcept { return pod_.value; }
        constexpr const T &get() const noexcept { return pod_.value; }

        constexpr operator T() const noexcept { return pod_.value; }

        // Compound assignment
        constexpr Scalar &operator+=(const Scalar &rhs) noexcept {
            pod_ += rhs.pod_;
            return *this;
        }
        constexpr Scalar &operator-=(const Scalar &rhs) noexcept {
            pod_ -= rhs.pod_;
            return *this;
        }
        constexpr Scalar &operator*=(const Scalar &rhs) noexcept {
            pod_ *= rhs.pod_;
            return *this;
        }
        constexpr Scalar &operator/=(const Scalar &rhs) noexcept {
            pod_ /= rhs.pod_;
            return *this;
        }

        // Unary
        constexpr Scalar operator-() const noexcept { return Scalar{-pod_.value}; }
        constexpr Scalar operator+() const noexcept { return *this; }

      private:
        pod_type pod_{};
    };

    // Binary ops
    template <typename T> constexpr Scalar<T> operator+(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept {
        return Scalar<T>{lhs.get() + rhs.get()};
    }

    template <typename T> constexpr Scalar<T> operator-(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept {
        return Scalar<T>{lhs.get() - rhs.get()};
    }

    template <typename T> constexpr Scalar<T> operator*(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept {
        return Scalar<T>{lhs.get() * rhs.get()};
    }

    template <typename T> constexpr Scalar<T> operator/(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept {
        return Scalar<T>{lhs.get() / rhs.get()};
    }

    // Comparisons
    template <typename T> constexpr bool operator==(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept {
        return lhs.get() == rhs.get();
    }

    template <typename T> constexpr bool operator!=(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept {
        return lhs.get() != rhs.get();
    }

    template <typename T> constexpr bool operator<(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept {
        return lhs.get() < rhs.get();
    }

    template <typename T> constexpr bool operator<=(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept {
        return lhs.get() <= rhs.get();
    }

    template <typename T> constexpr bool operator>(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept {
        return lhs.get() > rhs.get();
    }

    template <typename T> constexpr bool operator>=(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept {
        return lhs.get() >= rhs.get();
    }

} // namespace optinum::simd
