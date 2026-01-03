#pragma once

#include <datapod/matrix/scalar.hpp>

namespace optinum::simd {

    namespace dp = ::datapod;

    // Scalar: non-owning view over a single arithmetic value
    // Non-owning view over datapod::mat::Scalar<T> or raw T* data
    template <typename T> class Scalar {
        static_assert(std::is_arithmetic_v<T>, "Scalar<T> requires arithmetic type");

      public:
        using value_type = T;
        using pod_type = dp::mat::Scalar<T>;
        using pointer = T *;
        using const_pointer = const T *;
        using reference = T &;
        using const_reference = const T &;

        // Default constructor (null view)
        constexpr Scalar() noexcept : ptr_(nullptr) {}

        // Constructor from raw pointer (non-owning view)
        constexpr explicit Scalar(T *ptr) noexcept : ptr_(ptr) {}

        // Constructor from pod_type reference (non-owning view)
        constexpr Scalar(pod_type &pod) noexcept : ptr_(&pod.value) {}
        constexpr Scalar(const pod_type &pod) noexcept : ptr_(const_cast<T *>(&pod.value)) {}

        // Access to underlying pointer
        constexpr pointer data() noexcept { return ptr_; }
        constexpr const_pointer data() const noexcept { return ptr_; }

        // Check if view is valid (non-null)
        constexpr bool valid() const noexcept { return ptr_ != nullptr; }
        constexpr explicit operator bool() const noexcept { return valid(); }

        constexpr reference get() noexcept { return *ptr_; }
        constexpr const_reference get() const noexcept { return *ptr_; }

        // Implicit conversion to T (for convenience in expressions)
        constexpr operator T() const noexcept { return *ptr_; }

        // Assignment from value
        constexpr Scalar &operator=(T val) noexcept {
            *ptr_ = val;
            return *this;
        }

        // Compound assignment
        constexpr Scalar &operator+=(const Scalar &rhs) noexcept {
            *ptr_ += *rhs.ptr_;
            return *this;
        }
        constexpr Scalar &operator-=(const Scalar &rhs) noexcept {
            *ptr_ -= *rhs.ptr_;
            return *this;
        }
        constexpr Scalar &operator*=(const Scalar &rhs) noexcept {
            *ptr_ *= *rhs.ptr_;
            return *this;
        }
        constexpr Scalar &operator/=(const Scalar &rhs) noexcept {
            *ptr_ /= *rhs.ptr_;
            return *this;
        }

        // Compound assignment with raw T
        constexpr Scalar &operator+=(T rhs) noexcept {
            *ptr_ += rhs;
            return *this;
        }
        constexpr Scalar &operator-=(T rhs) noexcept {
            *ptr_ -= rhs;
            return *this;
        }
        constexpr Scalar &operator*=(T rhs) noexcept {
            *ptr_ *= rhs;
            return *this;
        }
        constexpr Scalar &operator/=(T rhs) noexcept {
            *ptr_ /= rhs;
            return *this;
        }

        // Negate to output pointer (non-owning pattern)
        constexpr void negate_to(T *out) const noexcept { *out = -(*ptr_); }

      private:
        T *ptr_;
    };

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
