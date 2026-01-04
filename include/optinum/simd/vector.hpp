#pragma once

#include <datapod/datapod.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/backend/norm.hpp>
#include <optinum/simd/backend/reduce.hpp>
#include <optinum/simd/debug.hpp>

#include <iostream>
#include <type_traits>
#include <variant>

namespace optinum::simd {

    // Import Dynamic constant from datapod
    using dp::mat::Dynamic;

    // Vector: Non-owning view over 1D arrays with SIMD-accelerated operations
    //
    // This is a lightweight view type that wraps existing data (datapod::mat::vector
    // or raw T* pointers). It does NOT own any data - the underlying storage must
    // outlive the view.
    //
    // Template parameter N can be:
    //   - Fixed size: Vector<double, 10>  (compile-time size)
    //   - Dynamic size: Vector<double, Dynamic> (runtime size)
    template <typename T, std::size_t N> class Vector {
        static_assert(N > 0 || N == Dynamic, "Vector size must be > 0 or Dynamic");
        static_assert(std::is_arithmetic_v<T>, "Vector<T, N> requires arithmetic type");

      public:
        using value_type = T;
        using pod_type = dp::mat::Vector<T, N>;
        using size_type = std::size_t;
        using reference = T &;
        using const_reference = const T &;
        using pointer = T *;
        using const_pointer = const T *;
        using iterator = T *;
        using const_iterator = const T *;

        static constexpr size_type extent = N;

      private:
        // Pointer to data (null for invalid/default view)
        T *ptr_ = nullptr;
        // For Dynamic vectors, store the size
        [[no_unique_address]] std::conditional_t<N == Dynamic, size_type, std::monostate> size_{};

      public:
        // Default constructor - creates null view
        constexpr Vector() noexcept = default;

        // Constructor from raw pointer + size (for Dynamic vectors)
        template <std::size_t M = N, typename = std::enable_if_t<M == Dynamic>>
        constexpr Vector(T *ptr, size_type size) noexcept : ptr_(ptr), size_(size) {}

        // Constructor from raw pointer (for fixed-size vectors)
        template <std::size_t M = N, typename = std::enable_if_t<M != Dynamic>>
        constexpr explicit Vector(T *ptr) noexcept : ptr_(ptr) {}

        // Constructor from non-const pod_type reference (creates view over pod's data)
        constexpr Vector(pod_type &pod) noexcept : ptr_(pod.data()) {
            if constexpr (N == Dynamic) {
                size_ = pod.size();
            }
        }

        // Constructor from const pod_type reference
        // Note: Creates mutable view; user must ensure const-correctness
        constexpr Vector(const pod_type &pod) noexcept : ptr_(const_cast<T *>(pod.data())) {
            if constexpr (N == Dynamic) {
                size_ = pod.size();
            }
        }

        // Copy constructor - copies the view (shallow copy)
        constexpr Vector(const Vector &other) noexcept = default;

        // Move constructor
        constexpr Vector(Vector &&other) noexcept = default;

        // Copy assignment - copies the view (shallow copy)
        constexpr Vector &operator=(const Vector &other) noexcept = default;

        // Move assignment
        constexpr Vector &operator=(Vector &&other) noexcept = default;

        // Check if view is valid (non-null)
        constexpr bool valid() const noexcept { return ptr_ != nullptr; }
        constexpr explicit operator bool() const noexcept { return valid(); }

        // Access to underlying pointer
        constexpr pointer data() noexcept { return ptr_; }
        constexpr const_pointer data() const noexcept { return ptr_; }

        constexpr reference operator[](size_type i) noexcept {
            OPTINUM_ASSERT_BOUNDS(i, size());
            return ptr_[i];
        }
        constexpr const_reference operator[](size_type i) const noexcept {
            OPTINUM_ASSERT_BOUNDS(i, size());
            return ptr_[i];
        }

        constexpr reference at(size_type i) {
            if (i >= size()) {
                throw std::out_of_range("Vector::at index out of range");
            }
            return ptr_[i];
        }
        constexpr const_reference at(size_type i) const {
            if (i >= size()) {
                throw std::out_of_range("Vector::at index out of range");
            }
            return ptr_[i];
        }

        constexpr reference front() noexcept { return ptr_[0]; }
        constexpr const_reference front() const noexcept { return ptr_[0]; }

        constexpr reference back() noexcept { return ptr_[size() - 1]; }
        constexpr const_reference back() const noexcept { return ptr_[size() - 1]; }

        constexpr size_type size() const noexcept {
            if constexpr (N == Dynamic) {
                return size_;
            } else {
                return N;
            }
        }

        constexpr bool empty() const noexcept {
            if constexpr (N == Dynamic) {
                return size_ == 0;
            } else {
                return false;
            }
        }

        constexpr iterator begin() noexcept { return ptr_; }
        constexpr const_iterator begin() const noexcept { return ptr_; }
        constexpr const_iterator cbegin() const noexcept { return ptr_; }

        constexpr iterator end() noexcept { return ptr_ + size(); }
        constexpr const_iterator end() const noexcept { return ptr_ + size(); }
        constexpr const_iterator cend() const noexcept { return ptr_ + size(); }

        // In-place fill (modifies underlying data)
        constexpr Vector &fill(const T &value) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i) {
                    ptr_[i] = value;
                }
            } else {
                if constexpr (N == Dynamic) {
                    backend::fill_runtime(ptr_, size(), value);
                } else {
                    backend::fill<T, N>(ptr_, value);
                }
            }
            return *this;
        }

        // Fill with sequential values: 0, 1, 2, ...
        constexpr Vector &iota() noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i) {
                    ptr_[i] = static_cast<T>(i);
                }
            } else {
                backend::iota<T, N>(ptr_, T{0}, T{1});
            }
            return *this;
        }

        // Fill with sequential values starting from 'start'
        constexpr Vector &iota(T start) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i) {
                    ptr_[i] = start + static_cast<T>(i);
                }
            } else {
                backend::iota<T, N>(ptr_, start, T{1});
            }
            return *this;
        }

        // Fill with sequential values with custom start and step
        constexpr Vector &iota(T start, T step) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i) {
                    ptr_[i] = start + static_cast<T>(i) * step;
                }
            } else {
                backend::iota<T, N>(ptr_, start, step);
            }
            return *this;
        }

        // Reverse elements in-place
        constexpr Vector &reverse() noexcept {
            const size_type n = size();
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < n / 2; ++i) {
                    std::swap(ptr_[i], ptr_[n - 1 - i]);
                }
            } else {
                backend::reverse<T, N>(ptr_);
            }
            return *this;
        }

        // Compound assignment operators (modify in-place)
        constexpr Vector &operator+=(const Vector &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i)
                    ptr_[i] += rhs.ptr_[i];
            } else {
                if constexpr (N == Dynamic) {
                    backend::add_runtime<T>(ptr_, ptr_, rhs.ptr_, size());
                } else {
                    backend::add<T, N>(ptr_, ptr_, rhs.ptr_);
                }
            }
            return *this;
        }

        constexpr Vector &operator-=(const Vector &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i)
                    ptr_[i] -= rhs.ptr_[i];
            } else {
                if constexpr (N == Dynamic) {
                    backend::sub_runtime<T>(ptr_, ptr_, rhs.ptr_, size());
                } else {
                    backend::sub<T, N>(ptr_, ptr_, rhs.ptr_);
                }
            }
            return *this;
        }

        constexpr Vector &operator*=(const Vector &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i)
                    ptr_[i] *= rhs.ptr_[i];
            } else {
                if constexpr (N == Dynamic) {
                    backend::mul_runtime<T>(ptr_, ptr_, rhs.ptr_, size());
                } else {
                    backend::mul<T, N>(ptr_, ptr_, rhs.ptr_);
                }
            }
            return *this;
        }

        constexpr Vector &operator/=(const Vector &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i)
                    ptr_[i] /= rhs.ptr_[i];
            } else {
                if constexpr (N == Dynamic) {
                    backend::div_runtime<T>(ptr_, ptr_, rhs.ptr_, size());
                } else {
                    backend::div<T, N>(ptr_, ptr_, rhs.ptr_);
                }
            }
            return *this;
        }

        constexpr Vector &operator*=(T scalar) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i)
                    ptr_[i] *= scalar;
            } else {
                backend::mul_scalar<T, N>(ptr_, ptr_, scalar);
            }
            return *this;
        }

        constexpr Vector &operator/=(T scalar) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i)
                    ptr_[i] /= scalar;
            } else {
                backend::div_scalar<T, N>(ptr_, ptr_, scalar);
            }
            return *this;
        }

        // Unary negation - writes negated values to output pointer
        constexpr void negate_to(T *out) const noexcept {
            for (size_type i = 0; i < size(); ++i)
                out[i] = -ptr_[i];
        }
    };

    // =============================================================================
    // Free functions that write to output pointers (non-allocating)
    // =============================================================================

    // Binary ops (element-wise) - write results to output pointer
    template <typename T, std::size_t N>
    constexpr void add(T *out, const Vector<T, N> &lhs, const Vector<T, N> &rhs) noexcept {
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < lhs.size(); ++i)
                out[i] = lhs[i] + rhs[i];
        } else {
            if constexpr (N == Dynamic) {
                backend::add_runtime<T>(out, lhs.data(), rhs.data(), lhs.size());
            } else {
                backend::add<T, N>(out, lhs.data(), rhs.data());
            }
        }
    }

    template <typename T, std::size_t N>
    constexpr void sub(T *out, const Vector<T, N> &lhs, const Vector<T, N> &rhs) noexcept {
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < lhs.size(); ++i)
                out[i] = lhs[i] - rhs[i];
        } else {
            if constexpr (N == Dynamic) {
                backend::sub_runtime<T>(out, lhs.data(), rhs.data(), lhs.size());
            } else {
                backend::sub<T, N>(out, lhs.data(), rhs.data());
            }
        }
    }

    template <typename T, std::size_t N>
    constexpr void mul(T *out, const Vector<T, N> &lhs, const Vector<T, N> &rhs) noexcept {
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < lhs.size(); ++i)
                out[i] = lhs[i] * rhs[i];
        } else {
            if constexpr (N == Dynamic) {
                backend::mul_runtime<T>(out, lhs.data(), rhs.data(), lhs.size());
            } else {
                backend::mul<T, N>(out, lhs.data(), rhs.data());
            }
        }
    }

    template <typename T, std::size_t N>
    constexpr void div(T *out, const Vector<T, N> &lhs, const Vector<T, N> &rhs) noexcept {
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < lhs.size(); ++i)
                out[i] = lhs[i] / rhs[i];
        } else {
            if constexpr (N == Dynamic) {
                backend::div_runtime<T>(out, lhs.data(), rhs.data(), lhs.size());
            } else {
                backend::div<T, N>(out, lhs.data(), rhs.data());
            }
        }
    }

    // Scalar ops - write results to output pointer
    template <typename T, std::size_t N> constexpr void mul_scalar(T *out, const Vector<T, N> &lhs, T scalar) noexcept {
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < lhs.size(); ++i)
                out[i] = lhs[i] * scalar;
        } else {
            if constexpr (N == Dynamic) {
                backend::mul_scalar_runtime<T>(out, lhs.data(), scalar, lhs.size());
            } else {
                backend::mul_scalar<T, N>(out, lhs.data(), scalar);
            }
        }
    }

    template <typename T, std::size_t N> constexpr void div_scalar(T *out, const Vector<T, N> &lhs, T scalar) noexcept {
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < lhs.size(); ++i)
                out[i] = lhs[i] / scalar;
        } else {
            if constexpr (N == Dynamic) {
                backend::div_scalar_runtime<T>(out, lhs.data(), scalar, lhs.size());
            } else {
                backend::div_scalar<T, N>(out, lhs.data(), scalar);
            }
        }
    }

    // Normalize - write result to output pointer
    template <typename T, std::size_t N> void normalize_to(T *out, const Vector<T, N> &v) noexcept {
        backend::normalize<T, N>(out, v.data());
    }

    // =============================================================================
    // Comparisons
    // =============================================================================

    template <typename T, std::size_t N>
    constexpr bool operator==(const Vector<T, N> &lhs, const Vector<T, N> &rhs) noexcept {
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            if (lhs[i] != rhs[i])
                return false;
        }
        return true;
    }

    template <typename T, std::size_t N>
    constexpr bool operator!=(const Vector<T, N> &lhs, const Vector<T, N> &rhs) noexcept {
        return !(lhs == rhs);
    }

    // =============================================================================
    // Common operations (return scalars, no allocation needed)
    // =============================================================================

    template <typename T, std::size_t N>
    [[nodiscard]] constexpr T dot(const Vector<T, N> &lhs, const Vector<T, N> &rhs) noexcept {
        if (std::is_constant_evaluated()) {
            T result{};
            for (std::size_t i = 0; i < lhs.size(); ++i)
                result += lhs[i] * rhs[i];
            return result;
        }
        if constexpr (N == Dynamic) {
            return backend::dot_runtime<T>(lhs.data(), rhs.data(), lhs.size());
        } else {
            return backend::dot<T, N>(lhs.data(), rhs.data());
        }
    }

    template <typename T, std::size_t N> [[nodiscard]] constexpr T sum(const Vector<T, N> &v) noexcept {
        if (std::is_constant_evaluated()) {
            T result{};
            for (std::size_t i = 0; i < v.size(); ++i)
                result += v[i];
            return result;
        }
        if constexpr (N == Dynamic) {
            return backend::reduce_sum_runtime<T>(v.data(), v.size());
        } else {
            return backend::reduce_sum<T, N>(v.data());
        }
    }

    template <typename T, std::size_t N> [[nodiscard]] T norm(const Vector<T, N> &v) noexcept {
        if constexpr (N == Dynamic) {
            return backend::norm_l2_runtime<T>(v.data(), v.size());
        } else {
            return backend::norm_l2<T, N>(v.data());
        }
    }

    template <typename T, std::size_t N> [[nodiscard]] T norm_squared(const Vector<T, N> &v) noexcept {
        return dot(v, v);
    }

    // =============================================================================
    // I/O - Stream output operator
    // =============================================================================

    template <typename T, std::size_t N> std::ostream &operator<<(std::ostream &os, const Vector<T, N> &v) {
        os << "[";
        for (std::size_t i = 0; i < v.size(); ++i) {
            os << v[i];
            if (i < v.size() - 1)
                os << ", ";
        }
        os << "]";
        return os;
    }

    // =============================================================================
    // Type conversion - cast_to() - writes to output pointer
    // =============================================================================

    template <typename U, typename T, std::size_t N> void cast_to(U *out, const Vector<T, N> &v) noexcept {
        for (std::size_t i = 0; i < v.size(); ++i) {
            out[i] = static_cast<U>(v[i]);
        }
    }

    // =============================================================================
    // Type aliases
    // =============================================================================

    template <typename T> using Vector1 = Vector<T, 1>;
    template <typename T> using Vector2 = Vector<T, 2>;
    template <typename T> using Vector3 = Vector<T, 3>;
    template <typename T> using Vector4 = Vector<T, 4>;
    template <typename T> using Vector6 = Vector<T, 6>;

    using Vector3f = Vector<float, 3>;
    using Vector3d = Vector<double, 3>;
    using Vector4f = Vector<float, 4>;
    using Vector4d = Vector<double, 4>;
    using Vector6f = Vector<float, 6>;
    using Vector6d = Vector<double, 6>;

    // Dynamic vector aliases
    template <typename T> using VectorX = Vector<T, Dynamic>;
    using VectorXf = Vector<float, Dynamic>;
    using VectorXd = Vector<double, Dynamic>;

} // namespace optinum::simd
