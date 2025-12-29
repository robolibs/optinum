#pragma once

#include <datapod/matrix/vector.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/backend/norm.hpp>
#include <optinum/simd/backend/reduce.hpp>
#include <optinum/simd/debug.hpp>

#include <iostream>
#include <random>
#include <type_traits>

namespace optinum::simd {

    namespace dp = ::datapod;

    // Import Dynamic constant from datapod
    using dp::mat::Dynamic;

    // Vector: 1D fixed-size array with SIMD-accelerated operations
    // Wraps datapod::mat::vector<T, N>
    // Template parameter N can be:
    //   - Fixed size: Vector<double, 10>  (compile-time size)
    //   - Dynamic size: Vector<double, Dynamic> (runtime size)
    template <typename T, std::size_t N> class Vector {
        static_assert(N > 0 || N == Dynamic, "Vector size must be > 0 or Dynamic");
        static_assert(std::is_arithmetic_v<T>, "Vector<T, N> requires arithmetic type");

      public:
        using value_type = T;
        using pod_type = dp::mat::vector<T, N>;
        using size_type = std::size_t;
        using reference = T &;
        using const_reference = const T &;
        using pointer = T *;
        using const_pointer = const T *;
        using iterator = T *;
        using const_iterator = const T *;

        static constexpr size_type extent = N;

        // Default constructor
        constexpr Vector() noexcept = default;

        // Runtime size constructor (only for Dynamic)
        template <std::size_t M = N, typename = std::enable_if_t<M == Dynamic>>
        explicit Vector(size_type size) : pod_(size) {}

        // Runtime size + fill constructor (only for Dynamic)
        template <std::size_t M = N, typename = std::enable_if_t<M == Dynamic>>
        Vector(size_type size, const T &value) : pod_(size, value) {}

        // Initializer list constructor (for fixed-size vectors)
        template <std::size_t M = N, typename = std::enable_if_t<M != Dynamic>>
        constexpr Vector(std::initializer_list<T> init) {
            OPTINUM_ASSERT(init.size() == N, "Initializer list size must match vector size");
            std::copy(init.begin(), init.end(), pod_.begin());
        }

        // POD constructors (implicit to allow dp::mat::vector -> simd::Vector conversion)
        constexpr Vector(const pod_type &pod) noexcept : pod_(pod) {}
        constexpr Vector(pod_type &&pod) noexcept : pod_(static_cast<pod_type &&>(pod)) {}

        constexpr pod_type &pod() noexcept { return pod_; }
        constexpr const pod_type &pod() const noexcept { return pod_; }

        // Implicit conversion to pod_type (allows simd::Vector -> dp::mat::vector)
        constexpr operator pod_type &() noexcept { return pod_; }
        constexpr operator const pod_type &() const noexcept { return pod_; }

        constexpr pointer data() noexcept { return pod_.data(); }
        constexpr const_pointer data() const noexcept { return pod_.data(); }

        constexpr reference operator[](size_type i) noexcept {
            OPTINUM_ASSERT_BOUNDS(i, N);
            return pod_[i];
        }
        constexpr const_reference operator[](size_type i) const noexcept {
            OPTINUM_ASSERT_BOUNDS(i, N);
            return pod_[i];
        }

        constexpr reference at(size_type i) { return pod_.at(i); }
        constexpr const_reference at(size_type i) const { return pod_.at(i); }

        constexpr reference front() noexcept { return pod_.front(); }
        constexpr const_reference front() const noexcept { return pod_.front(); }

        constexpr reference back() noexcept { return pod_.back(); }
        constexpr const_reference back() const noexcept { return pod_.back(); }

        constexpr size_type size() const noexcept {
            if constexpr (N == Dynamic) {
                return pod_.size();
            } else {
                return N;
            }
        }

        constexpr bool empty() const noexcept {
            if constexpr (N == Dynamic) {
                return pod_.empty();
            } else {
                return false;
            }
        }

        constexpr iterator begin() noexcept { return pod_.begin(); }
        constexpr const_iterator begin() const noexcept { return pod_.begin(); }
        constexpr const_iterator cbegin() const noexcept { return pod_.cbegin(); }

        constexpr iterator end() noexcept { return pod_.end(); }
        constexpr const_iterator end() const noexcept { return pod_.end(); }
        constexpr const_iterator cend() const noexcept { return pod_.cend(); }

        constexpr Vector &fill(const T &value) noexcept {
            if (std::is_constant_evaluated()) {
                pod_.fill(value);
            } else {
                // For Dynamic size, use runtime size instead of compile-time N
                if constexpr (N == Dynamic) {
                    backend::fill_runtime(data(), size(), value);
                } else {
                    backend::fill<T, N>(data(), value);
                }
            }
            return *this;
        }

        // Fill with sequential values: 0, 1, 2, ...
        constexpr Vector &iota() noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i) {
                    pod_[i] = static_cast<T>(i);
                }
            } else {
                backend::iota<T, N>(data(), T{0}, T{1});
            }
            return *this;
        }

        // Fill with sequential values starting from 'start'
        constexpr Vector &iota(T start) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i) {
                    pod_[i] = start + static_cast<T>(i);
                }
            } else {
                backend::iota<T, N>(data(), start, T{1});
            }
            return *this;
        }

        // Fill with sequential values with custom start and step
        constexpr Vector &iota(T start, T step) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i) {
                    pod_[i] = start + static_cast<T>(i) * step;
                }
            } else {
                backend::iota<T, N>(data(), start, step);
            }
            return *this;
        }

        // Reverse elements in-place
        constexpr Vector &reverse() noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < N / 2; ++i) {
                    std::swap(pod_[i], pod_[N - 1 - i]);
                }
            } else {
                backend::reverse<T, N>(data());
            }
            return *this;
        }

        // Static factory: create vector filled with zeros
        static constexpr Vector zeros() noexcept {
            Vector v;
            v.fill(T{0});
            return v;
        }

        // Static factory: create vector filled with ones
        static constexpr Vector ones() noexcept {
            Vector v;
            v.fill(T{1});
            return v;
        }

        // Static factory: create vector with sequential values
        static constexpr Vector arange() noexcept {
            Vector v;
            v.iota();
            return v;
        }

        // Static factory: create vector with sequential values from start
        static constexpr Vector arange(T start) noexcept {
            Vector v;
            v.iota(start);
            return v;
        }

        // Static factory: create vector with sequential values with custom start and step
        static constexpr Vector arange(T start, T step) noexcept {
            Vector v;
            v.iota(start, step);
            return v;
        }

        // Fill with uniform random values in [0, 1) for floating point, [0, max) for integers
        Vector &random() {
            static std::random_device rd;
            static std::mt19937 gen(rd());

            if constexpr (std::is_floating_point_v<T>) {
                std::uniform_real_distribution<T> dis(T{0}, T{1});
                for (size_type i = 0; i < size(); ++i) {
                    pod_[i] = dis(gen);
                }
            } else {
                std::uniform_int_distribution<T> dis(T{0}, std::numeric_limits<T>::max());
                for (size_type i = 0; i < size(); ++i) {
                    pod_[i] = dis(gen);
                }
            }
            return *this;
        }

        // Fill with random integers in [low, high]
        template <typename U = T> std::enable_if_t<std::is_integral_v<U>, Vector &> randint(T low, T high) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_int_distribution<T> dis(low, high);

            for (size_type i = 0; i < size(); ++i) {
                pod_[i] = dis(gen);
            }
            return *this;
        }

        // Compound assignment
        constexpr Vector &operator+=(const Vector &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i)
                    pod_[i] += rhs.pod_[i];
            } else {
                if constexpr (N == Dynamic) {
                    backend::add_runtime<T>(data(), data(), rhs.data(), size());
                } else {
                    backend::add<T, N>(data(), data(), rhs.data());
                }
            }
            return *this;
        }

        constexpr Vector &operator-=(const Vector &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i)
                    pod_[i] -= rhs.pod_[i];
            } else {
                if constexpr (N == Dynamic) {
                    backend::sub_runtime<T>(data(), data(), rhs.data(), size());
                } else {
                    backend::sub<T, N>(data(), data(), rhs.data());
                }
            }
            return *this;
        }

        constexpr Vector &operator*=(const Vector &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i)
                    pod_[i] *= rhs.pod_[i];
            } else {
                if constexpr (N == Dynamic) {
                    backend::mul_runtime<T>(data(), data(), rhs.data(), size());
                } else {
                    backend::mul<T, N>(data(), data(), rhs.data());
                }
            }
            return *this;
        }

        constexpr Vector &operator/=(const Vector &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i)
                    pod_[i] /= rhs.pod_[i];
            } else {
                if constexpr (N == Dynamic) {
                    backend::div_runtime<T>(data(), data(), rhs.data(), size());
                } else {
                    backend::div<T, N>(data(), data(), rhs.data());
                }
            }
            return *this;
        }

        constexpr Vector &operator*=(T scalar) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i)
                    pod_[i] *= scalar;
            } else {
                backend::mul_scalar<T, N>(data(), data(), scalar);
            }
            return *this;
        }

        constexpr Vector &operator/=(T scalar) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < size(); ++i)
                    pod_[i] /= scalar;
            } else {
                backend::div_scalar<T, N>(data(), data(), scalar);
            }
            return *this;
        }

        // Unary
        constexpr Vector operator-() const noexcept {
            Vector result;
            for (size_type i = 0; i < size(); ++i)
                result.pod_[i] = -pod_[i];
            return result;
        }

        constexpr Vector operator+() const noexcept { return *this; }

        // Resize (only for Dynamic vectors)
        template <std::size_t M = N, typename = std::enable_if_t<M == Dynamic>> void resize(size_type new_size) {
            pod_.resize(new_size);
        }

        template <std::size_t M = N, typename = std::enable_if_t<M == Dynamic>>
        void resize(size_type new_size, const T &value) {
            pod_.resize(new_size, value);
        }

      private:
        pod_type pod_{};
    };

    // Binary ops (element-wise)
    template <typename T, std::size_t N>
    constexpr Vector<T, N> operator+(const Vector<T, N> &lhs, const Vector<T, N> &rhs) noexcept {
        Vector<T, N> result;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                result[i] = lhs[i] + rhs[i];
        } else {
            if constexpr (N == Dynamic) {
                result.resize(lhs.size());
                backend::add_runtime<T>(result.data(), lhs.data(), rhs.data(), lhs.size());
            } else {
                backend::add<T, N>(result.data(), lhs.data(), rhs.data());
            }
        }
        return result;
    }

    template <typename T, std::size_t N>
    constexpr Vector<T, N> operator-(const Vector<T, N> &lhs, const Vector<T, N> &rhs) noexcept {
        Vector<T, N> result;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                result[i] = lhs[i] - rhs[i];
        } else {
            if constexpr (N == Dynamic) {
                result.resize(lhs.size());
                backend::sub_runtime<T>(result.data(), lhs.data(), rhs.data(), lhs.size());
            } else {
                backend::sub<T, N>(result.data(), lhs.data(), rhs.data());
            }
        }
        return result;
    }

    template <typename T, std::size_t N>
    constexpr Vector<T, N> operator*(const Vector<T, N> &lhs, const Vector<T, N> &rhs) noexcept {
        Vector<T, N> result;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                result[i] = lhs[i] * rhs[i];
        } else {
            if constexpr (N == Dynamic) {
                result.resize(lhs.size());
                backend::mul_runtime<T>(result.data(), lhs.data(), rhs.data(), lhs.size());
            } else {
                backend::mul<T, N>(result.data(), lhs.data(), rhs.data());
            }
        }
        return result;
    }

    template <typename T, std::size_t N>
    constexpr Vector<T, N> operator/(const Vector<T, N> &lhs, const Vector<T, N> &rhs) noexcept {
        Vector<T, N> result;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < N; ++i)
                result[i] = lhs[i] / rhs[i];
        } else {
            if constexpr (N == Dynamic) {
                result.resize(lhs.size());
                backend::div_runtime<T>(result.data(), lhs.data(), rhs.data(), lhs.size());
            } else {
                backend::div<T, N>(result.data(), lhs.data(), rhs.data());
            }
        }
        return result;
    }

    // Scalar ops
    template <typename T, std::size_t N> constexpr Vector<T, N> operator*(const Vector<T, N> &lhs, T scalar) noexcept {
        Vector<T, N> result;
        if constexpr (N == Dynamic) {
            result.resize(lhs.size());
        }
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < lhs.size(); ++i)
                result[i] = lhs[i] * scalar;
        } else {
            if constexpr (N == Dynamic) {
                backend::mul_scalar_runtime<T>(result.data(), lhs.data(), scalar, lhs.size());
            } else {
                backend::mul_scalar<T, N>(result.data(), lhs.data(), scalar);
            }
        }
        return result;
    }

    template <typename T, std::size_t N> constexpr Vector<T, N> operator*(T scalar, const Vector<T, N> &rhs) noexcept {
        return rhs * scalar;
    }

    template <typename T, std::size_t N> constexpr Vector<T, N> operator/(const Vector<T, N> &lhs, T scalar) noexcept {
        Vector<T, N> result;
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
    constexpr bool operator==(const Vector<T, N> &lhs, const Vector<T, N> &rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) {
            if (lhs[i] != rhs[i])
                return false;
        }
        return true;
    }

    template <typename T, std::size_t N>
    constexpr bool operator!=(const Vector<T, N> &lhs, const Vector<T, N> &rhs) noexcept {
        return !(lhs == rhs);
    }

    // Common operations
    template <typename T, std::size_t N> constexpr T dot(const Vector<T, N> &lhs, const Vector<T, N> &rhs) noexcept {
        if (std::is_constant_evaluated()) {
            T result{};
            for (std::size_t i = 0; i < N; ++i)
                result += lhs[i] * rhs[i];
            return result;
        }
        if constexpr (N == Dynamic) {
            return backend::dot_runtime<T>(lhs.data(), rhs.data(), lhs.size());
        } else {
            return backend::dot<T, N>(lhs.data(), rhs.data());
        }
    }

    template <typename T, std::size_t N> constexpr T sum(const Vector<T, N> &v) noexcept {
        if (std::is_constant_evaluated()) {
            T result{};
            for (std::size_t i = 0; i < N; ++i)
                result += v[i];
            return result;
        }
        if constexpr (N == Dynamic) {
            return backend::reduce_sum_runtime<T>(v.data(), v.size());
        } else {
            return backend::reduce_sum<T, N>(v.data());
        }
    }

    template <typename T, std::size_t N> T norm(const Vector<T, N> &v) noexcept {
        if constexpr (N == Dynamic) {
            return backend::norm_l2_runtime<T>(v.data(), v.size());
        } else {
            return backend::norm_l2<T, N>(v.data());
        }
    }

    template <typename T, std::size_t N> Vector<T, N> normalized(const Vector<T, N> &v) noexcept {
        Vector<T, N> result;
        backend::normalize<T, N>(result.data(), v.data());
        return result;
    }

    // =============================================================================
    // I/O - Stream output operator
    // =============================================================================

    template <typename T, std::size_t N> std::ostream &operator<<(std::ostream &os, const Vector<T, N> &v) {
        os << "[";
        for (std::size_t i = 0; i < N; ++i) {
            os << v[i];
            if (i < N - 1)
                os << ", ";
        }
        os << "]";
        return os;
    }

    // =============================================================================
    // Type conversion - cast<U>()
    // =============================================================================

    template <typename U, typename T, std::size_t N> Vector<U, N> cast(const Vector<T, N> &v) noexcept {
        Vector<U, N> result;
        for (std::size_t i = 0; i < N; ++i) {
            result[i] = static_cast<U>(v[i]);
        }
        return result;
    }

    // Type aliases
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

} // namespace optinum::simd
