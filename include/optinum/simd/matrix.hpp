#pragma once

#include <cmath>
#include <datapod/matrix/matrix.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/backend/matmul.hpp>
#include <optinum/simd/backend/transpose.hpp>
#include <optinum/simd/tensor.hpp>

namespace optinum::simd {

    namespace dp = ::datapod;

    template <typename T, std::size_t R, std::size_t C> class Matrix {
        static_assert(R > 0, "Matrix rows must be > 0");
        static_assert(C > 0, "Matrix cols must be > 0");
        static_assert(std::is_arithmetic_v<T>, "Matrix<T, R, C> requires arithmetic type");

      public:
        using value_type = T;
        using pod_type = dp::matrix<T, R, C>;
        using size_type = std::size_t;
        using reference = T &;
        using const_reference = const T &;
        using pointer = T *;
        using const_pointer = const T *;
        using iterator = T *;
        using const_iterator = const T *;

        static constexpr size_type rows_extent = R;
        static constexpr size_type cols_extent = C;
        static constexpr size_type extent = R * C;

        constexpr Matrix() noexcept = default;
        constexpr explicit Matrix(const pod_type &pod) noexcept : pod_(pod) {}
        constexpr explicit Matrix(pod_type &&pod) noexcept : pod_(static_cast<pod_type &&>(pod)) {}

        constexpr pod_type &pod() noexcept { return pod_; }
        constexpr const pod_type &pod() const noexcept { return pod_; }

        constexpr pointer data() noexcept { return pod_.data(); }
        constexpr const_pointer data() const noexcept { return pod_.data(); }

        // 2D indexing (row, col)
        constexpr reference operator()(size_type row, size_type col) noexcept { return pod_(row, col); }
        constexpr const_reference operator()(size_type row, size_type col) const noexcept { return pod_(row, col); }

        constexpr reference at(size_type row, size_type col) { return pod_.at(row, col); }
        constexpr const_reference at(size_type row, size_type col) const { return pod_.at(row, col); }

        // 1D indexing (linear)
        constexpr reference operator[](size_type i) noexcept { return pod_[i]; }
        constexpr const_reference operator[](size_type i) const noexcept { return pod_[i]; }

        static constexpr size_type rows() noexcept { return R; }
        static constexpr size_type cols() noexcept { return C; }
        static constexpr size_type size() noexcept { return R * C; }
        static constexpr bool empty() noexcept { return false; }

        constexpr iterator begin() noexcept { return pod_.begin(); }
        constexpr const_iterator begin() const noexcept { return pod_.begin(); }
        constexpr const_iterator cbegin() const noexcept { return pod_.cbegin(); }

        constexpr iterator end() noexcept { return pod_.end(); }
        constexpr const_iterator end() const noexcept { return pod_.end(); }
        constexpr const_iterator cend() const noexcept { return pod_.cend(); }

        constexpr Matrix &fill(const T &value) noexcept {
            pod_.fill(value);
            return *this;
        }

        // Identity (square matrices only)
        template <std::size_t R_ = R, std::size_t C_ = C>
        constexpr std::enable_if_t<R_ == C_, Matrix &> set_identity() noexcept {
            fill(T{});
            for (size_type i = 0; i < R; ++i)
                (*this)(i, i) = T{1};
            return *this;
        }

        // Compound assignment (element-wise)
        constexpr Matrix &operator+=(const Matrix &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < R * C; ++i)
                    pod_[i] += rhs.pod_[i];
            } else {
                backend::add<T, R * C>(data(), data(), rhs.data());
            }
            return *this;
        }

        constexpr Matrix &operator-=(const Matrix &rhs) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < R * C; ++i)
                    pod_[i] -= rhs.pod_[i];
            } else {
                backend::sub<T, R * C>(data(), data(), rhs.data());
            }
            return *this;
        }

        constexpr Matrix &operator*=(T scalar) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < R * C; ++i)
                    pod_[i] *= scalar;
            } else {
                backend::mul_scalar<T, R * C>(data(), data(), scalar);
            }
            return *this;
        }

        constexpr Matrix &operator/=(T scalar) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < R * C; ++i)
                    pod_[i] /= scalar;
            } else {
                backend::div_scalar<T, R * C>(data(), data(), scalar);
            }
            return *this;
        }

        // Unary
        constexpr Matrix operator-() const noexcept {
            Matrix result;
            for (size_type i = 0; i < R * C; ++i)
                result.pod_[i] = -pod_[i];
            return result;
        }

        constexpr Matrix operator+() const noexcept { return *this; }

      private:
        pod_type pod_{};
    };

    // Element-wise binary ops
    template <typename T, std::size_t R, std::size_t C>
    constexpr Matrix<T, R, C> operator+(const Matrix<T, R, C> &lhs, const Matrix<T, R, C> &rhs) noexcept {
        Matrix<T, R, C> result;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < R * C; ++i)
                result[i] = lhs[i] + rhs[i];
        } else {
            backend::add<T, R * C>(result.data(), lhs.data(), rhs.data());
        }
        return result;
    }

    template <typename T, std::size_t R, std::size_t C>
    constexpr Matrix<T, R, C> operator-(const Matrix<T, R, C> &lhs, const Matrix<T, R, C> &rhs) noexcept {
        Matrix<T, R, C> result;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < R * C; ++i)
                result[i] = lhs[i] - rhs[i];
        } else {
            backend::sub<T, R * C>(result.data(), lhs.data(), rhs.data());
        }
        return result;
    }

    // Scalar ops
    template <typename T, std::size_t R, std::size_t C>
    constexpr Matrix<T, R, C> operator*(const Matrix<T, R, C> &lhs, T scalar) noexcept {
        Matrix<T, R, C> result;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < R * C; ++i)
                result[i] = lhs[i] * scalar;
        } else {
            backend::mul_scalar<T, R * C>(result.data(), lhs.data(), scalar);
        }
        return result;
    }

    template <typename T, std::size_t R, std::size_t C>
    constexpr Matrix<T, R, C> operator*(T scalar, const Matrix<T, R, C> &rhs) noexcept {
        return rhs * scalar;
    }

    template <typename T, std::size_t R, std::size_t C>
    constexpr Matrix<T, R, C> operator/(const Matrix<T, R, C> &lhs, T scalar) noexcept {
        Matrix<T, R, C> result;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < R * C; ++i)
                result[i] = lhs[i] / scalar;
        } else {
            backend::div_scalar<T, R * C>(result.data(), lhs.data(), scalar);
        }
        return result;
    }

    // Matrix multiplication: (R x K) * (K x C) -> (R x C)
    template <typename T, std::size_t R, std::size_t K, std::size_t C>
    constexpr Matrix<T, R, C> operator*(const Matrix<T, R, K> &lhs, const Matrix<T, K, C> &rhs) noexcept {
        Matrix<T, R, C> result;
        if (std::is_constant_evaluated()) {
            result.fill(T{});
            for (std::size_t i = 0; i < R; ++i) {
                for (std::size_t j = 0; j < C; ++j) {
                    for (std::size_t k = 0; k < K; ++k) {
                        result(i, j) += lhs(i, k) * rhs(k, j);
                    }
                }
            }
        } else {
            backend::matmul<T, R, K, C>(result.data(), lhs.data(), rhs.data());
        }
        return result;
    }

    // Matrix-vector multiplication: (R x C) * Tensor<C> -> Tensor<R>
    template <typename T, std::size_t R, std::size_t C>
    constexpr Tensor<T, R> operator*(const Matrix<T, R, C> &mat, const Tensor<T, C> &vec) noexcept {
        Tensor<T, R> result;
        if (std::is_constant_evaluated()) {
            result.fill(T{});
            for (std::size_t i = 0; i < R; ++i) {
                for (std::size_t j = 0; j < C; ++j) {
                    result[i] += mat(i, j) * vec[j];
                }
            }
        } else {
            backend::matvec<T, R, C>(result.data(), mat.data(), vec.data());
        }
        return result;
    }

    // Comparisons
    template <typename T, std::size_t R, std::size_t C>
    constexpr bool operator==(const Matrix<T, R, C> &lhs, const Matrix<T, R, C> &rhs) noexcept {
        for (std::size_t i = 0; i < R * C; ++i) {
            if (lhs[i] != rhs[i])
                return false;
        }
        return true;
    }

    template <typename T, std::size_t R, std::size_t C>
    constexpr bool operator!=(const Matrix<T, R, C> &lhs, const Matrix<T, R, C> &rhs) noexcept {
        return !(lhs == rhs);
    }

    // Transpose
    template <typename T, std::size_t R, std::size_t C>
    constexpr Matrix<T, C, R> transpose(const Matrix<T, R, C> &mat) noexcept {
        Matrix<T, C, R> result;
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < R; ++i) {
                for (std::size_t j = 0; j < C; ++j) {
                    result(j, i) = mat(i, j);
                }
            }
        } else {
            backend::transpose<T, R, C>(result.data(), mat.data());
        }
        return result;
    }

    // Trace (square matrices only)
    template <typename T, std::size_t N> constexpr T trace(const Matrix<T, N, N> &mat) noexcept {
        T result{};
        for (std::size_t i = 0; i < N; ++i)
            result += mat(i, i);
        return result;
    }

    // Frobenius norm
    template <typename T, std::size_t R, std::size_t C> T frobenius_norm(const Matrix<T, R, C> &mat) noexcept {
        return std::sqrt(backend::dot<T, R * C>(mat.data(), mat.data()));
    }

    // Identity matrix factory
    template <typename T, std::size_t N> constexpr Matrix<T, N, N> identity() noexcept {
        Matrix<T, N, N> result;
        result.fill(T{});
        for (std::size_t i = 0; i < N; ++i)
            result(i, i) = T{1};
        return result;
    }

    // Type aliases
    template <typename T> using Matrix2x2 = Matrix<T, 2, 2>;
    template <typename T> using Matrix3x3 = Matrix<T, 3, 3>;
    template <typename T> using Matrix4x4 = Matrix<T, 4, 4>;
    template <typename T> using Matrix6x6 = Matrix<T, 6, 6>;

    using Matrix2x2f = Matrix<float, 2, 2>;
    using Matrix2x2d = Matrix<double, 2, 2>;
    using Matrix3x3f = Matrix<float, 3, 3>;
    using Matrix3x3d = Matrix<double, 3, 3>;
    using Matrix4x4f = Matrix<float, 4, 4>;
    using Matrix4x4d = Matrix<double, 4, 4>;
    using Matrix6x6f = Matrix<float, 6, 6>;
    using Matrix6x6d = Matrix<double, 6, 6>;

} // namespace optinum::simd
