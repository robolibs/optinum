#pragma once

#include <cmath>
#include <datapod/matrix/matrix.hpp>
#include <iostream>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/backend/matmul.hpp>
#include <optinum/simd/backend/reduce.hpp>
#include <optinum/simd/backend/transpose.hpp>
#include <optinum/simd/debug.hpp>
#include <optinum/simd/vector.hpp>
#include <random>

namespace optinum::simd {

    namespace dp = ::datapod;

    // Matrix: 2D array with SIMD-accelerated operations
    // Template parameters R, C can be:
    //   - Fixed size: Matrix<double, 3, 4>  (compile-time size)
    //   - Dynamic size: Matrix<double, Dynamic, Dynamic> (runtime size)
    template <typename T, std::size_t R, std::size_t C> class Matrix {
        static_assert(R > 0 || R == Dynamic, "Matrix rows must be > 0 or Dynamic");
        static_assert(C > 0 || C == Dynamic, "Matrix cols must be > 0 or Dynamic");
        static_assert(std::is_arithmetic_v<T>, "Matrix<T, R, C> requires arithmetic type");

      public:
        using value_type = T;
        using pod_type = dp::mat::matrix<T, R, C>;
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

        // Default constructor
        constexpr Matrix() noexcept = default;

        // Runtime size constructor (only for Dynamic)
        template <std::size_t RR = R, std::size_t CC = C, typename = std::enable_if_t<RR == Dynamic && CC == Dynamic>>
        explicit Matrix(size_type rows, size_type cols) : pod_(rows, cols) {}

        // POD constructors
        constexpr explicit Matrix(const pod_type &pod) noexcept : pod_(pod) {}
        constexpr explicit Matrix(pod_type &&pod) noexcept : pod_(static_cast<pod_type &&>(pod)) {}

        constexpr pod_type &pod() noexcept { return pod_; }
        constexpr const pod_type &pod() const noexcept { return pod_; }

        constexpr pointer data() noexcept { return pod_.data(); }
        constexpr const_pointer data() const noexcept { return pod_.data(); }

        // 2D indexing (row, col)
        constexpr reference operator()(size_type row, size_type col) noexcept {
            OPTINUM_ASSERT_BOUNDS_2D(row, col, R, C);
            return pod_(row, col);
        }
        constexpr const_reference operator()(size_type row, size_type col) const noexcept {
            OPTINUM_ASSERT_BOUNDS_2D(row, col, R, C);
            return pod_(row, col);
        }

        constexpr reference at(size_type row, size_type col) { return pod_.at(row, col); }
        constexpr const_reference at(size_type row, size_type col) const { return pod_.at(row, col); }

        // 1D indexing (linear)
        constexpr reference operator[](size_type i) noexcept {
            OPTINUM_ASSERT_BOUNDS(i, R * C);
            return pod_[i];
        }
        constexpr const_reference operator[](size_type i) const noexcept {
            OPTINUM_ASSERT_BOUNDS(i, R * C);
            return pod_[i];
        }

        constexpr size_type rows() const noexcept {
            if constexpr (R == Dynamic) {
                return pod_.rows();
            } else {
                return R;
            }
        }

        constexpr size_type cols() const noexcept {
            if constexpr (C == Dynamic) {
                return pod_.cols();
            } else {
                return C;
            }
        }
        static constexpr size_type size() noexcept { return R * C; }
        static constexpr bool empty() noexcept { return false; }

        constexpr iterator begin() noexcept { return pod_.begin(); }
        constexpr const_iterator begin() const noexcept { return pod_.begin(); }
        constexpr const_iterator cbegin() const noexcept { return pod_.cbegin(); }

        constexpr iterator end() noexcept { return pod_.end(); }
        constexpr const_iterator end() const noexcept { return pod_.end(); }
        constexpr const_iterator cend() const noexcept { return pod_.cend(); }

        constexpr Matrix &fill(const T &value) noexcept {
            if (std::is_constant_evaluated()) {
                pod_.fill(value);
            } else {
                backend::fill<T, R * C>(data(), value);
            }
            return *this;
        }

        // Fill with sequential values (row-major order): 0, 1, 2, ...
        constexpr Matrix &iota() noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < R * C; ++i) {
                    pod_[i] = static_cast<T>(i);
                }
            } else {
                backend::iota<T, R * C>(data(), T{0}, T{1});
            }
            return *this;
        }

        // Fill with sequential values starting from 'start'
        constexpr Matrix &iota(T start) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < R * C; ++i) {
                    pod_[i] = start + static_cast<T>(i);
                }
            } else {
                backend::iota<T, R * C>(data(), start, T{1});
            }
            return *this;
        }

        // Fill with sequential values with custom start and step
        constexpr Matrix &iota(T start, T step) noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < R * C; ++i) {
                    pod_[i] = start + static_cast<T>(i) * step;
                }
            } else {
                backend::iota<T, R * C>(data(), start, step);
            }
            return *this;
        }

        // Reverse elements in-place (linear order)
        constexpr Matrix &reverse() noexcept {
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < (R * C) / 2; ++i) {
                    std::swap(pod_[i], pod_[R * C - 1 - i]);
                }
            } else {
                backend::reverse<T, R * C>(data());
            }
            return *this;
        }

        // Flatten matrix to vector (row-major order)
        constexpr Vector<T, R * C> flatten() const noexcept {
            Vector<T, R * C> result;
            for (size_type i = 0; i < R * C; ++i) {
                result[i] = pod_[i];
            }
            return result;
        }

        // Static factory: create matrix filled with zeros
        static constexpr Matrix zeros() noexcept {
            Matrix m;
            m.fill(T{0});
            return m;
        }

        // Static factory: create matrix filled with ones
        static constexpr Matrix ones() noexcept {
            Matrix m;
            m.fill(T{1});
            return m;
        }

        // Static factory: create matrix with sequential values
        static constexpr Matrix arange() noexcept {
            Matrix m;
            m.iota();
            return m;
        }

        // Static factory: create matrix with sequential values from start
        static constexpr Matrix arange(T start) noexcept {
            Matrix m;
            m.iota(start);
            return m;
        }

        // Static factory: create matrix with sequential values with custom start and step
        static constexpr Matrix arange(T start, T step) noexcept {
            Matrix m;
            m.iota(start, step);
            return m;
        }

        // Fill with uniform random values in [0, 1) for floating point, [0, max) for integers
        Matrix &random() {
            static std::random_device rd;
            static std::mt19937 gen(rd());

            if constexpr (std::is_floating_point_v<T>) {
                std::uniform_real_distribution<T> dis(T{0}, T{1});
                for (size_type i = 0; i < R * C; ++i) {
                    pod_[i] = dis(gen);
                }
            } else {
                std::uniform_int_distribution<T> dis(T{0}, std::numeric_limits<T>::max());
                for (size_type i = 0; i < R * C; ++i) {
                    pod_[i] = dis(gen);
                }
            }
            return *this;
        }

        // Fill with random integers in [low, high]
        template <typename U = T> std::enable_if_t<std::is_integral_v<U>, Matrix &> randint(T low, T high) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_int_distribution<T> dis(low, high);

            for (size_type i = 0; i < R * C; ++i) {
                pod_[i] = dis(gen);
            }
            return *this;
        }

        // Identity (square matrices only)
        template <std::size_t R_ = R, std::size_t C_ = C>
        constexpr std::enable_if_t<R_ == C_, Matrix &> set_identity() noexcept {
            fill(T{});
            for (size_type i = 0; i < rows(); ++i)
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

        // Resize (only for Dynamic matrices)
        template <std::size_t RR = R, std::size_t CC = C, typename = std::enable_if_t<RR == Dynamic && CC == Dynamic>>
        void resize(size_type new_rows, size_type new_cols) {
            pod_.resize(new_rows, new_cols);
        }

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

    // Matrix-vector multiplication: (R x C) * Vector<C> -> Vector<R>
    template <typename T, std::size_t R, std::size_t C>
    constexpr Vector<T, R> operator*(const Matrix<T, R, C> &mat, const Vector<T, C> &vec) noexcept {
        Vector<T, R> result;
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

    // Trace (square matrices only) - SIMD optimized for N > 4
    template <typename T, std::size_t N> T trace(const Matrix<T, N, N> &mat) noexcept {
        if constexpr (N == 1) {
            // Trivial case: 1x1 matrix
            return mat(0, 0);
        } else if constexpr (N <= 4) {
            // For small matrices, scalar loop is efficient
            T result{};
            for (std::size_t i = 0; i < N; ++i)
                result += mat(i, i);
            return result;
        } else {
            // For larger matrices, extract diagonal to contiguous array and use SIMD reduction
            alignas(32) T diag[N];
            for (std::size_t i = 0; i < N; ++i) {
                diag[i] = mat(i, i);
            }
            // Use SIMD backend to sum the diagonal elements
            return backend::reduce_sum<T, N>(diag);
        }
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

    // =============================================================================
    // I/O - Stream output operator (column-major display)
    // =============================================================================

    template <typename T, std::size_t R, std::size_t C>
    std::ostream &operator<<(std::ostream &os, const Matrix<T, R, C> &m) {
        os << "[";
        for (std::size_t c = 0; c < C; ++c) {
            if (c > 0)
                os << " ";
            os << "[";
            for (std::size_t r = 0; r < R; ++r) {
                os << m(r, c);
                if (r < R - 1)
                    os << ", ";
            }
            os << "]";
            if (c < C - 1)
                os << "\n";
        }
        os << "]";
        return os;
    }

    // =============================================================================
    // Type conversion - cast<U>()
    // =============================================================================

    template <typename U, typename T, std::size_t R, std::size_t C>
    Matrix<U, R, C> cast(const Matrix<T, R, C> &m) noexcept {
        Matrix<U, R, C> result;
        for (std::size_t i = 0; i < R * C; ++i) {
            result[i] = static_cast<U>(m[i]);
        }
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
