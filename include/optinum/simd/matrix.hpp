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

#include <type_traits>
#include <variant>

namespace optinum::simd {

    namespace dp = ::datapod;

    // Matrix: Non-owning 2D view over arrays with SIMD-accelerated operations
    //
    // This is a lightweight view type that wraps existing data (datapod::mat::matrix
    // or raw T* pointers). It does NOT own any data - the underlying storage must
    // outlive the view.
    //
    // Template parameters R, C can be:
    //   - Fixed size: Matrix<double, 3, 3>  (compile-time size)
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

      private:
        // Pointer to data (null for invalid/default view)
        T *ptr_ = nullptr;
        // For Dynamic matrices, store the dimensions
        [[no_unique_address]] std::conditional_t<R == Dynamic, size_type, std::monostate> rows_{};
        [[no_unique_address]] std::conditional_t<C == Dynamic, size_type, std::monostate> cols_{};

      public:
        // Default constructor - creates null view
        constexpr Matrix() noexcept = default;

        // Constructor from raw pointer (for fixed-size matrices)
        template <std::size_t RR = R, std::size_t CC = C, typename = std::enable_if_t<RR != Dynamic && CC != Dynamic>>
        constexpr explicit Matrix(T *ptr) noexcept : ptr_(ptr) {}

        // Constructor from raw pointer + dimensions (for dynamic-size matrices)
        template <std::size_t RR = R, std::size_t CC = C, typename = std::enable_if_t<RR == Dynamic && CC == Dynamic>>
        constexpr Matrix(T *ptr, size_type rows, size_type cols) noexcept : ptr_(ptr), rows_(rows), cols_(cols) {}

        // Constructor from non-const pod_type reference (creates view over pod's data)
        constexpr Matrix(pod_type &pod) noexcept : ptr_(pod.data()) {
            if constexpr (R == Dynamic) {
                rows_ = pod.rows();
            }
            if constexpr (C == Dynamic) {
                cols_ = pod.cols();
            }
        }

        // Constructor from const pod_type reference
        // Note: Creates mutable view; user must ensure const-correctness
        constexpr Matrix(const pod_type &pod) noexcept : ptr_(const_cast<T *>(pod.data())) {
            if constexpr (R == Dynamic) {
                rows_ = pod.rows();
            }
            if constexpr (C == Dynamic) {
                cols_ = pod.cols();
            }
        }

        // Copy constructor - copies the view (shallow copy)
        constexpr Matrix(const Matrix &other) noexcept = default;

        // Move constructor
        constexpr Matrix(Matrix &&other) noexcept = default;

        // Copy assignment - copies the view (shallow copy)
        constexpr Matrix &operator=(const Matrix &other) noexcept = default;

        // Move assignment
        constexpr Matrix &operator=(Matrix &&other) noexcept = default;

        // Check if view is valid (non-null)
        constexpr bool valid() const noexcept { return ptr_ != nullptr; }
        constexpr explicit operator bool() const noexcept { return valid(); }

        // Access to underlying pointer
        constexpr pointer data() noexcept { return ptr_; }
        constexpr const_pointer data() const noexcept { return ptr_; }

        // Element access (column-major order: row + col * rows)
        constexpr reference operator()(size_type row, size_type col) noexcept {
            OPTINUM_ASSERT_BOUNDS_2D(row, col, rows(), cols());
            return ptr_[row + col * rows()];
        }
        constexpr const_reference operator()(size_type row, size_type col) const noexcept {
            OPTINUM_ASSERT_BOUNDS_2D(row, col, rows(), cols());
            return ptr_[row + col * rows()];
        }

        constexpr reference at(size_type row, size_type col) {
            if (row >= rows() || col >= cols())
                throw std::out_of_range("Matrix::at");
            return ptr_[row + col * rows()];
        }
        constexpr const_reference at(size_type row, size_type col) const {
            if (row >= rows() || col >= cols())
                throw std::out_of_range("Matrix::at");
            return ptr_[row + col * rows()];
        }

        // Linear element access
        constexpr reference operator[](size_type i) noexcept {
            OPTINUM_ASSERT_BOUNDS(i, size());
            return ptr_[i];
        }
        constexpr const_reference operator[](size_type i) const noexcept {
            OPTINUM_ASSERT_BOUNDS(i, size());
            return ptr_[i];
        }

        // Dimensions
        constexpr size_type rows() const noexcept {
            if constexpr (R == Dynamic) {
                return rows_;
            } else {
                return R;
            }
        }

        constexpr size_type cols() const noexcept {
            if constexpr (C == Dynamic) {
                return cols_;
            } else {
                return C;
            }
        }

        constexpr size_type size() const noexcept { return rows() * cols(); }

        constexpr bool empty() const noexcept {
            if constexpr (R == Dynamic && C == Dynamic) {
                return rows_ == 0 || cols_ == 0;
            } else if constexpr (R == Dynamic) {
                return rows_ == 0;
            } else if constexpr (C == Dynamic) {
                return cols_ == 0;
            } else {
                return false;
            }
        }

        // Iterators
        constexpr iterator begin() noexcept { return ptr_; }
        constexpr const_iterator begin() const noexcept { return ptr_; }
        constexpr const_iterator cbegin() const noexcept { return ptr_; }
        constexpr iterator end() noexcept { return ptr_ + size(); }
        constexpr const_iterator end() const noexcept { return ptr_ + size(); }
        constexpr const_iterator cend() const noexcept { return ptr_ + size(); }

        // In-place fill (modifies underlying data)
        constexpr Matrix &fill(const T &value) noexcept {
            const size_type n = size();
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < n; ++i)
                    ptr_[i] = value;
            } else {
                if constexpr (R == Dynamic || C == Dynamic) {
                    backend::fill_runtime<T>(ptr_, n, value);
                } else {
                    backend::fill<T, R * C>(ptr_, value);
                }
            }
            return *this;
        }

        // Fill with sequential values: 0, 1, 2, ...
        constexpr Matrix &iota() noexcept {
            const size_type n = size();
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < n; ++i)
                    ptr_[i] = static_cast<T>(i);
            } else {
                backend::iota<T, R * C>(ptr_, T{0}, T{1});
            }
            return *this;
        }

        // Fill with sequential values starting from 'start'
        constexpr Matrix &iota(T start) noexcept {
            const size_type n = size();
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < n; ++i)
                    ptr_[i] = start + static_cast<T>(i);
            } else {
                backend::iota<T, R * C>(ptr_, start, T{1});
            }
            return *this;
        }

        // Fill with sequential values with custom start and step
        constexpr Matrix &iota(T start, T step) noexcept {
            const size_type n = size();
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < n; ++i)
                    ptr_[i] = start + static_cast<T>(i) * step;
            } else {
                backend::iota<T, R * C>(ptr_, start, step);
            }
            return *this;
        }

        // Reverse elements in-place
        constexpr Matrix &reverse() noexcept {
            const size_type n = size();
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < n / 2; ++i)
                    std::swap(ptr_[i], ptr_[n - 1 - i]);
            } else {
                backend::reverse<T, R * C>(ptr_);
            }
            return *this;
        }

        // Set to identity matrix (square matrices only)
        template <std::size_t R_ = R, std::size_t C_ = C>
        constexpr std::enable_if_t<R_ == C_ && R_ != Dynamic, Matrix &> set_identity() noexcept {
            fill(T{});
            for (size_type i = 0; i < rows(); ++i)
                (*this)(i, i) = T{1};
            return *this;
        }

        // Compound assignment operators (modify in-place)
        constexpr Matrix &operator+=(const Matrix &rhs) noexcept {
            const size_type n = size();
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < n; ++i)
                    ptr_[i] += rhs.ptr_[i];
            } else {
                if constexpr (R == Dynamic || C == Dynamic) {
                    backend::add_runtime<T>(ptr_, ptr_, rhs.ptr_, n);
                } else {
                    backend::add<T, R * C>(ptr_, ptr_, rhs.ptr_);
                }
            }
            return *this;
        }

        constexpr Matrix &operator-=(const Matrix &rhs) noexcept {
            const size_type n = size();
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < n; ++i)
                    ptr_[i] -= rhs.ptr_[i];
            } else {
                if constexpr (R == Dynamic || C == Dynamic) {
                    backend::sub_runtime<T>(ptr_, ptr_, rhs.ptr_, n);
                } else {
                    backend::sub<T, R * C>(ptr_, ptr_, rhs.ptr_);
                }
            }
            return *this;
        }

        constexpr Matrix &operator*=(T scalar) noexcept {
            const size_type n = size();
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < n; ++i)
                    ptr_[i] *= scalar;
            } else {
                if constexpr (R == Dynamic || C == Dynamic) {
                    backend::mul_scalar_runtime<T>(ptr_, ptr_, scalar, n);
                } else {
                    backend::mul_scalar<T, R * C>(ptr_, ptr_, scalar);
                }
            }
            return *this;
        }

        constexpr Matrix &operator/=(T scalar) noexcept {
            const size_type n = size();
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < n; ++i)
                    ptr_[i] /= scalar;
            } else {
                if constexpr (R == Dynamic || C == Dynamic) {
                    backend::div_scalar_runtime<T>(ptr_, ptr_, scalar, n);
                } else {
                    backend::div_scalar<T, R * C>(ptr_, ptr_, scalar);
                }
            }
            return *this;
        }

        constexpr Matrix &operator*=(const Matrix &rhs) noexcept {
            const size_type n = size();
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < n; ++i)
                    ptr_[i] *= rhs.ptr_[i];
            } else {
                if constexpr (R == Dynamic || C == Dynamic) {
                    backend::mul_runtime<T>(ptr_, ptr_, rhs.ptr_, n);
                } else {
                    backend::mul<T, R * C>(ptr_, ptr_, rhs.ptr_);
                }
            }
            return *this;
        }

        constexpr Matrix &operator/=(const Matrix &rhs) noexcept {
            const size_type n = size();
            if (std::is_constant_evaluated()) {
                for (size_type i = 0; i < n; ++i)
                    ptr_[i] /= rhs.ptr_[i];
            } else {
                if constexpr (R == Dynamic || C == Dynamic) {
                    backend::div_runtime<T>(ptr_, ptr_, rhs.ptr_, n);
                } else {
                    backend::div<T, R * C>(ptr_, ptr_, rhs.ptr_);
                }
            }
            return *this;
        }

        // Unary negation - writes negated values to output pointer
        constexpr void negate_to(T *out) const noexcept {
            const size_type n = size();
            for (size_type i = 0; i < n; ++i)
                out[i] = -ptr_[i];
        }
    };

    // =============================================================================
    // Free functions that write to output pointers (non-allocating)
    // =============================================================================

    // Binary ops (element-wise) - write results to output pointer
    template <typename T, std::size_t R, std::size_t C>
    constexpr void add(T *out, const Matrix<T, R, C> &lhs, const Matrix<T, R, C> &rhs) noexcept {
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < lhs.size(); ++i)
                out[i] = lhs[i] + rhs[i];
        } else {
            if constexpr (R == Dynamic || C == Dynamic) {
                backend::add_runtime<T>(out, lhs.data(), rhs.data(), lhs.size());
            } else {
                backend::add<T, R * C>(out, lhs.data(), rhs.data());
            }
        }
    }

    template <typename T, std::size_t R, std::size_t C>
    constexpr void sub(T *out, const Matrix<T, R, C> &lhs, const Matrix<T, R, C> &rhs) noexcept {
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < lhs.size(); ++i)
                out[i] = lhs[i] - rhs[i];
        } else {
            if constexpr (R == Dynamic || C == Dynamic) {
                backend::sub_runtime<T>(out, lhs.data(), rhs.data(), lhs.size());
            } else {
                backend::sub<T, R * C>(out, lhs.data(), rhs.data());
            }
        }
    }

    template <typename T, std::size_t R, std::size_t C>
    constexpr void mul(T *out, const Matrix<T, R, C> &lhs, const Matrix<T, R, C> &rhs) noexcept {
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < lhs.size(); ++i)
                out[i] = lhs[i] * rhs[i];
        } else {
            if constexpr (R == Dynamic || C == Dynamic) {
                backend::mul_runtime<T>(out, lhs.data(), rhs.data(), lhs.size());
            } else {
                backend::mul<T, R * C>(out, lhs.data(), rhs.data());
            }
        }
    }

    template <typename T, std::size_t R, std::size_t C>
    constexpr void div(T *out, const Matrix<T, R, C> &lhs, const Matrix<T, R, C> &rhs) noexcept {
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < lhs.size(); ++i)
                out[i] = lhs[i] / rhs[i];
        } else {
            if constexpr (R == Dynamic || C == Dynamic) {
                backend::div_runtime<T>(out, lhs.data(), rhs.data(), lhs.size());
            } else {
                backend::div<T, R * C>(out, lhs.data(), rhs.data());
            }
        }
    }

    // Scalar ops - write results to output pointer
    template <typename T, std::size_t R, std::size_t C>
    constexpr void mul_scalar(T *out, const Matrix<T, R, C> &lhs, T scalar) noexcept {
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < lhs.size(); ++i)
                out[i] = lhs[i] * scalar;
        } else {
            if constexpr (R == Dynamic || C == Dynamic) {
                backend::mul_scalar_runtime<T>(out, lhs.data(), scalar, lhs.size());
            } else {
                backend::mul_scalar<T, R * C>(out, lhs.data(), scalar);
            }
        }
    }

    template <typename T, std::size_t R, std::size_t C>
    constexpr void div_scalar(T *out, const Matrix<T, R, C> &lhs, T scalar) noexcept {
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < lhs.size(); ++i)
                out[i] = lhs[i] / scalar;
        } else {
            if constexpr (R == Dynamic || C == Dynamic) {
                backend::div_scalar_runtime<T>(out, lhs.data(), scalar, lhs.size());
            } else {
                backend::div_scalar<T, R * C>(out, lhs.data(), scalar);
            }
        }
    }

    // Matrix multiplication - write result to output pointer
    template <typename T, std::size_t R, std::size_t K, std::size_t C>
    constexpr void matmul_to(T *out, const Matrix<T, R, K> &lhs, const Matrix<T, K, C> &rhs) noexcept {
        static_assert(R != Dynamic && K != Dynamic && C != Dynamic, "matmul_to() requires fixed-size matrices");
        if (std::is_constant_evaluated()) {
            // Initialize output to zero
            for (std::size_t i = 0; i < R * C; ++i)
                out[i] = T{0};
            // Column-major: out(i,j) = sum_k lhs(i,k) * rhs(k,j)
            for (std::size_t j = 0; j < C; ++j) {
                for (std::size_t k = 0; k < K; ++k) {
                    for (std::size_t i = 0; i < R; ++i) {
                        out[i + j * R] += lhs(i, k) * rhs(k, j);
                    }
                }
            }
        } else {
            backend::matmul<T, R, K, C>(out, lhs.data(), rhs.data());
        }
    }

    // Matrix-vector multiplication - write result to output pointer
    template <typename T, std::size_t R, std::size_t C>
    constexpr void matvec_to(T *out, const Matrix<T, R, C> &mat, const Vector<T, C> &vec) noexcept {
        static_assert(R != Dynamic && C != Dynamic, "matvec_to() requires fixed-size matrix");
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < R; ++i) {
                out[i] = T{0};
                for (std::size_t j = 0; j < C; ++j) {
                    out[i] += mat(i, j) * vec[j];
                }
            }
        } else {
            backend::matvec<T, R, C>(out, mat.data(), vec.data());
        }
    }

    // Transpose - write result to output pointer
    template <typename T, std::size_t R, std::size_t C>
    constexpr void transpose_to(T *out, const Matrix<T, R, C> &mat) noexcept {
        static_assert(R != Dynamic && C != Dynamic, "transpose_to() requires fixed-size matrix");
        if (std::is_constant_evaluated()) {
            for (std::size_t i = 0; i < R; ++i) {
                for (std::size_t j = 0; j < C; ++j) {
                    // Source: column-major, out is C x R
                    out[j + i * C] = mat(i, j);
                }
            }
        } else {
            backend::transpose<T, R, C>(out, mat.data());
        }
    }

    // Flatten matrix to vector - write result to output pointer
    template <typename T, std::size_t R, std::size_t C>
    constexpr void flatten_to(T *out, const Matrix<T, R, C> &mat) noexcept {
        const std::size_t n = mat.size();
        for (std::size_t i = 0; i < n; ++i)
            out[i] = mat[i];
    }

    // =============================================================================
    // Comparisons
    // =============================================================================

    template <typename T, std::size_t R, std::size_t C>
    constexpr bool operator==(const Matrix<T, R, C> &lhs, const Matrix<T, R, C> &rhs) noexcept {
        if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols())
            return false;
        for (std::size_t i = 0; i < lhs.size(); ++i)
            if (lhs[i] != rhs[i])
                return false;
        return true;
    }

    template <typename T, std::size_t R, std::size_t C>
    constexpr bool operator!=(const Matrix<T, R, C> &lhs, const Matrix<T, R, C> &rhs) noexcept {
        return !(lhs == rhs);
    }

    // =============================================================================
    // Common operations (return scalars, no allocation needed)
    // =============================================================================

    // Trace (square matrices only)
    template <typename T, std::size_t N> constexpr T trace(const Matrix<T, N, N> &mat) noexcept {
        static_assert(N != Dynamic, "trace() requires fixed-size matrix");
        if (std::is_constant_evaluated()) {
            T r{};
            for (std::size_t i = 0; i < N; ++i)
                r += mat(i, i);
            return r;
        } else {
            if constexpr (N <= 4) {
                T r{};
                for (std::size_t i = 0; i < N; ++i)
                    r += mat(i, i);
                return r;
            } else {
                alignas(32) T diag[N];
                for (std::size_t i = 0; i < N; ++i)
                    diag[i] = mat(i, i);
                return backend::reduce_sum<T, N>(diag);
            }
        }
    }

    // Frobenius norm
    template <typename T, std::size_t R, std::size_t C> T frobenius_norm(const Matrix<T, R, C> &mat) noexcept {
        static_assert(R != Dynamic && C != Dynamic, "frobenius_norm() requires fixed-size matrix");
        return std::sqrt(backend::dot<T, R * C>(mat.data(), mat.data()));
    }

    // =============================================================================
    // I/O - Stream output operator
    // =============================================================================

    template <typename T, std::size_t R, std::size_t C>
    std::ostream &operator<<(std::ostream &os, const Matrix<T, R, C> &m) {
        os << "[";
        for (std::size_t c = 0; c < m.cols(); ++c) {
            if (c > 0)
                os << " ";
            os << "[";
            for (std::size_t r = 0; r < m.rows(); ++r) {
                os << m(r, c);
                if (r < m.rows() - 1)
                    os << ", ";
            }
            os << "]";
            if (c < m.cols() - 1)
                os << "\n";
        }
        return os << "]";
    }

    // =============================================================================
    // Type conversion - cast_to() - writes to output pointer
    // =============================================================================

    template <typename U, typename T, std::size_t R, std::size_t C>
    void cast_to(U *out, const Matrix<T, R, C> &m) noexcept {
        for (std::size_t i = 0; i < m.size(); ++i)
            out[i] = static_cast<U>(m[i]);
    }

    // =============================================================================
    // Type aliases
    // =============================================================================

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

    // Dynamic matrix aliases
    template <typename T> using MatrixX = Matrix<T, Dynamic, Dynamic>;
    using MatrixXf = Matrix<float, Dynamic, Dynamic>;
    using MatrixXd = Matrix<double, Dynamic, Dynamic>;

} // namespace optinum::simd
