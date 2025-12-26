#pragma once

#include <datapod/matrix/math/complex.hpp>
#include <datapod/matrix/vector.hpp>
#include <optinum/simd/debug.hpp>

#include <cstddef>
#include <iostream>
#include <type_traits>

namespace optinum::simd {

    namespace dp = ::datapod;

    // Complex number array wrapper (owns data via datapod)
    template <typename T, std::size_t N> class Complex {
        static_assert(N > 0, "Complex size must be > 0");
        static_assert(std::is_floating_point_v<T>, "Complex<T, N> requires floating-point type");

      public:
        using value_type = dp::mat::complex<T>;
        using pod_type = dp::mat::vector<value_type, N>;
        using real_type = T;

        static constexpr std::size_t extent = N;
        static constexpr std::size_t rank = 1;

      private:
        pod_type pod_;

      public:
        // Constructors
        constexpr Complex() noexcept : pod_() {}
        constexpr Complex(const Complex &) = default;
        constexpr Complex(Complex &&) noexcept = default;
        constexpr Complex &operator=(const Complex &) = default;
        constexpr Complex &operator=(Complex &&) noexcept = default;

        // Direct initialization from pod
        constexpr explicit Complex(const pod_type &p) noexcept : pod_(p) {}

        // Element access
        [[nodiscard]] constexpr value_type &operator[](std::size_t i) noexcept { return pod_[i]; }

        [[nodiscard]] constexpr const value_type &operator[](std::size_t i) const noexcept { return pod_[i]; }

        // Raw data access
        [[nodiscard]] constexpr value_type *data() noexcept { return pod_.data(); }
        [[nodiscard]] constexpr const value_type *data() const noexcept { return pod_.data(); }

        // Get underlying pod
        [[nodiscard]] constexpr pod_type &pod() noexcept { return pod_; }
        [[nodiscard]] constexpr const pod_type &pod() const noexcept { return pod_; }

        // Fill operations
        constexpr void fill(const value_type &val) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                pod_[i] = val;
            }
        }

        constexpr void fill_real(T real_val) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                pod_[i] = value_type{real_val, T{}};
            }
        }

        // Factory functions
        [[nodiscard]] static constexpr Complex zeros() noexcept {
            Complex c;
            c.fill(value_type{T{}, T{}});
            return c;
        }

        [[nodiscard]] static constexpr Complex ones() noexcept {
            Complex c;
            c.fill(value_type{T{1}, T{}});
            return c;
        }

        [[nodiscard]] static constexpr Complex unit_imaginary() noexcept {
            Complex c;
            c.fill(value_type{T{}, T{1}});
            return c;
        }

        // Extract real/imaginary parts
        constexpr void real_parts(T *out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = pod_[i].real;
            }
        }

        constexpr void imag_parts(T *out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = pod_[i].imag;
            }
        }

        // Conjugate
        [[nodiscard]] constexpr Complex conjugate() const noexcept {
            Complex result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = pod_[i].conjugate();
            }
            return result;
        }

        // Magnitude
        constexpr void magnitude(T *out) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = pod_[i].magnitude();
            }
        }

        // Arithmetic operators
        [[nodiscard]] constexpr Complex operator+(const Complex &other) const noexcept {
            Complex result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = pod_[i] + other[i];
            }
            return result;
        }

        [[nodiscard]] constexpr Complex operator-(const Complex &other) const noexcept {
            Complex result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = pod_[i] - other[i];
            }
            return result;
        }

        [[nodiscard]] constexpr Complex operator*(const Complex &other) const noexcept {
            Complex result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = pod_[i] * other[i];
            }
            return result;
        }

        [[nodiscard]] Complex operator/(const Complex &other) const noexcept {
            Complex result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = pod_[i] / other[i];
            }
            return result;
        }

        // Scalar multiplication
        [[nodiscard]] constexpr Complex operator*(T scalar) const noexcept {
            Complex result;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = pod_[i] * scalar;
            }
            return result;
        }

        [[nodiscard]] friend constexpr Complex operator*(T scalar, const Complex &c) noexcept { return c * scalar; }

        // Compound assignment
        constexpr Complex &operator+=(const Complex &other) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                pod_[i] += other[i];
            }
            return *this;
        }

        constexpr Complex &operator-=(const Complex &other) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                pod_[i] -= other[i];
            }
            return *this;
        }

        constexpr Complex &operator*=(const Complex &other) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                pod_[i] *= other[i];
            }
            return *this;
        }

        Complex &operator/=(const Complex &other) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                pod_[i] /= other[i];
            }
            return *this;
        }

        // Stream output
        friend std::ostream &operator<<(std::ostream &os, const Complex &c) {
            os << "[";
            for (std::size_t i = 0; i < N; ++i) {
                os << "(" << c[i].real << "+" << c[i].imag << "i)";
                if (i + 1 < N)
                    os << ", ";
            }
            os << "]";
            return os;
        }
    };

} // namespace optinum::simd
