#pragma once

// =============================================================================
// optinum/simd/complex.hpp
// High-level complex array container with transparent SIMD
// =============================================================================
//
// This file provides a simple owning container for complex number arrays.
// For non-owning views with SIMD operations, use view/complex_view.hpp
// For low-level SIMD pack operations, use pack/complex.hpp
//
// Usage:
//   Complex<double, 8> nums;
//   nums[0] = dp::mat::complex<double>{3.0, 4.0};
//   nums.conjugate_inplace();  // SIMD under the hood
//
// Or use the view() bridge for existing arrays:
//   dp::mat::complex<double> nums[8];
//   auto cv = view(nums);  // auto-detect SIMD width
//   cv.conjugate_inplace();
// =============================================================================

#include <datapod/matrix/math/complex.hpp>
#include <datapod/matrix/vector.hpp>
#include <optinum/simd/bridge.hpp>

#include <cstddef>
#include <iostream>
#include <type_traits>

namespace optinum::simd {

    namespace dp = ::datapod;

    /**
     * @brief Complex array container with transparent SIMD operations
     *
     * High-level container for N complex numbers. Operations use SIMD internally
     * via complex_view. User works with dp::mat::complex<T> directly.
     *
     * Use cases:
     *   - FFT preprocessing
     *   - Signal processing
     *   - Quantum computing simulations
     *   - Electrical engineering (phasors)
     */
    template <typename T, std::size_t N> class Complex {
        static_assert(N > 0, "Complex size must be > 0");
        static_assert(std::is_floating_point_v<T>, "Complex<T, N> requires floating-point type");

      public:
        using value_type = dp::mat::complex<T>;
        using pod_type = dp::mat::vector<value_type, N>;
        using real_type = T;
        using view_type = complex_view<T, detail::default_width<T>()>;

        static constexpr std::size_t extent = N;
        static constexpr std::size_t rank = 1;

      private:
        pod_type pod_;

      public:
        // ===== CONSTRUCTORS =====

        constexpr Complex() noexcept : pod_() {}
        constexpr Complex(const Complex &) = default;
        constexpr Complex(Complex &&) noexcept = default;
        constexpr Complex &operator=(const Complex &) = default;
        constexpr Complex &operator=(Complex &&) noexcept = default;

        // POD constructors (implicit to allow dp::mat::vector<complex> -> simd::Complex conversion)
        constexpr Complex(const pod_type &p) noexcept : pod_(p) {}
        constexpr Complex(pod_type &&p) noexcept : pod_(static_cast<pod_type &&>(p)) {}

        // ===== ELEMENT ACCESS =====

        [[nodiscard]] constexpr value_type &operator[](std::size_t i) noexcept { return pod_[i]; }
        [[nodiscard]] constexpr const value_type &operator[](std::size_t i) const noexcept { return pod_[i]; }

        // ===== RAW DATA ACCESS =====

        [[nodiscard]] constexpr value_type *data() noexcept { return pod_.data(); }
        [[nodiscard]] constexpr const value_type *data() const noexcept { return pod_.data(); }

        // Get underlying pod
        [[nodiscard]] constexpr pod_type &pod() noexcept { return pod_; }
        [[nodiscard]] constexpr const pod_type &pod() const noexcept { return pod_; }

        // Implicit conversion to pod_type (allows simd::Complex -> dp::mat::vector<complex>)
        constexpr operator pod_type &() noexcept { return pod_; }
        constexpr operator const pod_type &() const noexcept { return pod_; }

        // Get SIMD view (for advanced operations)
        [[nodiscard]] view_type as_view() noexcept { return view(pod_); }
        [[nodiscard]] view_type as_view() const noexcept { return view(pod_); }

        // ===== SIZE =====

        [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }

        // ===== FILL OPERATIONS =====

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

        // ===== FACTORY FUNCTIONS =====

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

        // ===== IN-PLACE OPERATIONS (SIMD accelerated) =====

        void conjugate_inplace() noexcept { as_view().conjugate_inplace(); }

        void normalize_inplace() noexcept { as_view().normalize_inplace(); }

        void negate_inplace() noexcept { as_view().negate_inplace(); }

        void scale_inplace(T scalar) noexcept { as_view().scale_inplace(scalar); }

        // ===== OPERATIONS RETURNING NEW ARRAY =====

        [[nodiscard]] Complex conjugate() const noexcept {
            Complex result;
            (void)as_view().conjugate_to(result.data());
            return result;
        }

        [[nodiscard]] Complex normalized() const noexcept {
            Complex result;
            (void)as_view().normalized_to(result.data());
            return result;
        }

        // ===== BINARY OPERATIONS =====

        [[nodiscard]] Complex operator+(const Complex &other) const noexcept {
            Complex result;
            (void)as_view().add_to(other.as_view(), result.data());
            return result;
        }

        [[nodiscard]] Complex operator-(const Complex &other) const noexcept {
            Complex result;
            (void)as_view().subtract_to(other.as_view(), result.data());
            return result;
        }

        [[nodiscard]] Complex operator*(const Complex &other) const noexcept {
            Complex result;
            (void)as_view().multiply_to(other.as_view(), result.data());
            return result;
        }

        [[nodiscard]] Complex operator/(const Complex &other) const noexcept {
            Complex result;
            (void)as_view().divide_to(other.as_view(), result.data());
            return result;
        }

        // Scalar multiplication
        [[nodiscard]] Complex operator*(T scalar) const noexcept {
            Complex result = *this;
            result.scale_inplace(scalar);
            return result;
        }

        [[nodiscard]] friend Complex operator*(T scalar, const Complex &c) noexcept { return c * scalar; }

        // ===== COMPOUND ASSIGNMENT =====

        Complex &operator+=(const Complex &other) noexcept {
            as_view().add_to(other.as_view(), data());
            return *this;
        }

        Complex &operator-=(const Complex &other) noexcept {
            as_view().subtract_to(other.as_view(), data());
            return *this;
        }

        Complex &operator*=(const Complex &other) noexcept {
            as_view().multiply_to(other.as_view(), data());
            return *this;
        }

        Complex &operator/=(const Complex &other) noexcept {
            as_view().divide_to(other.as_view(), data());
            return *this;
        }

        Complex &operator*=(T scalar) noexcept {
            scale_inplace(scalar);
            return *this;
        }

        // ===== REDUCTION OPERATIONS =====

        void magnitudes(T *out) const noexcept { as_view().magnitudes_to(out); }

        void phases(T *out) const noexcept { as_view().phases_to(out); }

        void real_parts(T *out) const noexcept { as_view().real_parts_to(out); }

        void imag_parts(T *out) const noexcept { as_view().imag_parts_to(out); }

        [[nodiscard]] value_type sum() const noexcept { return as_view().sum(); }

        [[nodiscard]] value_type dot(const Complex &other) const noexcept { return as_view().dot(other.as_view()); }

        // ===== STREAM OUTPUT =====

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

        // ===== ITERATORS =====

        value_type *begin() noexcept { return pod_.data(); }
        value_type *end() noexcept { return pod_.data() + N; }
        const value_type *begin() const noexcept { return pod_.data(); }
        const value_type *end() const noexcept { return pod_.data() + N; }
    };

    // ===== TYPE ALIASES =====

    template <std::size_t N> using Complexf = Complex<float, N>;
    template <std::size_t N> using Complexd = Complex<double, N>;

} // namespace optinum::simd
