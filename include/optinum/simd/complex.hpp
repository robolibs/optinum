#pragma once

// =============================================================================
// optinum/simd/complex.hpp
// Non-owning view over complex number arrays with transparent SIMD operations
// =============================================================================
//
// This file provides a non-owning view over complex number arrays.
// For low-level SIMD pack operations, use pack/complex.hpp
// For the underlying complex_view with SIMD width, use view/complex_view.hpp
//
// Usage:
//   dp::mat::vector<dp::mat::complex<double>, 8> data;
//   Complex<double, 8> view(data);  // Non-owning view
//   view.conjugate_inplace();       // SIMD under the hood
//
// Or from raw pointer:
//   dp::mat::complex<double> nums[8];
//   Complex<double, 8> view(nums);
//   view.conjugate_inplace();
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
     * @brief Non-owning view over complex number arrays with transparent SIMD operations
     *
     * Non-owning view for N complex numbers. Operations use SIMD internally
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
        using size_type = std::size_t;
        using reference = value_type &;
        using const_reference = const value_type &;
        using pointer = value_type *;
        using const_pointer = const value_type *;
        using iterator = value_type *;
        using const_iterator = const value_type *;

        static constexpr std::size_t extent = N;
        static constexpr std::size_t rank = 1;

      private:
        value_type *ptr_;

      public:
        // ===== CONSTRUCTORS =====

        // Default constructor (null view)
        constexpr Complex() noexcept : ptr_(nullptr) {}

        // Constructor from raw pointer (non-owning view)
        constexpr explicit Complex(value_type *ptr) noexcept : ptr_(ptr) {}

        // Constructor from pod_type reference (non-owning view)
        constexpr Complex(pod_type &pod) noexcept : ptr_(pod.data()) {}
        constexpr Complex(const pod_type &pod) noexcept : ptr_(const_cast<value_type *>(pod.data())) {}

        // Copy/move (shallow - just copies pointer)
        constexpr Complex(const Complex &) = default;
        constexpr Complex(Complex &&) noexcept = default;
        constexpr Complex &operator=(const Complex &) = default;
        constexpr Complex &operator=(Complex &&) noexcept = default;

        // ===== VALIDITY CHECK =====

        /// Check if view is valid (non-null)
        [[nodiscard]] constexpr bool valid() const noexcept { return ptr_ != nullptr; }
        constexpr explicit operator bool() const noexcept { return valid(); }

        // ===== ELEMENT ACCESS =====

        [[nodiscard]] constexpr reference operator[](std::size_t i) noexcept { return ptr_[i]; }
        [[nodiscard]] constexpr const_reference operator[](std::size_t i) const noexcept { return ptr_[i]; }

        // ===== RAW DATA ACCESS =====

        [[nodiscard]] constexpr pointer data() noexcept { return ptr_; }
        [[nodiscard]] constexpr const_pointer data() const noexcept { return ptr_; }

        // Get SIMD view (for advanced operations)
        [[nodiscard]] view_type as_view() noexcept { return view_type(ptr_, N); }
        [[nodiscard]] view_type as_view() const noexcept { return view_type(const_cast<value_type *>(ptr_), N); }

        // ===== SIZE =====

        [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }

        // ===== FILL OPERATIONS =====

        constexpr Complex &fill(const value_type &val) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                ptr_[i] = val;
            }
            return *this;
        }

        constexpr Complex &fill_real(T real_val) noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                ptr_[i] = value_type{real_val, T{}};
            }
            return *this;
        }

        // ===== IN-PLACE OPERATIONS (SIMD accelerated) =====

        Complex &conjugate_inplace() noexcept {
            as_view().conjugate_inplace();
            return *this;
        }

        Complex &normalize_inplace() noexcept {
            as_view().normalize_inplace();
            return *this;
        }

        Complex &negate_inplace() noexcept {
            as_view().negate_inplace();
            return *this;
        }

        Complex &scale_inplace(T scalar) noexcept {
            as_view().scale_inplace(scalar);
            return *this;
        }

        // ===== OPERATIONS WRITING TO OUTPUT =====

        void conjugate_to(value_type *out) const noexcept { (void)as_view().conjugate_to(out); }

        void normalize_to(value_type *out) const noexcept { (void)as_view().normalize_to(out); }

        // ===== BINARY OPERATIONS (writing to output) =====

        void add_to(const Complex &other, value_type *out) const noexcept {
            (void)as_view().add_to(other.as_view(), out);
        }

        void subtract_to(const Complex &other, value_type *out) const noexcept {
            (void)as_view().subtract_to(other.as_view(), out);
        }

        void multiply_to(const Complex &other, value_type *out) const noexcept {
            (void)as_view().multiply_to(other.as_view(), out);
        }

        void divide_to(const Complex &other, value_type *out) const noexcept {
            (void)as_view().divide_to(other.as_view(), out);
        }

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

        Complex &operator/=(T scalar) noexcept {
            scale_inplace(T{1} / scalar);
            return *this;
        }

        // ===== REDUCTION OPERATIONS =====

        void magnitudes_to(T *out) const noexcept { as_view().magnitudes_to(out); }

        void phases_to(T *out) const noexcept { as_view().phases_to(out); }

        void real_parts_to(T *out) const noexcept { as_view().real_parts_to(out); }

        void imag_parts_to(T *out) const noexcept { as_view().imag_parts_to(out); }

        [[nodiscard]] value_type sum() const noexcept { return as_view().sum(); }

        [[nodiscard]] value_type dot(const Complex &other) const noexcept { return as_view().dot(other.as_view()); }

        // ===== STREAM OUTPUT =====

        friend std::ostream &operator<<(std::ostream &os, const Complex &c) {
            os << "[";
            if (c.valid()) {
                for (std::size_t i = 0; i < N; ++i) {
                    os << "(" << c[i].real << "+" << c[i].imag << "i)";
                    if (i + 1 < N)
                        os << ", ";
                }
            } else {
                os << "null";
            }
            os << "]";
            return os;
        }

        // ===== ITERATORS =====

        iterator begin() noexcept { return ptr_; }
        iterator end() noexcept { return ptr_ + N; }
        const_iterator begin() const noexcept { return ptr_; }
        const_iterator end() const noexcept { return ptr_ + N; }
        const_iterator cbegin() const noexcept { return ptr_; }
        const_iterator cend() const noexcept { return ptr_ + N; }
    };

    // ===== TYPE ALIASES =====

    template <std::size_t N> using Complexf = Complex<float, N>;
    template <std::size_t N> using Complexd = Complex<double, N>;

} // namespace optinum::simd
