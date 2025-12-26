#pragma once

// =============================================================================
// optinum/simd/pack/complex.hpp
// SIMD pack specialization for complex numbers (split representation)
// =============================================================================

#include <datapod/matrix/math/complex.hpp>
#include <optinum/simd/pack/pack.hpp>

namespace optinum::simd {

    namespace dp = ::datapod;

    /**
     * @brief SIMD pack for complex numbers using split representation
     *
     * Stores real and imaginary parts in separate SIMD registers for efficiency:
     * - pack_real: [r0, r1, r2, ..., rW-1]
     * - pack_imag: [i0, i1, i2, ..., iW-1]
     *
     * This is more efficient than interleaved [r0, i0, r1, i1, ...] for most operations.
     */
    template <typename T, std::size_t W> class pack<dp::mat::complex<T>, W> {
        static_assert(std::is_floating_point_v<T>, "Complex pack requires floating-point type");

      public:
        using value_type = dp::mat::complex<T>;
        using real_type = T;
        using real_pack = pack<T, W>;

        static constexpr std::size_t width = W;

      private:
        real_pack real_; // Real parts
        real_pack imag_; // Imaginary parts

      public:
        // Constructors
        pack() noexcept = default;

        // Construct from real and imaginary packs
        pack(const real_pack &r, const real_pack &i) noexcept : real_(r), imag_(i) {}

        // Broadcast single complex value to all lanes
        explicit pack(const value_type &val) noexcept : real_(val.real), imag_(val.imag) {}

        // Construct from scalar (real part only)
        explicit pack(T real_val) noexcept : real_(real_val), imag_(T{}) {}

        // Zero initialization
        static pack zero() noexcept { return pack(real_pack(T{}), real_pack(T{})); }

        // Access real and imaginary parts
        [[nodiscard]] const real_pack &real() const noexcept { return real_; }
        [[nodiscard]] const real_pack &imag() const noexcept { return imag_; }
        [[nodiscard]] real_pack &real() noexcept { return real_; }
        [[nodiscard]] real_pack &imag() noexcept { return imag_; }

        // Load from interleaved memory: [r0, i0, r1, i1, ...]
        static pack loadu_interleaved(const value_type *ptr) noexcept {
            alignas(32) T real_vals[W];
            alignas(32) T imag_vals[W];
            for (std::size_t i = 0; i < W; ++i) {
                real_vals[i] = ptr[i].real;
                imag_vals[i] = ptr[i].imag;
            }
            return pack(real_pack::loadu(real_vals), real_pack::loadu(imag_vals));
        }

        // Store to interleaved memory: [r0, i0, r1, i1, ...]
        void storeu_interleaved(value_type *ptr) const noexcept {
            alignas(32) T real_vals[W];
            alignas(32) T imag_vals[W];
            real_.storeu(real_vals);
            imag_.storeu(imag_vals);
            for (std::size_t i = 0; i < W; ++i) {
                ptr[i].real = real_vals[i];
                ptr[i].imag = imag_vals[i];
            }
        }

        // Load from split memory: separate arrays for real and imaginary
        static pack loadu_split(const T *real_ptr, const T *imag_ptr) noexcept {
            return pack(real_pack::loadu(real_ptr), real_pack::loadu(imag_ptr));
        }

        // Store to split memory
        void storeu_split(T *real_ptr, T *imag_ptr) const noexcept {
            real_.storeu(real_ptr);
            imag_.storeu(imag_ptr);
        }

        // ===== Arithmetic operations =====

        // Addition: (a + bi) + (c + di) = (a+c) + (b+d)i
        pack operator+(const pack &other) const noexcept { return pack(real_ + other.real_, imag_ + other.imag_); }

        // Subtraction: (a + bi) - (c + di) = (a-c) + (b-d)i
        pack operator-(const pack &other) const noexcept { return pack(real_ - other.real_, imag_ - other.imag_); }

        // Multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        pack operator*(const pack &other) const noexcept {
            auto ac = real_ * other.real_;
            auto bd = imag_ * other.imag_;
            auto ad = real_ * other.imag_;
            auto bc = imag_ * other.real_;
            return pack(ac - bd, ad + bc);
        }

        // Scalar multiplication: (a + bi) * s = (as) + (bs)i
        pack operator*(T scalar) const noexcept { return pack(real_ * scalar, imag_ * scalar); }

        friend pack operator*(T scalar, const pack &p) noexcept { return p * scalar; }

        // Division: (a + bi) / (c + di) = [(ac + bd) + (bc - ad)i] / (c² + d²)
        pack operator/(const pack &other) const noexcept {
            auto c2_d2 = other.real_ * other.real_ + other.imag_ * other.imag_;
            auto ac = real_ * other.real_;
            auto bd = imag_ * other.imag_;
            auto bc = imag_ * other.real_;
            auto ad = real_ * other.imag_;
            return pack((ac + bd) / c2_d2, (bc - ad) / c2_d2);
        }

        // Unary negation
        pack operator-() const noexcept { return pack(-real_, -imag_); }

        // Compound assignment
        pack &operator+=(const pack &other) noexcept {
            real_ += other.real_;
            imag_ += other.imag_;
            return *this;
        }

        pack &operator-=(const pack &other) noexcept {
            real_ -= other.real_;
            imag_ -= other.imag_;
            return *this;
        }

        pack &operator*=(const pack &other) noexcept {
            *this = *this * other;
            return *this;
        }

        pack &operator*=(T scalar) noexcept {
            real_ *= scalar;
            imag_ *= scalar;
            return *this;
        }

        // ===== Complex operations =====

        // Conjugate: conj(a + bi) = a - bi
        [[nodiscard]] pack conjugate() const noexcept { return pack(real_, -imag_); }

        // Magnitude squared: |a + bi|² = a² + b²
        [[nodiscard]] real_pack magnitude_squared() const noexcept { return real_ * real_ + imag_ * imag_; }

        // Magnitude: |a + bi| = sqrt(a² + b²)
        [[nodiscard]] real_pack magnitude() const noexcept { return sqrt(magnitude_squared()); }

        // Phase/argument: arg(a + bi) = atan2(b, a)
        [[nodiscard]] real_pack phase() const noexcept { return atan2(imag_, real_); }

        // Horizontal sum (for reductions)
        [[nodiscard]] value_type hsum() const noexcept { return value_type{real_.hsum(), imag_.hsum()}; }
    };

} // namespace optinum::simd
