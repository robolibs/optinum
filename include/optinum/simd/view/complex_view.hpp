#pragma once

// =============================================================================
// optinum/simd/view/complex_view.hpp
// complex_view<T,W> - Non-owning view over complex array with SIMD access
// =============================================================================

#include <datapod/matrix/math/complex.hpp>
#include <optinum/simd/pack/complex.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace optinum::simd {

    namespace dp = ::datapod;

    // =============================================================================
    // complex_view<T, W> - SIMD view over complex number arrays
    //
    // Provides transparent SIMD access to dp::mat::complex<T> arrays.
    // User works with complex numbers directly; SIMD is handled internally.
    //
    // Usage:
    //   dp::mat::complex<double> nums[8];
    //   auto cv = complex_view<double, 4>(nums, 8);
    //   cv.conjugate_inplace();  // SIMD under the hood
    //
    // Or via bridge:
    //   auto cv = view(nums);  // auto-detect width
    // =============================================================================

    template <typename T, std::size_t W> class complex_view {
        static_assert(std::is_floating_point_v<T>, "complex_view requires floating-point type");
        static_assert(W > 0, "complex_view requires W > 0");

      public:
        using value_type = dp::mat::complex<T>;
        using real_type = T;
        using pack_type = pack<value_type, W>;
        using real_pack = pack<T, W>;

        static constexpr std::size_t width = W;

      private:
        value_type *ptr_ = nullptr;
        std::size_t size_ = 0;

      public:
        // ===== CONSTRUCTORS =====

        OPTINUM_INLINE constexpr complex_view() noexcept = default;

        OPTINUM_INLINE constexpr complex_view(value_type *ptr, std::size_t n) noexcept : ptr_(ptr), size_(n) {}

        // From const pointer (creates const view)
        OPTINUM_INLINE constexpr complex_view(const value_type *ptr, std::size_t n) noexcept
            : ptr_(const_cast<value_type *>(ptr)), size_(n) {}

        // ===== SIZE QUERIES =====

        [[nodiscard]] OPTINUM_INLINE constexpr std::size_t size() const noexcept { return size_; }

        [[nodiscard]] OPTINUM_INLINE constexpr std::size_t num_packs() const noexcept { return (size_ + W - 1) / W; }

        [[nodiscard]] OPTINUM_INLINE constexpr std::size_t tail_size() const noexcept {
            const std::size_t rem = size_ % W;
            return (rem == 0) ? W : rem;
        }

        [[nodiscard]] OPTINUM_INLINE constexpr bool empty() const noexcept { return size_ == 0; }

        // ===== ELEMENT ACCESS (scalar) =====

        [[nodiscard]] OPTINUM_INLINE value_type &operator[](std::size_t i) noexcept { return ptr_[i]; }

        [[nodiscard]] OPTINUM_INLINE const value_type &operator[](std::size_t i) const noexcept { return ptr_[i]; }

        [[nodiscard]] OPTINUM_INLINE value_type *data() noexcept { return ptr_; }
        [[nodiscard]] OPTINUM_INLINE const value_type *data() const noexcept { return ptr_; }

        // ===== PACK ACCESS (SIMD) =====

        // Load pack at index (loads W complex numbers)
        [[nodiscard]] OPTINUM_INLINE pack_type load_pack(std::size_t pack_idx) const noexcept {
            return pack_type::loadu_interleaved(ptr_ + pack_idx * W);
        }

        // Store pack at index
        OPTINUM_INLINE void store_pack(std::size_t pack_idx, const pack_type &p) noexcept {
            p.storeu_interleaved(ptr_ + pack_idx * W);
        }

        // Tail-safe load (handles partial packs at end)
        [[nodiscard]] OPTINUM_INLINE pack_type load_pack_safe(std::size_t pack_idx) const noexcept {
            const std::size_t start = pack_idx * W;
            const std::size_t valid = (pack_idx == num_packs() - 1) ? tail_size() : W;

            if (valid == W) {
                return pack_type::loadu_interleaved(ptr_ + start);
            }

            // Partial load - load valid elements, zero the rest
            alignas(32) T real_vals[W] = {};
            alignas(32) T imag_vals[W] = {};

            for (std::size_t i = 0; i < valid; ++i) {
                real_vals[i] = ptr_[start + i].real;
                imag_vals[i] = ptr_[start + i].imag;
            }

            return pack_type::loadu_split(real_vals, imag_vals);
        }

        // Tail-safe store
        OPTINUM_INLINE void store_pack_safe(std::size_t pack_idx, const pack_type &p) noexcept {
            const std::size_t start = pack_idx * W;
            const std::size_t valid = (pack_idx == num_packs() - 1) ? tail_size() : W;

            if (valid == W) {
                p.storeu_interleaved(ptr_ + start);
                return;
            }

            // Partial store
            alignas(32) T real_vals[W];
            alignas(32) T imag_vals[W];

            p.storeu_split(real_vals, imag_vals);

            for (std::size_t i = 0; i < valid; ++i) {
                ptr_[start + i].real = real_vals[i];
                ptr_[start + i].imag = imag_vals[i];
            }
        }

        // ===== FILL OPERATIONS =====

        // Fill all elements with a value
        void fill(const value_type &val) noexcept {
            pack_type p(val);
            for (std::size_t i = 0; i < num_packs(); ++i) {
                store_pack_safe(i, p);
            }
        }

        // Fill with real value only (imag = 0)
        void fill_real(T real_val) noexcept { fill(value_type{real_val, T{}}); }

        // ===== IN-PLACE OPERATIONS =====

        // Conjugate in place
        void conjugate_inplace() noexcept {
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);
                store_pack_safe(i, p.conjugate());
            }
        }

        // Normalize in place (make unit magnitude)
        void normalize_inplace() noexcept {
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);
                auto mag = p.magnitude();
                store_pack_safe(i, pack_type(p.real() / mag, p.imag() / mag));
            }
        }

        // Negate in place
        void negate_inplace() noexcept {
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);
                store_pack_safe(i, -p);
            }
        }

        // Scale by real scalar
        void scale_inplace(T scalar) noexcept {
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);
                store_pack_safe(i, p * scalar);
            }
        }

        // ===== OPERATIONS RETURNING NEW ARRAY =====

        // Conjugate to output
        [[nodiscard]] complex_view conjugate_to(value_type *out) const noexcept {
            complex_view result(out, size_);
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);
                result.store_pack_safe(i, p.conjugate());
            }
            return result;
        }

        // Normalize to output
        [[nodiscard]] complex_view normalize_to(value_type *out) const noexcept {
            complex_view result(out, size_);
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);
                auto mag = p.magnitude();
                result.store_pack_safe(i, pack_type(p.real() / mag, p.imag() / mag));
            }
            return result;
        }

        // ===== BINARY OPERATIONS =====

        // Element-wise addition: this + other -> out
        [[nodiscard]] complex_view add_to(const complex_view &other, value_type *out) const noexcept {
            complex_view result(out, size_);
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto a = load_pack_safe(i);
                auto b = other.load_pack_safe(i);
                result.store_pack_safe(i, a + b);
            }
            return result;
        }

        // Element-wise subtraction: this - other -> out
        [[nodiscard]] complex_view subtract_to(const complex_view &other, value_type *out) const noexcept {
            complex_view result(out, size_);
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto a = load_pack_safe(i);
                auto b = other.load_pack_safe(i);
                result.store_pack_safe(i, a - b);
            }
            return result;
        }

        // Element-wise multiplication: this * other -> out
        [[nodiscard]] complex_view multiply_to(const complex_view &other, value_type *out) const noexcept {
            complex_view result(out, size_);
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto a = load_pack_safe(i);
                auto b = other.load_pack_safe(i);
                result.store_pack_safe(i, a * b);
            }
            return result;
        }

        // Element-wise division: this / other -> out
        [[nodiscard]] complex_view divide_to(const complex_view &other, value_type *out) const noexcept {
            complex_view result(out, size_);
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto a = load_pack_safe(i);
                auto b = other.load_pack_safe(i);
                result.store_pack_safe(i, a / b);
            }
            return result;
        }

        // ===== REDUCTION OPERATIONS =====

        // Compute magnitudes of all complex numbers -> output array
        void magnitudes_to(T *out) const noexcept {
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);
                auto mag = p.magnitude();

                alignas(32) T vals[W];
                mag.storeu(vals);

                const std::size_t start = i * W;
                const std::size_t valid = (i == num_packs() - 1) ? tail_size() : W;
                for (std::size_t j = 0; j < valid; ++j) {
                    out[start + j] = vals[j];
                }
            }
        }

        // Compute phases/arguments of all complex numbers -> output array
        void phases_to(T *out) const noexcept {
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);
                auto ph = p.phase();

                alignas(32) T vals[W];
                ph.storeu(vals);

                const std::size_t start = i * W;
                const std::size_t valid = (i == num_packs() - 1) ? tail_size() : W;
                for (std::size_t j = 0; j < valid; ++j) {
                    out[start + j] = vals[j];
                }
            }
        }

        // Extract real parts -> output array
        void real_parts_to(T *out) const noexcept {
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);

                alignas(32) T vals[W];
                p.real().storeu(vals);

                const std::size_t start = i * W;
                const std::size_t valid = (i == num_packs() - 1) ? tail_size() : W;
                for (std::size_t j = 0; j < valid; ++j) {
                    out[start + j] = vals[j];
                }
            }
        }

        // Extract imaginary parts -> output array
        void imag_parts_to(T *out) const noexcept {
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);

                alignas(32) T vals[W];
                p.imag().storeu(vals);

                const std::size_t start = i * W;
                const std::size_t valid = (i == num_packs() - 1) ? tail_size() : W;
                for (std::size_t j = 0; j < valid; ++j) {
                    out[start + j] = vals[j];
                }
            }
        }

        // Sum all complex numbers (horizontal reduction)
        [[nodiscard]] value_type sum() const noexcept {
            T real_sum = T{};
            T imag_sum = T{};

            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);
                auto h = p.hsum();

                const std::size_t valid = (i == num_packs() - 1) ? tail_size() : W;
                if (valid == W) {
                    real_sum += h.real;
                    imag_sum += h.imag;
                } else {
                    // For partial pack, sum only valid elements
                    alignas(32) T real_vals[W];
                    alignas(32) T imag_vals[W];
                    p.real().storeu(real_vals);
                    p.imag().storeu(imag_vals);
                    for (std::size_t j = 0; j < valid; ++j) {
                        real_sum += real_vals[j];
                        imag_sum += imag_vals[j];
                    }
                }
            }

            return value_type{real_sum, imag_sum};
        }

        // Dot product (conjugate first, then multiply, then sum)
        [[nodiscard]] value_type dot(const complex_view &other) const noexcept {
            T real_sum = T{};
            T imag_sum = T{};

            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto a = load_pack_safe(i);
                auto b = other.load_pack_safe(i);

                // Hermitian inner product: conj(a) * b
                auto prod = a.conjugate() * b;
                auto h = prod.hsum();

                const std::size_t valid = (i == num_packs() - 1) ? tail_size() : W;
                if (valid == W) {
                    real_sum += h.real;
                    imag_sum += h.imag;
                } else {
                    alignas(32) T real_vals[W];
                    alignas(32) T imag_vals[W];
                    prod.real().storeu(real_vals);
                    prod.imag().storeu(imag_vals);
                    for (std::size_t j = 0; j < valid; ++j) {
                        real_sum += real_vals[j];
                        imag_sum += imag_vals[j];
                    }
                }
            }

            return value_type{real_sum, imag_sum};
        }

        // ===== SUBVIEW =====

        [[nodiscard]] complex_view subview(std::size_t offset, std::size_t count) const noexcept {
            return complex_view(ptr_ + offset, count);
        }

        // ===== ITERATORS (for range-based for) =====

        value_type *begin() noexcept { return ptr_; }
        value_type *end() noexcept { return ptr_ + size_; }
        const value_type *begin() const noexcept { return ptr_; }
        const value_type *end() const noexcept { return ptr_ + size_; }
    };

    // ===== TYPE ALIASES =====

    template <std::size_t W> using complex_viewf = complex_view<float, W>;
    template <std::size_t W> using complex_viewd = complex_view<double, W>;

} // namespace optinum::simd
