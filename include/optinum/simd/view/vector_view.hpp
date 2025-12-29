#pragma once

// =============================================================================
// optinum/simd/view/vector_view.hpp
// vector_view<T,W> - Non-owning view over a 1D array with SIMD access
// =============================================================================

#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/backend/norm.hpp>
#include <optinum/simd/backend/reduce.hpp>
#include <optinum/simd/kernel.hpp>
#include <optinum/simd/view/slice.hpp>

#include <cmath>

namespace optinum::simd {

    // =============================================================================
    // vector_view<T, W> - Rank-1 view (1D array)
    //
    // Provides pack-based iteration over a 1D array:
    //   - size(): number of elements
    //   - operator[]: scalar element access
    //   - load_pack(i): load the i-th pack
    //   - store_pack(i, v): store to the i-th pack
    //   - num_packs(): total number of complete + partial packs
    //   - tail_size(): number of valid elements in the last pack
    // =============================================================================

    template <typename T, std::size_t W> struct vector_view {
        using value_type = std::remove_const_t<T>;
        using kernel_type = Kernel<T, W, 1>;
        static constexpr std::size_t width = W;
        static constexpr std::size_t rank = 1;

        kernel_type kernel_;

        // ==========================================================================
        // Constructors
        // ==========================================================================

        OPTINUM_INLINE constexpr vector_view() noexcept = default;

        OPTINUM_INLINE constexpr vector_view(T *ptr, std::size_t n) noexcept : kernel_(ptr, {n}, {1}) {}

        OPTINUM_INLINE constexpr explicit vector_view(const kernel_type &k) noexcept : kernel_(k) {}

        // ==========================================================================
        // Size queries
        // ==========================================================================

        OPTINUM_INLINE constexpr std::size_t size() const noexcept { return kernel_.extent(0); }

        OPTINUM_INLINE constexpr std::size_t num_packs() const noexcept { return kernel_.num_packs(); }

        OPTINUM_INLINE constexpr std::size_t tail_size() const noexcept { return kernel_.tail_size(); }

        OPTINUM_INLINE constexpr bool is_contiguous() const noexcept { return kernel_.is_contiguous(); }

        // ==========================================================================
        // Element access (scalar) - integer index
        // ==========================================================================

        OPTINUM_INLINE value_type &operator[](std::size_t i) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot get non-const reference from const view");
            return kernel_.at_linear(i);
        }

        OPTINUM_INLINE const value_type &at(std::size_t i) const noexcept { return kernel_.at_linear_const(i); }

        // ==========================================================================
        // Pack access (SIMD)
        // ==========================================================================

        OPTINUM_INLINE pack<value_type, W> load_pack(std::size_t pack_idx) const noexcept {
            return kernel_.load_pack(pack_idx);
        }

        OPTINUM_INLINE void store_pack(std::size_t pack_idx, const pack<value_type, W> &v) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot store to const view");
            kernel_.store_pack(pack_idx, v);
        }

        // Tail-safe pack access (loads/stores only valid elements)
        OPTINUM_INLINE pack<value_type, W> load_pack_tail(std::size_t pack_idx) const noexcept {
            const std::size_t valid = (pack_idx == num_packs() - 1) ? tail_size() : W;
            return kernel_.load_pack_tail(pack_idx, valid);
        }

        OPTINUM_INLINE void store_pack_tail(std::size_t pack_idx, const pack<value_type, W> &v) const noexcept {
            static_assert(!std::is_const_v<T>, "Cannot store to const view");
            const std::size_t valid = (pack_idx == num_packs() - 1) ? tail_size() : W;
            kernel_.store_pack_tail(pack_idx, v, valid);
        }

        // ==========================================================================
        // Data access
        // ==========================================================================

        OPTINUM_INLINE T *data() const noexcept { return kernel_.data(); }

        OPTINUM_INLINE const value_type *data_const() const noexcept { return kernel_.data_const(); }

        // ==========================================================================
        // Subview (slice)
        // ==========================================================================

        OPTINUM_INLINE vector_view subview(std::size_t offset, std::size_t count) const noexcept {
            return vector_view(kernel_.data() + offset, count);
        }

        // Slicing with seq/fseq/all - use slice() method
        template <typename Slice> OPTINUM_INLINE vector_view slice(const Slice &s) const noexcept {
            static_assert(is_slice_v<Slice> || is_fixed_index_v<Slice>, "Invalid slice type");

            // Resolve slice to concrete indices
            seq sl = resolve_slice(s, size());

            // If step == 1, we can create a contiguous subview
            if (sl.step == 1) {
                return vector_view(kernel_.data() + sl.start, sl.size());
            }

            // For strided slicing, create a view with custom stride
            // Note: This requires the Kernel to support non-unit stride
            // For now, just return a contiguous copy of the indices (simplified)
            return vector_view(kernel_.data() + sl.start, sl.size());
        }

        // ==========================================================================
        // Fill operations (in-place)
        // ==========================================================================

        OPTINUM_INLINE vector_view &fill(value_type value) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot fill const view");
            backend::fill_runtime<value_type>(data(), size(), value);
            return *this;
        }

        OPTINUM_INLINE vector_view &iota(value_type start = value_type{0}, value_type step = value_type{1}) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot iota const view");
            const std::size_t n = size();
            const std::size_t main = backend::main_loop_count_runtime(n, W);

            // Create pack with sequential offsets: [0, 1, 2, 3, ...] * step
            alignas(32) value_type offsets[W];
            for (std::size_t j = 0; j < W; ++j) {
                offsets[j] = static_cast<value_type>(j) * step;
            }
            const pack<value_type, W> offset_pack = pack<value_type, W>::loadu(offsets);
            const pack<value_type, W> step_pack(static_cast<value_type>(W) * step);

            pack<value_type, W> current(start);
            current = current + offset_pack;

            for (std::size_t i = 0; i < main; i += W) {
                current.storeu(data() + i);
                current = current + step_pack;
            }

            // Tail elements
            for (std::size_t i = main; i < n; ++i) {
                data()[i] = start + static_cast<value_type>(i) * step;
            }
            return *this;
        }

        // ==========================================================================
        // Static factory functions (write to output pointer)
        // ==========================================================================

        static OPTINUM_INLINE void zeros(T *dst, std::size_t n) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot write to const pointer");
            backend::fill_runtime<value_type>(dst, n, value_type{0});
        }

        static OPTINUM_INLINE void ones(T *dst, std::size_t n) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot write to const pointer");
            backend::fill_runtime<value_type>(dst, n, value_type{1});
        }

        static OPTINUM_INLINE void arange(T *dst, std::size_t n, value_type start = value_type{0},
                                          value_type step = value_type{1}) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot write to const pointer");
            vector_view(dst, n).iota(start, step);
        }

        // ==========================================================================
        // Compound assignment operators (in-place, SIMD-accelerated)
        // ==========================================================================

        template <typename U, std::size_t W2>
        OPTINUM_INLINE vector_view &operator+=(const vector_view<U, W2> &rhs) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            backend::add_runtime<value_type>(data(), data(), rhs.data_const(), size());
            return *this;
        }

        template <typename U, std::size_t W2>
        OPTINUM_INLINE vector_view &operator-=(const vector_view<U, W2> &rhs) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            backend::sub_runtime<value_type>(data(), data(), rhs.data_const(), size());
            return *this;
        }

        template <typename U, std::size_t W2>
        OPTINUM_INLINE vector_view &operator*=(const vector_view<U, W2> &rhs) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            backend::mul_runtime<value_type>(data(), data(), rhs.data_const(), size());
            return *this;
        }

        template <typename U, std::size_t W2>
        OPTINUM_INLINE vector_view &operator/=(const vector_view<U, W2> &rhs) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            backend::div_runtime<value_type>(data(), data(), rhs.data_const(), size());
            return *this;
        }

        // Scalar compound assignment
        OPTINUM_INLINE vector_view &operator*=(value_type scalar) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            backend::mul_scalar_runtime<value_type>(data(), data(), scalar, size());
            return *this;
        }

        OPTINUM_INLINE vector_view &operator/=(value_type scalar) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            backend::div_scalar_runtime<value_type>(data(), data(), scalar, size());
            return *this;
        }

        OPTINUM_INLINE vector_view &operator+=(value_type scalar) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            const std::size_t n = size();
            const std::size_t main = backend::main_loop_count_runtime(n, W);
            const pack<value_type, W> s(scalar);

            for (std::size_t i = 0; i < main; i += W) {
                auto v = pack<value_type, W>::loadu(data() + i);
                (v + s).storeu(data() + i);
            }
            for (std::size_t i = main; i < n; ++i) {
                data()[i] += scalar;
            }
            return *this;
        }

        OPTINUM_INLINE vector_view &operator-=(value_type scalar) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot modify const view");
            const std::size_t n = size();
            const std::size_t main = backend::main_loop_count_runtime(n, W);
            const pack<value_type, W> s(scalar);

            for (std::size_t i = 0; i < main; i += W) {
                auto v = pack<value_type, W>::loadu(data() + i);
                (v - s).storeu(data() + i);
            }
            for (std::size_t i = main; i < n; ++i) {
                data()[i] -= scalar;
            }
            return *this;
        }

        // ==========================================================================
        // Reduction operations (return scalar)
        // ==========================================================================

        [[nodiscard]] OPTINUM_INLINE value_type sum() const noexcept {
            return backend::reduce_sum_runtime<value_type>(data_const(), size());
        }

        [[nodiscard]] OPTINUM_INLINE value_type min() const noexcept {
            const std::size_t n = size();
            if (n == 0)
                return value_type{};

            const std::size_t main = backend::main_loop_count_runtime(n, W);
            value_type result = data_const()[0];

            if (n >= W) {
                pack<value_type, W> acc = pack<value_type, W>::loadu(data_const());
                for (std::size_t i = W; i < main; i += W) {
                    acc = pack<value_type, W>::min(acc, pack<value_type, W>::loadu(data_const() + i));
                }
                result = acc.hmin();
            }

            for (std::size_t i = main; i < n; ++i) {
                result = (data_const()[i] < result) ? data_const()[i] : result;
            }
            return result;
        }

        [[nodiscard]] OPTINUM_INLINE value_type max() const noexcept {
            const std::size_t n = size();
            if (n == 0)
                return value_type{};

            const std::size_t main = backend::main_loop_count_runtime(n, W);
            value_type result = data_const()[0];

            if (n >= W) {
                pack<value_type, W> acc = pack<value_type, W>::loadu(data_const());
                for (std::size_t i = W; i < main; i += W) {
                    acc = pack<value_type, W>::max(acc, pack<value_type, W>::loadu(data_const() + i));
                }
                result = acc.hmax();
            }

            for (std::size_t i = main; i < n; ++i) {
                result = (data_const()[i] > result) ? data_const()[i] : result;
            }
            return result;
        }

        template <typename U, std::size_t W2>
        [[nodiscard]] OPTINUM_INLINE value_type dot(const vector_view<U, W2> &rhs) const noexcept {
            return backend::dot_runtime<value_type>(data_const(), rhs.data_const(), size());
        }

        [[nodiscard]] OPTINUM_INLINE value_type norm_squared() const noexcept {
            return backend::dot_runtime<value_type>(data_const(), data_const(), size());
        }

        [[nodiscard]] OPTINUM_INLINE value_type norm() const noexcept { return std::sqrt(norm_squared()); }

        // ==========================================================================
        // Normalization (in-place)
        // ==========================================================================

        OPTINUM_INLINE vector_view &normalize_inplace() noexcept {
            static_assert(!std::is_const_v<T>, "Cannot normalize const view");
            const value_type n = norm();
            if (n > value_type{}) {
                backend::div_scalar_runtime<value_type>(data(), data(), n, size());
            }
            return *this;
        }

        // Normalize and write result to output
        OPTINUM_INLINE void normalize_to(value_type *dst) const noexcept {
            backend::normalize_runtime<value_type>(dst, data_const(), size());
        }

        // ==========================================================================
        // Element-wise operations (write to output pointer)
        // ==========================================================================

        template <typename U, std::size_t W2>
        OPTINUM_INLINE void add_to(value_type *dst, const vector_view<U, W2> &rhs) const noexcept {
            backend::add_runtime<value_type>(dst, data_const(), rhs.data_const(), size());
        }

        template <typename U, std::size_t W2>
        OPTINUM_INLINE void sub_to(value_type *dst, const vector_view<U, W2> &rhs) const noexcept {
            backend::sub_runtime<value_type>(dst, data_const(), rhs.data_const(), size());
        }

        template <typename U, std::size_t W2>
        OPTINUM_INLINE void mul_to(value_type *dst, const vector_view<U, W2> &rhs) const noexcept {
            backend::mul_runtime<value_type>(dst, data_const(), rhs.data_const(), size());
        }

        template <typename U, std::size_t W2>
        OPTINUM_INLINE void div_to(value_type *dst, const vector_view<U, W2> &rhs) const noexcept {
            backend::div_runtime<value_type>(dst, data_const(), rhs.data_const(), size());
        }

        OPTINUM_INLINE void mul_scalar_to(value_type *dst, value_type scalar) const noexcept {
            backend::mul_scalar_runtime<value_type>(dst, data_const(), scalar, size());
        }

        OPTINUM_INLINE void div_scalar_to(value_type *dst, value_type scalar) const noexcept {
            backend::div_scalar_runtime<value_type>(dst, data_const(), scalar, size());
        }

        // ==========================================================================
        // Copy operations
        // ==========================================================================

        OPTINUM_INLINE void copy_to(value_type *dst) const noexcept {
            const std::size_t n = size();
            const std::size_t main = backend::main_loop_count_runtime(n, W);

            for (std::size_t i = 0; i < main; i += W) {
                pack<value_type, W>::loadu(data_const() + i).storeu(dst + i);
            }
            for (std::size_t i = main; i < n; ++i) {
                dst[i] = data_const()[i];
            }
        }

        OPTINUM_INLINE void copy_from(const value_type *src) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot copy to const view");
            const std::size_t n = size();
            const std::size_t main = backend::main_loop_count_runtime(n, W);

            for (std::size_t i = 0; i < main; i += W) {
                pack<value_type, W>::loadu(src + i).storeu(data() + i);
            }
            for (std::size_t i = main; i < n; ++i) {
                data()[i] = src[i];
            }
        }

        template <typename U, std::size_t W2> OPTINUM_INLINE void copy_from(const vector_view<U, W2> &src) noexcept {
            static_assert(!std::is_const_v<T>, "Cannot copy to const view");
            copy_from(src.data_const());
        }

        // ==========================================================================
        // Negate (in-place)
        // ==========================================================================

        OPTINUM_INLINE vector_view &negate_inplace() noexcept {
            static_assert(!std::is_const_v<T>, "Cannot negate const view");
            const std::size_t n = size();
            const std::size_t main = backend::main_loop_count_runtime(n, W);

            for (std::size_t i = 0; i < main; i += W) {
                auto v = pack<value_type, W>::loadu(data() + i);
                (-v).storeu(data() + i);
            }
            for (std::size_t i = main; i < n; ++i) {
                data()[i] = -data()[i];
            }
            return *this;
        }

        OPTINUM_INLINE void negate_to(value_type *dst) const noexcept {
            const std::size_t n = size();
            const std::size_t main = backend::main_loop_count_runtime(n, W);

            for (std::size_t i = 0; i < main; i += W) {
                auto v = pack<value_type, W>::loadu(data_const() + i);
                (-v).storeu(dst + i);
            }
            for (std::size_t i = main; i < n; ++i) {
                dst[i] = -data_const()[i];
            }
        }
    };

    // =============================================================================
    // Free function operators for vector_view (write to output pointer)
    // =============================================================================

    // Element-wise addition: dst = lhs + rhs
    template <typename T, std::size_t W, typename U, std::size_t W2>
    OPTINUM_INLINE void add(typename vector_view<T, W>::value_type *dst, const vector_view<T, W> &lhs,
                            const vector_view<U, W2> &rhs) noexcept {
        lhs.add_to(dst, rhs);
    }

    // Element-wise subtraction: dst = lhs - rhs
    template <typename T, std::size_t W, typename U, std::size_t W2>
    OPTINUM_INLINE void sub(typename vector_view<T, W>::value_type *dst, const vector_view<T, W> &lhs,
                            const vector_view<U, W2> &rhs) noexcept {
        lhs.sub_to(dst, rhs);
    }

    // Element-wise multiplication: dst = lhs * rhs
    template <typename T, std::size_t W, typename U, std::size_t W2>
    OPTINUM_INLINE void mul(typename vector_view<T, W>::value_type *dst, const vector_view<T, W> &lhs,
                            const vector_view<U, W2> &rhs) noexcept {
        lhs.mul_to(dst, rhs);
    }

    // Element-wise division: dst = lhs / rhs
    template <typename T, std::size_t W, typename U, std::size_t W2>
    OPTINUM_INLINE void div(typename vector_view<T, W>::value_type *dst, const vector_view<T, W> &lhs,
                            const vector_view<U, W2> &rhs) noexcept {
        lhs.div_to(dst, rhs);
    }

    // Scalar multiplication: dst = v * scalar
    template <typename T, std::size_t W>
    OPTINUM_INLINE void mul(typename vector_view<T, W>::value_type *dst, const vector_view<T, W> &v,
                            typename vector_view<T, W>::value_type scalar) noexcept {
        v.mul_scalar_to(dst, scalar);
    }

    // Scalar division: dst = v / scalar
    template <typename T, std::size_t W>
    OPTINUM_INLINE void div(typename vector_view<T, W>::value_type *dst, const vector_view<T, W> &v,
                            typename vector_view<T, W>::value_type scalar) noexcept {
        v.div_scalar_to(dst, scalar);
    }

    // Dot product
    template <typename T, std::size_t W, typename U, std::size_t W2>
    [[nodiscard]] OPTINUM_INLINE typename vector_view<T, W>::value_type dot(const vector_view<T, W> &lhs,
                                                                            const vector_view<U, W2> &rhs) noexcept {
        return lhs.dot(rhs);
    }

    // Sum
    template <typename T, std::size_t W>
    [[nodiscard]] OPTINUM_INLINE typename vector_view<T, W>::value_type sum(const vector_view<T, W> &v) noexcept {
        return v.sum();
    }

    // Norm
    template <typename T, std::size_t W>
    [[nodiscard]] OPTINUM_INLINE typename vector_view<T, W>::value_type norm(const vector_view<T, W> &v) noexcept {
        return v.norm();
    }

    // Norm squared
    template <typename T, std::size_t W>
    [[nodiscard]] OPTINUM_INLINE typename vector_view<T, W>::value_type
    norm_squared(const vector_view<T, W> &v) noexcept {
        return v.norm_squared();
    }

    // Normalize to output
    template <typename T, std::size_t W>
    OPTINUM_INLINE void normalize(typename vector_view<T, W>::value_type *dst, const vector_view<T, W> &v) noexcept {
        v.normalize_to(dst);
    }

    // Min/Max
    template <typename T, std::size_t W>
    [[nodiscard]] OPTINUM_INLINE typename vector_view<T, W>::value_type min(const vector_view<T, W> &v) noexcept {
        return v.min();
    }

    template <typename T, std::size_t W>
    [[nodiscard]] OPTINUM_INLINE typename vector_view<T, W>::value_type max(const vector_view<T, W> &v) noexcept {
        return v.max();
    }

} // namespace optinum::simd
