#pragma once

// =============================================================================
// optinum/simd/algo/reduce.hpp
// Reduction SIMD algorithms operating on views
// =============================================================================

#include <cmath>
#include <cstddef>
#include <optinum/simd/pack/pack.hpp>
#include <optinum/simd/view/vector_view.hpp>

namespace optinum::simd {

    // =============================================================================
    // dot: dot product of two vectors
    // Returns: sum(x[i] * y[i])
    // =============================================================================

    template <typename T, std::size_t W>
    OPTINUM_INLINE T dot(const vector_view<const T, W> &x, const vector_view<const T, W> &y) noexcept {
        const std::size_t n = x.size();
        if (n != y.size())
            return T(0); // Size mismatch

        const std::size_t num_packs = x.num_packs();
        pack<T, W> acc(T(0));

        // Process full packs
        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            auto x_pack = x.load_pack(i);
            auto y_pack = y.load_pack(i);
            acc = fma(x_pack, y_pack, acc); // acc += x * y
        }

        // Handle tail
        if (num_packs > 0 && x.tail_size() > 0) {
            const std::size_t last_idx = num_packs - 1;
            auto x_pack = x.load_pack_tail(last_idx);
            auto y_pack = y.load_pack_tail(last_idx);

            // Create mask for valid elements
            auto m = mask<T, W>::first_n(x.tail_size());
            auto x_masked = blend(pack<T, W>(T(0)), x_pack, m);
            auto y_masked = blend(pack<T, W>(T(0)), y_pack, m);
            acc = fma(x_masked, y_masked, acc);
        } else if (num_packs > 0) {
            // Tail is full pack
            const std::size_t last_idx = num_packs - 1;
            auto x_pack = x.load_pack(last_idx);
            auto y_pack = y.load_pack(last_idx);
            acc = fma(x_pack, y_pack, acc);
        }

        return horizontal_sum(acc);
    }

    // Overload accepting non-const views
    template <typename T, std::size_t W>
    OPTINUM_INLINE T dot(const vector_view<T, W> &x, const vector_view<T, W> &y) noexcept {
        return dot(vector_view<const T, W>(x.data(), x.size()), vector_view<const T, W>(y.data(), y.size()));
    }

    // =============================================================================
    // sum: sum of all elements
    // Returns: sum(x[i])
    // =============================================================================

    template <typename T, std::size_t W> OPTINUM_INLINE T sum(const vector_view<const T, W> &x) noexcept {
        const std::size_t num_packs = x.num_packs();
        pack<T, W> acc(T(0));

        // Process full packs
        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            acc = acc + x.load_pack(i);
        }

        // Handle tail
        if (num_packs > 0 && x.tail_size() > 0) {
            const std::size_t last_idx = num_packs - 1;
            auto x_pack = x.load_pack_tail(last_idx);

            // Mask out invalid elements
            auto m = mask<T, W>::first_n(x.tail_size());
            auto x_masked = blend(pack<T, W>(T(0)), x_pack, m);
            acc = acc + x_masked;
        } else if (num_packs > 0) {
            // Tail is full pack
            const std::size_t last_idx = num_packs - 1;
            acc = acc + x.load_pack(last_idx);
        }

        return horizontal_sum(acc);
    }

    // Overload accepting non-const view
    template <typename T, std::size_t W> OPTINUM_INLINE T sum(const vector_view<T, W> &x) noexcept {
        return sum(vector_view<const T, W>(x.data(), x.size()));
    }

    // =============================================================================
    // norm2: L2 norm (Euclidean norm)
    // Returns: sqrt(sum(x[i]^2))
    // =============================================================================

    template <typename T, std::size_t W> OPTINUM_INLINE T norm2(const vector_view<const T, W> &x) noexcept {
        const std::size_t num_packs = x.num_packs();
        pack<T, W> acc(T(0));

        // Process full packs
        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            auto x_pack = x.load_pack(i);
            acc = fma(x_pack, x_pack, acc); // acc += x * x
        }

        // Handle tail
        if (num_packs > 0 && x.tail_size() > 0) {
            const std::size_t last_idx = num_packs - 1;
            auto x_pack = x.load_pack_tail(last_idx);

            // Mask out invalid elements
            auto m = mask<T, W>::first_n(x.tail_size());
            auto x_masked = blend(pack<T, W>(T(0)), x_pack, m);
            acc = fma(x_masked, x_masked, acc);
        } else if (num_packs > 0) {
            // Tail is full pack
            const std::size_t last_idx = num_packs - 1;
            auto x_pack = x.load_pack(last_idx);
            acc = fma(x_pack, x_pack, acc);
        }

        return std::sqrt(horizontal_sum(acc));
    }

    // Overload accepting non-const view
    template <typename T, std::size_t W> OPTINUM_INLINE T norm2(const vector_view<T, W> &x) noexcept {
        return norm2(vector_view<const T, W>(x.data(), x.size()));
    }

    // =============================================================================
    // norm2_squared: Squared L2 norm (without sqrt)
    // Returns: sum(x[i]^2)
    // =============================================================================

    template <typename T, std::size_t W> OPTINUM_INLINE T norm2_squared(const vector_view<const T, W> &x) noexcept {
        const std::size_t num_packs = x.num_packs();
        pack<T, W> acc(T(0));

        // Process full packs
        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            auto x_pack = x.load_pack(i);
            acc = fma(x_pack, x_pack, acc); // acc += x * x
        }

        // Handle tail
        if (num_packs > 0 && x.tail_size() > 0) {
            const std::size_t last_idx = num_packs - 1;
            auto x_pack = x.load_pack_tail(last_idx);

            // Mask out invalid elements
            auto m = mask<T, W>::first_n(x.tail_size());
            auto x_masked = blend(pack<T, W>(T(0)), x_pack, m);
            acc = fma(x_masked, x_masked, acc);
        } else if (num_packs > 0) {
            // Tail is full pack
            const std::size_t last_idx = num_packs - 1;
            auto x_pack = x.load_pack(last_idx);
            acc = fma(x_pack, x_pack, acc);
        }

        return horizontal_sum(acc);
    }

    // Overload accepting non-const view
    template <typename T, std::size_t W> OPTINUM_INLINE T norm2_squared(const vector_view<T, W> &x) noexcept {
        return norm2_squared(vector_view<const T, W>(x.data(), x.size()));
    }

    // =============================================================================
    // norm1: L1 norm (Manhattan norm)
    // Returns: sum(|x[i]|)
    // =============================================================================

    template <typename T, std::size_t W> OPTINUM_INLINE T norm1(const vector_view<const T, W> &x) noexcept {
        const std::size_t num_packs = x.num_packs();
        pack<T, W> acc(T(0));

        // Process full packs
        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            auto x_pack = x.load_pack(i);
            acc = acc + abs(x_pack);
        }

        // Handle tail
        if (num_packs > 0 && x.tail_size() > 0) {
            const std::size_t last_idx = num_packs - 1;
            auto x_pack = x.load_pack_tail(last_idx);

            // Mask out invalid elements
            auto m = mask<T, W>::first_n(x.tail_size());
            auto x_masked = blend(pack<T, W>(T(0)), x_pack, m);
            acc = acc + abs(x_masked);
        } else if (num_packs > 0) {
            // Tail is full pack
            const std::size_t last_idx = num_packs - 1;
            acc = acc + abs(x.load_pack(last_idx));
        }

        return horizontal_sum(acc);
    }

    // Overload accepting non-const view
    template <typename T, std::size_t W> OPTINUM_INLINE T norm1(const vector_view<T, W> &x) noexcept {
        return norm1(vector_view<const T, W>(x.data(), x.size()));
    }

    // =============================================================================
    // norminf: L-infinity norm (max absolute value)
    // Returns: max(|x[i]|)
    // =============================================================================

    template <typename T, std::size_t W> OPTINUM_INLINE T norminf(const vector_view<const T, W> &x) noexcept {
        if (x.size() == 0)
            return T(0);

        const std::size_t num_packs = x.num_packs();
        pack<T, W> max_pack(T(0));

        // Process full packs
        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            auto x_pack = abs(x.load_pack(i));
            max_pack = max(max_pack, x_pack);
        }

        // Handle tail
        if (num_packs > 0 && x.tail_size() > 0) {
            const std::size_t last_idx = num_packs - 1;
            auto x_pack = x.load_pack_tail(last_idx);

            // Mask out invalid elements with zeros
            auto m = mask<T, W>::first_n(x.tail_size());
            auto x_masked = blend(pack<T, W>(T(0)), abs(x_pack), m);
            max_pack = max(max_pack, x_masked);
        } else if (num_packs > 0) {
            // Tail is full pack
            const std::size_t last_idx = num_packs - 1;
            max_pack = max(max_pack, abs(x.load_pack(last_idx)));
        }

        return horizontal_max(max_pack);
    }

    // Overload accepting non-const view
    template <typename T, std::size_t W> OPTINUM_INLINE T norminf(const vector_view<T, W> &x) noexcept {
        return norminf(vector_view<const T, W>(x.data(), x.size()));
    }

    // =============================================================================
    // min: minimum element
    // =============================================================================

    template <typename T, std::size_t W> OPTINUM_INLINE T min(const vector_view<const T, W> &x) noexcept {
        if (x.size() == 0)
            return T(0);

        const std::size_t num_packs = x.num_packs();

        // Initialize with first element broadcast
        pack<T, W> min_pack(x.at(0));

        // Process full packs
        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            auto x_pack = x.load_pack(i);
            min_pack = min(min_pack, x_pack);
        }

        // Handle tail
        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            const std::size_t tail = x.tail_size();

            if (tail > 0) {
                // Partial pack - need careful handling
                for (std::size_t i = last_idx * W; i < x.size(); ++i) {
                    T val = x.at(i);
                    if (val < min_pack[0]) {
                        min_pack = pack<T, W>(val);
                    }
                }
            } else {
                // Full pack
                min_pack = min(min_pack, x.load_pack(last_idx));
            }
        }

        return horizontal_min(min_pack);
    }

    // Overload accepting non-const view
    template <typename T, std::size_t W> OPTINUM_INLINE T min(const vector_view<T, W> &x) noexcept {
        return min(vector_view<const T, W>(x.data(), x.size()));
    }

    // =============================================================================
    // max: maximum element
    // =============================================================================

    template <typename T, std::size_t W> OPTINUM_INLINE T max(const vector_view<const T, W> &x) noexcept {
        if (x.size() == 0)
            return T(0);

        const std::size_t num_packs = x.num_packs();

        // Initialize with first element broadcast
        pack<T, W> max_pack(x.at(0));

        // Process full packs
        for (std::size_t i = 0; i < num_packs - 1; ++i) {
            auto x_pack = x.load_pack(i);
            max_pack = max(max_pack, x_pack);
        }

        // Handle tail
        if (num_packs > 0) {
            const std::size_t last_idx = num_packs - 1;
            const std::size_t tail = x.tail_size();

            if (tail > 0) {
                // Partial pack - need careful handling
                for (std::size_t i = last_idx * W; i < x.size(); ++i) {
                    T val = x.at(i);
                    if (val > max_pack[0]) {
                        max_pack = pack<T, W>(val);
                    }
                }
            } else {
                // Full pack
                max_pack = max(max_pack, x.load_pack(last_idx));
            }
        }

        return horizontal_max(max_pack);
    }

    // Overload accepting non-const view
    template <typename T, std::size_t W> OPTINUM_INLINE T max(const vector_view<T, W> &x) noexcept {
        return max(vector_view<const T, W>(x.data(), x.size()));
    }

} // namespace optinum::simd
