#pragma once

// =============================================================================
// optinum/simd/algo/random.hpp
// SIMD-accelerated random number generation utilities
// =============================================================================

#include <optinum/simd/math/cos.hpp>
#include <optinum/simd/math/log.hpp>
#include <optinum/simd/math/sin.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/pack/pack.hpp>

#include <cmath>
#include <random>
#include <utility>

namespace optinum::simd {

    // =========================================================================
    // Constants
    // =========================================================================

    namespace detail {
        template <typename T> inline constexpr T two_pi = T{6.283185307179586476925286766559};
    } // namespace detail

    // =========================================================================
    // random_uniform: Generate W uniform random values in [lo, hi)
    // =========================================================================

    /**
     * Generate a pack of uniform random values in [lo, hi)
     *
     * @tparam T Scalar type (float or double)
     * @tparam W SIMD width
     * @param rng Random number generator (e.g., std::mt19937)
     * @param lo Lower bound (inclusive)
     * @param hi Upper bound (exclusive)
     * @return pack<T,W> with uniform random values
     */
    template <typename T, std::size_t W, typename RNG>
    inline pack<T, W> random_uniform(RNG &rng, T lo = T{0}, T hi = T{1}) noexcept {
        std::uniform_real_distribution<T> dist(lo, hi);
        alignas(64) T values[W];
        for (std::size_t i = 0; i < W; ++i) {
            values[i] = dist(rng);
        }
        return pack<T, W>::loadu(values);
    }

    /**
     * Fill an array with uniform random values using SIMD
     *
     * @param dst Destination array
     * @param n Number of elements
     * @param rng Random number generator
     * @param lo Lower bound
     * @param hi Upper bound
     */
    template <typename T, typename RNG>
    inline void random_uniform_fill(T *dst, std::size_t n, RNG &rng, T lo = T{0}, T hi = T{1}) noexcept {
        std::uniform_real_distribution<T> dist(lo, hi);

        constexpr std::size_t W = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = pack<T, W>;

        const std::size_t main = (n / W) * W;

        // Generate W values at a time
        for (std::size_t i = 0; i < main; i += W) {
            alignas(64) T values[W];
            for (std::size_t j = 0; j < W; ++j) {
                values[j] = dist(rng);
            }
            pack_t::loadu(values).storeu(dst + i);
        }

        // Handle tail
        for (std::size_t i = main; i < n; ++i) {
            dst[i] = dist(rng);
        }
    }

    // =========================================================================
    // random_normal: Generate W normal random values using Box-Muller
    // =========================================================================

    /**
     * Generate a pair of packs with normal random values using Box-Muller transform
     *
     * Box-Muller transform:
     *   z0 = sqrt(-2 * ln(u1)) * cos(2π * u2)
     *   z1 = sqrt(-2 * ln(u1)) * sin(2π * u2)
     *
     * @tparam T Scalar type (float or double)
     * @tparam W SIMD width
     * @param rng Random number generator
     * @param mean Mean of the distribution (default 0)
     * @param stddev Standard deviation (default 1)
     * @return pair of pack<T,W> with normal random values
     */
    template <typename T, std::size_t W, typename RNG>
    inline std::pair<pack<T, W>, pack<T, W>> random_normal_pair(RNG &rng, T mean = T{0}, T stddev = T{1}) noexcept {
        // Generate uniform random values in (0, 1]
        // Note: We use (0, 1] to avoid log(0)
        std::uniform_real_distribution<T> dist(std::numeric_limits<T>::min(), T{1});

        alignas(64) T u1_vals[W];
        alignas(64) T u2_vals[W];
        for (std::size_t i = 0; i < W; ++i) {
            u1_vals[i] = dist(rng);
            u2_vals[i] = dist(rng);
        }

        auto u1 = pack<T, W>::loadu(u1_vals);
        auto u2 = pack<T, W>::loadu(u2_vals);

        // Box-Muller transform using SIMD math functions
        // r = sqrt(-2 * ln(u1))
        auto neg_two = pack<T, W>(T{-2});
        auto r = sqrt(neg_two * log(u1));

        // theta = 2π * u2
        auto two_pi_pack = pack<T, W>(detail::two_pi<T>);
        auto theta = two_pi_pack * u2;

        // z0 = r * cos(theta), z1 = r * sin(theta)
        auto z0 = r * cos(theta);
        auto z1 = r * sin(theta);

        // Scale and shift: result = mean + stddev * z
        if (mean != T{0} || stddev != T{1}) {
            auto mean_pack = pack<T, W>(mean);
            auto stddev_pack = pack<T, W>(stddev);
            z0 = pack<T, W>::fma(z0, stddev_pack, mean_pack);
            z1 = pack<T, W>::fma(z1, stddev_pack, mean_pack);
        }

        return {z0, z1};
    }

    /**
     * Generate a single pack of normal random values
     *
     * This is less efficient than random_normal_pair since it discards half the values.
     * Use random_normal_pair when you need multiple values.
     *
     * @tparam T Scalar type (float or double)
     * @tparam W SIMD width
     * @param rng Random number generator
     * @param mean Mean of the distribution (default 0)
     * @param stddev Standard deviation (default 1)
     * @return pack<T,W> with normal random values
     */
    template <typename T, std::size_t W, typename RNG>
    inline pack<T, W> random_normal(RNG &rng, T mean = T{0}, T stddev = T{1}) noexcept {
        return random_normal_pair<T, W>(rng, mean, stddev).first;
    }

    /**
     * Fill an array with normal random values using SIMD Box-Muller
     *
     * @param dst Destination array
     * @param n Number of elements
     * @param rng Random number generator
     * @param mean Mean of the distribution
     * @param stddev Standard deviation
     */
    template <typename T, typename RNG>
    inline void random_normal_fill(T *dst, std::size_t n, RNG &rng, T mean = T{0}, T stddev = T{1}) noexcept {
        constexpr std::size_t W = std::is_same_v<T, double> ? 4 : 8;

        const std::size_t main = (n / (2 * W)) * (2 * W);

        // Generate 2*W values at a time using Box-Muller
        for (std::size_t i = 0; i < main; i += 2 * W) {
            auto [z0, z1] = random_normal_pair<T, W>(rng, mean, stddev);
            z0.storeu(dst + i);
            z1.storeu(dst + i + W);
        }

        // Handle remaining elements with scalar generation
        std::normal_distribution<T> dist(mean, stddev);
        for (std::size_t i = main; i < n; ++i) {
            dst[i] = dist(rng);
        }
    }

    // =========================================================================
    // random_uniform_int: Generate W uniform random integers in [lo, hi]
    // =========================================================================

    /**
     * Generate a pack of uniform random integers in [lo, hi]
     *
     * @tparam T Integer type
     * @tparam W SIMD width
     * @param rng Random number generator
     * @param lo Lower bound (inclusive)
     * @param hi Upper bound (inclusive)
     * @return Array of W random integers
     */
    template <typename T, std::size_t W, typename RNG>
    inline void random_uniform_int(T *dst, RNG &rng, T lo, T hi) noexcept {
        std::uniform_int_distribution<T> dist(lo, hi);
        for (std::size_t i = 0; i < W; ++i) {
            dst[i] = dist(rng);
        }
    }

} // namespace optinum::simd
