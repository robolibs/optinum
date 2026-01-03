#pragma once

#include <datapod/matrix/vector.hpp>
#include <optinum/simd/backend/elementwise.hpp>

namespace optinum::opti {

    namespace dp = ::datapod;

    /**
     * Vanilla update policy for gradient descent
     *
     * Simple gradient descent update:
     *   x_{k+1} = x_k - α * ∇f(x_k)
     *
     * where α is the step size (learning rate).
     *
     * This is the most basic update policy with no momentum or acceleration.
     * Uses SIMD-optimized operations for performance.
     */
    struct VanillaUpdate {
        /**
         * Update the iterate using vanilla gradient descent
         *
         * @param x Current iterate (modified in-place)
         * @param step_size Learning rate α
         * @param gradient Current gradient ∇f(x)
         */
        template <typename T, std::size_t N>
        void update(dp::mat::Vector<T, N> &x, T step_size, const dp::mat::Vector<T, N> &gradient) const noexcept {
            // x = x - step_size * gradient using SIMD
            simd::backend::scale_sub_runtime<T>(x.data(), step_size, gradient.data(), x.size());
        }

        /// Reset state (vanilla has no state)
        void reset() noexcept {}

        /// Initialize (vanilla needs no initialization)
        template <typename T, std::size_t N> void initialize(std::size_t) noexcept {}
    };

} // namespace optinum::opti
