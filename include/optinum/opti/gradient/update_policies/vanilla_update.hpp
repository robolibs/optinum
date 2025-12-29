#pragma once

#include <datapod/matrix/vector.hpp>

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
        void update(dp::mat::vector<T, N> &x, T step_size, const dp::mat::vector<T, N> &gradient) const noexcept {
            // x = x - step_size * gradient (element-wise)
            for (std::size_t i = 0; i < x.size(); ++i) {
                x[i] -= step_size * gradient[i];
            }
        }

        /// Reset state (vanilla has no state)
        void reset() noexcept {}

        /// Initialize (vanilla needs no initialization)
        template <typename T, std::size_t N> void initialize(std::size_t) noexcept {}
    };

} // namespace optinum::opti
