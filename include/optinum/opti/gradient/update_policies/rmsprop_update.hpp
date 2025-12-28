#pragma once

#include <cmath>
#include <limits>

#include <optinum/simd/backend/backend.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::opti {

    /**
     * RMSProp update policy for gradient descent
     *
     * Adaptive learning rate method that divides the learning rate by an exponentially
     * decaying average of squared gradients. This helps deal with the problem of
     * learning rates that decrease too aggressively.
     *
     * Update equations:
     *   v_t = α * v_{t-1} + (1 - α) * g_t²
     *   x_{t+1} = x_t - (η / √(v_t + ε)) * g_t
     *
     * where:
     *   - α is the decay rate (typically 0.99)
     *   - η is the learning rate
     *   - ε is a small constant for numerical stability (typically 1e-8)
     *   - v is the moving average of squared gradients
     *
     * Reference:
     *   Tieleman & Hinton (2012)
     *   "Lecture 6.5 - RMSprop, COURSERA: Neural Networks for Machine Learning"
     */
    struct RMSPropUpdate {
        double alpha = 0.99;   ///< Decay rate α ∈ (0, 1)
        double epsilon = 1e-8; ///< Numerical stability constant ε

        /// Constructor with alpha and epsilon parameters
        explicit RMSPropUpdate(double a = 0.99, double eps = 1e-8) : alpha(a), epsilon(eps) {}

        /**
         * Update the iterate using RMSprop (SIMD-optimized)
         *
         * @param x Current iterate (modified in-place)
         * @param step_size Learning rate η
         * @param gradient Current gradient g_t
         */
        template <typename T, std::size_t N>
        void update(simd::Vector<T, N> &x, T step_size, const simd::Vector<T, N> &gradient) noexcept {
            const std::size_t n = x.size();

            // Lazy initialization on first use
            if (mean_squared_grad.size() != n) {
                mean_squared_grad.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    mean_squared_grad[i] = 0.0;
                }
            }

            double one_minus_alpha = 1.0 - alpha;

            // Get raw pointers
            double *v_ptr = mean_squared_grad.data();
            const T *g_ptr = gradient.data();
            T *x_ptr = x.data();

            if constexpr (N == simd::Dynamic) {
                for (std::size_t i = 0; i < n; ++i) {
                    double g_i = double(g_ptr[i]);
                    v_ptr[i] = alpha * v_ptr[i] + one_minus_alpha * g_i * g_i;
                    x_ptr[i] -= T(double(step_size) * g_i / (std::sqrt(v_ptr[i]) + epsilon));
                }
            } else {
                for (std::size_t i = 0; i < N; ++i) {
                    double g_i = double(g_ptr[i]);
                    v_ptr[i] = alpha * v_ptr[i] + one_minus_alpha * g_i * g_i;
                    x_ptr[i] -= T(double(step_size) * g_i / (std::sqrt(v_ptr[i]) + epsilon));
                }
            }
        }

        /// Reset mean squared gradient to zero
        void reset() noexcept { mean_squared_grad.resize(0); }

        /// Initialize (for compatibility with GradientDescent interface)
        template <typename T, std::size_t N> void initialize(std::size_t) noexcept {
            mean_squared_grad.resize(0); // Will re-initialize on first update
        }

      private:
        simd::Vector<double, simd::Dynamic> mean_squared_grad; ///< Moving average of squared gradients
    };

} // namespace optinum::opti
