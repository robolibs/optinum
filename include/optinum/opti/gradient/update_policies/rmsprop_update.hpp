#pragma once

#include <cmath>
#include <limits>
#include <vector>

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
            // Lazy initialization on first use
            if (mean_squared_grad.size() != N) {
                mean_squared_grad.resize(N, T{0});
            }

            T alpha_t = T(alpha);
            T eps_t = T(epsilon);
            T one_minus_alpha = T{1} - alpha_t;

            // Catch underflow
            if (eps_t == T{0} && epsilon != 0.0) {
                eps_t = T{10} * std::numeric_limits<T>::epsilon();
            }

            // Get raw pointers for SIMD operations
            T *v_ptr = mean_squared_grad.data();
            const T *g_ptr = gradient.data();
            T *x_ptr = x.data();

            // SIMD width for this type and size
            constexpr std::size_t W = simd::backend::preferred_simd_lanes<T, N>();
            constexpr std::size_t main = simd::backend::main_loop_count<N, W>();

            using pack_t = simd::pack<T, W>;

            const pack_t alpha_vec(alpha_t);
            const pack_t one_minus_alpha_vec(one_minus_alpha);
            const pack_t step_vec(step_size);
            const pack_t eps_vec(eps_t);

            // Update mean squared gradient: v = alpha * v + (1 - alpha) * g² (SIMD)
            for (std::size_t i = 0; i < main; i += W) {
                auto v_val = pack_t::loadu(v_ptr + i);
                auto g_val = pack_t::loadu(g_ptr + i);
                auto g_squared = g_val * g_val;
                auto result = alpha_vec * v_val + one_minus_alpha_vec * g_squared;
                result.storeu(v_ptr + i);
            }
            // Tail
            for (std::size_t i = main; i < N; ++i) {
                T g_i = g_ptr[i];
                v_ptr[i] = alpha_t * v_ptr[i] + one_minus_alpha * g_i * g_i;
            }

            // Update iterate: x = x - (step_size / sqrt(v + eps)) * g (SIMD)
            for (std::size_t i = 0; i < main; i += W) {
                auto x_val = pack_t::loadu(x_ptr + i);
                auto v_val = pack_t::loadu(v_ptr + i);
                auto g_val = pack_t::loadu(g_ptr + i);

                // Compute sqrt(v) + eps
                auto sqrt_v = simd::sqrt(v_val);
                auto denom = sqrt_v + eps_vec;

                // Compute update: step_size * g / denom
                auto update = step_vec * g_val / denom;

                // Apply update: x -= update
                auto result = x_val - update;
                result.storeu(x_ptr + i);
            }
            // Tail
            for (std::size_t i = main; i < N; ++i) {
                x_ptr[i] -= step_size * g_ptr[i] / (std::sqrt(v_ptr[i]) + eps_t);
            }
        }

        /// Reset mean squared gradient to zero
        void reset() noexcept { mean_squared_grad.clear(); }

        /// Initialize (for compatibility with GradientDescent interface)
        template <typename T, std::size_t N> void initialize(std::size_t n) noexcept {
            mean_squared_grad.clear(); // Will re-initialize on first update
        }

      private:
        std::vector<double> mean_squared_grad; ///< Moving average of squared gradients
    };

} // namespace optinum::opti
