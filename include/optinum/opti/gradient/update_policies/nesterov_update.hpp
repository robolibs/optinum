#pragma once

#include <optinum/simd/backend/backend.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::opti {

    /**
     * Nesterov Accelerated Gradient (NAG) update policy
     *
     * Nesterov momentum improves upon classical momentum by computing the gradient
     * at the "lookahead" position, providing better convergence properties.
     *
     * The standard formulation computes gradient at x + μv:
     *   v_t = μ * v_{t-1} - α * ∇f(x_t + μ * v_{t-1})
     *   x_{t+1} = x_t + v_t
     *
     * Since we receive the gradient at x_t (not the lookahead), we use the
     * equivalent reformulation that achieves the same effect:
     *   v_t = μ * v_{t-1} - α * g_t
     *   x_{t+1} = x_t + μ * v_t - α * g_t
     *
     * This is mathematically equivalent to the lookahead formulation and provides
     * the same accelerated convergence.
     *
     * where:
     *   - μ is the momentum coefficient (typically 0.9)
     *   - α is the learning rate
     *   - v is the velocity vector
     *   - g_t is the gradient at current position
     *
     * Reference:
     *   Nesterov (1983) "A method for unconstrained convex minimization problem
     *   with the rate of convergence O(1/k^2)"
     *
     *   Sutskever et al. (2013) "On the importance of initialization and momentum
     *   in deep learning"
     */
    struct NesterovUpdate {
        double momentum = 0.9; ///< Momentum coefficient μ ∈ [0, 1)

        /// Constructor with momentum parameter
        explicit NesterovUpdate(double mu = 0.9) : momentum(mu) {}

        /**
         * Update the iterate using Nesterov momentum (SIMD-optimized)
         *
         * @param x Current iterate (modified in-place)
         * @param step_size Learning rate α
         * @param gradient Current gradient ∇f(x)
         */
        template <typename T, std::size_t N>
        void update(simd::Vector<T, N> &x, T step_size, const simd::Vector<T, N> &gradient) noexcept {
            const std::size_t n = x.size();

            // Lazy initialization of velocity on first use
            if (velocity.size() != n) {
                velocity.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    velocity[i] = 0.0;
                }
            }

            // Get raw pointers
            double *v_ptr = velocity.data();
            const T *g_ptr = gradient.data();
            T *x_ptr = x.data();

            // Nesterov update:
            // 1. v_new = μ * v - α * g
            // 2. x_new = x + μ * v_new - α * g
            if constexpr (N == simd::Dynamic) {
                for (std::size_t i = 0; i < n; ++i) {
                    double v_new = momentum * v_ptr[i] - double(step_size) * double(g_ptr[i]);
                    v_ptr[i] = v_new;
                    x_ptr[i] = x_ptr[i] + T(momentum * v_new) - step_size * g_ptr[i];
                }
            } else {
                for (std::size_t i = 0; i < N; ++i) {
                    double v_new = momentum * v_ptr[i] - double(step_size) * double(g_ptr[i]);
                    v_ptr[i] = v_new;
                    x_ptr[i] = x_ptr[i] + T(momentum * v_new) - step_size * g_ptr[i];
                }
            }
        }

        /// Reset velocity to zero
        void reset() noexcept { velocity.resize(0); }

        /// Initialize (for compatibility with GradientDescent interface)
        template <typename T, std::size_t N> void initialize(std::size_t) noexcept {
            velocity.resize(0); // Will re-initialize on first update
        }

      private:
        simd::Vector<double, simd::Dynamic> velocity; ///< Velocity storage (double precision)
    };

} // namespace optinum::opti
