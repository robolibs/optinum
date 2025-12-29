#pragma once

#include <datapod/matrix/vector.hpp>

namespace optinum::opti {

    namespace dp = ::datapod;

    /**
     * Momentum update policy for gradient descent
     *
     * Adds velocity/momentum to accelerate convergence, especially in ravines
     * where the surface curves more steeply in one dimension than another.
     *
     * Update equations:
     *   v_t = μ * v_{t-1} - α * ∇f(x_t)
     *   x_{t+1} = x_t + v_t
     *
     * where:
     *   - μ is the momentum coefficient (typically 0.9)
     *   - α is the learning rate
     *   - v is the velocity vector
     *
     * The momentum term increases for dimensions whose gradients point in the same
     * direction and reduces updates for dimensions whose gradients change directions.
     *
     * Reference:
     *   Rumelhart, Hinton & Williams (1988)
     *   "Learning representations by back-propagating errors"
     */
    struct MomentumUpdate {
        double momentum = 0.9; ///< Momentum coefficient μ ∈ [0, 1)

        /// Constructor with momentum parameter
        explicit MomentumUpdate(double mu = 0.9) : momentum(mu) {}

        /**
         * Update the iterate using momentum
         *
         * @param x Current iterate (modified in-place)
         * @param step_size Learning rate α
         * @param gradient Current gradient ∇f(x)
         */
        template <typename T, std::size_t N>
        void update(dp::mat::vector<T, N> &x, T step_size, const dp::mat::vector<T, N> &gradient) noexcept {
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

            // Update velocity and position
            for (std::size_t i = 0; i < n; ++i) {
                v_ptr[i] = momentum * v_ptr[i] - double(step_size) * double(g_ptr[i]);
                x_ptr[i] += T(v_ptr[i]);
            }
        }

        /// Reset velocity to zero
        void reset() noexcept { velocity.resize(0); }

        /// Initialize (for compatibility with GradientDescent interface)
        template <typename T, std::size_t N> void initialize(std::size_t) noexcept {
            velocity.resize(0); // Will re-initialize on first update
        }

      private:
        dp::mat::vector<double, dp::mat::Dynamic> velocity; ///< Velocity storage (double precision)
    };

} // namespace optinum::opti
