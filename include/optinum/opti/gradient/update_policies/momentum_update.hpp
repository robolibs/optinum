#pragma once

#include <optinum/simd/backend/backend.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/vector.hpp>
#include <vector>

namespace optinum::opti {

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
         * Update the iterate using momentum (SIMD-optimized)
         *
         * @param x Current iterate (modified in-place)
         * @param step_size Learning rate α
         * @param gradient Current gradient ∇f(x)
         */
        template <typename T, std::size_t N>
        void update(simd::Vector<T, N> &x, T step_size, const simd::Vector<T, N> &gradient) noexcept {
            // Lazy initialization of velocity on first use
            if (velocity.size() != N) {
                velocity.resize(N, T{0});
            }

            T mu = T(momentum);

            // Get raw pointers for SIMD operations
            T *v_ptr = velocity.data();
            const T *g_ptr = gradient.data();
            T *x_ptr = x.data();

            // SIMD width for this type and size
            constexpr std::size_t W = simd::backend::preferred_simd_lanes<T, N>();
            constexpr std::size_t main = simd::backend::main_loop_count<N, W>();

            using pack_t = simd::pack<T, W>;

            const pack_t mu_vec(mu);
            const pack_t step_vec(step_size);

            // Update velocity: v = momentum * v - step_size * gradient (SIMD)
            for (std::size_t i = 0; i < main; i += W) {
                auto v_val = pack_t::loadu(v_ptr + i);
                auto g_val = pack_t::loadu(g_ptr + i);
                auto result = mu_vec * v_val - step_vec * g_val;
                result.storeu(v_ptr + i);
            }
            // Tail
            for (std::size_t i = main; i < N; ++i) {
                v_ptr[i] = mu * v_ptr[i] - step_size * g_ptr[i];
            }

            // Update iterate: x = x + v (SIMD)
            for (std::size_t i = 0; i < main; i += W) {
                auto x_val = pack_t::loadu(x_ptr + i);
                auto v_val = pack_t::loadu(v_ptr + i);
                auto result = x_val + v_val;
                result.storeu(x_ptr + i);
            }
            // Tail
            for (std::size_t i = main; i < N; ++i) {
                x_ptr[i] += v_ptr[i];
            }
        }

        /// Reset velocity to zero
        void reset() noexcept { velocity.clear(); }

        /// Initialize (for compatibility with GradientDescent interface)
        template <typename T, std::size_t N> void initialize(std::size_t n) noexcept {
            velocity.clear(); // Will re-initialize on first update
        }

      private:
        std::vector<double> velocity; ///< Velocity storage (dynamically sized)
    };

} // namespace optinum::opti
