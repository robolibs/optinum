#pragma once

#include <cmath>

#include <datapod/datapod.hpp>
#include <optinum/simd/backend/backend.hpp>

namespace optinum::opti {

    /**
     * AdaGrad (Adaptive Gradient) update policy
     *
     * Adapts the learning rate for each parameter based on the historical sum of
     * squared gradients. Parameters with large gradients receive smaller updates,
     * while parameters with small gradients receive larger updates.
     *
     * Update equations:
     *   G_t = G_{t-1} + g_t²                           // Accumulate squared gradients
     *   x_{t+1} = x_t - (η / √(G_t + ε)) * g_t         // Update with adapted learning rate
     *
     * where:
     *   - G is the accumulated sum of squared gradients
     *   - η is the initial learning rate
     *   - ε is a small constant for numerical stability (typically 1e-8)
     *   - g_t is the gradient at time t
     *
     * Properties:
     *   - Automatically adapts learning rate per-parameter
     *   - Well-suited for sparse gradients (e.g., NLP, recommender systems)
     *   - Learning rate monotonically decreases (can be too aggressive for non-convex)
     *
     * Reference:
     *   Duchi, Hazan & Singer (2011)
     *   "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
     *   Journal of Machine Learning Research 12 (2011) 2121-2159
     */
    struct AdaGradUpdate {
        double epsilon = 1e-8; ///< Numerical stability constant ε

        /// Constructor with epsilon parameter
        explicit AdaGradUpdate(double eps = 1e-8) : epsilon(eps) {}

        /**
         * Update the iterate using AdaGrad
         *
         * @param x Current iterate (modified in-place)
         * @param step_size Learning rate η
         * @param gradient Current gradient g_t
         */
        template <typename T, std::size_t N>
        void update(dp::mat::Vector<T, N> &x, T step_size, const dp::mat::Vector<T, N> &gradient) noexcept {
            const std::size_t n = x.size();

            // Lazy initialization on first use
            if (sum_squared_grad.size() != n) {
                sum_squared_grad.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    sum_squared_grad[i] = 0.0;
                }
            }

            // Get raw pointers
            double *G_ptr = sum_squared_grad.data();
            const T *g_ptr = gradient.data();
            T *x_ptr = x.data();

            // SIMD-optimized AdaGrad update
            constexpr std::size_t W = simd::backend::default_pack_width<double>();
            using pack_t = simd::pack<double, W>;

            const double step_val = double(step_size);
            const pack_t step_pack(step_val);
            const pack_t eps(epsilon);

            const std::size_t main = simd::backend::main_loop_count_runtime(n, W);

            // SIMD loop
            for (std::size_t i = 0; i < main; i += W) {
                // Load gradient (convert to double if needed)
                pack_t g;
                if constexpr (std::is_same_v<T, double>) {
                    g = pack_t::loadu(g_ptr + i);
                } else {
                    alignas(32) double g_tmp[W];
                    for (std::size_t j = 0; j < W; ++j)
                        g_tmp[j] = double(g_ptr[i + j]);
                    g = pack_t::loadu(g_tmp);
                }

                // Load accumulated squared gradient
                auto G_pack = pack_t::loadu(G_ptr + i);

                // Accumulate squared gradient: G = G + g²
                G_pack = G_pack + g * g;

                // Store updated G
                G_pack.storeu(G_ptr + i);

                // Compute update: (η / √(G + ε)) * g
                auto update = step_pack * g / (G_pack.sqrt() + eps);

                // Apply update to x
                if constexpr (std::is_same_v<T, double>) {
                    auto x_pack = pack_t::loadu(x_ptr + i);
                    (x_pack - update).storeu(x_ptr + i);
                } else {
                    for (std::size_t j = 0; j < W; ++j) {
                        x_ptr[i + j] -= T(update[j]);
                    }
                }
            }

            // Scalar tail
            for (std::size_t i = main; i < n; ++i) {
                double g_i = double(g_ptr[i]);
                G_ptr[i] += g_i * g_i;
                x_ptr[i] -= T(double(step_size) * g_i / (std::sqrt(G_ptr[i]) + epsilon));
            }
        }

        /// Reset accumulated squared gradients to zero
        void reset() noexcept { sum_squared_grad.resize(0); }

        /// Initialize (for compatibility with GradientDescent interface)
        template <typename T, std::size_t N> void initialize(std::size_t) noexcept {
            sum_squared_grad.resize(0); // Will re-initialize on first update
        }

      private:
        dp::mat::Vector<double, dp::mat::Dynamic> sum_squared_grad; ///< Accumulated sum of squared gradients G
    };

} // namespace optinum::opti
