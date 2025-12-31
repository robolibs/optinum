#pragma once

#include <cmath>

#include <datapod/matrix/vector.hpp>
#include <optinum/simd/backend/backend.hpp>

namespace optinum::opti {

    namespace dp = ::datapod;

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
         * Update the iterate using RMSprop
         *
         * @param x Current iterate (modified in-place)
         * @param step_size Learning rate η
         * @param gradient Current gradient g_t
         */
        template <typename T, std::size_t N>
        void update(dp::mat::vector<T, N> &x, T step_size, const dp::mat::vector<T, N> &gradient) noexcept {
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

            // SIMD-optimized RMSProp update
            constexpr std::size_t W = simd::backend::default_pack_width<double>();
            using pack_t = simd::pack<double, W>;

            const double step_val = double(step_size);
            const pack_t alpha_pack(alpha);
            const pack_t one_m_alpha(one_minus_alpha);
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

                // Load mean squared gradient
                auto v_pack = pack_t::loadu(v_ptr + i);

                // Update: v = alpha * v + (1 - alpha) * g²
                v_pack = pack_t::fma(alpha_pack, v_pack, one_m_alpha * g * g);

                // Store updated v
                v_pack.storeu(v_ptr + i);

                // Compute update: (η / √(v + ε)) * g
                auto update = step_pack * g / (v_pack.sqrt() + eps);

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
                v_ptr[i] = alpha * v_ptr[i] + one_minus_alpha * g_i * g_i;
                x_ptr[i] -= T(double(step_size) * g_i / (std::sqrt(v_ptr[i]) + epsilon));
            }
        }

        /// Reset mean squared gradient to zero
        void reset() noexcept { mean_squared_grad.resize(0); }

        /// Initialize (for compatibility with GradientDescent interface)
        template <typename T, std::size_t N> void initialize(std::size_t) noexcept {
            mean_squared_grad.resize(0); // Will re-initialize on first update
        }

      private:
        dp::mat::vector<double, dp::mat::Dynamic> mean_squared_grad; ///< Moving average of squared gradients
    };

} // namespace optinum::opti
