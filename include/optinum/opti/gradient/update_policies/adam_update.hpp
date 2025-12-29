#pragma once

#include <cmath>

#include <datapod/matrix/vector.hpp>

namespace optinum::opti {

    namespace dp = ::datapod;

    /**
     * Adam (Adaptive Moment Estimation) update policy
     *
     * Computes adaptive learning rates for each parameter from estimates of first
     * and second moments of the gradients. Combines the benefits of Momentum and
     * RMSprop with bias correction.
     *
     * Update equations:
     *   m_t = β₁ * m_{t-1} + (1 - β₁) * g_t         // First moment (momentum)
     *   v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²        // Second moment (variance)
     *   m̂_t = m_t / (1 - β₁^t)                      // Bias-corrected first moment
     *   v̂_t = v_t / (1 - β₂^t)                      // Bias-corrected second moment
     *   x_{t+1} = x_t - η * m̂_t / (√v̂_t + ε)       // Update
     *
     * where:
     *   - β₁ is the exponential decay rate for first moment (typically 0.9)
     *   - β₂ is the exponential decay rate for second moment (typically 0.999)
     *   - η is the learning rate
     *   - ε is a small constant for numerical stability (typically 1e-8)
     *   - t is the iteration/timestep
     *
     * Reference:
     *   Kingma & Ba (2014)
     *   "Adam: A Method for Stochastic Optimization"
     *   https://arxiv.org/abs/1412.6980
     */
    struct AdamUpdate {
        double beta1 = 0.9;    ///< First moment decay rate β₁ ∈ [0, 1)
        double beta2 = 0.999;  ///< Second moment decay rate β₂ ∈ [0, 1)
        double epsilon = 1e-8; ///< Numerical stability constant ε

        /// Constructor with beta1, beta2, and epsilon parameters
        explicit AdamUpdate(double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
            : beta1(b1), beta2(b2), epsilon(eps) {}

        /**
         * Update the iterate using Adam
         *
         * @param x Current iterate (modified in-place)
         * @param step_size Learning rate η
         * @param gradient Current gradient g_t
         */
        template <typename T, std::size_t N>
        void update(dp::mat::vector<T, N> &x, T step_size, const dp::mat::vector<T, N> &gradient) noexcept {
            const std::size_t n = x.size();

            // Lazy initialization on first use
            if (m.size() != n) {
                m.resize(n);
                v.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    m[i] = 0.0;
                    v[i] = 0.0;
                }
                iteration = 0;
            }

            // Increment iteration counter
            ++iteration;

            double one_minus_beta1 = 1.0 - beta1;
            double one_minus_beta2 = 1.0 - beta2;

            // Compute bias correction terms
            double bias_correction1 = 1.0 - std::pow(beta1, double(iteration));
            double bias_correction2 = 1.0 - std::pow(beta2, double(iteration));
            double step_correction = double(step_size) * std::sqrt(bias_correction2) / bias_correction1;

            // Get raw pointers
            double *m_ptr = m.data();
            double *v_ptr = v.data();
            const T *g_ptr = gradient.data();
            T *x_ptr = x.data();

            for (std::size_t i = 0; i < n; ++i) {
                double g_i = double(g_ptr[i]);
                // Update biased first moment: m = beta1 * m + (1 - beta1) * g
                m_ptr[i] = beta1 * m_ptr[i] + one_minus_beta1 * g_i;
                // Update biased second moment: v = beta2 * v + (1 - beta2) * g²
                v_ptr[i] = beta2 * v_ptr[i] + one_minus_beta2 * g_i * g_i;
                // Update iterate: x = x - step_correction * m / (sqrt(v) + eps)
                x_ptr[i] -= T(step_correction * m_ptr[i] / (std::sqrt(v_ptr[i]) + epsilon));
            }
        }

        /// Reset moments and iteration counter
        void reset() noexcept {
            m.resize(0);
            v.resize(0);
            iteration = 0;
        }

        /// Initialize (for compatibility with GradientDescent interface)
        template <typename T, std::size_t N> void initialize(std::size_t) noexcept {
            m.resize(0);
            v.resize(0);
            iteration = 0;
        }

      private:
        dp::mat::vector<double, dp::mat::Dynamic> m; ///< First moment estimate (momentum)
        dp::mat::vector<double, dp::mat::Dynamic> v; ///< Second moment estimate (variance)
        std::size_t iteration = 0;                   ///< Iteration counter for bias correction
    };

} // namespace optinum::opti
