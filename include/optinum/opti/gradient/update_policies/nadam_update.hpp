#pragma once

#include <cmath>
#include <limits>

#include <optinum/simd/backend/backend.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::opti {

    /**
     * NAdam (Nesterov-accelerated Adaptive Moment Estimation) update policy
     *
     * NAdam incorporates Nesterov momentum into Adam, providing faster convergence
     * by using the lookahead gradient in the momentum term. This combines the
     * benefits of Nesterov accelerated gradient with Adam's adaptive learning rates.
     *
     * The key insight is that instead of using the current momentum estimate m_t
     * in the update, NAdam uses a "lookahead" momentum that incorporates the
     * next step's momentum contribution.
     *
     * Update equations:
     *   m_t = β₁ * m_{t-1} + (1 - β₁) * g_t         // First moment (momentum)
     *   v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²        // Second moment (variance)
     *
     *   // Bias-corrected estimates
     *   m̂_t = m_t / (1 - β₁^t)
     *   v̂_t = v_t / (1 - β₂^t)
     *
     *   // NAdam uses Nesterov-style lookahead momentum:
     *   // Instead of m̂_t, use: β₁ * m̂_t + (1 - β₁) * g_t / (1 - β₁^t)
     *   m̄_t = β₁ * m̂_t + (1 - β₁) * g_t / (1 - β₁^t)
     *
     *   x_{t+1} = x_t - η * m̄_t / (√v̂_t + ε)
     *
     * Simplified form (used in implementation):
     *   m̄_t = (β₁ * m_t + (1 - β₁) * g_t) / (1 - β₁^t)
     *       = β₁ * m̂_t + (1 - β₁) / (1 - β₁^t) * g_t
     *
     * where:
     *   - β₁ is the exponential decay rate for first moment (typically 0.9)
     *   - β₂ is the exponential decay rate for second moment (typically 0.999)
     *   - η is the learning rate
     *   - ε is a small constant for numerical stability (typically 1e-8)
     *   - t is the iteration/timestep
     *
     * Reference:
     *   Dozat (2016)
     *   "Incorporating Nesterov Momentum into Adam"
     *   ICLR Workshop
     *   https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
     */
    struct NAdamUpdate {
        double beta1 = 0.9;    ///< First moment decay rate β₁ ∈ [0, 1)
        double beta2 = 0.999;  ///< Second moment decay rate β₂ ∈ [0, 1)
        double epsilon = 1e-8; ///< Numerical stability constant ε

        /// Constructor with beta1, beta2, and epsilon parameters
        explicit NAdamUpdate(double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
            : beta1(b1), beta2(b2), epsilon(eps) {}

        /**
         * Update the iterate using NAdam (SIMD-optimized)
         *
         * @param x Current iterate (modified in-place)
         * @param step_size Learning rate η
         * @param gradient Current gradient g_t
         */
        template <typename T, std::size_t N>
        void update(simd::Vector<T, N> &x, T step_size, const simd::Vector<T, N> &gradient) noexcept {
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
            double beta1_t = std::pow(beta1, double(iteration));
            double beta2_t = std::pow(beta2, double(iteration));
            double bias_correction1 = 1.0 - beta1_t;
            double bias_correction2 = 1.0 - beta2_t;

            // For NAdam, we also need the next step's bias correction
            double beta1_t_next = beta1_t * beta1; // β₁^(t+1)
            double bias_correction1_next = 1.0 - beta1_t_next;

            // Precompute step size correction for second moment
            double step_correction = double(step_size) / std::sqrt(bias_correction2);

            // Get raw pointers
            double *m_ptr = m.data();
            double *v_ptr = v.data();
            const T *g_ptr = gradient.data();
            T *x_ptr = x.data();

            if constexpr (N == simd::Dynamic) {
                for (std::size_t i = 0; i < n; ++i) {
                    double g_i = double(g_ptr[i]);

                    // Update biased first moment: m = β₁ * m + (1 - β₁) * g
                    m_ptr[i] = beta1 * m_ptr[i] + one_minus_beta1 * g_i;

                    // Update biased second moment: v = β₂ * v + (1 - β₂) * g²
                    v_ptr[i] = beta2 * v_ptr[i] + one_minus_beta2 * g_i * g_i;

                    // NAdam lookahead momentum:
                    // m̄ = β₁ * m / (1 - β₁^(t+1)) + (1 - β₁) * g / (1 - β₁^t)
                    // This is equivalent to using the "future" momentum estimate
                    double m_hat_next = beta1 * m_ptr[i] / bias_correction1_next;
                    double g_hat = one_minus_beta1 * g_i / bias_correction1;
                    double m_bar = m_hat_next + g_hat;

                    // Bias-corrected second moment (already incorporated in step_correction)
                    double v_sqrt = std::sqrt(v_ptr[i]) + epsilon;

                    // Update: x = x - η * m̄ / (√v̂ + ε)
                    x_ptr[i] -= T(step_correction * m_bar / v_sqrt);
                }
            } else {
                for (std::size_t i = 0; i < N; ++i) {
                    double g_i = double(g_ptr[i]);

                    // Update biased first moment
                    m_ptr[i] = beta1 * m_ptr[i] + one_minus_beta1 * g_i;

                    // Update biased second moment
                    v_ptr[i] = beta2 * v_ptr[i] + one_minus_beta2 * g_i * g_i;

                    // NAdam lookahead momentum
                    double m_hat_next = beta1 * m_ptr[i] / bias_correction1_next;
                    double g_hat = one_minus_beta1 * g_i / bias_correction1;
                    double m_bar = m_hat_next + g_hat;

                    // Bias-corrected second moment
                    double v_sqrt = std::sqrt(v_ptr[i]) + epsilon;

                    // Update
                    x_ptr[i] -= T(step_correction * m_bar / v_sqrt);
                }
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
        simd::Vector<double, simd::Dynamic> m; ///< First moment estimate (momentum)
        simd::Vector<double, simd::Dynamic> v; ///< Second moment estimate (variance)
        std::size_t iteration = 0;             ///< Iteration counter for bias correction
    };

} // namespace optinum::opti
