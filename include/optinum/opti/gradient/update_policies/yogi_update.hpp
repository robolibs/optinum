#pragma once

#include <cmath>
#include <limits>

#include <datapod/matrix/vector.hpp>
#include <optinum/simd/backend/backend.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::opti {

    namespace dp = ::datapod;

    /**
     * Yogi update policy
     *
     * A variant of Adam that uses additive updates for the second moment instead
     * of exponential moving average. This provides better control of the effective
     * learning rate and prevents the second moment from growing too quickly, which
     * can cause Adam to converge to suboptimal solutions.
     *
     * Update equations:
     *   m_t = β₁ * m_{t-1} + (1 - β₁) * g_t                           // First moment (same as Adam)
     *   v_t = v_{t-1} + (1 - β₂) * sign(g_t² - v_{t-1}) * g_t²        // KEY DIFFERENCE: additive update
     *   m̂_t = m_t / (1 - β₁^t)                                        // Bias-corrected first moment
     *   v̂_t = v_t / (1 - β₂^t)                                        // Bias-corrected second moment
     *   x_{t+1} = x_t - η * m̂_t / (√v̂_t + ε)                         // Update
     *
     * The key insight is that the additive update for v_t prevents the second moment
     * from growing too quickly when gradients are large, and allows it to decrease
     * when gradients become small. This leads to more stable training.
     *
     * Reference:
     *   Zaheer et al. (2018)
     *   "Adaptive Methods for Nonconvex Optimization"
     *   NeurIPS 2018
     *   https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization
     */
    struct YogiUpdate {
        double beta1 = 0.9;    ///< First moment decay rate β₁ ∈ [0, 1)
        double beta2 = 0.999;  ///< Second moment decay rate β₂ ∈ [0, 1)
        double epsilon = 1e-8; ///< Numerical stability constant ε

        /// Constructor with beta1, beta2, and epsilon parameters
        explicit YogiUpdate(double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
            : beta1(b1), beta2(b2), epsilon(eps) {}

        /**
         * Update the iterate using Yogi (SIMD-optimized)
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
            double bias_correction1 = 1.0 - std::pow(beta1, double(iteration));
            double bias_correction2 = 1.0 - std::pow(beta2, double(iteration));
            double step_correction = double(step_size) * std::sqrt(bias_correction2) / bias_correction1;

            // Get raw pointers
            double *m_ptr = m.data();
            double *v_ptr = v.data();
            const T *g_ptr = gradient.data();
            T *x_ptr = x.data();

            if constexpr (N == simd::Dynamic) {
                for (std::size_t i = 0; i < n; ++i) {
                    double g_i = double(g_ptr[i]);
                    double g_sq = g_i * g_i;

                    // Update biased first moment: m = beta1 * m + (1 - beta1) * g
                    m_ptr[i] = beta1 * m_ptr[i] + one_minus_beta1 * g_i;

                    // Yogi update for second moment: v = v + (1 - beta2) * sign(g² - v) * g²
                    // sign(g² - v) determines whether to increase or decrease v
                    double diff = g_sq - v_ptr[i];
                    double sign_diff = (diff > 0.0) ? 1.0 : ((diff < 0.0) ? -1.0 : 0.0);
                    v_ptr[i] = v_ptr[i] + one_minus_beta2 * sign_diff * g_sq;

                    // Ensure v stays non-negative (numerical stability)
                    if (v_ptr[i] < 0.0) {
                        v_ptr[i] = 0.0;
                    }

                    // Update iterate: x = x - step_correction * m / (sqrt(v) + eps)
                    x_ptr[i] -= T(step_correction * m_ptr[i] / (std::sqrt(v_ptr[i]) + epsilon));
                }
            } else {
                for (std::size_t i = 0; i < N; ++i) {
                    double g_i = double(g_ptr[i]);
                    double g_sq = g_i * g_i;

                    m_ptr[i] = beta1 * m_ptr[i] + one_minus_beta1 * g_i;

                    double diff = g_sq - v_ptr[i];
                    double sign_diff = (diff > 0.0) ? 1.0 : ((diff < 0.0) ? -1.0 : 0.0);
                    v_ptr[i] = v_ptr[i] + one_minus_beta2 * sign_diff * g_sq;

                    if (v_ptr[i] < 0.0) {
                        v_ptr[i] = 0.0;
                    }

                    x_ptr[i] -= T(step_correction * m_ptr[i] / (std::sqrt(v_ptr[i]) + epsilon));
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
        dp::mat::vector<double, dp::mat::Dynamic> m; ///< First moment estimate (momentum)
        dp::mat::vector<double, dp::mat::Dynamic> v; ///< Second moment estimate (variance)
        std::size_t iteration = 0;                   ///< Iteration counter for bias correction
    };

} // namespace optinum::opti
