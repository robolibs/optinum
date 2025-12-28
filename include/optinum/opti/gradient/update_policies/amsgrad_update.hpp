#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <optinum/simd/backend/backend.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::opti {

    /**
     * AMSGrad update policy
     *
     * A variant of Adam that fixes convergence issues by using the maximum of
     * past squared gradients rather than the exponential moving average. This
     * ensures the effective learning rate is non-increasing, guaranteeing
     * convergence in the convex setting.
     *
     * Update equations:
     *   m_t = β₁ * m_{t-1} + (1 - β₁) * g_t         // First moment (same as Adam)
     *   v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²        // Second moment (same as Adam)
     *   v̂_t = max(v̂_{t-1}, v_t)                     // KEY DIFFERENCE: max instead of EMA
     *   m̂_t = m_t / (1 - β₁^t)                      // Bias-corrected first moment
     *   x_{t+1} = x_t - η * m̂_t / (√v̂_t + ε)       // Update using v̂ instead of v
     *
     * The key insight is that v̂_t is monotonically non-decreasing, which ensures
     * the effective step size η / √v̂_t is non-increasing. This fixes cases where
     * Adam can diverge due to the second moment estimate decreasing.
     *
     * Reference:
     *   Reddi, Kale & Kumar (2018)
     *   "On the Convergence of Adam and Beyond"
     *   https://arxiv.org/abs/1904.09237
     */
    struct AMSGradUpdate {
        double beta1 = 0.9;    ///< First moment decay rate β₁ ∈ [0, 1)
        double beta2 = 0.999;  ///< Second moment decay rate β₂ ∈ [0, 1)
        double epsilon = 1e-8; ///< Numerical stability constant ε

        /// Constructor with beta1, beta2, and epsilon parameters
        explicit AMSGradUpdate(double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
            : beta1(b1), beta2(b2), epsilon(eps) {}

        /**
         * Update the iterate using AMSGrad (SIMD-optimized)
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
                v_hat.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    m[i] = 0.0;
                    v[i] = 0.0;
                    v_hat[i] = 0.0;
                }
                iteration = 0;
            }

            // Increment iteration counter
            ++iteration;

            double one_minus_beta1 = 1.0 - beta1;
            double one_minus_beta2 = 1.0 - beta2;

            // Compute bias correction for first moment only
            // Note: AMSGrad typically doesn't bias-correct v_hat since it's a max
            double bias_correction1 = 1.0 - std::pow(beta1, double(iteration));
            double step_correction = double(step_size) / bias_correction1;

            // Get raw pointers
            double *m_ptr = m.data();
            double *v_ptr = v.data();
            double *v_hat_ptr = v_hat.data();
            const T *g_ptr = gradient.data();
            T *x_ptr = x.data();

            if constexpr (N == simd::Dynamic) {
                for (std::size_t i = 0; i < n; ++i) {
                    double g_i = double(g_ptr[i]);
                    // Update biased first moment: m = beta1 * m + (1 - beta1) * g
                    m_ptr[i] = beta1 * m_ptr[i] + one_minus_beta1 * g_i;
                    // Update biased second moment: v = beta2 * v + (1 - beta2) * g²
                    v_ptr[i] = beta2 * v_ptr[i] + one_minus_beta2 * g_i * g_i;
                    // AMSGrad: v_hat = max(v_hat, v)
                    v_hat_ptr[i] = std::max(v_hat_ptr[i], v_ptr[i]);
                    // Update iterate: x = x - step_correction * m / (sqrt(v_hat) + eps)
                    x_ptr[i] -= T(step_correction * m_ptr[i] / (std::sqrt(v_hat_ptr[i]) + epsilon));
                }
            } else {
                for (std::size_t i = 0; i < N; ++i) {
                    double g_i = double(g_ptr[i]);
                    m_ptr[i] = beta1 * m_ptr[i] + one_minus_beta1 * g_i;
                    v_ptr[i] = beta2 * v_ptr[i] + one_minus_beta2 * g_i * g_i;
                    v_hat_ptr[i] = std::max(v_hat_ptr[i], v_ptr[i]);
                    x_ptr[i] -= T(step_correction * m_ptr[i] / (std::sqrt(v_hat_ptr[i]) + epsilon));
                }
            }
        }

        /// Reset moments and iteration counter
        void reset() noexcept {
            m.resize(0);
            v.resize(0);
            v_hat.resize(0);
            iteration = 0;
        }

        /// Initialize (for compatibility with GradientDescent interface)
        template <typename T, std::size_t N> void initialize(std::size_t) noexcept {
            m.resize(0);
            v.resize(0);
            v_hat.resize(0);
            iteration = 0;
        }

      private:
        simd::Vector<double, simd::Dynamic> m;     ///< First moment estimate (momentum)
        simd::Vector<double, simd::Dynamic> v;     ///< Second moment estimate (variance)
        simd::Vector<double, simd::Dynamic> v_hat; ///< Max of second moment estimates (AMSGrad key)
        std::size_t iteration = 0;                 ///< Iteration counter for bias correction
    };

} // namespace optinum::opti
