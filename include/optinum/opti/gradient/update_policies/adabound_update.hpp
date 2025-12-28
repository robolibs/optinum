#pragma once

#include <cmath>
#include <limits>

#include <optinum/simd/backend/backend.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::opti {

    /**
     * AdaBound update policy
     *
     * An optimizer that starts like Adam but gradually transitions to SGD by
     * applying dynamic bounds to the learning rate. This combines the fast
     * initial convergence of Adam with the good generalization of SGD.
     *
     * Update equations:
     *   m_t = β₁ * m_{t-1} + (1 - β₁) * g_t                    // First moment
     *   v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²                   // Second moment
     *   m̂_t = m_t / (1 - β₁^t)                                 // Bias-corrected first moment
     *   v̂_t = v_t / (1 - β₂^t)                                 // Bias-corrected second moment
     *   η_t = clip(α / √v̂_t, η_l(t), η_u(t))                   // Bounded learning rate
     *   x_{t+1} = x_t - η_t * m̂_t                              // Update
     *
     * Dynamic bounds:
     *   η_l(t) = α_final * (1 - 1/(γt + 1))                    // Lower bound
     *   η_u(t) = α_final * (1 + 1/(γt))                        // Upper bound
     *
     * where:
     *   - α is the initial learning rate
     *   - α_final is the final (SGD) learning rate
     *   - γ controls the convergence speed of bounds (typically 1e-3)
     *   - β₁, β₂ are Adam's moment decay rates
     *
     * Properties:
     *   - Starts with Adam-like adaptive learning rates
     *   - Gradually transitions to SGD as training progresses
     *   - Better generalization than pure Adam
     *   - Bounds prevent extreme learning rates
     *
     * Reference:
     *   Luo, Xiong, Liu, et al. (2019)
     *   "Adaptive Gradient Methods with Dynamic Bound of Learning Rate"
     *   ICLR 2019
     *   https://arxiv.org/abs/1902.09843
     */
    struct AdaBoundUpdate {
        double beta1 = 0.9;    ///< First moment decay rate β₁ ∈ [0, 1)
        double beta2 = 0.999;  ///< Second moment decay rate β₂ ∈ [0, 1)
        double epsilon = 1e-8; ///< Numerical stability constant ε
        double final_lr = 0.1; ///< Final learning rate α_final (SGD rate)
        double gamma = 1e-3;   ///< Bound convergence rate γ

        /// Constructor with parameters
        explicit AdaBoundUpdate(double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, double final = 0.1,
                                double g = 1e-3)
            : beta1(b1), beta2(b2), epsilon(eps), final_lr(final), gamma(g) {}

        /**
         * Update the iterate using AdaBound (SIMD-optimized)
         *
         * @param x Current iterate (modified in-place)
         * @param step_size Initial learning rate α
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

            // Compute dynamic bounds
            double t = double(iteration);
            double lower_bound = final_lr * (1.0 - 1.0 / (gamma * t + 1.0));
            double upper_bound = final_lr * (1.0 + 1.0 / (gamma * t));

            // Get raw pointers
            double *m_ptr = m.data();
            double *v_ptr = v.data();
            const T *g_ptr = gradient.data();
            T *x_ptr = x.data();

            double alpha = double(step_size);

            if constexpr (N == simd::Dynamic) {
                for (std::size_t i = 0; i < n; ++i) {
                    double g_i = double(g_ptr[i]);

                    // Update biased first moment: m = β₁ * m + (1 - β₁) * g
                    m_ptr[i] = beta1 * m_ptr[i] + one_minus_beta1 * g_i;

                    // Update biased second moment: v = β₂ * v + (1 - β₂) * g²
                    v_ptr[i] = beta2 * v_ptr[i] + one_minus_beta2 * g_i * g_i;

                    // Bias-corrected estimates
                    double m_hat = m_ptr[i] / bias_correction1;
                    double v_hat = v_ptr[i] / bias_correction2;

                    // Compute adaptive learning rate and clip to bounds
                    double adaptive_lr = alpha / (std::sqrt(v_hat) + epsilon);
                    double bounded_lr = std::max(lower_bound, std::min(upper_bound, adaptive_lr));

                    // Update iterate
                    x_ptr[i] -= T(bounded_lr * m_hat);
                }
            } else {
                for (std::size_t i = 0; i < N; ++i) {
                    double g_i = double(g_ptr[i]);

                    m_ptr[i] = beta1 * m_ptr[i] + one_minus_beta1 * g_i;
                    v_ptr[i] = beta2 * v_ptr[i] + one_minus_beta2 * g_i * g_i;

                    double m_hat = m_ptr[i] / bias_correction1;
                    double v_hat = v_ptr[i] / bias_correction2;

                    double adaptive_lr = alpha / (std::sqrt(v_hat) + epsilon);
                    double bounded_lr = std::max(lower_bound, std::min(upper_bound, adaptive_lr));

                    x_ptr[i] -= T(bounded_lr * m_hat);
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
        std::size_t iteration = 0;             ///< Iteration counter for bias correction and bounds
    };

} // namespace optinum::opti
