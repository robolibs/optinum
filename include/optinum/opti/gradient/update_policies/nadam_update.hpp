#pragma once

#include <cmath>

#include <datapod/datapod.hpp>
#include <optinum/simd/backend/backend.hpp>

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
         * Update the iterate using NAdam
         *
         * @param x Current iterate (modified in-place)
         * @param step_size Learning rate η
         * @param gradient Current gradient g_t
         */
        template <typename T, std::size_t N>
        void update(dp::mat::Vector<T, N> &x, T step_size, const dp::mat::Vector<T, N> &gradient) noexcept {
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

            // SIMD-optimized NAdam update
            constexpr std::size_t W = simd::backend::default_pack_width<double>();
            using pack_t = simd::pack<double, W>;

            const pack_t beta1_pack(beta1);
            const pack_t beta2_pack(beta2);
            const pack_t one_m_beta1(one_minus_beta1);
            const pack_t one_m_beta2(one_minus_beta2);
            const pack_t step_corr(step_correction);
            const pack_t eps(epsilon);
            const pack_t bc1_next(bias_correction1_next);
            const pack_t bc1(bias_correction1);

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

                // Load moments
                auto m_pack = pack_t::loadu(m_ptr + i);
                auto v_pack = pack_t::loadu(v_ptr + i);

                // Update first moment: m = β₁ * m + (1 - β₁) * g
                m_pack = pack_t::fma(beta1_pack, m_pack, one_m_beta1 * g);

                // Update second moment: v = β₂ * v + (1 - β₂) * g²
                v_pack = pack_t::fma(beta2_pack, v_pack, one_m_beta2 * g * g);

                // Store updated moments
                m_pack.storeu(m_ptr + i);
                v_pack.storeu(v_ptr + i);

                // NAdam lookahead momentum: m̄ = β₁ * m / bc1_next + (1 - β₁) * g / bc1
                auto m_hat_next = beta1_pack * m_pack / bc1_next;
                auto g_hat = one_m_beta1 * g / bc1;
                auto m_bar = m_hat_next + g_hat;

                // Compute update: step_correction * m̄ / (√v + ε)
                auto update = step_corr * m_bar / (v_pack.sqrt() + eps);

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
                m_ptr[i] = beta1 * m_ptr[i] + one_minus_beta1 * g_i;
                v_ptr[i] = beta2 * v_ptr[i] + one_minus_beta2 * g_i * g_i;
                double m_hat_next = beta1 * m_ptr[i] / bias_correction1_next;
                double g_hat = one_minus_beta1 * g_i / bias_correction1;
                double m_bar = m_hat_next + g_hat;
                double v_sqrt = std::sqrt(v_ptr[i]) + epsilon;
                x_ptr[i] -= T(step_correction * m_bar / v_sqrt);
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
        dp::mat::Vector<double, dp::mat::Dynamic> m; ///< First moment estimate (momentum)
        dp::mat::Vector<double, dp::mat::Dynamic> v; ///< Second moment estimate (variance)
        std::size_t iteration = 0;                   ///< Iteration counter for bias correction
    };

} // namespace optinum::opti
