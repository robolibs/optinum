#pragma once

#include <cmath>

#include <datapod/datapod.hpp>
#include <optinum/simd/backend/backend.hpp>

namespace optinum::opti {

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
         * Update the iterate using Yogi
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
            double bias_correction1 = 1.0 - std::pow(beta1, double(iteration));
            double bias_correction2 = 1.0 - std::pow(beta2, double(iteration));
            double step_correction = double(step_size) * std::sqrt(bias_correction2) / bias_correction1;

            // Get raw pointers
            double *m_ptr = m.data();
            double *v_ptr = v.data();
            const T *g_ptr = gradient.data();
            T *x_ptr = x.data();

            // SIMD-optimized Yogi update
            constexpr std::size_t W = simd::backend::default_pack_width<double>();
            using pack_t = simd::pack<double, W>;

            const pack_t beta1_pack(beta1);
            const pack_t one_m_beta1(one_minus_beta1);
            const pack_t one_m_beta2(one_minus_beta2);
            const pack_t step_corr(step_correction);
            const pack_t eps(epsilon);
            const pack_t zero(0.0);

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

                auto g_sq = g * g;

                // Load moments
                auto m_pack = pack_t::loadu(m_ptr + i);
                auto v_pack = pack_t::loadu(v_ptr + i);

                // Update first moment: m = β₁ * m + (1 - β₁) * g
                m_pack = pack_t::fma(beta1_pack, m_pack, one_m_beta1 * g);
                m_pack.storeu(m_ptr + i);

                // Yogi update: v = v + (1 - β₂) * sign(g² - v) * g²
                // Compute sign using comparisons: sign(x) = (x > 0) ? 1 : ((x < 0) ? -1 : 0)
                auto diff = g_sq - v_pack;
                // For each lane, compute sign manually
                alignas(32) double sign_vals[W];
                alignas(32) double diff_vals[W];
                diff.storeu(diff_vals);
                for (std::size_t j = 0; j < W; ++j) {
                    sign_vals[j] = (diff_vals[j] > 0.0) ? 1.0 : ((diff_vals[j] < 0.0) ? -1.0 : 0.0);
                }
                auto sign_diff = pack_t::loadu(sign_vals);
                v_pack = v_pack + one_m_beta2 * sign_diff * g_sq;

                // Clamp v to non-negative
                alignas(32) double v_vals[W];
                v_pack.storeu(v_vals);
                for (std::size_t j = 0; j < W; ++j) {
                    if (v_vals[j] < 0.0)
                        v_vals[j] = 0.0;
                }
                v_pack = pack_t::loadu(v_vals);
                v_pack.storeu(v_ptr + i);

                // Compute update: step_correction * m / (√v + ε)
                auto update = step_corr * m_pack / (v_pack.sqrt() + eps);

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
                double g_sq = g_i * g_i;
                m_ptr[i] = beta1 * m_ptr[i] + one_minus_beta1 * g_i;
                double diff = g_sq - v_ptr[i];
                double sign_diff = (diff > 0.0) ? 1.0 : ((diff < 0.0) ? -1.0 : 0.0);
                v_ptr[i] = v_ptr[i] + one_minus_beta2 * sign_diff * g_sq;
                if (v_ptr[i] < 0.0)
                    v_ptr[i] = 0.0;
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
        dp::mat::Vector<double, dp::mat::Dynamic> m; ///< First moment estimate (momentum)
        dp::mat::Vector<double, dp::mat::Dynamic> v; ///< Second moment estimate (variance)
        std::size_t iteration = 0;                   ///< Iteration counter for bias correction
    };

} // namespace optinum::opti
