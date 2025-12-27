#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include <optinum/simd/backend/backend.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::opti {

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
         * Update the iterate using Adam (SIMD-optimized)
         *
         * @param x Current iterate (modified in-place)
         * @param step_size Learning rate η
         * @param gradient Current gradient g_t
         */
        template <typename T, std::size_t N>
        void update(simd::Vector<T, N> &x, T step_size, const simd::Vector<T, N> &gradient) noexcept {
            // Lazy initialization on first use
            if (m.size() != N) {
                m.resize(N, T{0});
                v.resize(N, T{0});
                iteration = 0;
            }

            // Increment iteration counter
            ++iteration;

            T beta1_t = T(beta1);
            T beta2_t = T(beta2);
            T eps_t = T(epsilon);
            T one_minus_beta1 = T{1} - beta1_t;
            T one_minus_beta2 = T{1} - beta2_t;

            // Catch underflow
            if (eps_t == T{0} && epsilon != 0.0) {
                eps_t = T{10} * std::numeric_limits<T>::epsilon();
            }

            // Get raw pointers for SIMD operations
            T *m_ptr = m.data();
            T *v_ptr = v.data();
            const T *g_ptr = gradient.data();
            T *x_ptr = x.data();

            // SIMD width for this type and size
            constexpr std::size_t W = simd::backend::preferred_simd_lanes<T, N>();
            constexpr std::size_t main = simd::backend::main_loop_count<N, W>();

            using pack_t = simd::pack<T, W>;

            const pack_t beta1_vec(beta1_t);
            const pack_t beta2_vec(beta2_t);
            const pack_t one_minus_beta1_vec(one_minus_beta1);
            const pack_t one_minus_beta2_vec(one_minus_beta2);

            // Update biased first moment: m = beta1 * m + (1 - beta1) * g (SIMD)
            for (std::size_t i = 0; i < main; i += W) {
                auto m_val = pack_t::loadu(m_ptr + i);
                auto g_val = pack_t::loadu(g_ptr + i);
                auto result = beta1_vec * m_val + one_minus_beta1_vec * g_val;
                result.storeu(m_ptr + i);
            }
            // Tail
            for (std::size_t i = main; i < N; ++i) {
                m_ptr[i] = beta1_t * m_ptr[i] + one_minus_beta1 * g_ptr[i];
            }

            // Update biased second moment: v = beta2 * v + (1 - beta2) * g² (SIMD)
            for (std::size_t i = 0; i < main; i += W) {
                auto v_val = pack_t::loadu(v_ptr + i);
                auto g_val = pack_t::loadu(g_ptr + i);
                auto g_squared = g_val * g_val;
                auto result = beta2_vec * v_val + one_minus_beta2_vec * g_squared;
                result.storeu(v_ptr + i);
            }
            // Tail
            for (std::size_t i = main; i < N; ++i) {
                T g_i = g_ptr[i];
                v_ptr[i] = beta2_t * v_ptr[i] + one_minus_beta2 * g_i * g_i;
            }

            // Compute bias correction terms
            T bias_correction1 = T{1} - std::pow(beta1_t, T(iteration));
            T bias_correction2 = T{1} - std::pow(beta2_t, T(iteration));
            T step_correction = step_size * std::sqrt(bias_correction2) / bias_correction1;

            const pack_t step_correction_vec(step_correction);
            const pack_t eps_vec(eps_t);

            // Update iterate: x = x - step_correction * m / (sqrt(v) + eps) (SIMD)
            for (std::size_t i = 0; i < main; i += W) {
                auto x_val = pack_t::loadu(x_ptr + i);
                auto m_val = pack_t::loadu(m_ptr + i);
                auto v_val = pack_t::loadu(v_ptr + i);

                // Compute sqrt(v) + eps
                auto sqrt_v = simd::sqrt(v_val);
                auto denom = sqrt_v + eps_vec;

                // Compute update: step_correction * m / denom
                auto update = step_correction_vec * m_val / denom;

                // Apply update: x -= update
                auto result = x_val - update;
                result.storeu(x_ptr + i);
            }
            // Tail
            for (std::size_t i = main; i < N; ++i) {
                x_ptr[i] -= step_correction * m_ptr[i] / (std::sqrt(v_ptr[i]) + eps_t);
            }
        }

        /// Reset moments and iteration counter
        void reset() noexcept {
            m.clear();
            v.clear();
            iteration = 0;
        }

        /// Initialize (for compatibility with GradientDescent interface)
        template <typename T, std::size_t N> void initialize(std::size_t n) noexcept {
            m.clear();
            v.clear();
            iteration = 0;
        }

      private:
        std::vector<double> m;     ///< First moment estimate (momentum)
        std::vector<double> v;     ///< Second moment estimate (variance)
        std::size_t iteration = 0; ///< Iteration counter for bias correction
    };

} // namespace optinum::opti
