#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

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
            // Lazy initialization on first use (use runtime size for Dynamic)
            const std::size_t n = x.size();
            if (m.size() != n) {
                m.resize(n, T{0});
                v.resize(n, T{0});
                v_hat.resize(n, T{0}); // AMSGrad: max of second moments
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

            // Get raw pointers
            T *m_ptr = m.data();
            T *v_ptr = v.data();
            T *v_hat_ptr = v_hat.data();
            const T *g_ptr = gradient.data();
            T *x_ptr = x.data();

            // Choose SIMD path based on whether N is Dynamic or fixed
            if constexpr (N == simd::Dynamic) {
                // RUNTIME SIMD for Dynamic size
                const std::size_t n = x.size();
                const std::size_t W = simd::backend::preferred_simd_lanes_runtime<T>();
                const std::size_t main = simd::backend::main_loop_count_runtime(n, W);

                constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
                using pack_t = simd::pack<T, pack_width>;

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
                for (std::size_t i = main; i < n; ++i) {
                    m_ptr[i] = beta1_t * m_ptr[i] + one_minus_beta1 * g_ptr[i];
                }

                // Update biased second moment: v = beta2 * v + (1 - beta2) * g² (SIMD)
                // Then update v_hat = max(v_hat, v) - THE KEY AMSGRAD DIFFERENCE
                for (std::size_t i = 0; i < main; i += W) {
                    auto v_val = pack_t::loadu(v_ptr + i);
                    auto v_hat_val = pack_t::loadu(v_hat_ptr + i);
                    auto g_val = pack_t::loadu(g_ptr + i);
                    auto g_squared = g_val * g_val;
                    auto v_new = beta2_vec * v_val + one_minus_beta2_vec * g_squared;
                    v_new.storeu(v_ptr + i);
                    // AMSGrad: v_hat = max(v_hat, v)
                    auto v_hat_new = simd::max(v_hat_val, v_new);
                    v_hat_new.storeu(v_hat_ptr + i);
                }
                // Tail
                for (std::size_t i = main; i < n; ++i) {
                    T g_i = g_ptr[i];
                    v_ptr[i] = beta2_t * v_ptr[i] + one_minus_beta2 * g_i * g_i;
                    v_hat_ptr[i] = std::max(v_hat_ptr[i], v_ptr[i]);
                }

                // Compute bias correction for first moment only
                // Note: AMSGrad typically doesn't bias-correct v_hat since it's a max
                T bias_correction1 = T{1} - std::pow(beta1_t, T(iteration));
                T step_correction = step_size / bias_correction1;

                const pack_t step_correction_vec(step_correction);
                const pack_t eps_vec(eps_t);

                // Update iterate: x = x - step_correction * m / (sqrt(v_hat) + eps) (SIMD)
                for (std::size_t i = 0; i < main; i += W) {
                    auto x_val = pack_t::loadu(x_ptr + i);
                    auto m_val = pack_t::loadu(m_ptr + i);
                    auto v_hat_val = pack_t::loadu(v_hat_ptr + i);

                    auto sqrt_v_hat = simd::sqrt(v_hat_val);
                    auto denom = sqrt_v_hat + eps_vec;
                    auto update = step_correction_vec * m_val / denom;
                    auto result = x_val - update;
                    result.storeu(x_ptr + i);
                }
                // Tail
                for (std::size_t i = main; i < n; ++i) {
                    x_ptr[i] -= step_correction * m_ptr[i] / (std::sqrt(v_hat_ptr[i]) + eps_t);
                }
            } else {
                // COMPILE-TIME SIMD for fixed size
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
                // Then update v_hat = max(v_hat, v) - THE KEY AMSGRAD DIFFERENCE
                for (std::size_t i = 0; i < main; i += W) {
                    auto v_val = pack_t::loadu(v_ptr + i);
                    auto v_hat_val = pack_t::loadu(v_hat_ptr + i);
                    auto g_val = pack_t::loadu(g_ptr + i);
                    auto g_squared = g_val * g_val;
                    auto v_new = beta2_vec * v_val + one_minus_beta2_vec * g_squared;
                    v_new.storeu(v_ptr + i);
                    // AMSGrad: v_hat = max(v_hat, v)
                    auto v_hat_new = simd::max(v_hat_val, v_new);
                    v_hat_new.storeu(v_hat_ptr + i);
                }
                // Tail
                for (std::size_t i = main; i < N; ++i) {
                    T g_i = g_ptr[i];
                    v_ptr[i] = beta2_t * v_ptr[i] + one_minus_beta2 * g_i * g_i;
                    v_hat_ptr[i] = std::max(v_hat_ptr[i], v_ptr[i]);
                }

                // Compute bias correction for first moment only
                T bias_correction1 = T{1} - std::pow(beta1_t, T(iteration));
                T step_correction = step_size / bias_correction1;

                const pack_t step_correction_vec(step_correction);
                const pack_t eps_vec(eps_t);

                // Update iterate: x = x - step_correction * m / (sqrt(v_hat) + eps) (SIMD)
                for (std::size_t i = 0; i < main; i += W) {
                    auto x_val = pack_t::loadu(x_ptr + i);
                    auto m_val = pack_t::loadu(m_ptr + i);
                    auto v_hat_val = pack_t::loadu(v_hat_ptr + i);

                    auto sqrt_v_hat = simd::sqrt(v_hat_val);
                    auto denom = sqrt_v_hat + eps_vec;
                    auto update = step_correction_vec * m_val / denom;
                    auto result = x_val - update;
                    result.storeu(x_ptr + i);
                }
                // Tail
                for (std::size_t i = main; i < N; ++i) {
                    x_ptr[i] -= step_correction * m_ptr[i] / (std::sqrt(v_hat_ptr[i]) + eps_t);
                }
            }
        }

        /// Reset moments and iteration counter
        void reset() noexcept {
            m.clear();
            v.clear();
            v_hat.clear();
            iteration = 0;
        }

        /// Initialize (for compatibility with GradientDescent interface)
        template <typename T, std::size_t N> void initialize(std::size_t n) noexcept {
            m.clear();
            v.clear();
            v_hat.clear();
            iteration = 0;
        }

      private:
        std::vector<double> m;     ///< First moment estimate (momentum)
        std::vector<double> v;     ///< Second moment estimate (variance)
        std::vector<double> v_hat; ///< Max of second moment estimates (AMSGrad key)
        std::size_t iteration = 0; ///< Iteration counter for bias correction
    };

} // namespace optinum::opti
