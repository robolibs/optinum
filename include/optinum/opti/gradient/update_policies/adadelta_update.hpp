#pragma once

#include <cmath>

#include <datapod/matrix/vector.hpp>
#include <optinum/simd/backend/backend.hpp>

namespace optinum::opti {

    namespace dp = ::datapod;

    /**
     * AdaDelta update policy
     *
     * An extension of AdaGrad that seeks to reduce its aggressive, monotonically
     * decreasing learning rate. Instead of accumulating all past squared gradients,
     * AdaDelta restricts the window of accumulated past gradients to a fixed size.
     *
     * Key innovation: AdaDelta does not require a learning rate to be set manually.
     * It uses the ratio of RMS of parameter updates to RMS of gradients.
     *
     * Update equations:
     *   E[g²]_t = ρ * E[g²]_{t-1} + (1 - ρ) * g_t²           // Accumulated gradient
     *   Δx_t = -(RMS[Δx]_{t-1} / RMS[g]_t) * g_t             // Compute update
     *   E[Δx²]_t = ρ * E[Δx²]_{t-1} + (1 - ρ) * Δx_t²        // Accumulated updates
     *   x_{t+1} = x_t + Δx_t                                  // Apply update
     *
     * where:
     *   - RMS[y] = √(E[y²] + ε)
     *   - ρ is the decay rate (typically 0.95)
     *   - ε is a small constant for numerical stability (typically 1e-6)
     *
     * Properties:
     *   - No manual learning rate tuning required
     *   - Robust to large gradients and noisy problems
     *   - Corrects units mismatch in AdaGrad update rule
     *   - Learning rate adapts based on curvature information
     *
     * Reference:
     *   Zeiler (2012)
     *   "ADADELTA: An Adaptive Learning Rate Method"
     *   https://arxiv.org/abs/1212.5701
     */
    struct AdaDeltaUpdate {
        double rho = 0.95;     ///< Decay rate ρ ∈ (0, 1)
        double epsilon = 1e-6; ///< Numerical stability constant ε

        /// Constructor with rho and epsilon parameters
        explicit AdaDeltaUpdate(double r = 0.95, double eps = 1e-6) : rho(r), epsilon(eps) {}

        /**
         * Update the iterate using AdaDelta
         *
         * Note: The step_size parameter is ignored in AdaDelta as it computes
         * its own adaptive learning rate. It's kept for API compatibility.
         *
         * @param x Current iterate (modified in-place)
         * @param step_size Ignored (AdaDelta computes its own learning rate)
         * @param gradient Current gradient g_t
         */
        template <typename T, std::size_t N>
        void update(dp::mat::vector<T, N> &x, T /*step_size*/, const dp::mat::vector<T, N> &gradient) noexcept {
            const std::size_t n = x.size();

            // Lazy initialization on first use
            if (avg_squared_grad.size() != n) {
                avg_squared_grad.resize(n);
                avg_squared_delta.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    avg_squared_grad[i] = 0.0;
                    avg_squared_delta[i] = 0.0;
                }
            }

            double one_minus_rho = 1.0 - rho;

            // Get raw pointers
            double *E_g2_ptr = avg_squared_grad.data();
            double *E_dx2_ptr = avg_squared_delta.data();
            const T *g_ptr = gradient.data();
            T *x_ptr = x.data();

            // SIMD-optimized AdaDelta update
            constexpr std::size_t W = 4; // AVX: 4 doubles
            using pack_t = simd::pack<double, W>;

            const pack_t rho_pack(rho);
            const pack_t one_m_rho(one_minus_rho);
            const pack_t eps(epsilon);
            const pack_t neg_one(-1.0);

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

                // Load running averages
                auto E_g2 = pack_t::loadu(E_g2_ptr + i);
                auto E_dx2 = pack_t::loadu(E_dx2_ptr + i);

                // Update E[g²] = ρ * E[g²] + (1-ρ) * g²
                E_g2 = pack_t::fma(rho_pack, E_g2, one_m_rho * g * g);
                E_g2.storeu(E_g2_ptr + i);

                // Compute RMS values: RMS[g] = √(E[g²] + ε), RMS[Δx] = √(E[Δx²] + ε)
                auto rms_g = (E_g2 + eps).sqrt();
                auto rms_dx = (E_dx2 + eps).sqrt();

                // Compute update: Δx = -(RMS[Δx] / RMS[g]) * g
                auto delta_x = neg_one * (rms_dx / rms_g) * g;

                // Update E[Δx²] = ρ * E[Δx²] + (1-ρ) * Δx²
                E_dx2 = pack_t::fma(rho_pack, E_dx2, one_m_rho * delta_x * delta_x);
                E_dx2.storeu(E_dx2_ptr + i);

                // Apply update to x
                if constexpr (std::is_same_v<T, double>) {
                    auto x_pack = pack_t::loadu(x_ptr + i);
                    (x_pack + delta_x).storeu(x_ptr + i);
                } else {
                    for (std::size_t j = 0; j < W; ++j) {
                        x_ptr[i + j] += T(delta_x[j]);
                    }
                }
            }

            // Scalar tail
            for (std::size_t i = main; i < n; ++i) {
                double g_i = double(g_ptr[i]);
                E_g2_ptr[i] = rho * E_g2_ptr[i] + one_minus_rho * g_i * g_i;
                double rms_g = std::sqrt(E_g2_ptr[i] + epsilon);
                double rms_dx = std::sqrt(E_dx2_ptr[i] + epsilon);
                double delta_x = -(rms_dx / rms_g) * g_i;
                E_dx2_ptr[i] = rho * E_dx2_ptr[i] + one_minus_rho * delta_x * delta_x;
                x_ptr[i] += T(delta_x);
            }
        }

        /// Reset accumulated averages to zero
        void reset() noexcept {
            avg_squared_grad.resize(0);
            avg_squared_delta.resize(0);
        }

        /// Initialize (for compatibility with GradientDescent interface)
        template <typename T, std::size_t N> void initialize(std::size_t) noexcept {
            avg_squared_grad.resize(0);
            avg_squared_delta.resize(0);
        }

      private:
        dp::mat::vector<double, dp::mat::Dynamic> avg_squared_grad;  ///< Running average of squared gradients E[g²]
        dp::mat::vector<double, dp::mat::Dynamic> avg_squared_delta; ///< Running average of squared updates E[Δx²]
    };

} // namespace optinum::opti
