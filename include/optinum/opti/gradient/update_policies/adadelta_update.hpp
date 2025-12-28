#pragma once

#include <cmath>
#include <limits>

#include <optinum/simd/backend/backend.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::opti {

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
         * Update the iterate using AdaDelta (SIMD-optimized)
         *
         * Note: The step_size parameter is ignored in AdaDelta as it computes
         * its own adaptive learning rate. It's kept for API compatibility.
         *
         * @param x Current iterate (modified in-place)
         * @param step_size Ignored (AdaDelta computes its own learning rate)
         * @param gradient Current gradient g_t
         */
        template <typename T, std::size_t N>
        void update(simd::Vector<T, N> &x, T /*step_size*/, const simd::Vector<T, N> &gradient) noexcept {
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

            if constexpr (N == simd::Dynamic) {
                for (std::size_t i = 0; i < n; ++i) {
                    double g_i = double(g_ptr[i]);

                    // Update running average of squared gradients: E[g²] = ρ * E[g²] + (1-ρ) * g²
                    E_g2_ptr[i] = rho * E_g2_ptr[i] + one_minus_rho * g_i * g_i;

                    // Compute RMS of gradients: RMS[g] = √(E[g²] + ε)
                    double rms_g = std::sqrt(E_g2_ptr[i] + epsilon);

                    // Compute RMS of previous updates: RMS[Δx] = √(E[Δx²] + ε)
                    double rms_dx = std::sqrt(E_dx2_ptr[i] + epsilon);

                    // Compute update: Δx = -(RMS[Δx] / RMS[g]) * g
                    double delta_x = -(rms_dx / rms_g) * g_i;

                    // Update running average of squared updates: E[Δx²] = ρ * E[Δx²] + (1-ρ) * Δx²
                    E_dx2_ptr[i] = rho * E_dx2_ptr[i] + one_minus_rho * delta_x * delta_x;

                    // Apply update: x = x + Δx
                    x_ptr[i] += T(delta_x);
                }
            } else {
                for (std::size_t i = 0; i < N; ++i) {
                    double g_i = double(g_ptr[i]);

                    E_g2_ptr[i] = rho * E_g2_ptr[i] + one_minus_rho * g_i * g_i;

                    double rms_g = std::sqrt(E_g2_ptr[i] + epsilon);
                    double rms_dx = std::sqrt(E_dx2_ptr[i] + epsilon);

                    double delta_x = -(rms_dx / rms_g) * g_i;

                    E_dx2_ptr[i] = rho * E_dx2_ptr[i] + one_minus_rho * delta_x * delta_x;

                    x_ptr[i] += T(delta_x);
                }
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
        simd::Vector<double, simd::Dynamic> avg_squared_grad;  ///< Running average of squared gradients E[g²]
        simd::Vector<double, simd::Dynamic> avg_squared_delta; ///< Running average of squared updates E[Δx²]
    };

} // namespace optinum::opti
