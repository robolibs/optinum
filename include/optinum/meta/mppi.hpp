#pragma once

/**
 * Model Predictive Path Integral (MPPI) Controller
 *
 * Sampling-based stochastic optimal control for trajectory optimization.
 * MPPI samples K trajectories by adding noise to a nominal control sequence,
 * evaluates costs, and updates controls using importance-weighted averaging.
 *
 * Algorithm:
 *   1. Sample K trajectories by adding Gaussian noise to control sequence
 *   2. Rollout dynamics for each trajectory: x_{t+1} = f(x_t, u_t)
 *   3. Evaluate total cost for each trajectory: J_k = sum_t c(x_t, u_t)
 *   4. Compute importance weights: w_k = exp(-J_k / lambda)
 *   5. Update control: u = sum(w_k * u_k) / sum(w_k)
 *   6. Shift control sequence for receding horizon
 *
 * References:
 *   - Williams et al. (2017) "Information Theoretic MPC for Model-Based RL"
 *     https://arxiv.org/abs/1707.02342
 *
 * @file mppi.hpp
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <datapod/matrix/vector.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::meta {

    namespace dp = ::datapod;

    /**
     * Result of MPPI optimization step
     */
    template <typename T> struct MPPIResult {
        dp::mat::vector<T, dp::mat::Dynamic> optimal_control; ///< First control in optimized sequence
        T best_cost;                                          ///< Cost of best sampled trajectory
        std::size_t iterations;                               ///< Number of optimization iterations
        bool valid;                                           ///< Whether optimization succeeded
    };

    /**
     * Model Predictive Path Integral (MPPI) Controller
     *
     * Usage:
     * @code
     * MPPI<double> mppi;
     * mppi.config.num_samples = 1000;
     * mppi.config.horizon = 50;
     * mppi.config.lambda = 1.0;
     *
     * // Define dynamics: x_{t+1} = f(x_t, u_t)
     * auto dynamics = [](const auto& state, const auto& control) {
     *     simd::Vector<double, simd::Dynamic> next_state(state.size());
     *     // ... integrate dynamics ...
     *     return next_state;
     * };
     *
     * // Define cost: c(x, u)
     * auto cost = [](const auto& state, const auto& control) {
     *     return state[0]*state[0] + 0.1*control[0]*control[0];
     * };
     *
     * auto result = mppi.step(dynamics, cost, current_state);
     * @endcode
     *
     * @tparam T Scalar type (float or double)
     */
    template <typename T = double> class MPPI {
      public:
        /**
         * MPPI configuration parameters
         */
        struct Config {
            std::size_t num_samples = 1000; ///< Number of trajectory samples (K)
            std::size_t horizon = 50;       ///< Prediction horizon (time steps)
            std::size_t state_dim = 0;      ///< State dimension (set automatically if 0)
            std::size_t control_dim = 0;    ///< Control dimension (set automatically if 0)
            T lambda = T{1.0};              ///< Temperature parameter (higher = more exploration)
            T noise_sigma = T{1.0};         ///< Standard deviation of control noise
            T dt = T{0.02};                 ///< Time step for dynamics integration
            bool warm_start = true;         ///< Use previous solution as initial guess
            bool track_history = false;     ///< Record best cost per iteration
        };

        /**
         * Control bounds for clamping
         */
        struct Bounds {
            dp::mat::vector<T, dp::mat::Dynamic> lower; ///< Lower bounds for each control dimension
            dp::mat::vector<T, dp::mat::Dynamic> upper; ///< Upper bounds for each control dimension

            bool valid() const { return lower.size() > 0 && lower.size() == upper.size(); }
        };

        Config config; ///< Configuration parameters
        Bounds bounds; ///< Control bounds (optional)

        /// Default constructor
        MPPI() = default;

        /// Constructor with custom config
        explicit MPPI(const Config &cfg) : config(cfg) { initialize(); }

        /**
         * Initialize or reinitialize the controller
         *
         * Call this after changing config.horizon or config.control_dim
         */
        void initialize() {
            if (config.control_dim == 0 || config.horizon == 0) {
                return; // Will be initialized on first step
            }

            // Initialize mean control sequence to zeros
            mean_controls_.resize(config.horizon);
            for (std::size_t t = 0; t < config.horizon; ++t) {
                mean_controls_[t] = dp::mat::vector<T, dp::mat::Dynamic>(config.control_dim);
                mean_controls_[t].fill(T{0});
            }

            initialized_ = true;
        }

        /**
         * Reset the controller state
         *
         * Clears the mean control sequence (useful when starting a new trajectory)
         */
        void reset() {
            for (auto &u : mean_controls_) {
                u.fill(T{0});
            }
        }

        /**
         * Perform one MPPI optimization step
         *
         * @tparam Dynamics Callable: (state, control) -> next_state
         * @tparam Cost Callable: (state, control) -> scalar cost
         * @param dynamics System dynamics function
         * @param cost Stage cost function
         * @param initial_state Current state of the system
         * @return MPPIResult with optimal control and diagnostics
         */
        template <typename Dynamics, typename Cost>
        MPPIResult<T> step(Dynamics &&dynamics, Cost &&cost, const simd::Vector<T, simd::Dynamic> &initial_state) {
            const std::size_t state_dim = initial_state.size();

            // Auto-detect control dimension from first dynamics call if needed
            if (config.control_dim == 0) {
                // Try to infer control dimension - use a dummy control
                // This is a fallback; users should set control_dim explicitly
                config.control_dim = state_dim; // Common case: control_dim == state_dim
            }

            const std::size_t control_dim = config.control_dim;
            const std::size_t K = config.num_samples;
            const std::size_t H = config.horizon;

            // Initialize if needed
            if (!initialized_ || mean_controls_.size() != H) {
                config.state_dim = state_dim;
                initialize();
            }

            // Validate
            if (K == 0 || H == 0 || control_dim == 0) {
                return MPPIResult<T>{{}, std::numeric_limits<T>::max(), 0, false};
            }

            // Random number generator
            std::random_device rd;
            std::mt19937 rng(rd());
            std::normal_distribution<T> noise_dist(T{0}, config.noise_sigma);

            // Storage for sampled noise and costs
            // noise_samples[k][t] = noise vector for sample k at time t
            std::vector<std::vector<dp::mat::vector<T, dp::mat::Dynamic>>> noise_samples(K);
            std::vector<T> costs(K, T{0});

            // Best trajectory tracking
            T best_cost = std::numeric_limits<T>::max();
            std::size_t best_idx = 0;

            // Sample and rollout all K trajectories
            for (std::size_t k = 0; k < K; ++k) {
                noise_samples[k].resize(H);

                // Initialize state for this rollout
                dp::mat::vector<T, dp::mat::Dynamic> state(initial_state.size());
                for (std::size_t i = 0; i < initial_state.size(); ++i) {
                    state[i] = initial_state[i];
                }
                T trajectory_cost = T{0};

                // Rollout trajectory
                for (std::size_t t = 0; t < H; ++t) {
                    // Sample control noise
                    noise_samples[k][t] = dp::mat::vector<T, dp::mat::Dynamic>(control_dim);
                    for (std::size_t d = 0; d < control_dim; ++d) {
                        noise_samples[k][t][d] = noise_dist(rng);
                    }

                    // Compute noisy control: u = mean + noise
                    dp::mat::vector<T, dp::mat::Dynamic> control(control_dim);
                    for (std::size_t d = 0; d < control_dim; ++d) {
                        control[d] = mean_controls_[t][d] + noise_samples[k][t][d];
                    }

                    // Apply control bounds if specified
                    if (bounds.valid()) {
                        for (std::size_t d = 0; d < control_dim; ++d) {
                            control[d] = std::clamp(control[d], bounds.lower[d], bounds.upper[d]);
                        }
                    }

                    // Evaluate stage cost
                    T stage_cost = cost(state, control);
                    trajectory_cost += stage_cost * config.dt;

                    // Integrate dynamics
                    state = dynamics(state, control);
                }

                costs[k] = trajectory_cost;

                // Track best trajectory
                if (trajectory_cost < best_cost) {
                    best_cost = trajectory_cost;
                    best_idx = k;
                }
            }

            // Compute importance weights
            // w_k = exp(-1/lambda * (J_k - J_min))
            const T temperature = std::max(config.lambda, T{1e-6});
            const T beta = T{1} / temperature;

            T min_cost = *std::min_element(costs.begin(), costs.end());
            std::vector<T> weights(K);
            T weight_sum = T{0};

            for (std::size_t k = 0; k < K; ++k) {
                // Subtract min_cost for numerical stability
                T exponent = -beta * (costs[k] - min_cost);
                // Clamp exponent to prevent underflow (exp(-60) ≈ 0)
                exponent = std::max(exponent, T{-60});
                weights[k] = std::exp(exponent);
                weight_sum += weights[k];
            }

            // Handle degenerate case
            if (weight_sum < T{1e-12}) {
                weight_sum = T{1};
            }

            // Update mean control sequence using weighted noise
            for (std::size_t t = 0; t < H; ++t) {
                dp::mat::vector<T, dp::mat::Dynamic> delta_u(control_dim);
                delta_u.fill(T{0});

                for (std::size_t k = 0; k < K; ++k) {
                    T w = weights[k] / weight_sum;
                    // delta_u += w * noise_samples[k][t]
                    for (std::size_t d = 0; d < control_dim; ++d) {
                        delta_u[d] += w * noise_samples[k][t][d];
                    }
                }

                // Update mean control
                for (std::size_t d = 0; d < control_dim; ++d) {
                    mean_controls_[t][d] += delta_u[d];
                }

                // Apply bounds to updated mean
                if (bounds.valid()) {
                    for (std::size_t d = 0; d < control_dim; ++d) {
                        mean_controls_[t][d] = std::clamp(mean_controls_[t][d], bounds.lower[d], bounds.upper[d]);
                    }
                }
            }

            // Extract first control (receding horizon)
            dp::mat::vector<T, dp::mat::Dynamic> optimal_control = mean_controls_[0];

            // Shift control sequence for next iteration (warm start)
            if (config.warm_start) {
                for (std::size_t t = 0; t + 1 < H; ++t) {
                    mean_controls_[t] = mean_controls_[t + 1];
                }
                // Initialize last control to zero
                mean_controls_[H - 1].fill(T{0});
            }

            // Track history if enabled
            if (config.track_history) {
                history_.push_back(best_cost);
            }

            return MPPIResult<T>{optimal_control, best_cost, 1, true};
        }

        /**
         * Perform multiple MPPI iterations for better convergence
         *
         * @tparam Dynamics Callable: (state, control) -> next_state
         * @tparam Cost Callable: (state, control) -> scalar cost
         * @param dynamics System dynamics function
         * @param cost Stage cost function
         * @param initial_state Current state of the system
         * @param num_iterations Number of optimization iterations
         * @return MPPIResult with optimal control and diagnostics
         */
        template <typename Dynamics, typename Cost>
        MPPIResult<T> optimize(Dynamics &&dynamics, Cost &&cost, const simd::Vector<T, simd::Dynamic> &initial_state,
                               std::size_t num_iterations = 1) {
            MPPIResult<T> result{{}, std::numeric_limits<T>::max(), 0, false};

            for (std::size_t iter = 0; iter < num_iterations; ++iter) {
                result = step(std::forward<Dynamics>(dynamics), std::forward<Cost>(cost), initial_state);
                if (!result.valid) {
                    break;
                }
                result.iterations = iter + 1;
            }

            return result;
        }

        /**
         * Get the full optimized control sequence
         *
         * @return Vector of control vectors for each time step
         */
        const std::vector<dp::mat::vector<T, dp::mat::Dynamic>> &get_control_sequence() const { return mean_controls_; }

        /**
         * Get the cost history (if tracking enabled)
         *
         * @return Vector of best costs per iteration
         */
        const std::vector<T> &get_history() const { return history_; }

        /**
         * Set the control sequence (for external initialization)
         *
         * @param controls Initial control sequence
         */
        void set_control_sequence(const std::vector<dp::mat::vector<T, dp::mat::Dynamic>> &controls) {
            mean_controls_ = controls;
            if (!controls.empty()) {
                config.horizon = controls.size();
                config.control_dim = controls[0].size();
                initialized_ = true;
            }
        }

      private:
        std::vector<dp::mat::vector<T, dp::mat::Dynamic>> mean_controls_; ///< Mean control sequence
        std::vector<T> history_;                                          ///< Best cost per iteration
        bool initialized_ = false;                                        ///< Whether controller is initialized
    };

} // namespace optinum::meta
