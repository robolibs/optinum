#pragma once

/**
 * Simulated Annealing (SA)
 *
 * Probabilistic optimization algorithm inspired by the annealing process in metallurgy.
 * SA explores the solution space by accepting worse solutions with a probability that
 * decreases as the "temperature" cools, allowing escape from local minima.
 *
 * Algorithm:
 *   1. Initialize with starting solution, set initial temperature T
 *   2. For each iteration:
 *      - Generate neighbor solution by random perturbation
 *      - Compute delta = f(neighbor) - f(current)
 *      - If delta < 0: accept (better solution)
 *      - Else: accept with probability exp(-delta/T)
 *      - Decrease temperature according to cooling schedule
 *   3. Return best solution found
 *
 * References:
 *   - Kirkpatrick, Gelatt, Vecchi (1983) "Optimization by Simulated Annealing"
 *   - ensmallen SA: xtra/ensmallen/include/ensmallen_bits/sa/
 *
 * @file sa.hpp
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <datapod/datapod.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/bridge.hpp>

namespace optinum::meta {

    /**
     * Cooling schedule types for temperature reduction
     */
    enum class CoolingSchedule {
        Geometric,   ///< T = T * cooling_rate (exponential decay)
        Linear,      ///< T = T - (T_init - T_final) / max_iter
        Logarithmic, ///< T = T_init / log(1 + iter)
        Adaptive     ///< Adjust based on acceptance ratio
    };

    /**
     * Result of Simulated Annealing optimization
     */
    template <typename T> struct SAResult {
        dp::mat::Vector<T, dp::mat::Dynamic> best_position; ///< Best solution found
        T best_value;                                       ///< Objective value at best position
        std::size_t iterations;                             ///< Number of iterations performed
        std::size_t accepted_moves;                         ///< Number of accepted moves
        std::size_t function_evaluations;                   ///< Total function evaluations
        bool converged;                                     ///< Whether convergence criteria met
        std::vector<T> history;                             ///< Best value per iteration
    };

    /**
     * Simulated Annealing Optimizer
     *
     * Usage:
     * @code
     * SimulatedAnnealing<double> sa;
     * sa.config.max_iterations = 10000;
     * sa.config.initial_temperature = 100.0;
     * sa.config.cooling_rate = 0.99;
     *
     * auto objective = [](const auto& x) { return x[0]*x[0] + x[1]*x[1]; };
     *
     * dp::mat::Vector<double, dp::mat::Dynamic> initial{1.0, 1.0};
     * auto result = sa.optimize(objective, initial);
     * @endcode
     *
     * @tparam T Scalar type (float or double)
     */
    template <typename T = double> class SimulatedAnnealing {
      public:
        /**
         * SA configuration parameters
         */
        struct Config {
            std::size_t max_iterations = 10000; ///< Maximum iterations
            std::size_t horizon_size = 100;     ///< Window for convergence check
            T initial_temperature = T{100.0};   ///< Starting temperature
            T final_temperature = T{1e-6};      ///< Minimum temperature
            T cooling_rate = T{0.99};           ///< Geometric cooling factor
            T step_size = T{0.1};               ///< Initial perturbation size
            T step_size_min = T{1e-6};          ///< Minimum step size
            T tolerance = T{1e-8};              ///< Convergence tolerance
            T target_acceptance = T{0.44};      ///< Target acceptance ratio (optimal ~0.44)
            bool adaptive_step = true;          ///< Adapt step size to maintain acceptance
            CoolingSchedule schedule = CoolingSchedule::Geometric;
            bool track_history = false; ///< Record best value each iteration
        };

        Config config; ///< Configuration parameters

        /// Default constructor
        SimulatedAnnealing() = default;

        /// Constructor with custom config
        explicit SimulatedAnnealing(const Config &cfg) : config(cfg) {}

        /**
         * Optimize an objective function starting from initial point
         *
         * @tparam F Callable with signature T(const simd::Vector<T, Dynamic>&)
         * @param objective Function to minimize
         * @param initial Starting point
         * @return SAResult with best solution and convergence info
         */
        template <typename F> SAResult<T> optimize(F &&objective, const dp::mat::Vector<T, dp::mat::Dynamic> &initial) {
            const std::size_t dim = initial.size();

            if (dim == 0) {
                return SAResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, 0, false, {}};
            }

            // No bounds - use large range
            dp::mat::Vector<T, dp::mat::Dynamic> lower(dim);
            dp::mat::Vector<T, dp::mat::Dynamic> upper(dim);
            lower.fill(-std::numeric_limits<T>::max());
            upper.fill(std::numeric_limits<T>::max());

            return optimize_impl(std::forward<F>(objective), initial, lower, upper);
        }

        /**
         * Optimize an objective function within bounds
         *
         * @tparam F Callable with signature T(const simd::Vector<T, Dynamic>&)
         * @param objective Function to minimize
         * @param initial Starting point
         * @param lower_bounds Lower bounds for each dimension
         * @param upper_bounds Upper bounds for each dimension
         * @return SAResult with best solution and convergence info
         */
        template <typename F>
        SAResult<T> optimize(F &&objective, const dp::mat::Vector<T, dp::mat::Dynamic> &initial,
                             const dp::mat::Vector<T, dp::mat::Dynamic> &lower_bounds,
                             const dp::mat::Vector<T, dp::mat::Dynamic> &upper_bounds) {
            const std::size_t dim = initial.size();

            if (dim == 0 || lower_bounds.size() != dim || upper_bounds.size() != dim) {
                return SAResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, 0, false, {}};
            }

            return optimize_impl(std::forward<F>(objective), initial, lower_bounds, upper_bounds);
        }

        /**
         * Optimize within bounds, starting from center
         *
         * @tparam F Callable with signature T(const simd::Vector<T, Dynamic>&)
         * @param objective Function to minimize
         * @param lower_bounds Lower bounds for each dimension
         * @param upper_bounds Upper bounds for each dimension
         * @return SAResult with best solution and convergence info
         */
        template <typename F>
        SAResult<T> optimize(F &&objective, const dp::mat::Vector<T, dp::mat::Dynamic> &lower_bounds,
                             const dp::mat::Vector<T, dp::mat::Dynamic> &upper_bounds) {
            const std::size_t dim = lower_bounds.size();

            if (dim == 0 || upper_bounds.size() != dim) {
                return SAResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, 0, false, {}};
            }

            // Start from center of bounds
            dp::mat::Vector<T, dp::mat::Dynamic> initial(dim);
            for (std::size_t d = 0; d < dim; ++d) {
                initial[d] = (lower_bounds[d] + upper_bounds[d]) / T{2};
            }

            return optimize_impl(std::forward<F>(objective), initial, lower_bounds, upper_bounds);
        }

      private:
        /**
         * Core optimization implementation
         */
        template <typename F>
        SAResult<T> optimize_impl(F &&objective, const dp::mat::Vector<T, dp::mat::Dynamic> &initial,
                                  const dp::mat::Vector<T, dp::mat::Dynamic> &lower_bounds,
                                  const dp::mat::Vector<T, dp::mat::Dynamic> &upper_bounds) {
            const std::size_t dim = initial.size();

            // Random number generator
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_real_distribution<T> uniform(T{0}, T{1});
            std::uniform_int_distribution<std::size_t> dim_dist(0, dim - 1);

            // Initialize current and best solutions using SIMD copy
            dp::mat::Vector<T, dp::mat::Dynamic> current(initial.size());
            simd::backend::copy_runtime<T>(current.data(), initial.data(), dim);
            T current_value = objective(current);
            std::size_t total_evals = 1;

            dp::mat::Vector<T, dp::mat::Dynamic> best(dim);
            simd::backend::copy_runtime<T>(best.data(), current.data(), dim);
            T best_value = current_value;

            // Temperature and step size
            T temperature = config.initial_temperature;
            T step_size = config.step_size;

            // Compute step size based on range if bounds are finite
            dp::mat::Vector<T, dp::mat::Dynamic> range(dim);
            for (std::size_t d = 0; d < dim; ++d) {
                if (std::isfinite(upper_bounds[d]) && std::isfinite(lower_bounds[d])) {
                    range[d] = (upper_bounds[d] - lower_bounds[d]) * step_size;
                } else {
                    range[d] = step_size;
                }
            }

            // Acceptance tracking for adaptive step size
            std::size_t accepted_moves = 0;
            std::size_t total_moves = 0;
            std::size_t acceptance_window = 100;
            std::size_t window_accepted = 0;

            // History tracking
            std::vector<T> history;
            if (config.track_history) {
                history.reserve(config.max_iterations);
            }

            // Convergence tracking
            std::vector<T> horizon_buffer(config.horizon_size, best_value);
            std::size_t horizon_idx = 0;

            // Main optimization loop
            std::size_t iteration = 0;
            bool converged = false;

            for (; iteration < config.max_iterations; ++iteration) {
                // Generate neighbor by perturbing one random dimension
                dp::mat::Vector<T, dp::mat::Dynamic> neighbor = current;
                std::size_t perturb_dim = dim_dist(rng);

                // Gaussian perturbation
                std::normal_distribution<T> normal(T{0}, range[perturb_dim]);
                T perturbation = normal(rng);
                neighbor[perturb_dim] += perturbation;

                // Clamp to bounds
                neighbor[perturb_dim] =
                    std::clamp(neighbor[perturb_dim], lower_bounds[perturb_dim], upper_bounds[perturb_dim]);

                // Evaluate neighbor
                T neighbor_value = objective(neighbor);
                ++total_evals;

                // Compute energy difference
                T delta = neighbor_value - current_value;

                // Metropolis acceptance criterion
                bool accept = false;
                if (delta < T{0}) {
                    // Always accept improvements
                    accept = true;
                } else {
                    // Accept worse solutions with probability exp(-delta/T)
                    T prob = std::exp(-delta / temperature);
                    accept = (uniform(rng) < prob);
                }

                if (accept) {
                    current = neighbor;
                    current_value = neighbor_value;
                    ++accepted_moves;
                    ++window_accepted;

                    // Update best if improved
                    if (current_value < best_value) {
                        simd::backend::copy_runtime<T>(best.data(), current.data(), dim);
                        best_value = current_value;
                    }
                }

                ++total_moves;

                // Adaptive step size adjustment
                if (config.adaptive_step && total_moves % acceptance_window == 0) {
                    T acceptance_ratio = static_cast<T>(window_accepted) / static_cast<T>(acceptance_window);

                    // Adjust step size to target acceptance ratio using SIMD
                    if (acceptance_ratio > config.target_acceptance + T{0.1}) {
                        // Too many acceptances - increase step size
                        simd::backend::mul_scalar_runtime<T>(range.data(), range.data(), T{1.1}, dim);
                    } else if (acceptance_ratio < config.target_acceptance - T{0.1}) {
                        // Too few acceptances - decrease step size
                        simd::backend::mul_scalar_runtime<T>(range.data(), range.data(), T{0.9}, dim);
                        // Clamp to minimum (scalar loop needed for max)
                        for (std::size_t d = 0; d < dim; ++d) {
                            range[d] = std::max(range[d], config.step_size_min);
                        }
                    }

                    window_accepted = 0;
                }

                // Cool temperature
                temperature = cool_temperature(temperature, iteration);

                // Track history
                if (config.track_history) {
                    history.push_back(best_value);
                }

                // Update horizon buffer for convergence check
                horizon_buffer[horizon_idx] = best_value;
                horizon_idx = (horizon_idx + 1) % config.horizon_size;

                // Check convergence
                if (iteration >= config.horizon_size) {
                    T horizon_min = horizon_buffer[0];
                    T horizon_max = horizon_buffer[0];
                    for (std::size_t h = 1; h < config.horizon_size; ++h) {
                        horizon_min = std::min(horizon_min, horizon_buffer[h]);
                        horizon_max = std::max(horizon_max, horizon_buffer[h]);
                    }

                    if (horizon_max - horizon_min < config.tolerance) {
                        converged = true;
                        break;
                    }
                }

                // Check if frozen (temperature too low)
                if (temperature < config.final_temperature) {
                    converged = true;
                    break;
                }
            }

            return SAResult<T>{best,        best_value, iteration + 1,     accepted_moves,
                               total_evals, converged,  std::move(history)};
        }

        /**
         * Compute new temperature based on cooling schedule
         */
        T cool_temperature(T current_temp, std::size_t iteration) const {
            switch (config.schedule) {
            case CoolingSchedule::Geometric:
                return current_temp * config.cooling_rate;

            case CoolingSchedule::Linear: {
                T delta =
                    (config.initial_temperature - config.final_temperature) / static_cast<T>(config.max_iterations);
                return std::max(current_temp - delta, config.final_temperature);
            }

            case CoolingSchedule::Logarithmic:
                return config.initial_temperature / std::log(T{2} + static_cast<T>(iteration));

            case CoolingSchedule::Adaptive:
                // Adaptive cooling based on iteration progress
                // Slower cooling at start, faster at end
                {
                    T progress = static_cast<T>(iteration) / static_cast<T>(config.max_iterations);
                    T rate = config.cooling_rate + (T{1} - config.cooling_rate) * (T{1} - progress) * T{0.5};
                    return current_temp * rate;
                }

            default:
                return current_temp * config.cooling_rate;
            }
        }
    };

} // namespace optinum::meta
