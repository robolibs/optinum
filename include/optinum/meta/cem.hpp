#pragma once

/**
 * Cross-Entropy Method (CEM)
 *
 * Population-based optimization via importance sampling. CEM maintains a
 * probability distribution over solutions, samples candidates, evaluates
 * them, and updates the distribution to focus on elite (best) samples.
 *
 * Algorithm:
 *   1. Initialize distribution: mean = initial, std = initial_std
 *   2. For each iteration:
 *      - Sample N candidates from N(mean, diag(std^2))
 *      - Evaluate fitness for all candidates
 *      - Select top elite_fraction candidates (elite set)
 *      - Update mean = mean of elite samples
 *      - Update std = std of elite samples (with smoothing)
 *   3. Return mean of final distribution
 *
 * References:
 *   - Rubinstein (1999) "The Cross-Entropy Method for Combinatorial
 *     and Continuous Optimization"
 *   - De Boer et al. (2005) "A Tutorial on the Cross-Entropy Method"
 *
 * @file cem.hpp
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::meta {

    /**
     * Result of CEM optimization
     */
    template <typename T> struct CEMResult {
        simd::Vector<T, simd::Dynamic> best_position; ///< Best solution found
        T best_value;                                 ///< Objective value at best position
        std::size_t iterations;                       ///< Number of iterations performed
        std::size_t function_evaluations;             ///< Total function evaluations
        bool converged;                               ///< Whether convergence criteria met
        std::vector<T> history;                       ///< Best value per iteration
    };

    /**
     * Cross-Entropy Method (CEM) Optimizer
     *
     * Usage:
     * @code
     * CEM<double> cem;
     * cem.config.population_size = 100;
     * cem.config.elite_fraction = 0.1;
     * cem.config.max_iterations = 100;
     *
     * auto objective = [](const auto& x) { return x[0]*x[0] + x[1]*x[1]; };
     *
     * simd::Vector<double, simd::Dynamic> lower{-5.0, -5.0};
     * simd::Vector<double, simd::Dynamic> upper{5.0, 5.0};
     *
     * auto result = cem.optimize(objective, lower, upper);
     * @endcode
     *
     * @tparam T Scalar type (float or double)
     */
    template <typename T = double> class CEM {
      public:
        /**
         * CEM configuration parameters
         */
        struct Config {
            std::size_t population_size = 100; ///< Number of samples per iteration
            std::size_t max_iterations = 100;  ///< Maximum iterations
            std::size_t horizon_size = 10;     ///< Window for convergence check
            T elite_fraction = T{0.1};         ///< Fraction of top samples to use (0-1)
            T initial_std = T{1.0};            ///< Initial standard deviation
            T min_std = T{1e-4};               ///< Minimum standard deviation
            T std_decay = T{0.99};             ///< Std decay factor per iteration
            T std_smoothing = T{0.5};          ///< Smoothing for std update (0=no smooth, 1=full smooth)
            T tolerance = T{1e-8};             ///< Convergence tolerance
            bool track_history = false;        ///< Record best value each iteration
        };

        Config config; ///< Configuration parameters

        /// Default constructor
        CEM() = default;

        /// Constructor with custom config
        explicit CEM(const Config &cfg) : config(cfg) {}

        /**
         * Optimize an objective function within bounds
         *
         * @tparam F Callable with signature T(const simd::Vector<T, Dynamic>&)
         * @param objective Function to minimize
         * @param lower_bounds Lower bounds for each dimension
         * @param upper_bounds Upper bounds for each dimension
         * @return CEMResult with best solution and convergence info
         */
        template <typename F>
        CEMResult<T> optimize(F &&objective, const simd::Vector<T, simd::Dynamic> &lower_bounds,
                              const simd::Vector<T, simd::Dynamic> &upper_bounds) {
            const std::size_t dim = lower_bounds.size();

            // Validate inputs
            if (dim == 0) {
                return CEMResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }
            if (upper_bounds.size() != dim) {
                return CEMResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }

            // Initialize mean at center of bounds
            simd::Vector<T, simd::Dynamic> mean(dim);
            for (std::size_t d = 0; d < dim; ++d) {
                mean[d] = (lower_bounds[d] + upper_bounds[d]) / T{2};
            }

            // Initialize std based on range
            simd::Vector<T, simd::Dynamic> std_dev(dim);
            for (std::size_t d = 0; d < dim; ++d) {
                std_dev[d] = config.initial_std * (upper_bounds[d] - lower_bounds[d]) / T{4};
            }

            return optimize_impl(std::forward<F>(objective), mean, std_dev, lower_bounds, upper_bounds);
        }

        /**
         * Optimize starting from an initial mean
         *
         * @tparam F Callable with signature T(const simd::Vector<T, Dynamic>&)
         * @param objective Function to minimize
         * @param initial_mean Initial mean of the distribution
         * @return CEMResult with best solution and convergence info
         */
        template <typename F> CEMResult<T> optimize(F &&objective, const simd::Vector<T, simd::Dynamic> &initial_mean) {
            const std::size_t dim = initial_mean.size();

            if (dim == 0) {
                return CEMResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }

            // Initialize std
            simd::Vector<T, simd::Dynamic> std_dev(dim);
            std_dev.fill(config.initial_std);

            // No bounds
            simd::Vector<T, simd::Dynamic> lower_bounds(dim);
            simd::Vector<T, simd::Dynamic> upper_bounds(dim);
            lower_bounds.fill(-std::numeric_limits<T>::max());
            upper_bounds.fill(std::numeric_limits<T>::max());

            return optimize_impl(std::forward<F>(objective), initial_mean, std_dev, lower_bounds, upper_bounds);
        }

        /**
         * Optimize with initial mean and bounds
         *
         * @tparam F Callable with signature T(const simd::Vector<T, Dynamic>&)
         * @param objective Function to minimize
         * @param initial_mean Initial mean of the distribution
         * @param lower_bounds Lower bounds for each dimension
         * @param upper_bounds Upper bounds for each dimension
         * @return CEMResult with best solution and convergence info
         */
        template <typename F>
        CEMResult<T> optimize(F &&objective, const simd::Vector<T, simd::Dynamic> &initial_mean,
                              const simd::Vector<T, simd::Dynamic> &lower_bounds,
                              const simd::Vector<T, simd::Dynamic> &upper_bounds) {
            const std::size_t dim = initial_mean.size();

            if (dim == 0 || lower_bounds.size() != dim || upper_bounds.size() != dim) {
                return CEMResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }

            // Initialize std based on range
            simd::Vector<T, simd::Dynamic> std_dev(dim);
            for (std::size_t d = 0; d < dim; ++d) {
                std_dev[d] = config.initial_std * (upper_bounds[d] - lower_bounds[d]) / T{4};
            }

            return optimize_impl(std::forward<F>(objective), initial_mean, std_dev, lower_bounds, upper_bounds);
        }

      private:
        /**
         * Core optimization implementation
         */
        template <typename F>
        CEMResult<T> optimize_impl(F &&objective, simd::Vector<T, simd::Dynamic> mean,
                                   simd::Vector<T, simd::Dynamic> std_dev,
                                   const simd::Vector<T, simd::Dynamic> &lower_bounds,
                                   const simd::Vector<T, simd::Dynamic> &upper_bounds) {
            const std::size_t dim = mean.size();
            const std::size_t n_samples = config.population_size;
            const std::size_t n_elite =
                std::max(std::size_t{1}, static_cast<std::size_t>(n_samples * config.elite_fraction));

            // Random number generator
            std::random_device rd;
            std::mt19937 rng(rd());

            // Storage for samples and their fitness values
            std::vector<simd::Vector<T, simd::Dynamic>> samples(n_samples);
            std::vector<T> fitness(n_samples);
            std::vector<std::size_t> indices(n_samples);

            // Best solution tracking
            simd::Vector<T, simd::Dynamic> best_position = mean;
            T best_value = std::numeric_limits<T>::max();
            std::size_t total_evals = 0;

            // History tracking
            std::vector<T> history;
            if (config.track_history) {
                history.reserve(config.max_iterations);
            }

            // Convergence tracking: sliding window of best values
            std::vector<T> horizon_buffer(config.horizon_size, std::numeric_limits<T>::max());
            std::size_t horizon_idx = 0;

            // Main optimization loop
            std::size_t iteration = 0;
            bool converged = false;

            for (; iteration < config.max_iterations; ++iteration) {
                // Sample population from current distribution
                for (std::size_t i = 0; i < n_samples; ++i) {
                    samples[i] = simd::Vector<T, simd::Dynamic>(dim);

                    for (std::size_t d = 0; d < dim; ++d) {
                        std::normal_distribution<T> dist(mean[d], std_dev[d]);
                        T value = dist(rng);

                        // Clamp to bounds
                        value = std::clamp(value, lower_bounds[d], upper_bounds[d]);
                        samples[i][d] = value;
                    }

                    // Evaluate fitness
                    fitness[i] = objective(samples[i]);
                    ++total_evals;

                    // Track best
                    if (fitness[i] < best_value) {
                        best_value = fitness[i];
                        best_position = samples[i];
                    }
                }

                // Sort indices by fitness (ascending - minimization)
                std::iota(indices.begin(), indices.end(), 0);
                std::sort(indices.begin(), indices.end(),
                          [&fitness](std::size_t a, std::size_t b) { return fitness[a] < fitness[b]; });

                // Compute elite mean
                simd::Vector<T, simd::Dynamic> elite_mean(dim);
                elite_mean.fill(T{0});

                for (std::size_t i = 0; i < n_elite; ++i) {
                    std::size_t idx = indices[i];
                    for (std::size_t d = 0; d < dim; ++d) {
                        elite_mean[d] += samples[idx][d];
                    }
                }
                for (std::size_t d = 0; d < dim; ++d) {
                    elite_mean[d] /= static_cast<T>(n_elite);
                }

                // Compute elite standard deviation
                simd::Vector<T, simd::Dynamic> elite_std(dim);
                elite_std.fill(T{0});

                for (std::size_t i = 0; i < n_elite; ++i) {
                    std::size_t idx = indices[i];
                    for (std::size_t d = 0; d < dim; ++d) {
                        T diff = samples[idx][d] - elite_mean[d];
                        elite_std[d] += diff * diff;
                    }
                }
                for (std::size_t d = 0; d < dim; ++d) {
                    elite_std[d] = std::sqrt(elite_std[d] / static_cast<T>(n_elite));
                    // Ensure minimum std
                    elite_std[d] = std::max(elite_std[d], config.min_std);
                }

                // Update distribution with smoothing
                for (std::size_t d = 0; d < dim; ++d) {
                    mean[d] = elite_mean[d];
                    // Smooth std update: new_std = (1-alpha)*elite_std + alpha*old_std
                    std_dev[d] = (T{1} - config.std_smoothing) * elite_std[d] + config.std_smoothing * std_dev[d];
                    // Apply decay
                    std_dev[d] *= config.std_decay;
                    // Ensure minimum
                    std_dev[d] = std::max(std_dev[d], config.min_std);
                }

                // Track history
                if (config.track_history) {
                    history.push_back(best_value);
                }

                // Update horizon buffer for convergence check
                horizon_buffer[horizon_idx] = best_value;
                horizon_idx = (horizon_idx + 1) % config.horizon_size;

                // Check convergence after horizon is filled
                if (iteration >= config.horizon_size) {
                    // Find min and max in horizon
                    T horizon_min = horizon_buffer[0];
                    T horizon_max = horizon_buffer[0];
                    for (std::size_t h = 1; h < config.horizon_size; ++h) {
                        horizon_min = std::min(horizon_min, horizon_buffer[h]);
                        horizon_max = std::max(horizon_max, horizon_buffer[h]);
                    }

                    // Converged if improvement over horizon is below tolerance
                    if (horizon_max - horizon_min < config.tolerance) {
                        converged = true;
                        break;
                    }
                }

                // Also check if std has collapsed
                T max_std = T{0};
                for (std::size_t d = 0; d < dim; ++d) {
                    max_std = std::max(max_std, std_dev[d]);
                }
                if (max_std <= config.min_std * T{1.01}) {
                    converged = true;
                    break;
                }
            }

            return CEMResult<T>{best_position, best_value, iteration, total_evals, converged, std::move(history)};
        }
    };

} // namespace optinum::meta
