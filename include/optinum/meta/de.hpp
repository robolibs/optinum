#pragma once

/**
 * Differential Evolution (DE)
 *
 * Population-based stochastic optimization for continuous global optimization.
 * DE evolves a population of candidate solutions using mutation, crossover,
 * and selection operators.
 *
 * Algorithm:
 *   1. Initialize population randomly within bounds
 *   2. For each generation:
 *      - For each individual i:
 *        - Select distinct random individuals for mutation
 *        - Create mutant vector based on strategy
 *        - Crossover: mix target and mutant with probability CR
 *        - Selection: keep trial if f(trial) < f(target)
 *   3. Return best individual
 *
 * Strategies:
 *   - Rand1:   v = r1 + F*(r2 - r3)
 *   - Best1:   v = best + F*(r1 - r2)
 *   - CurrentToBest1: v = x + F*(best - x) + F*(r1 - r2)
 *
 * References:
 *   - Storn & Price (1997) "Differential Evolution"
 *   - ensmallen DE: xtra/ensmallen/include/ensmallen_bits/de/
 *
 * @file de.hpp
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::meta {

    /**
     * Mutation strategy for Differential Evolution
     */
    enum class DEStrategy {
        Rand1,         ///< v = r1 + F*(r2 - r3) - classic random mutation
        Best1,         ///< v = best + F*(r1 - r2) - exploit best solution
        CurrentToBest1 ///< v = x + F*(best - x) + F*(r1 - r2) - balanced
    };

    /**
     * Result of DE optimization
     */
    template <typename T> struct DEResult {
        simd::Vector<T, simd::Dynamic> best_position; ///< Best solution found
        T best_value;                                 ///< Objective value at best position
        std::size_t generations;                      ///< Number of generations performed
        std::size_t function_evaluations;             ///< Total function evaluations
        bool converged;                               ///< Whether convergence criteria met
        std::vector<T> history;                       ///< Best value per generation
    };

    /**
     * Differential Evolution Optimizer
     *
     * Usage:
     * @code
     * DifferentialEvolution<double> de;
     * de.config.population_size = 50;
     * de.config.max_generations = 1000;
     * de.config.mutation_factor = 0.8;
     * de.config.crossover_prob = 0.9;
     *
     * auto objective = [](const auto& x) { return x[0]*x[0] + x[1]*x[1]; };
     *
     * simd::Vector<double, simd::Dynamic> lower{-5.0, -5.0};
     * simd::Vector<double, simd::Dynamic> upper{5.0, 5.0};
     *
     * auto result = de.optimize(objective, lower, upper);
     * @endcode
     *
     * @tparam T Scalar type (float or double)
     */
    template <typename T = double> class DifferentialEvolution {
      public:
        /**
         * DE configuration parameters
         */
        struct Config {
            std::size_t population_size = 50;        ///< Number of individuals in population
            std::size_t max_generations = 1000;      ///< Maximum generations
            std::size_t horizon_size = 50;           ///< Window for convergence check
            T mutation_factor = T{0.8};              ///< Differential weight F (0.4-1.0 typical)
            T crossover_prob = T{0.9};               ///< Crossover probability CR (0.1-1.0)
            T tolerance = T{1e-8};                   ///< Convergence tolerance
            DEStrategy strategy = DEStrategy::Best1; ///< Mutation strategy
            bool track_history = false;              ///< Record best value each generation
        };

        Config config; ///< Configuration parameters

        /// Default constructor
        DifferentialEvolution() = default;

        /// Constructor with custom config
        explicit DifferentialEvolution(const Config &cfg) : config(cfg) {}

        /**
         * Optimize an objective function within bounds
         *
         * @tparam F Callable with signature T(const simd::Vector<T, Dynamic>&)
         * @param objective Function to minimize
         * @param lower_bounds Lower bounds for each dimension
         * @param upper_bounds Upper bounds for each dimension
         * @return DEResult with best solution and convergence info
         */
        template <typename F>
        DEResult<T> optimize(F &&objective, const simd::Vector<T, simd::Dynamic> &lower_bounds,
                             const simd::Vector<T, simd::Dynamic> &upper_bounds) {
            const std::size_t dim = lower_bounds.size();
            const std::size_t pop_size = config.population_size;

            // Validate inputs
            if (dim == 0) {
                return DEResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }
            if (upper_bounds.size() != dim) {
                return DEResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }
            if (pop_size < 4) {
                // Need at least 4 for mutation (target + 3 distinct others)
                return DEResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }

            // Random number generator
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_real_distribution<T> uniform(T{0}, T{1});

            // Initialize population uniformly within bounds
            std::vector<simd::Vector<T, simd::Dynamic>> population(pop_size);
            std::vector<T> fitness(pop_size);
            std::size_t total_evals = 0;

            // Best solution tracking
            simd::Vector<T, simd::Dynamic> best_position(dim);
            T best_value = std::numeric_limits<T>::max();
            std::size_t best_idx = 0;

            // Initialize population
            for (std::size_t i = 0; i < pop_size; ++i) {
                population[i] = simd::Vector<T, simd::Dynamic>(dim);
                for (std::size_t d = 0; d < dim; ++d) {
                    T r = uniform(rng);
                    population[i][d] = lower_bounds[d] + r * (upper_bounds[d] - lower_bounds[d]);
                }

                // Evaluate fitness
                fitness[i] = objective(population[i]);
                ++total_evals;

                // Track best
                if (fitness[i] < best_value) {
                    best_value = fitness[i];
                    best_position = population[i];
                    best_idx = i;
                }
            }

            // History tracking
            std::vector<T> history;
            if (config.track_history) {
                history.reserve(config.max_generations);
            }

            // Convergence tracking: sliding window of best values
            std::vector<T> horizon_buffer(config.horizon_size, best_value);
            std::size_t horizon_idx = 0;

            // Trial vector for mutation/crossover
            simd::Vector<T, simd::Dynamic> trial(dim);

            // Main evolution loop
            std::size_t generation = 0;
            bool converged = false;

            for (; generation < config.max_generations; ++generation) {
                // For each individual in population
                for (std::size_t i = 0; i < pop_size; ++i) {
                    // Select distinct random indices for mutation
                    std::size_t r1, r2, r3;
                    do {
                        r1 = static_cast<std::size_t>(uniform(rng) * pop_size) % pop_size;
                    } while (r1 == i);

                    do {
                        r2 = static_cast<std::size_t>(uniform(rng) * pop_size) % pop_size;
                    } while (r2 == i || r2 == r1);

                    do {
                        r3 = static_cast<std::size_t>(uniform(rng) * pop_size) % pop_size;
                    } while (r3 == i || r3 == r1 || r3 == r2);

                    // Mutation: create mutant vector based on strategy
                    simd::Vector<T, simd::Dynamic> mutant(dim);

                    switch (config.strategy) {
                    case DEStrategy::Rand1:
                        // v = r1 + F*(r2 - r3)
                        for (std::size_t d = 0; d < dim; ++d) {
                            mutant[d] =
                                population[r1][d] + config.mutation_factor * (population[r2][d] - population[r3][d]);
                        }
                        break;

                    case DEStrategy::Best1:
                        // v = best + F*(r1 - r2)
                        for (std::size_t d = 0; d < dim; ++d) {
                            mutant[d] =
                                best_position[d] + config.mutation_factor * (population[r1][d] - population[r2][d]);
                        }
                        break;

                    case DEStrategy::CurrentToBest1:
                        // v = x + F*(best - x) + F*(r1 - r2)
                        for (std::size_t d = 0; d < dim; ++d) {
                            mutant[d] = population[i][d] +
                                        config.mutation_factor * (best_position[d] - population[i][d]) +
                                        config.mutation_factor * (population[r1][d] - population[r2][d]);
                        }
                        break;
                    }

                    // Crossover: binomial crossover
                    // At least one dimension must come from mutant (j_rand)
                    std::size_t j_rand = static_cast<std::size_t>(uniform(rng) * dim) % dim;

                    for (std::size_t d = 0; d < dim; ++d) {
                        if (uniform(rng) < config.crossover_prob || d == j_rand) {
                            trial[d] = mutant[d];
                        } else {
                            trial[d] = population[i][d];
                        }

                        // Clamp to bounds
                        trial[d] = std::clamp(trial[d], lower_bounds[d], upper_bounds[d]);
                    }

                    // Selection: greedy selection
                    T trial_fitness = objective(trial);
                    ++total_evals;

                    if (trial_fitness < fitness[i]) {
                        population[i] = trial;
                        fitness[i] = trial_fitness;

                        // Update global best
                        if (trial_fitness < best_value) {
                            best_value = trial_fitness;
                            best_position = trial;
                            best_idx = i;
                        }
                    }
                }

                // Track history
                if (config.track_history) {
                    history.push_back(best_value);
                }

                // Update horizon buffer for convergence check
                horizon_buffer[horizon_idx] = best_value;
                horizon_idx = (horizon_idx + 1) % config.horizon_size;

                // Check convergence after horizon is filled
                if (generation >= config.horizon_size) {
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
            }

            return DEResult<T>{best_position, best_value, generation, total_evals, converged, std::move(history)};
        }

        /**
         * Optimize starting from an initial point
         *
         * The initial point is used to seed one member of the population,
         * with the rest initialized randomly within bounds.
         *
         * @tparam F Callable with signature T(const simd::Vector<T, Dynamic>&)
         * @param objective Function to minimize
         * @param initial Initial point (seeds one population member)
         * @param lower_bounds Lower bounds for each dimension
         * @param upper_bounds Upper bounds for each dimension
         * @return DEResult with best solution and convergence info
         */
        template <typename F>
        DEResult<T> optimize(F &&objective, const simd::Vector<T, simd::Dynamic> &initial,
                             const simd::Vector<T, simd::Dynamic> &lower_bounds,
                             const simd::Vector<T, simd::Dynamic> &upper_bounds) {
            const std::size_t dim = initial.size();
            const std::size_t pop_size = config.population_size;

            // Validate inputs
            if (dim == 0) {
                return DEResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }
            if (lower_bounds.size() != dim || upper_bounds.size() != dim) {
                return DEResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }
            if (pop_size < 4) {
                return DEResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }

            // Random number generator
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_real_distribution<T> uniform(T{0}, T{1});

            // Initialize population - first member is the initial point
            std::vector<simd::Vector<T, simd::Dynamic>> population(pop_size);
            std::vector<T> fitness(pop_size);
            std::size_t total_evals = 0;

            // Best solution tracking
            simd::Vector<T, simd::Dynamic> best_position = initial;
            T best_value = objective(initial);
            ++total_evals;
            std::size_t best_idx = 0;

            // First member is the initial point
            population[0] = initial;
            fitness[0] = best_value;

            // Rest of population initialized randomly
            for (std::size_t i = 1; i < pop_size; ++i) {
                population[i] = simd::Vector<T, simd::Dynamic>(dim);
                for (std::size_t d = 0; d < dim; ++d) {
                    T r = uniform(rng);
                    population[i][d] = lower_bounds[d] + r * (upper_bounds[d] - lower_bounds[d]);
                }

                fitness[i] = objective(population[i]);
                ++total_evals;

                if (fitness[i] < best_value) {
                    best_value = fitness[i];
                    best_position = population[i];
                    best_idx = i;
                }
            }

            // History tracking
            std::vector<T> history;
            if (config.track_history) {
                history.reserve(config.max_generations);
            }

            // Convergence tracking
            std::vector<T> horizon_buffer(config.horizon_size, best_value);
            std::size_t horizon_idx = 0;

            // Trial vector
            simd::Vector<T, simd::Dynamic> trial(dim);

            // Main evolution loop
            std::size_t generation = 0;
            bool converged = false;

            for (; generation < config.max_generations; ++generation) {
                for (std::size_t i = 0; i < pop_size; ++i) {
                    // Select distinct random indices
                    std::size_t r1, r2, r3;
                    do {
                        r1 = static_cast<std::size_t>(uniform(rng) * pop_size) % pop_size;
                    } while (r1 == i);

                    do {
                        r2 = static_cast<std::size_t>(uniform(rng) * pop_size) % pop_size;
                    } while (r2 == i || r2 == r1);

                    do {
                        r3 = static_cast<std::size_t>(uniform(rng) * pop_size) % pop_size;
                    } while (r3 == i || r3 == r1 || r3 == r2);

                    // Mutation
                    simd::Vector<T, simd::Dynamic> mutant(dim);

                    switch (config.strategy) {
                    case DEStrategy::Rand1:
                        for (std::size_t d = 0; d < dim; ++d) {
                            mutant[d] =
                                population[r1][d] + config.mutation_factor * (population[r2][d] - population[r3][d]);
                        }
                        break;

                    case DEStrategy::Best1:
                        for (std::size_t d = 0; d < dim; ++d) {
                            mutant[d] =
                                best_position[d] + config.mutation_factor * (population[r1][d] - population[r2][d]);
                        }
                        break;

                    case DEStrategy::CurrentToBest1:
                        for (std::size_t d = 0; d < dim; ++d) {
                            mutant[d] = population[i][d] +
                                        config.mutation_factor * (best_position[d] - population[i][d]) +
                                        config.mutation_factor * (population[r1][d] - population[r2][d]);
                        }
                        break;
                    }

                    // Crossover
                    std::size_t j_rand = static_cast<std::size_t>(uniform(rng) * dim) % dim;

                    for (std::size_t d = 0; d < dim; ++d) {
                        if (uniform(rng) < config.crossover_prob || d == j_rand) {
                            trial[d] = mutant[d];
                        } else {
                            trial[d] = population[i][d];
                        }
                        trial[d] = std::clamp(trial[d], lower_bounds[d], upper_bounds[d]);
                    }

                    // Selection
                    T trial_fitness = objective(trial);
                    ++total_evals;

                    if (trial_fitness < fitness[i]) {
                        population[i] = trial;
                        fitness[i] = trial_fitness;

                        if (trial_fitness < best_value) {
                            best_value = trial_fitness;
                            best_position = trial;
                            best_idx = i;
                        }
                    }
                }

                if (config.track_history) {
                    history.push_back(best_value);
                }

                horizon_buffer[horizon_idx] = best_value;
                horizon_idx = (horizon_idx + 1) % config.horizon_size;

                if (generation >= config.horizon_size) {
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
            }

            // Suppress unused variable warning
            (void)best_idx;

            return DEResult<T>{best_position, best_value, generation, total_evals, converged, std::move(history)};
        }
    };

} // namespace optinum::meta
