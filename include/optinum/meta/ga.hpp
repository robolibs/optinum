#pragma once

/**
 * Genetic Algorithm (GA)
 *
 * Population-based evolutionary optimization inspired by natural selection.
 * GA evolves a population of candidate solutions through selection, crossover,
 * and mutation operators.
 *
 * Algorithm:
 *   1. Initialize population randomly within bounds
 *   2. For each generation:
 *      - Evaluate fitness for all individuals
 *      - Selection: choose parents (tournament, roulette, rank)
 *      - Crossover: combine parents to create offspring
 *      - Mutation: randomly perturb offspring
 *      - Replacement: form new population (with elitism)
 *   3. Return best individual
 *
 * Operators:
 *   - Selection: Tournament (default), Roulette Wheel, Rank-based
 *   - Crossover: SBX (default), Uniform, Single-point
 *   - Mutation: Gaussian (default), Polynomial
 *
 * References:
 *   - Holland (1975) "Adaptation in Natural and Artificial Systems"
 *   - Deb & Agrawal (1995) "Simulated Binary Crossover"
 *   - ensmallen GA: xtra/ensmallen/include/ensmallen_bits/
 *
 * @file ga.hpp
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
     * Selection strategy for parent selection
     */
    enum class GASelection {
        Tournament,    ///< Tournament selection (default)
        RouletteWheel, ///< Fitness-proportionate selection
        Rank           ///< Rank-based selection
    };

    /**
     * Crossover operator for combining parents
     */
    enum class GACrossover {
        SBX,        ///< Simulated Binary Crossover (default, for real-valued)
        Uniform,    ///< Uniform crossover
        SinglePoint ///< Single-point crossover
    };

    /**
     * Mutation operator
     */
    enum class GAMutation {
        Gaussian,  ///< Gaussian mutation (default)
        Polynomial ///< Polynomial mutation
    };

    /**
     * Result of GA optimization
     */
    template <typename T> struct GAResult {
        simd::Vector<T, simd::Dynamic> best_position; ///< Best solution found
        T best_value;                                 ///< Objective value at best position
        std::size_t generations;                      ///< Number of generations performed
        std::size_t function_evaluations;             ///< Total function evaluations
        bool converged;                               ///< Whether convergence criteria met
        std::vector<T> history;                       ///< Best value per generation
    };

    /**
     * Genetic Algorithm Optimizer
     *
     * Usage:
     * @code
     * GeneticAlgorithm<double> ga;
     * ga.config.population_size = 100;
     * ga.config.max_generations = 500;
     * ga.config.crossover_prob = 0.9;
     * ga.config.mutation_prob = 0.1;
     *
     * auto objective = [](const auto& x) { return x[0]*x[0] + x[1]*x[1]; };
     *
     * simd::Vector<double, simd::Dynamic> lower{-5.0, -5.0};
     * simd::Vector<double, simd::Dynamic> upper{5.0, 5.0};
     *
     * auto result = ga.optimize(objective, lower, upper);
     * @endcode
     *
     * @tparam T Scalar type (float or double)
     */
    template <typename T = double> class GeneticAlgorithm {
      public:
        /**
         * GA configuration parameters
         */
        struct Config {
            std::size_t population_size = 100; ///< Number of individuals
            std::size_t max_generations = 500; ///< Maximum generations
            std::size_t horizon_size = 50;     ///< Window for convergence check
            T crossover_prob = T{0.9};         ///< Probability of crossover
            T mutation_prob = T{0.1};          ///< Probability of mutation per gene
            T mutation_strength = T{0.1};      ///< Mutation step size (fraction of range)
            std::size_t tournament_size = 3;   ///< Tournament size for selection
            std::size_t elitism = 2;           ///< Number of elite individuals to preserve
            T tolerance = T{1e-8};             ///< Convergence tolerance
            T sbx_eta = T{20};                 ///< SBX distribution index
            T polynomial_eta = T{20};          ///< Polynomial mutation distribution index
            GASelection selection = GASelection::Tournament;
            GACrossover crossover = GACrossover::SBX;
            GAMutation mutation = GAMutation::Gaussian;
            bool track_history = false; ///< Record best value each generation
        };

        Config config; ///< Configuration parameters

        /// Default constructor
        GeneticAlgorithm() = default;

        /// Constructor with custom config
        explicit GeneticAlgorithm(const Config &cfg) : config(cfg) {}

        /**
         * Optimize an objective function within bounds
         *
         * @tparam F Callable with signature T(const simd::Vector<T, Dynamic>&)
         * @param objective Function to minimize
         * @param lower_bounds Lower bounds for each dimension
         * @param upper_bounds Upper bounds for each dimension
         * @return GAResult with best solution and convergence info
         */
        template <typename F>
        GAResult<T> optimize(F &&objective, const simd::Vector<T, simd::Dynamic> &lower_bounds,
                             const simd::Vector<T, simd::Dynamic> &upper_bounds) {
            const std::size_t dim = lower_bounds.size();
            const std::size_t pop_size = config.population_size;

            // Validate inputs
            if (dim == 0) {
                return GAResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }
            if (upper_bounds.size() != dim) {
                return GAResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }
            if (pop_size < 4) {
                return GAResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }

            // Random number generator
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_real_distribution<T> uniform(T{0}, T{1});

            // Initialize population
            std::vector<simd::Vector<T, simd::Dynamic>> population(pop_size);
            std::vector<T> fitness(pop_size);
            std::size_t total_evals = 0;

            // Best solution tracking
            simd::Vector<T, simd::Dynamic> best_position(dim);
            T best_value = std::numeric_limits<T>::max();

            // Initialize population randomly
            for (std::size_t i = 0; i < pop_size; ++i) {
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

            // Offspring storage
            std::vector<simd::Vector<T, simd::Dynamic>> offspring(pop_size);
            std::vector<T> offspring_fitness(pop_size);

            // Ranking for selection
            std::vector<std::size_t> ranking(pop_size);

            // Main evolution loop
            std::size_t generation = 0;
            bool converged = false;

            for (; generation < config.max_generations; ++generation) {
                // Sort population by fitness for elitism and rank-based selection
                std::iota(ranking.begin(), ranking.end(), 0);
                std::sort(ranking.begin(), ranking.end(),
                          [&fitness](std::size_t a, std::size_t b) { return fitness[a] < fitness[b]; });

                // Elitism: copy best individuals directly to offspring
                for (std::size_t i = 0; i < config.elitism && i < pop_size; ++i) {
                    offspring[i] = population[ranking[i]];
                    offspring_fitness[i] = fitness[ranking[i]];
                }

                // Generate rest of offspring through selection, crossover, mutation
                for (std::size_t i = config.elitism; i < pop_size; i += 2) {
                    // Selection: choose two parents
                    std::size_t parent1 = select_parent(fitness, ranking, rng, uniform);
                    std::size_t parent2 = select_parent(fitness, ranking, rng, uniform);

                    // Ensure different parents
                    while (parent2 == parent1 && pop_size > 1) {
                        parent2 = select_parent(fitness, ranking, rng, uniform);
                    }

                    // Create offspring
                    simd::Vector<T, simd::Dynamic> child1(dim);
                    simd::Vector<T, simd::Dynamic> child2(dim);

                    // Crossover
                    if (uniform(rng) < config.crossover_prob) {
                        crossover_op(population[parent1], population[parent2], child1, child2, lower_bounds,
                                     upper_bounds, rng, uniform);
                    } else {
                        child1 = population[parent1];
                        child2 = population[parent2];
                    }

                    // Mutation
                    mutate(child1, lower_bounds, upper_bounds, rng, uniform);
                    mutate(child2, lower_bounds, upper_bounds, rng, uniform);

                    // Store offspring
                    offspring[i] = child1;
                    if (i + 1 < pop_size) {
                        offspring[i + 1] = child2;
                    }
                }

                // Evaluate offspring fitness (skip elites already evaluated)
                for (std::size_t i = config.elitism; i < pop_size; ++i) {
                    offspring_fitness[i] = objective(offspring[i]);
                    ++total_evals;

                    if (offspring_fitness[i] < best_value) {
                        best_value = offspring_fitness[i];
                        best_position = offspring[i];
                    }
                }

                // Replace population with offspring
                std::swap(population, offspring);
                std::swap(fitness, offspring_fitness);

                // Track history
                if (config.track_history) {
                    history.push_back(best_value);
                }

                // Update horizon buffer for convergence check
                horizon_buffer[horizon_idx] = best_value;
                horizon_idx = (horizon_idx + 1) % config.horizon_size;

                // Check convergence
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

            return GAResult<T>{best_position, best_value, generation, total_evals, converged, std::move(history)};
        }

        /**
         * Optimize starting from an initial point
         */
        template <typename F>
        GAResult<T> optimize(F &&objective, const simd::Vector<T, simd::Dynamic> &initial,
                             const simd::Vector<T, simd::Dynamic> &lower_bounds,
                             const simd::Vector<T, simd::Dynamic> &upper_bounds) {
            const std::size_t dim = initial.size();
            const std::size_t pop_size = config.population_size;

            if (dim == 0 || lower_bounds.size() != dim || upper_bounds.size() != dim) {
                return GAResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }
            if (pop_size < 4) {
                return GAResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }

            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_real_distribution<T> uniform(T{0}, T{1});

            std::vector<simd::Vector<T, simd::Dynamic>> population(pop_size);
            std::vector<T> fitness(pop_size);
            std::size_t total_evals = 0;

            simd::Vector<T, simd::Dynamic> best_position = initial;
            T best_value = objective(initial);
            ++total_evals;

            // First individual is the initial point
            population[0] = initial;
            fitness[0] = best_value;

            // Rest initialized randomly
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
                }
            }

            std::vector<T> history;
            if (config.track_history) {
                history.reserve(config.max_generations);
            }

            std::vector<T> horizon_buffer(config.horizon_size, best_value);
            std::size_t horizon_idx = 0;

            std::vector<simd::Vector<T, simd::Dynamic>> offspring(pop_size);
            std::vector<T> offspring_fitness(pop_size);
            std::vector<std::size_t> ranking(pop_size);

            std::size_t generation = 0;
            bool converged = false;

            for (; generation < config.max_generations; ++generation) {
                std::iota(ranking.begin(), ranking.end(), 0);
                std::sort(ranking.begin(), ranking.end(),
                          [&fitness](std::size_t a, std::size_t b) { return fitness[a] < fitness[b]; });

                for (std::size_t i = 0; i < config.elitism && i < pop_size; ++i) {
                    offspring[i] = population[ranking[i]];
                    offspring_fitness[i] = fitness[ranking[i]];
                }

                for (std::size_t i = config.elitism; i < pop_size; i += 2) {
                    std::size_t parent1 = select_parent(fitness, ranking, rng, uniform);
                    std::size_t parent2 = select_parent(fitness, ranking, rng, uniform);

                    while (parent2 == parent1 && pop_size > 1) {
                        parent2 = select_parent(fitness, ranking, rng, uniform);
                    }

                    simd::Vector<T, simd::Dynamic> child1(dim);
                    simd::Vector<T, simd::Dynamic> child2(dim);

                    if (uniform(rng) < config.crossover_prob) {
                        crossover_op(population[parent1], population[parent2], child1, child2, lower_bounds,
                                     upper_bounds, rng, uniform);
                    } else {
                        child1 = population[parent1];
                        child2 = population[parent2];
                    }

                    mutate(child1, lower_bounds, upper_bounds, rng, uniform);
                    mutate(child2, lower_bounds, upper_bounds, rng, uniform);

                    offspring[i] = child1;
                    if (i + 1 < pop_size) {
                        offspring[i + 1] = child2;
                    }
                }

                for (std::size_t i = config.elitism; i < pop_size; ++i) {
                    offspring_fitness[i] = objective(offspring[i]);
                    ++total_evals;

                    if (offspring_fitness[i] < best_value) {
                        best_value = offspring_fitness[i];
                        best_position = offspring[i];
                    }
                }

                std::swap(population, offspring);
                std::swap(fitness, offspring_fitness);

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

            return GAResult<T>{best_position, best_value, generation, total_evals, converged, std::move(history)};
        }

      private:
        /**
         * Select a parent using the configured selection strategy
         */
        template <typename RNG, typename Dist>
        std::size_t select_parent(const std::vector<T> &fitness, const std::vector<std::size_t> &ranking, RNG &rng,
                                  Dist &uniform) {
            const std::size_t pop_size = fitness.size();

            switch (config.selection) {
            case GASelection::Tournament: {
                // Tournament selection
                std::size_t best = static_cast<std::size_t>(uniform(rng) * pop_size) % pop_size;
                for (std::size_t t = 1; t < config.tournament_size; ++t) {
                    std::size_t candidate = static_cast<std::size_t>(uniform(rng) * pop_size) % pop_size;
                    if (fitness[candidate] < fitness[best]) {
                        best = candidate;
                    }
                }
                return best;
            }

            case GASelection::RouletteWheel: {
                // Fitness-proportionate selection (for minimization, invert fitness)
                T max_fitness = *std::max_element(fitness.begin(), fitness.end());
                T total = T{0};
                for (std::size_t i = 0; i < pop_size; ++i) {
                    total += max_fitness - fitness[i] + T{1}; // +1 to avoid zero
                }

                T r = uniform(rng) * total;
                T cumsum = T{0};
                for (std::size_t i = 0; i < pop_size; ++i) {
                    cumsum += max_fitness - fitness[i] + T{1};
                    if (cumsum >= r) {
                        return i;
                    }
                }
                return pop_size - 1;
            }

            case GASelection::Rank: {
                // Rank-based selection (linear ranking)
                // ranking[0] is best, ranking[pop_size-1] is worst
                // Probability proportional to rank position
                T total = static_cast<T>(pop_size * (pop_size + 1)) / T{2};
                T r = uniform(rng) * total;
                T cumsum = T{0};
                for (std::size_t i = 0; i < pop_size; ++i) {
                    cumsum += static_cast<T>(pop_size - i); // Best gets highest weight
                    if (cumsum >= r) {
                        return ranking[i];
                    }
                }
                return ranking[0];
            }
            }

            return 0;
        }

        /**
         * Crossover operation
         */
        template <typename RNG, typename Dist>
        void crossover_op(const simd::Vector<T, simd::Dynamic> &parent1, const simd::Vector<T, simd::Dynamic> &parent2,
                          simd::Vector<T, simd::Dynamic> &child1, simd::Vector<T, simd::Dynamic> &child2,
                          const simd::Vector<T, simd::Dynamic> &lower_bounds,
                          const simd::Vector<T, simd::Dynamic> &upper_bounds, RNG &rng, Dist &uniform) {
            const std::size_t dim = parent1.size();

            switch (config.crossover) {
            case GACrossover::SBX: {
                // Simulated Binary Crossover
                for (std::size_t d = 0; d < dim; ++d) {
                    if (uniform(rng) < T{0.5}) {
                        // Apply SBX to this gene
                        T y1 = std::min(parent1[d], parent2[d]);
                        T y2 = std::max(parent1[d], parent2[d]);

                        if (std::abs(y2 - y1) > T{1e-14}) {
                            T beta;
                            T u = uniform(rng);

                            // Compute beta
                            T eta = config.sbx_eta;
                            if (u <= T{0.5}) {
                                beta = std::pow(T{2} * u, T{1} / (eta + T{1}));
                            } else {
                                beta = std::pow(T{1} / (T{2} * (T{1} - u)), T{1} / (eta + T{1}));
                            }

                            child1[d] = T{0.5} * ((y1 + y2) - beta * (y2 - y1));
                            child2[d] = T{0.5} * ((y1 + y2) + beta * (y2 - y1));
                        } else {
                            child1[d] = parent1[d];
                            child2[d] = parent2[d];
                        }
                    } else {
                        child1[d] = parent1[d];
                        child2[d] = parent2[d];
                    }

                    // Clamp to bounds
                    child1[d] = std::clamp(child1[d], lower_bounds[d], upper_bounds[d]);
                    child2[d] = std::clamp(child2[d], lower_bounds[d], upper_bounds[d]);
                }
                break;
            }

            case GACrossover::Uniform: {
                // Uniform crossover
                for (std::size_t d = 0; d < dim; ++d) {
                    if (uniform(rng) < T{0.5}) {
                        child1[d] = parent1[d];
                        child2[d] = parent2[d];
                    } else {
                        child1[d] = parent2[d];
                        child2[d] = parent1[d];
                    }
                }
                break;
            }

            case GACrossover::SinglePoint: {
                // Single-point crossover
                std::size_t point = static_cast<std::size_t>(uniform(rng) * dim) % dim;
                for (std::size_t d = 0; d < dim; ++d) {
                    if (d < point) {
                        child1[d] = parent1[d];
                        child2[d] = parent2[d];
                    } else {
                        child1[d] = parent2[d];
                        child2[d] = parent1[d];
                    }
                }
                break;
            }
            }
        }

        /**
         * Mutation operation
         */
        template <typename RNG, typename Dist>
        void mutate(simd::Vector<T, simd::Dynamic> &individual, const simd::Vector<T, simd::Dynamic> &lower_bounds,
                    const simd::Vector<T, simd::Dynamic> &upper_bounds, RNG &rng, Dist &uniform) {
            const std::size_t dim = individual.size();
            std::normal_distribution<T> normal(T{0}, T{1});

            for (std::size_t d = 0; d < dim; ++d) {
                if (uniform(rng) < config.mutation_prob) {
                    T range = upper_bounds[d] - lower_bounds[d];

                    switch (config.mutation) {
                    case GAMutation::Gaussian: {
                        // Gaussian mutation
                        T delta = normal(rng) * config.mutation_strength * range;
                        individual[d] += delta;
                        break;
                    }

                    case GAMutation::Polynomial: {
                        // Polynomial mutation
                        T u = uniform(rng);
                        T eta = config.polynomial_eta;
                        T delta;

                        if (u < T{0.5}) {
                            delta = std::pow(T{2} * u, T{1} / (eta + T{1})) - T{1};
                        } else {
                            delta = T{1} - std::pow(T{2} * (T{1} - u), T{1} / (eta + T{1}));
                        }

                        individual[d] += delta * range;
                        break;
                    }
                    }

                    // Clamp to bounds
                    individual[d] = std::clamp(individual[d], lower_bounds[d], upper_bounds[d]);
                }
            }
        }
    };

} // namespace optinum::meta
