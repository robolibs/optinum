#pragma once

/**
 * Particle Swarm Optimization (PSO)
 *
 * Population-based stochastic optimization inspired by social behavior of birds/fish.
 * Each particle maintains position, velocity, and personal best. The swarm collectively
 * converges toward the global optimum through information sharing.
 *
 * Algorithm:
 *   1. Initialize particles with random positions and velocities within bounds
 *   2. For each iteration:
 *      - Evaluate fitness for all particles
 *      - Update personal best (pbest) if current position is better
 *      - Update global best (gbest) across all particles
 *      - Update velocities: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
 *      - Update positions: x = x + v
 *      - Clamp positions to bounds
 *   3. Return global best position and value
 *
 * References:
 *   - Kennedy & Eberhart (1995) "Particle Swarm Optimization"
 *   - ensmallen PSO: xtra/ensmallen/include/ensmallen_bits/pso/
 *
 * @file pso.hpp
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <datapod/matrix.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/bridge.hpp>

namespace optinum::meta {

    namespace dp = ::datapod;

    /**
     * Result of PSO optimization
     */
    template <typename T> struct PSOResult {
        dp::mat::vector<T, dp::mat::Dynamic> best_position; ///< Best solution found
        T best_value;                                       ///< Objective value at best position
        std::size_t iterations;                             ///< Number of iterations performed
        bool converged;                                     ///< Whether convergence criteria met
        std::vector<T> history;                             ///< Best value per iteration (for analysis)
    };

    /**
     * Particle Swarm Optimization
     *
     * Usage:
     * @code
     * PSO<double> pso;
     * pso.config.population_size = 50;
     * pso.config.max_iterations = 1000;
     *
     * auto objective = [](const auto& x) { return x[0]*x[0] + x[1]*x[1]; };
     *
     * dp::mat::vector<double, dp::mat::Dynamic> lower{-5.0, -5.0};
     * dp::mat::vector<double, dp::mat::Dynamic> upper{5.0, 5.0};
     *
     * auto result = pso.optimize(objective, lower, upper);
     * @endcode
     *
     * @tparam T Scalar type (float or double)
     */
    template <typename T = double> class PSO {
      public:
        /**
         * PSO configuration parameters
         */
        struct Config {
            std::size_t population_size = 50;  ///< Number of particles in swarm
            std::size_t max_iterations = 1000; ///< Maximum iterations
            std::size_t horizon_size = 50;     ///< Window for convergence check
            T inertia_weight = T{0.7};         ///< Velocity inertia (w)
            T cognitive_coeff = T{1.5};        ///< Personal best attraction (c1)
            T social_coeff = T{1.5};           ///< Global best attraction (c2)
            T tolerance = T{1e-8};             ///< Convergence tolerance
            T velocity_clamp = T{0.5};         ///< Max velocity as fraction of range
            bool track_history = false;        ///< Record best value each iteration
        };

        Config config; ///< Configuration parameters

        /// Default constructor
        PSO() = default;

        /// Constructor with custom config
        explicit PSO(const Config &cfg) : config(cfg) {}

        /**
         * Optimize an objective function within bounds
         *
         * @tparam F Callable with signature T(const simd::Vector<T, Dynamic>&)
         * @param objective Function to minimize
         * @param lower_bounds Lower bounds for each dimension
         * @param upper_bounds Upper bounds for each dimension
         * @return PSOResult with best solution and convergence info
         */
        template <typename F>
        PSOResult<T> optimize(F &&objective, const dp::mat::vector<T, dp::mat::Dynamic> &lower_bounds,
                              const dp::mat::vector<T, dp::mat::Dynamic> &upper_bounds) {
            const std::size_t dim = lower_bounds.size();
            const std::size_t n_particles = config.population_size;

            // Validate inputs
            if (dim == 0) {
                return PSOResult<T>{{}, std::numeric_limits<T>::max(), 0, false, {}};
            }
            if (upper_bounds.size() != dim) {
                return PSOResult<T>{{}, std::numeric_limits<T>::max(), 0, false, {}};
            }
            if (config.max_iterations < config.horizon_size) {
                return PSOResult<T>{{}, std::numeric_limits<T>::max(), 0, false, {}};
            }

            // Random number generator
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_real_distribution<T> uniform(T{0}, T{1});

            // Compute velocity limits based on range
            dp::mat::vector<T, dp::mat::Dynamic> velocity_max(dim);
            for (std::size_t d = 0; d < dim; ++d) {
                velocity_max[d] = config.velocity_clamp * (upper_bounds[d] - lower_bounds[d]);
            }

            // Initialize particles: positions, velocities, personal bests
            // Using std::vector of simd::Vector for population (each particle is a vector)
            std::vector<dp::mat::vector<T, dp::mat::Dynamic>> positions(n_particles);
            std::vector<dp::mat::vector<T, dp::mat::Dynamic>> velocities(n_particles);
            std::vector<dp::mat::vector<T, dp::mat::Dynamic>> pbest_positions(n_particles);
            std::vector<T> pbest_values(n_particles);
            std::vector<T> current_values(n_particles);

            // Initialize each particle
            for (std::size_t i = 0; i < n_particles; ++i) {
                positions[i] = dp::mat::vector<T, dp::mat::Dynamic>(dim);
                velocities[i] = dp::mat::vector<T, dp::mat::Dynamic>(dim);
                pbest_positions[i] = dp::mat::vector<T, dp::mat::Dynamic>(dim);

                for (std::size_t d = 0; d < dim; ++d) {
                    // Random position within bounds
                    T r = uniform(rng);
                    positions[i][d] = lower_bounds[d] + r * (upper_bounds[d] - lower_bounds[d]);

                    // Random initial velocity (small)
                    T rv = uniform(rng) * T{2} - T{1}; // [-1, 1]
                    velocities[i][d] = rv * velocity_max[d] * T{0.1};

                    // Personal best starts at initial position
                    pbest_positions[i][d] = positions[i][d];
                }

                // Evaluate initial fitness
                pbest_values[i] = objective(positions[i]);
                current_values[i] = pbest_values[i];
            }

            // Find initial global best
            std::size_t gbest_idx = 0;
            T gbest_value = pbest_values[0];
            for (std::size_t i = 1; i < n_particles; ++i) {
                if (pbest_values[i] < gbest_value) {
                    gbest_value = pbest_values[i];
                    gbest_idx = i;
                }
            }
            dp::mat::vector<T, dp::mat::Dynamic> gbest_position = pbest_positions[gbest_idx];

            // History tracking
            std::vector<T> history;
            if (config.track_history) {
                history.reserve(config.max_iterations);
            }

            // Convergence tracking: sliding window of best values
            std::vector<T> horizon_buffer(config.horizon_size, gbest_value);
            std::size_t horizon_idx = 0;

            // Pre-allocate temporary vectors for SIMD operations
            dp::mat::vector<T, dp::mat::Dynamic> r1_vec(dim);
            dp::mat::vector<T, dp::mat::Dynamic> r2_vec(dim);
            dp::mat::vector<T, dp::mat::Dynamic> temp1(dim);
            dp::mat::vector<T, dp::mat::Dynamic> temp2(dim);

            // Main optimization loop
            std::size_t iteration = 0;
            bool converged = false;

            for (; iteration < config.max_iterations; ++iteration) {
                // Update each particle
                for (std::size_t i = 0; i < n_particles; ++i) {
                    // Generate random coefficient vectors for SIMD processing
                    for (std::size_t d = 0; d < dim; ++d) {
                        r1_vec[d] = uniform(rng);
                        r2_vec[d] = uniform(rng);
                    }

                    // SIMD-optimized velocity update: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
                    // Step 1: Scale velocity by inertia weight: v *= w
                    simd::backend::mul_scalar_runtime<T>(velocities[i].data(), velocities[i].data(),
                                                         config.inertia_weight, dim);

                    // Step 2: Compute cognitive component: temp1 = pbest - x
                    simd::backend::sub_runtime<T>(temp1.data(), pbest_positions[i].data(), positions[i].data(), dim);
                    // temp1 *= c1 * r1 (element-wise)
                    simd::backend::mul_runtime<T>(temp1.data(), temp1.data(), r1_vec.data(), dim);
                    simd::backend::mul_scalar_runtime<T>(temp1.data(), temp1.data(), config.cognitive_coeff, dim);

                    // Step 3: Compute social component: temp2 = gbest - x
                    simd::backend::sub_runtime<T>(temp2.data(), gbest_position.data(), positions[i].data(), dim);
                    // temp2 *= c2 * r2 (element-wise)
                    simd::backend::mul_runtime<T>(temp2.data(), temp2.data(), r2_vec.data(), dim);
                    simd::backend::mul_scalar_runtime<T>(temp2.data(), temp2.data(), config.social_coeff, dim);

                    // Step 4: v += temp1 + temp2
                    simd::backend::add_runtime<T>(velocities[i].data(), velocities[i].data(), temp1.data(), dim);
                    simd::backend::add_runtime<T>(velocities[i].data(), velocities[i].data(), temp2.data(), dim);

                    // Clamp velocity and update position (scalar loop for clamping)
                    for (std::size_t d = 0; d < dim; ++d) {
                        velocities[i][d] = std::clamp(velocities[i][d], -velocity_max[d], velocity_max[d]);
                        positions[i][d] += velocities[i][d];
                        positions[i][d] = std::clamp(positions[i][d], lower_bounds[d], upper_bounds[d]);
                    }

                    // Evaluate fitness
                    current_values[i] = objective(positions[i]);

                    // Update personal best
                    if (current_values[i] < pbest_values[i]) {
                        pbest_values[i] = current_values[i];
                        simd::backend::copy_runtime<T>(pbest_positions[i].data(), positions[i].data(), dim);

                        // Update global best
                        if (current_values[i] < gbest_value) {
                            gbest_value = current_values[i];
                            simd::backend::copy_runtime<T>(gbest_position.data(), positions[i].data(), dim);
                        }
                    }
                }

                // Track history
                if (config.track_history) {
                    history.push_back(gbest_value);
                }

                // Update horizon buffer for convergence check
                horizon_buffer[horizon_idx] = gbest_value;
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
            }

            return PSOResult<T>{gbest_position, gbest_value, iteration, converged, std::move(history)};
        }

        /**
         * Optimize starting from an initial point (uses point to infer dimension)
         *
         * @tparam F Callable with signature T(const simd::Vector<T, Dynamic>&)
         * @param objective Function to minimize
         * @param initial Initial point (also determines dimension)
         * @param lower_bounds Lower bounds for each dimension
         * @param upper_bounds Upper bounds for each dimension
         * @return PSOResult with best solution and convergence info
         */
        template <typename F>
        PSOResult<T> optimize(F &&objective, const dp::mat::vector<T, dp::mat::Dynamic> &initial,
                              const dp::mat::vector<T, dp::mat::Dynamic> &lower_bounds,
                              const dp::mat::vector<T, dp::mat::Dynamic> &upper_bounds) {
            // Use initial point as one of the particles by seeding the RNG
            // For now, just delegate to the bounds-only version
            // The initial point could be used to bias the initial population
            (void)initial; // TODO: Use initial point to seed one particle
            return optimize(std::forward<F>(objective), lower_bounds, upper_bounds);
        }
    };

} // namespace optinum::meta
