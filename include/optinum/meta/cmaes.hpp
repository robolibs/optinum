#pragma once

/**
 * CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
 *
 * A powerful derivative-free optimization algorithm for continuous domains.
 * CMA-ES adapts the covariance matrix of a multivariate normal distribution
 * to learn the structure of the objective function landscape.
 *
 * Algorithm:
 *   1. Initialize mean m, step-size sigma, covariance C = I
 *   2. For each generation:
 *      - Sample lambda offspring: x_k = m + sigma * N(0, C)
 *      - Evaluate and rank offspring by fitness
 *      - Update mean: m = weighted sum of mu best offspring
 *      - Update evolution paths p_c, p_sigma
 *      - Update covariance: C = (1-c1-cmu)*C + c1*p_c*p_c^T + cmu*sum(w_i*y_i*y_i^T)
 *      - Update step-size via cumulative path length
 *   3. Return best solution
 *
 * Key features:
 *   - Auto-computed hyperparameters based on dimension
 *   - Invariant to monotonic transformations of objective
 *   - Handles ill-conditioned problems well
 *   - Robust to local optima (population-based)
 *
 * References:
 *   - Hansen & Ostermeier (2001) "Completely Derandomized Self-Adaptation
 *     in Evolution Strategies"
 *   - Hansen (2016) "The CMA Evolution Strategy: A Tutorial"
 *   - ensmallen CMA-ES: xtra/ensmallen/include/ensmallen_bits/cmaes/
 *
 * @file cmaes.hpp
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <datapod/matrix.hpp>
#include <optinum/simd/bridge.hpp>

namespace optinum::meta {

    namespace dp = ::datapod;

    /**
     * Result of CMA-ES optimization
     */
    template <typename T> struct CMAESResult {
        dp::mat::vector<T, dp::mat::Dynamic> best_position; ///< Best solution found
        T best_value;                                       ///< Objective value at best position
        std::size_t generations;                            ///< Number of generations performed
        std::size_t function_evaluations;                   ///< Total function evaluations
        bool converged;                                     ///< Whether convergence criteria met
        std::vector<T> history;                             ///< Best value per generation
    };

    /**
     * CMA-ES Optimizer
     *
     * Covariance Matrix Adaptation Evolution Strategy for derivative-free
     * optimization of continuous functions. Particularly effective for
     * non-separable, ill-conditioned problems.
     *
     * Usage:
     * @code
     * CMAES<double> cmaes;
     * cmaes.config.max_generations = 500;
     * cmaes.config.sigma0 = 0.5;
     *
     * auto objective = [](const auto& x) { return x[0]*x[0] + x[1]*x[1]; };
     *
     * simd::Vector<double, simd::Dynamic> lower{-5.0, -5.0};
     * simd::Vector<double, simd::Dynamic> upper{5.0, 5.0};
     *
     * auto result = cmaes.optimize(objective, lower, upper);
     * @endcode
     *
     * @tparam T Scalar type (float or double)
     */
    template <typename T = double> class CMAES {
      public:
        /**
         * CMA-ES configuration parameters
         *
         * Most parameters are auto-computed from dimension if left at 0.
         */
        struct Config {
            std::size_t population_size = 0;    ///< Lambda (0 = auto: 4 + floor(3*ln(n)))
            std::size_t max_generations = 1000; ///< Maximum generations
            T sigma0 = T{0.3};                  ///< Initial step-size (fraction of range)
            T tolerance = T{1e-8};              ///< Convergence tolerance
            std::size_t patience = 0;           ///< Generations without improvement (0 = auto)
            bool track_history = false;         ///< Record best value each generation
        };

        Config config; ///< Configuration parameters

        /// Default constructor
        CMAES() = default;

        /// Constructor with custom config
        explicit CMAES(const Config &cfg) : config(cfg) {}

        /**
         * Optimize an objective function within bounds
         *
         * @tparam F Callable with signature T(const simd::Vector<T, Dynamic>&)
         * @param objective Function to minimize
         * @param lower_bounds Lower bounds for each dimension
         * @param upper_bounds Upper bounds for each dimension
         * @return CMAESResult with best solution and convergence info
         */
        template <typename F>
        CMAESResult<T> optimize(F &&objective, const dp::mat::vector<T, dp::mat::Dynamic> &lower_bounds,
                                const dp::mat::vector<T, dp::mat::Dynamic> &upper_bounds) {
            const std::size_t n = lower_bounds.size();

            // Validate inputs
            if (n == 0) {
                return CMAESResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }
            if (upper_bounds.size() != n) {
                return CMAESResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }

            // Initialize mean at center of bounds
            dp::mat::vector<T, dp::mat::Dynamic> mean(n);
            for (std::size_t i = 0; i < n; ++i) {
                mean[i] = (lower_bounds[i] + upper_bounds[i]) / T{2};
            }

            return optimize_impl(std::forward<F>(objective), mean, lower_bounds, upper_bounds);
        }

        /**
         * Optimize starting from an initial point
         *
         * @tparam F Callable with signature T(const simd::Vector<T, Dynamic>&)
         * @param objective Function to minimize
         * @param initial Initial mean position
         * @param lower_bounds Lower bounds for each dimension
         * @param upper_bounds Upper bounds for each dimension
         * @return CMAESResult with best solution and convergence info
         */
        template <typename F>
        CMAESResult<T> optimize(F &&objective, const dp::mat::vector<T, dp::mat::Dynamic> &initial,
                                const dp::mat::vector<T, dp::mat::Dynamic> &lower_bounds,
                                const dp::mat::vector<T, dp::mat::Dynamic> &upper_bounds) {
            const std::size_t n = initial.size();

            // Validate inputs
            if (n == 0) {
                return CMAESResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }
            if (lower_bounds.size() != n || upper_bounds.size() != n) {
                return CMAESResult<T>{{}, std::numeric_limits<T>::max(), 0, 0, false, {}};
            }

            return optimize_impl(std::forward<F>(objective), initial, lower_bounds, upper_bounds);
        }

      private:
        /**
         * Core CMA-ES optimization implementation
         */
        template <typename F>
        CMAESResult<T> optimize_impl(F &&objective, const dp::mat::vector<T, dp::mat::Dynamic> &initial_mean,
                                     const dp::mat::vector<T, dp::mat::Dynamic> &lower_bounds,
                                     const dp::mat::vector<T, dp::mat::Dynamic> &upper_bounds) {
            const std::size_t n = initial_mean.size();

            // Random number generator
            std::random_device rd;
            std::mt19937 rng(rd());
            std::normal_distribution<T> normal(T{0}, T{1});

            // ============================================================
            // Strategy parameter setting (auto-computed from dimension n)
            // ============================================================

            // Population size: lambda = 4 + floor(3 * ln(n))
            const std::size_t lambda =
                (config.population_size > 0)
                    ? config.population_size
                    : static_cast<std::size_t>(4 + std::floor(3.0 * std::log(static_cast<double>(n))));

            // Number of parents/selected points
            const std::size_t mu = lambda / 2;

            // Recombination weights: log-linear decreasing
            std::vector<T> weights(mu);
            T weight_sum = T{0};
            for (std::size_t i = 0; i < mu; ++i) {
                weights[i] = std::log(static_cast<T>(mu) + T{0.5}) - std::log(static_cast<T>(i + 1));
                weight_sum += weights[i];
            }
            // Normalize weights
            for (std::size_t i = 0; i < mu; ++i) {
                weights[i] /= weight_sum;
            }

            // Variance-effectiveness of sum w_i x_i
            T mu_eff = T{0};
            for (std::size_t i = 0; i < mu; ++i) {
                mu_eff += weights[i] * weights[i];
            }
            mu_eff = T{1} / mu_eff;

            // Step-size control parameters
            const T cs = (mu_eff + T{2}) / (static_cast<T>(n) + mu_eff + T{5});
            const T ds =
                T{1} + cs + T{2} * std::max(std::sqrt((mu_eff - T{1}) / (static_cast<T>(n) + T{1})) - T{1}, T{0});

            // Expected length of N(0,I) vector
            const T chi_n = std::sqrt(static_cast<T>(n)) * (T{1} - T{1} / (T{4} * static_cast<T>(n)) +
                                                            T{1} / (T{21} * static_cast<T>(n) * static_cast<T>(n)));

            // Covariance matrix adaptation parameters
            const T cc =
                (T{4} + mu_eff / static_cast<T>(n)) / (static_cast<T>(n) + T{4} + T{2} * mu_eff / static_cast<T>(n));
            const T c1 = T{2} / ((static_cast<T>(n) + T{1.3}) * (static_cast<T>(n) + T{1.3}) + mu_eff);
            const T cmu = std::min(T{1} - c1, T{2} * (mu_eff - T{2} + T{1} / mu_eff) /
                                                  ((static_cast<T>(n) + T{2}) * (static_cast<T>(n) + T{2}) + mu_eff));

            // ============================================================
            // Initialize dynamic state
            // ============================================================

            // Mean of distribution
            dp::mat::vector<T, dp::mat::Dynamic> m(initial_mean.size());
            for (std::size_t i = 0; i < initial_mean.size(); ++i) {
                m[i] = initial_mean[i];
            }

            // Step-size (initial based on range)
            T avg_range = T{0};
            for (std::size_t i = 0; i < n; ++i) {
                avg_range += upper_bounds[i] - lower_bounds[i];
            }
            avg_range /= static_cast<T>(n);
            T sigma = config.sigma0 * avg_range;

            // Covariance matrix C = I (stored as full matrix for simplicity)
            // We store C as a flat vector in row-major order
            std::vector<T> C(n * n, T{0});
            for (std::size_t i = 0; i < n; ++i) {
                C[i * n + i] = T{1}; // Identity matrix
            }

            // Evolution paths
            dp::mat::vector<T, dp::mat::Dynamic> p_sigma(n); // For step-size control
            dp::mat::vector<T, dp::mat::Dynamic> p_c(n);     // For covariance adaptation
            for (std::size_t i = 0; i < n; ++i) {
                p_sigma[i] = T{0};
                p_c[i] = T{0};
            }

            // Eigendecomposition of C: C = B * D^2 * B^T
            // B = eigenvectors, D = sqrt(eigenvalues)
            std::vector<T> B(n * n); // Eigenvectors (columns)
            std::vector<T> D(n);     // sqrt(eigenvalues)
            for (std::size_t i = 0; i < n; ++i) {
                D[i] = T{1};
                for (std::size_t j = 0; j < n; ++j) {
                    B[i * n + j] = (i == j) ? T{1} : T{0};
                }
            }

            // Best solution tracking
            dp::mat::vector<T, dp::mat::Dynamic> best_position = m;
            T best_value = objective(m);
            std::size_t total_evals = 1;

            // History tracking
            std::vector<T> history;
            if (config.track_history) {
                history.reserve(config.max_generations);
            }

            // Patience for early stopping
            const std::size_t patience =
                (config.patience > 0) ? config.patience : (10 + static_cast<std::size_t>(30 * n / lambda));
            std::size_t no_improvement_count = 0;
            T last_best = best_value;

            // Population storage
            std::vector<dp::mat::vector<T, dp::mat::Dynamic>> population(lambda);
            std::vector<T> fitness(lambda);
            std::vector<std::size_t> ranking(lambda);

            // Temporary vectors for offspring generation
            std::vector<T> z(n); // N(0,I) sample
            std::vector<T> y(n); // B*D*z

            // ============================================================
            // Main optimization loop
            // ============================================================

            std::size_t generation = 0;
            bool converged = false;

            for (; generation < config.max_generations; ++generation) {
                // Generate and evaluate lambda offspring
                for (std::size_t k = 0; k < lambda; ++k) {
                    population[k] = dp::mat::vector<T, dp::mat::Dynamic>(n);

                    // Sample z ~ N(0, I)
                    for (std::size_t i = 0; i < n; ++i) {
                        z[i] = normal(rng);
                    }

                    // Compute y = B * D * z
                    for (std::size_t i = 0; i < n; ++i) {
                        y[i] = T{0};
                        for (std::size_t j = 0; j < n; ++j) {
                            y[i] += B[i * n + j] * D[j] * z[j];
                        }
                    }

                    // x_k = m + sigma * y
                    for (std::size_t i = 0; i < n; ++i) {
                        T val = m[i] + sigma * y[i];
                        // Clamp to bounds
                        population[k][i] = std::clamp(val, lower_bounds[i], upper_bounds[i]);
                    }

                    // Evaluate fitness
                    fitness[k] = objective(population[k]);
                    ++total_evals;

                    // Update best
                    if (fitness[k] < best_value) {
                        best_value = fitness[k];
                        best_position = population[k];
                    }
                }

                // Sort population by fitness (ascending - minimization)
                std::iota(ranking.begin(), ranking.end(), 0);
                std::sort(ranking.begin(), ranking.end(),
                          [&fitness](std::size_t a, std::size_t b) { return fitness[a] < fitness[b]; });

                // ============================================================
                // Update mean: m = sum(w_i * x_i) for i = 1..mu
                // ============================================================

                dp::mat::vector<T, dp::mat::Dynamic> m_old = m;
                for (std::size_t i = 0; i < n; ++i) {
                    m[i] = T{0};
                    for (std::size_t j = 0; j < mu; ++j) {
                        m[i] += weights[j] * population[ranking[j]][i];
                    }
                }

                // ============================================================
                // Update evolution paths
                // ============================================================

                // Compute C^(-1/2) * (m - m_old) / sigma
                // C^(-1/2) = B * D^(-1) * B^T
                std::vector<T> diff(n);
                for (std::size_t i = 0; i < n; ++i) {
                    diff[i] = (m[i] - m_old[i]) / sigma;
                }

                // y_w = B^T * diff
                std::vector<T> y_w(n);
                for (std::size_t i = 0; i < n; ++i) {
                    y_w[i] = T{0};
                    for (std::size_t j = 0; j < n; ++j) {
                        y_w[i] += B[j * n + i] * diff[j]; // B^T
                    }
                }

                // C^(-1/2) * diff = B * D^(-1) * y_w
                std::vector<T> c_inv_sqrt_diff(n);
                for (std::size_t i = 0; i < n; ++i) {
                    c_inv_sqrt_diff[i] = T{0};
                    for (std::size_t j = 0; j < n; ++j) {
                        T d_inv = (D[j] > T{1e-14}) ? (T{1} / D[j]) : T{0};
                        c_inv_sqrt_diff[i] += B[i * n + j] * d_inv * y_w[j];
                    }
                }

                // Update p_sigma (conjugate evolution path)
                T sqrt_cs = std::sqrt(cs * (T{2} - cs) * mu_eff);
                for (std::size_t i = 0; i < n; ++i) {
                    p_sigma[i] = (T{1} - cs) * p_sigma[i] + sqrt_cs * c_inv_sqrt_diff[i];
                }

                // Compute ||p_sigma||
                T p_sigma_norm = T{0};
                for (std::size_t i = 0; i < n; ++i) {
                    p_sigma_norm += p_sigma[i] * p_sigma[i];
                }
                p_sigma_norm = std::sqrt(p_sigma_norm);

                // Heaviside function for stalling
                T h_sigma_threshold = (T{1.4} + T{2} / (static_cast<T>(n) + T{1})) * chi_n;
                T expected_p_sigma_norm = std::sqrt(T{1} - std::pow(T{1} - cs, T{2} * static_cast<T>(generation + 1)));
                bool h_sigma = (p_sigma_norm / expected_p_sigma_norm) < h_sigma_threshold;

                // Update p_c (evolution path for covariance)
                T sqrt_cc = std::sqrt(cc * (T{2} - cc) * mu_eff);
                T h_sigma_val = h_sigma ? T{1} : T{0};
                for (std::size_t i = 0; i < n; ++i) {
                    p_c[i] = (T{1} - cc) * p_c[i] + h_sigma_val * sqrt_cc * diff[i];
                }

                // ============================================================
                // Update covariance matrix C
                // ============================================================

                // C = (1 - c1 - cmu) * C + c1 * (p_c * p_c^T + delta_h * C) + cmu * sum(w_i * y_i * y_i^T)
                T delta_h = h_sigma ? T{0} : (cc * (T{2} - cc));

                // Compute weighted sum of outer products
                std::vector<T> C_mu(n * n, T{0});
                for (std::size_t k = 0; k < mu; ++k) {
                    // y_k = (x_k - m_old) / sigma
                    for (std::size_t i = 0; i < n; ++i) {
                        T y_ki = (population[ranking[k]][i] - m_old[i]) / sigma;
                        for (std::size_t j = 0; j < n; ++j) {
                            T y_kj = (population[ranking[k]][j] - m_old[j]) / sigma;
                            C_mu[i * n + j] += weights[k] * y_ki * y_kj;
                        }
                    }
                }

                // Update C
                for (std::size_t i = 0; i < n; ++i) {
                    for (std::size_t j = 0; j < n; ++j) {
                        C[i * n + j] = (T{1} - c1 - cmu + delta_h * c1) * C[i * n + j] + c1 * p_c[i] * p_c[j] +
                                       cmu * C_mu[i * n + j];
                    }
                }

                // ============================================================
                // Update step-size sigma
                // ============================================================

                sigma = sigma * std::exp((cs / ds) * (p_sigma_norm / chi_n - T{1}));

                // Check for step-size divergence
                if (std::isnan(sigma) || sigma > T{1e14}) {
                    // Step size diverged - terminate
                    break;
                }

                // ============================================================
                // Eigendecomposition of C for next generation
                // ============================================================

                // Simple power iteration / Jacobi-like update for eigendecomposition
                // For robustness, we use a simplified approach:
                // Ensure C is symmetric and positive semi-definite
                eigendecompose(C, B, D, n);

                // ============================================================
                // Track history and check convergence
                // ============================================================

                if (config.track_history) {
                    history.push_back(best_value);
                }

                // Check for improvement
                if (best_value < last_best - config.tolerance) {
                    no_improvement_count = 0;
                    last_best = best_value;
                } else {
                    ++no_improvement_count;
                }

                // Convergence check: no improvement for patience generations
                if (no_improvement_count >= patience) {
                    converged = true;
                    break;
                }

                // Also check if sigma is very small (converged)
                if (sigma < config.tolerance * avg_range) {
                    converged = true;
                    break;
                }
            }

            return CMAESResult<T>{best_position, best_value, generation, total_evals, converged, std::move(history)};
        }

        /**
         * Simple eigendecomposition for symmetric positive semi-definite matrix
         * Uses Jacobi iteration for small matrices
         */
        void eigendecompose(std::vector<T> &C, std::vector<T> &B, std::vector<T> &D, std::size_t n) {
            const std::size_t max_iter = 100;
            const T tol = T{1e-12};

            // Initialize B to identity
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    B[i * n + j] = (i == j) ? T{1} : T{0};
                }
            }

            // Work on a copy of C
            std::vector<T> A = C;

            // Jacobi iteration
            for (std::size_t iter = 0; iter < max_iter; ++iter) {
                // Find largest off-diagonal element
                T max_off = T{0};
                std::size_t p = 0, q = 1;
                for (std::size_t i = 0; i < n; ++i) {
                    for (std::size_t j = i + 1; j < n; ++j) {
                        T val = std::abs(A[i * n + j]);
                        if (val > max_off) {
                            max_off = val;
                            p = i;
                            q = j;
                        }
                    }
                }

                if (max_off < tol) {
                    break; // Converged
                }

                // Compute rotation angle
                T app = A[p * n + p];
                T aqq = A[q * n + q];
                T apq = A[p * n + q];

                T theta;
                if (std::abs(app - aqq) < tol) {
                    theta = (apq > 0) ? T{0.7853981633974483} : T{-0.7853981633974483}; // pi/4
                } else {
                    theta = T{0.5} * std::atan2(T{2} * apq, aqq - app);
                }

                T c = std::cos(theta);
                T s = std::sin(theta);

                // Apply rotation to A: A = G^T * A * G
                for (std::size_t i = 0; i < n; ++i) {
                    if (i != p && i != q) {
                        T aip = A[i * n + p];
                        T aiq = A[i * n + q];
                        A[i * n + p] = c * aip - s * aiq;
                        A[p * n + i] = A[i * n + p];
                        A[i * n + q] = s * aip + c * aiq;
                        A[q * n + i] = A[i * n + q];
                    }
                }

                T new_app = c * c * app - T{2} * c * s * apq + s * s * aqq;
                T new_aqq = s * s * app + T{2} * c * s * apq + c * c * aqq;
                A[p * n + p] = new_app;
                A[q * n + q] = new_aqq;
                A[p * n + q] = T{0};
                A[q * n + p] = T{0};

                // Apply rotation to B: B = B * G
                for (std::size_t i = 0; i < n; ++i) {
                    T bip = B[i * n + p];
                    T biq = B[i * n + q];
                    B[i * n + p] = c * bip - s * biq;
                    B[i * n + q] = s * bip + c * biq;
                }
            }

            // Extract eigenvalues (diagonal of A) and take sqrt
            for (std::size_t i = 0; i < n; ++i) {
                T eigval = A[i * n + i];
                // Ensure non-negative (numerical stability)
                if (eigval < T{0}) {
                    eigval = T{0};
                }
                D[i] = std::sqrt(eigval);
            }

            // Update C to be reconstructed from B and D (ensures positive semi-definiteness)
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    C[i * n + j] = T{0};
                    for (std::size_t k = 0; k < n; ++k) {
                        C[i * n + j] += B[i * n + k] * D[k] * D[k] * B[j * n + k];
                    }
                }
            }
        }
    };

} // namespace optinum::meta
