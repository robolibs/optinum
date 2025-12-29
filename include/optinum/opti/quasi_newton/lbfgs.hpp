#pragma once

// =============================================================================
// optinum/opti/quasi_newton/lbfgs.hpp
// L-BFGS (Limited-memory BFGS) optimizer for unconstrained optimization
// =============================================================================

#include <datapod/matrix/vector.hpp>
#include <optinum/opti/core/callbacks.hpp>
#include <optinum/opti/core/types.hpp>
#include <optinum/opti/line_search/line_search.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/bridge.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace optinum::opti {

    namespace dp = ::datapod;

    /**
     * @brief L-BFGS (Limited-memory BFGS) optimizer for unconstrained optimization
     *
     * Minimizes a smooth function f: R^n -> R using the L-BFGS algorithm,
     * a quasi-Newton method that approximates the inverse Hessian using
     * a limited history of position and gradient differences.
     *
     * **Algorithm:**
     * L-BFGS maintains the m most recent pairs (s_k, y_k) where:
     *   - s_k = x_{k+1} - x_k (position difference)
     *   - y_k = g_{k+1} - g_k (gradient difference)
     *
     * The search direction is computed using the two-loop recursion:
     *   d_k = -H_k * g_k
     *
     * where H_k is the inverse Hessian approximation implicitly defined by
     * the stored pairs. This avoids storing the full n×n matrix.
     *
     * **Two-Loop Recursion (Nocedal & Wright, Algorithm 7.4):**
     * ```
     * q = g_k
     * for i = k-1, ..., k-m:
     *     alpha_i = rho_i * s_i^T * q
     *     q = q - alpha_i * y_i
     * r = H_0 * q  (initial Hessian approximation)
     * for i = k-m, ..., k-1:
     *     beta = rho_i * y_i^T * r
     *     r = r + s_i * (alpha_i - beta)
     * return d = -r
     * ```
     *
     * **Key Features:**
     * - Memory efficient: O(m*n) storage vs O(n²) for full BFGS
     * - Superlinear convergence on smooth problems
     * - Strong Wolfe line search for guaranteed convergence
     * - Automatic scaling of initial Hessian approximation
     * - Robust handling of numerical issues
     *
     * **Use Cases:**
     * - Large-scale unconstrained optimization
     * - Machine learning (logistic regression, neural networks)
     * - Computer vision (bundle adjustment)
     * - Robotics (trajectory optimization)
     * - Any smooth optimization problem
     *
     * **Convergence:**
     * - Superlinear convergence rate on strongly convex functions
     * - Much faster than gradient descent (typically 10-100x fewer iterations)
     * - Requires smooth objective (continuous second derivatives)
     *
     * @tparam T Scalar type (float, double)
     *
     * Reference:
     *   Liu & Nocedal (1989)
     *   "On the Limited Memory BFGS Method for Large Scale Optimization"
     *   Mathematical Programming 45, pp. 503-528
     *
     *   Nocedal & Wright (2006)
     *   "Numerical Optimization", 2nd Edition, Chapter 7
     *
     * @example
     * // Define objective function
     * struct Rosenbrock {
     *     double evaluate(const dp::mat::vector<double, 2>& x) const {
     *         double a = 1.0 - x[0];
     *         double b = x[1] - x[0] * x[0];
     *         return a * a + 100.0 * b * b;
     *     }
     *     void gradient(const dp::mat::vector<double, 2>& x, dp::mat::vector<double, 2>& g) const {
     *         g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
     *         g[1] = 200.0 * (x[1] - x[0] * x[0]);
     *     }
     *     double evaluate_with_gradient(const dp::mat::vector<double, 2>& x, dp::mat::vector<double, 2>& g) const {
     *         gradient(x, g);
     *         return evaluate(x);
     *     }
     * };
     *
     * LBFGS<double> optimizer;
     * optimizer.max_iterations = 100;
     * optimizer.tolerance = 1e-8;
     * optimizer.history_size = 10;
     *
     * dp::mat::vector<double, 2> x0{-1.0, 1.0};
     * Rosenbrock func;
     * auto result = optimizer.optimize(func, x0);
     * // result.x should be close to (1, 1)
     */
    template <typename T = double> class LBFGS {
      public:
        // =============================================================================
        // Configuration Parameters
        // =============================================================================

        /// Maximum number of iterations
        std::size_t max_iterations = 100;

        /// Convergence tolerance on gradient norm
        T gradient_tolerance = T(1e-6);

        /// Convergence tolerance on function value change
        T function_tolerance = T(1e-12);

        /// Convergence tolerance on step size
        T step_tolerance = T(1e-12);

        /// Number of correction pairs to store (memory = O(m*n))
        /// Typical values: 3-20, default 10
        std::size_t history_size = 10;

        /// Initial inverse Hessian scaling (0 = automatic)
        /// If 0, uses gamma_k = (s_{k-1}^T * y_{k-1}) / (y_{k-1}^T * y_{k-1})
        T initial_hessian_scale = T(0);

        /// Print iteration info
        bool verbose = false;

        /// Line search type: "wolfe" (default), "armijo"
        std::string line_search_type = "wolfe";

        // Line search parameters
        T line_search_c1 = T(1e-4);             ///< Armijo parameter (sufficient decrease)
        T line_search_c2 = T(0.9);              ///< Wolfe curvature parameter
        T line_search_alpha_init = T(1.0);      ///< Initial step size
        std::size_t line_search_max_iters = 25; ///< Max line search iterations

        // =============================================================================
        // Public Methods
        // =============================================================================

        /// Default constructor
        LBFGS() = default;

        /**
         * @brief Optimize a function starting from initial point
         *
         * @param function Objective function with evaluate/gradient methods
         * @param x_init Initial point (will NOT be modified)
         * @param callback Optional callback for monitoring
         * @return OptimizationResult with solution and diagnostics
         *
         * The function must provide:
         * - T evaluate(const Vector& x) - function value
         * - void gradient(const Vector& x, Vector& g) - gradient
         * - T evaluate_with_gradient(const Vector& x, Vector& g) - both (optional but efficient)
         */
        template <typename FunctionType, std::size_t N, typename CallbackType = NoCallback>
        OptimizationResult<T, N> optimize(FunctionType &function, const dp::mat::vector<T, N> &x_init,
                                          CallbackType callback = NoCallback{}) {
            using vector_type = dp::mat::vector<T, N>;

            const std::size_t n = x_init.size();

            // Working variables
            vector_type x = x_init;
            vector_type gradient;
            vector_type gradient_prev;
            vector_type direction;
            vector_type x_new;
            vector_type gradient_new;

            // Allocate for Dynamic vectors
            if constexpr (N == dp::mat::Dynamic) {
                gradient.resize(n);
                gradient_prev.resize(n);
                direction.resize(n);
                x_new.resize(n);
                gradient_new.resize(n);
            }

            // History storage for L-BFGS
            // s_history[i] = x_{k-m+i+1} - x_{k-m+i}
            // y_history[i] = g_{k-m+i+1} - g_{k-m+i}
            // rho_history[i] = 1 / (y_i^T * s_i)
            dp::mat::vector<T, dp::mat::Dynamic> s_flat; // Flattened s vectors
            dp::mat::vector<T, dp::mat::Dynamic> y_flat; // Flattened y vectors
            dp::mat::vector<T, dp::mat::Dynamic> rho;    // rho values
            dp::mat::vector<T, dp::mat::Dynamic> alpha;  // alpha values for two-loop recursion

            s_flat.resize(history_size * n);
            y_flat.resize(history_size * n);
            rho.resize(history_size);
            alpha.resize(history_size);

            // Initialize history to zero
            simd::view(s_flat).fill(T(0));
            simd::view(y_flat).fill(T(0));
            simd::view(rho).fill(T(0));

            std::size_t history_count = 0; // Number of stored pairs
            std::size_t history_start = 0; // Circular buffer start index

            // Evaluate initial function and gradient
            T f_current = function.evaluate_with_gradient(x, gradient);
            T f_prev = std::numeric_limits<T>::max();

            T grad_norm = simd::view(gradient).norm();

            if (verbose) {
                std::cout << "=== L-BFGS Optimization ===" << std::endl;
                std::cout << "Variables: " << n << std::endl;
                std::cout << "History size: " << history_size << std::endl;
                std::cout << "Initial f(x): " << f_current << std::endl;
                std::cout << "Initial ||g||: " << grad_norm << std::endl;
            }

            // Callback: begin
            callback.on_begin(x);

            std::size_t iteration = 0;
            bool converged = false;
            std::string termination_reason;

            // Check initial convergence
            if (grad_norm < gradient_tolerance) {
                converged = true;
                termination_reason = "Converged: initial gradient norm < tolerance";
            }

            // Main L-BFGS loop
            for (; iteration < max_iterations && !converged; ++iteration) {
                // Callback: iteration info
                IterationInfo<T> info(iteration, f_current, grad_norm, T(1.0));
                bool should_stop = callback.on_iteration(info, x);
                if (should_stop) {
                    converged = true;
                    termination_reason = termination::CALLBACK_STOP;
                    break;
                }

                if (verbose) {
                    std::cout << "Iter " << iteration << ": f = " << f_current << ", ||g|| = " << grad_norm
                              << std::endl;
                }

                // Step 1: Compute search direction using two-loop recursion
                compute_direction(gradient, s_flat, y_flat, rho, alpha, history_count, history_start, n, direction);

                // Check if direction is descent
                T directional_derivative = simd::view(gradient).dot(simd::view(direction));
                if (directional_derivative >= T(0)) {
                    // Not a descent direction - reset to steepest descent
                    if (verbose) {
                        std::cout << "  Warning: Not a descent direction, resetting to steepest descent" << std::endl;
                    }
                    // direction = -gradient using SIMD
                    simd::backend::negate_runtime<T>(direction.data(), gradient.data(), n);
                    directional_derivative = -grad_norm * grad_norm;
                    history_count = 0; // Reset history
                }

                // Step 2: Line search
                T alpha_step;
                bool line_search_success;

                if (line_search_type == "wolfe") {
                    WolfeLineSearch<T> ls(line_search_c1, line_search_c2, line_search_alpha_init);
                    ls.max_bracket_iters = line_search_max_iters;
                    ls.max_zoom_iters = line_search_max_iters;

                    auto ls_result = ls.search(function, x, direction, f_current, gradient, gradient_new);
                    alpha_step = ls_result.alpha;
                    line_search_success = ls_result.success;

                    if (line_search_success) {
                        // x_new = x + alpha * direction using SIMD
                        simd::backend::axpy_runtime<T>(x_new.data(), x.data(), alpha_step, direction.data(), n);
                        // gradient_new already computed by Wolfe line search
                    }
                } else {
                    // Armijo line search
                    ArmijoLineSearch<T> ls(line_search_c1, T(0.5), line_search_alpha_init);
                    ls.max_iters = line_search_max_iters;

                    auto ls_result = ls.search(function, x, direction, f_current, gradient);
                    alpha_step = ls_result.alpha;
                    line_search_success = ls_result.success;

                    if (line_search_success) {
                        // x_new = x + alpha * direction using SIMD
                        simd::backend::axpy_runtime<T>(x_new.data(), x.data(), alpha_step, direction.data(), n);
                        function.gradient(x_new, gradient_new);
                    }
                }

                if (!line_search_success) {
                    if (verbose) {
                        std::cout << "  Line search failed" << std::endl;
                    }
                    termination_reason = "Failed: line search did not converge";
                    converged = false;
                    break;
                }

                // Evaluate function at new point
                T f_new = function.evaluate(x_new);

                // Check for NaN/Inf
                if (std::isnan(f_new) || std::isinf(f_new)) {
                    termination_reason = termination::NAN_INF;
                    converged = false;
                    break;
                }

                // Step 3: Update history (s_k = x_new - x, y_k = g_new - g)
                std::size_t idx = (history_start + history_count) % history_size;
                if (history_count < history_size) {
                    ++history_count;
                } else {
                    history_start = (history_start + 1) % history_size;
                    idx = (history_start + history_count - 1) % history_size;
                }

                // Store s_k and y_k using SIMD
                T *s_ptr = s_flat.data() + idx * n;
                T *y_ptr = y_flat.data() + idx * n;

                // s = x_new - x, y = gradient_new - gradient
                simd::backend::sub_runtime<T>(s_ptr, x_new.data(), x.data(), n);
                simd::backend::sub_runtime<T>(y_ptr, gradient_new.data(), gradient.data(), n);

                // Compute ys = y^T * s and yy = y^T * y using SIMD
                T ys = simd::backend::dot_runtime<T>(y_ptr, s_ptr, n);
                T yy = simd::backend::dot_runtime<T>(y_ptr, y_ptr, n);

                // Check curvature condition: y^T * s > 0
                if (ys > T(1e-10) * yy) {
                    rho[idx] = T(1) / ys;
                } else {
                    // Skip this update (curvature condition violated)
                    if (verbose) {
                        std::cout << "  Warning: Curvature condition violated, skipping update" << std::endl;
                    }
                    if (history_count > 0) {
                        --history_count;
                    }
                }

                // Step 4: Check convergence
                T step_norm = alpha_step * simd::view(direction).norm();
                T f_decrease = f_current - f_new;

                // Update state
                x = x_new;
                gradient = gradient_new;
                f_prev = f_current;
                f_current = f_new;
                grad_norm = simd::view(gradient).norm();

                // Convergence checks
                if (grad_norm < gradient_tolerance) {
                    converged = true;
                    termination_reason = "Converged: gradient norm < " + std::to_string(gradient_tolerance);
                    break;
                }

                if (std::abs(f_decrease) < function_tolerance * (T(1) + std::abs(f_current))) {
                    converged = true;
                    termination_reason = "Converged: function change < tolerance";
                    break;
                }

                if (step_norm < step_tolerance * (T(1) + simd::view(x).norm())) {
                    converged = true;
                    termination_reason = "Converged: step size < tolerance";
                    break;
                }
            }

            // Check if stopped due to max iterations
            if (iteration >= max_iterations && !converged) {
                termination_reason = termination::MAX_ITERATIONS;
                converged = false;
            }

            if (verbose) {
                std::cout << "=== Optimization Complete ===" << std::endl;
                std::cout << "Converged: " << (converged ? "YES" : "NO") << std::endl;
                std::cout << "Reason: " << termination_reason << std::endl;
                std::cout << "Iterations: " << iteration << std::endl;
                std::cout << "Final f(x): " << f_current << std::endl;
                std::cout << "Final ||g||: " << grad_norm << std::endl;
            }

            // Create result
            OptimizationResult<T, N> result(x, f_current, grad_norm, iteration, converged, termination_reason);

            // Callback: end
            callback.on_end(result);

            return result;
        }

      private:
        // =============================================================================
        // Two-Loop Recursion for Computing Search Direction
        // =============================================================================

        /**
         * @brief Compute search direction using L-BFGS two-loop recursion
         *
         * Implements Algorithm 7.4 from Nocedal & Wright (2006).
         *
         * @param gradient Current gradient g_k
         * @param s_flat Flattened s vectors (position differences)
         * @param y_flat Flattened y vectors (gradient differences)
         * @param rho rho values (1 / y^T * s)
         * @param alpha Working array for alpha values
         * @param history_count Number of stored pairs
         * @param history_start Start index in circular buffer
         * @param n Problem dimension
         * @param direction Output: search direction d = -H_k * g_k
         */
        template <std::size_t N>
        void compute_direction(const dp::mat::vector<T, N> &gradient,
                               const dp::mat::vector<T, dp::mat::Dynamic> &s_flat,
                               const dp::mat::vector<T, dp::mat::Dynamic> &y_flat,
                               const dp::mat::vector<T, dp::mat::Dynamic> &rho,
                               dp::mat::vector<T, dp::mat::Dynamic> &alpha, std::size_t history_count,
                               std::size_t history_start, std::size_t n, dp::mat::vector<T, N> &direction) const {
            // If no history, use steepest descent
            if (history_count == 0) {
                // direction = -gradient using SIMD
                simd::backend::negate_runtime<T>(direction.data(), gradient.data(), n);
                return;
            }

            // q = gradient (copy) using SIMD
            dp::mat::vector<T, dp::mat::Dynamic> q;
            q.resize(n);
            simd::backend::copy_runtime<T>(q.data(), gradient.data(), n);

            // First loop: iterate from newest to oldest
            for (std::size_t j = 0; j < history_count; ++j) {
                // Index from newest to oldest
                std::size_t idx = (history_start + history_count - 1 - j) % history_size;
                const T *s_ptr = s_flat.data() + idx * n;
                const T *y_ptr = y_flat.data() + idx * n;

                // alpha_i = rho_i * s_i^T * q using SIMD dot product
                T dot_sq = simd::backend::dot_runtime<T>(s_ptr, q.data(), n);
                alpha[idx] = rho[idx] * dot_sq;

                // q = q - alpha_i * y_i using SIMD (q += (-alpha) * y)
                simd::backend::axpy_inplace_runtime<T>(q.data(), -alpha[idx], y_ptr, n);
            }

            // Compute initial Hessian scaling: gamma = (s_{k-1}^T * y_{k-1}) / (y_{k-1}^T * y_{k-1})
            T gamma;
            if (initial_hessian_scale > T(0)) {
                gamma = initial_hessian_scale;
            } else {
                // Use most recent pair for scaling
                std::size_t newest_idx = (history_start + history_count - 1) % history_size;
                const T *s_ptr = s_flat.data() + newest_idx * n;
                const T *y_ptr = y_flat.data() + newest_idx * n;

                // Compute sy and yy using SIMD dot products
                T sy = simd::backend::dot_runtime<T>(s_ptr, y_ptr, n);
                T yy = simd::backend::dot_runtime<T>(y_ptr, y_ptr, n);
                gamma = (yy > T(1e-15)) ? (sy / yy) : T(1);
            }

            // r = gamma * q (initial Hessian approximation H_0 = gamma * I) using SIMD
            dp::mat::vector<T, dp::mat::Dynamic> r;
            r.resize(n);
            simd::backend::mul_scalar_runtime<T>(r.data(), q.data(), gamma, n);

            // Second loop: iterate from oldest to newest
            for (std::size_t j = 0; j < history_count; ++j) {
                // Index from oldest to newest
                std::size_t idx = (history_start + j) % history_size;
                const T *s_ptr = s_flat.data() + idx * n;
                const T *y_ptr = y_flat.data() + idx * n;

                // beta = rho_i * y_i^T * r using SIMD dot product
                T dot_yr = simd::backend::dot_runtime<T>(y_ptr, r.data(), n);
                T beta = rho[idx] * dot_yr;

                // r = r + s_i * (alpha_i - beta) using SIMD
                T diff = alpha[idx] - beta;
                simd::backend::axpy_inplace_runtime<T>(r.data(), diff, s_ptr, n);
            }

            // direction = -r using SIMD
            simd::backend::negate_runtime<T>(direction.data(), r.data(), n);
        }
    };

    // =============================================================================
    // Type aliases for convenience
    // =============================================================================

    using LBFGSf = LBFGS<float>;
    using LBFGSd = LBFGS<double>;

} // namespace optinum::opti
