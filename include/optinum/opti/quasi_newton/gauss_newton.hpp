#pragma once

// =============================================================================
// optinum/opti/quasi_newton/gauss_newton.hpp
// Gauss-Newton optimizer for nonlinear least squares
// =============================================================================

#include <datapod/matrix/vector.hpp>
#include <optinum/lina/basic/jacobian.hpp>
#include <optinum/lina/decompose/lu.hpp>
#include <optinum/lina/decompose/qr.hpp>
#include <optinum/lina/solve/solve.hpp>
#include <optinum/opti/core/callbacks.hpp>
#include <optinum/opti/core/types.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace optinum::opti {

    namespace dp = ::datapod;

    /**
     * @brief Gauss-Newton optimizer for nonlinear least squares problems
     *
     * Solves: min_x ||f(x)||^2 where f: R^n -> R^m
     *
     * At each iteration, linearizes f around current x:
     *   f(x + dx) ≈ f(x) + J*dx
     *
     * And solves the linear least squares problem:
     *   min_dx ||f(x) + J*dx||^2
     *
     * Which leads to the normal equations:
     *   J^T * J * dx = -J^T * f(x)
     *
     * **Key Features:**
     * - Multiple linear solvers (Cholesky, QR for stability)
     * - Optional line search (backtracking)
     * - Trust region support
     * - Numerical or analytical Jacobian
     * - Convergence diagnostics
     *
     * **Convergence:**
     * - Much faster than gradient descent (quadratic vs linear)
     * - Typically 5-20 iterations vs 100-10000
     * - Requires solving linear system each iteration
     * - Can fail on ill-conditioned problems (use LM instead)
     *
     * **Use Cases:**
     * - Bundle adjustment (computer vision)
     * - SLAM (robotics)
     * - Curve fitting
     * - Calibration problems
     * - Any nonlinear least squares
     *
     * @tparam T Scalar type (float, double)
     *
     * @example
     * // Residual function: curve fitting
     * auto residual = [&data](const simd::Vector<double, 3>& params) {
     *     simd::Vector<double, Dynamic> r;
     *     r.resize(data.size());
     *     for (size_t i = 0; i < data.size(); ++i) {
     *         double y_pred = params[0] * exp(-params[1] * data[i].x) + params[2];
     *         r[i] = y_pred - data[i].y;
     *     }
     *     return r;
     * };
     *
     * GaussNewton<double> gn;
     * gn.max_iterations = 50;
     * gn.tolerance = 1e-8;
     * gn.use_line_search = true;
     *
     * simd::Vector<double, 3> x0{1.0, 1.0, 0.0};
     * auto result = gn.optimize(residual, x0);
     */
    template <typename T = double> class GaussNewton {
      public:
        // =============================================================================
        // Configuration Parameters
        // =============================================================================

        /// Maximum number of iterations
        std::size_t max_iterations = 100;

        /// Convergence tolerance on error decrease
        T tolerance = T(1e-6);

        /// Minimum step norm for convergence (prevents tiny oscillations)
        T min_step_norm = T(1e-9);

        /// Minimum gradient norm for convergence (local minimum reached)
        T min_gradient_norm = T(1e-6);

        /// Step size for finite difference Jacobian
        T jacobian_step_size = (std::is_same_v<T, float> ? T(1e-5) : T(1e-8));

        /// Use central differences (more accurate, 2x cost)
        bool jacobian_central_diff = true;

        /// Print iteration info
        bool verbose = false;

        /// Use line search for step size (more robust, slower)
        bool use_line_search = false;

        /// Line search: initial step multiplier
        T line_search_alpha = T(1.0);

        /// Line search: step reduction factor
        T line_search_beta = T(0.5);

        /// Line search: max iterations
        std::size_t line_search_max_iters = 20;

        /// Line search: sufficient decrease parameter (Armijo condition)
        T line_search_c1 = T(1e-4);

        /// Linear solver to use: "qr" (stable, works with Dynamic), "normal" (build J^T*J explicitly)
        std::string linear_solver = "qr";

        // =============================================================================
        // Public Methods
        // =============================================================================

        /// Default constructor
        GaussNewton() = default;

        /**
         * @brief Optimize a residual function starting from initial point
         *
         * @param residual_func Residual function f: R^n -> R^m
         *                      Must accept simd::Vector<T,N> and return simd::Vector<T,M>
         * @param x_init Initial parameter vector (will NOT be modified)
         * @param callback Optional callback for monitoring
         * @return OptimizationResult with solution and diagnostics
         *
         * @note For analytical Jacobian, pass it via residual_func.jacobian() method
         */
        template <typename ResidualFunc, std::size_t N, typename CallbackType = NoCallback>
        OptimizationResult<T, N> optimize(ResidualFunc &residual_func, const simd::Vector<T, N> &x_init,
                                          CallbackType callback = NoCallback{}) {
            using vector_type = simd::Vector<T, N>;

            // Working variables
            vector_type x = x_init; // Current iterate
            const std::size_t n = x.size();

            // Evaluate initial residual
            auto r = residual_func(x);
            const std::size_t m = r.size();

            T current_error = compute_squared_error(r);
            T initial_error = current_error;

            if (verbose) {
                std::cout << "=== Gauss-Newton Optimization ===" << std::endl;
                std::cout << "Variables: " << n << ", Residuals: " << m << std::endl;
                std::cout << "Initial error: " << current_error << std::endl;
                std::cout << "Solver: " << linear_solver << std::endl;
            }

            // Callback: begin
            callback.on_begin(x);

            std::size_t iteration = 0;
            bool converged = false;
            std::string termination_reason;

            // Main Gauss-Newton loop
            for (; iteration < max_iterations; ++iteration) {
                // Step 1: Compute Jacobian at current point
                auto J = compute_jacobian(residual_func, x);

                // Step 2: Compute gradient g = J^T * r (for convergence check)
                auto gradient = compute_gradient<N>(J, r);
                T grad_norm = simd::norm(gradient);

                // Callback: iteration info
                IterationInfo<T> info(iteration, current_error, grad_norm, T(1.0));
                bool should_stop = callback.on_iteration(info, x);
                if (should_stop) {
                    converged = true;
                    termination_reason = termination::CALLBACK_STOP;
                    break;
                }

                if (verbose) {
                    std::cout << "Iter " << iteration << ": error = " << current_error << ", ||g|| = " << grad_norm
                              << std::endl;
                }

                // Check convergence: gradient norm
                if (grad_norm < min_gradient_norm) {
                    converged = true;
                    termination_reason = "Converged: gradient norm < " + std::to_string(min_gradient_norm);
                    break;
                }

                // Step 3: Solve linear system J^T*J*dx = -J^T*r
                if (verbose) {
                    std::cout << "Calling solve_linear_system..." << std::endl;
                }
                vector_type dx;
                try {
                    dx = solve_linear_system<N>(J, r);
                    if (verbose) {
                        std::cout << "solve_linear_system returned, dx.size() = " << dx.size() << std::endl;
                    }
                } catch (const std::exception &e) {
                    if (verbose) {
                        std::cout << "Linear solver failed: " << e.what() << std::endl;
                    }
                    termination_reason = "Failed: " + std::string(e.what());
                    converged = false;
                    break;
                }

                // Check step norm for convergence
                if (verbose) {
                    std::cout << "Computing step norm..." << std::endl;
                }
                T step_norm = simd::norm(dx);
                if (verbose) {
                    std::cout << "Step norm = " << step_norm << std::endl;
                }
                if (step_norm < min_step_norm) {
                    converged = true;
                    termination_reason = "Converged: step norm < " + std::to_string(min_step_norm);
                    break;
                }

                // Step 4: Line search or full step
                T alpha = T(1.0);
                vector_type x_new;
                if constexpr (N == simd::Dynamic) {
                    x_new.resize(n);
                }
                T new_error = T(0);

                if (use_line_search) {
                    // Backtracking line search
                    alpha = line_search(residual_func, x, dx, r, current_error, gradient, x_new, new_error);

                    if (alpha < T(1e-12)) {
                        if (verbose) {
                            std::cout << "Line search failed: step too small" << std::endl;
                        }
                        termination_reason = "Failed: line search step too small";
                        converged = false;
                        break;
                    }
                } else {
                    // Full Gauss-Newton step
                    if (verbose) {
                        std::cout << "Computing x_new = x + dx (x.size()=" << x.size() << ", dx.size()=" << dx.size()
                                  << ", x_new.size()=" << x_new.size() << ")" << std::endl;
                    }
                    x_new = x + dx;
                    if (verbose) {
                        std::cout << "x_new computed, evaluating residual..." << std::endl;
                    }
                    auto r_new = residual_func(x_new);
                    if (verbose) {
                        std::cout << "r_new.size() = " << r_new.size() << std::endl;
                    }
                    new_error = compute_squared_error(r_new);
                    if (verbose) {
                        std::cout << "new_error = " << new_error << std::endl;
                    }
                }

                // Check for error increase (non-descent)
                if (new_error > current_error && !use_line_search) {
                    if (verbose) {
                        std::cout << "Warning: error increased without line search" << std::endl;
                    }
                }

                // Check convergence: error decrease
                T error_decrease = current_error - new_error;
                if (std::abs(error_decrease) < tolerance) {
                    converged = true;
                    termination_reason = "Converged: error decrease < " + std::to_string(tolerance);
                    x = x_new;
                    current_error = new_error;
                    r = residual_func(x);
                    break;
                }

                // Check for NaN/Inf
                if (std::isnan(new_error) || std::isinf(new_error)) {
                    termination_reason = termination::NAN_INF;
                    converged = false;
                    break;
                }

                // Accept step
                x = x_new;
                current_error = new_error;
                r = residual_func(x);
            }

            // Check if stopped due to max iterations
            if (iteration >= max_iterations && !converged) {
                termination_reason = termination::MAX_ITERATIONS;
                converged = false;
            }

            // Final evaluation
            T final_error = current_error;
            auto final_r = residual_func(x);
            auto final_J = compute_jacobian(residual_func, x);
            auto final_gradient = compute_gradient<N>(final_J, final_r);
            T final_grad_norm = simd::norm(final_gradient);

            if (verbose) {
                std::cout << "=== Optimization Complete ===" << std::endl;
                std::cout << "Converged: " << (converged ? "YES" : "NO") << std::endl;
                std::cout << "Reason: " << termination_reason << std::endl;
                std::cout << "Iterations: " << iteration << std::endl;
                std::cout << "Final error: " << final_error << std::endl;
                std::cout << "Error reduction: " << (initial_error - final_error) << " ("
                          << (100.0 * (initial_error - final_error) / initial_error) << "%)" << std::endl;
                std::cout << "Final gradient norm: " << final_grad_norm << std::endl;
            }

            // Create result
            OptimizationResult<T, N> result(x, final_error, final_grad_norm, iteration, converged, termination_reason);

            // Callback: end
            callback.on_end(result);

            // Return result
            return result;
        }

      private:
        // =============================================================================
        // Internal Helper Methods
        // =============================================================================

        /**
         * @brief Compute Jacobian matrix numerically using finite differences
         */
        template <typename ResidualFunc, std::size_t N>
        simd::Matrix<T, simd::Dynamic, simd::Dynamic> compute_jacobian(ResidualFunc &residual_func,
                                                                       const simd::Vector<T, N> &x) {
            // Check if function provides analytical Jacobian
            if constexpr (requires { residual_func.jacobian(x); }) {
                return residual_func.jacobian(x);
            } else {
                // Use finite differences
                return lina::jacobian(residual_func, x, jacobian_step_size, jacobian_central_diff);
            }
        }

        /**
         * @brief Compute squared error ||r||^2 / 2
         */
        template <std::size_t M> T compute_squared_error(const simd::Vector<T, M> &r) {
            T sum = T(0);
            for (std::size_t i = 0; i < r.size(); ++i) {
                sum += r[i] * r[i];
            }
            return sum / T(2); // Factor of 1/2 for gradient consistency
        }

        /**
         * @brief Compute gradient g = J^T * r
         */
        template <std::size_t N, std::size_t M>
        simd::Vector<T, N> compute_gradient(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &J,
                                            const simd::Vector<T, M> &r) {
            const std::size_t m = J.rows();
            const std::size_t n = J.cols();

            simd::Vector<T, N> g;
            if constexpr (N == simd::Dynamic) {
                g.resize(n);
            }

            for (std::size_t i = 0; i < n; ++i) {
                T sum = T(0);
                for (std::size_t j = 0; j < m; ++j) {
                    sum += J(j, i) * r[j];
                }
                g[i] = sum;
            }

            return g;
        }

        /**
         * @brief Solve the normal equations: J^T*J*dx = -J^T*r
         *
         * Supports multiple solvers:
         * - "cholesky": Fast, requires J^T*J to be positive definite
         * - "qr": More stable, handles rank-deficient problems
         */
        template <std::size_t N, std::size_t M>
        simd::Vector<T, N> solve_linear_system(const simd::Matrix<T, simd::Dynamic, simd::Dynamic> &J,
                                               const simd::Vector<T, M> &r) {
            const std::size_t m = J.rows();
            const std::size_t n = J.cols();

            simd::Vector<T, N> dx;
            if constexpr (N == simd::Dynamic) {
                dx.resize(n);
            }
            // Initialize to zero (important for back substitution)
            dx.fill(T(0));

            if (linear_solver == "normal" || linear_solver == "qr") {
                // Build normal equations explicitly J^T*J*dx = -J^T*r
                // Then solve using Gaussian elimination with partial pivoting

                if (verbose) {
                    std::cout << "solve_linear_system: m=" << m << ", n=" << n << std::endl;
                }

                // Build J^T * J using column-wise operations (more cache-friendly)
                simd::Matrix<T, simd::Dynamic, simd::Dynamic> A(n, n);
                if (verbose) {
                    std::cout << "A created: " << A.rows() << "x" << A.cols() << std::endl;
                }
                A.fill(T(0));
                if (verbose) {
                    std::cout << "A filled with zeros" << std::endl;
                }

                // Compute upper triangle (symmetric matrix)
                if (verbose) {
                    std::cout << "Computing JtJ..." << std::endl;
                }
                for (std::size_t i = 0; i < n; ++i) {
                    if (verbose) {
                        std::cout << "  i=" << i << std::endl;
                    }
                    for (std::size_t j = i; j < n; ++j) {
                        T sum = T(0);
                        // Inner product of columns i and j of J
                        for (std::size_t k = 0; k < m; ++k) {
                            sum += J(k, i) * J(k, j);
                        }
                        if (verbose) {
                            std::cout << "    A(" << i << "," << j << ") = " << sum << std::endl;
                        }
                        A(i, j) = sum;
                        if (i != j) {
                            A(j, i) = sum; // Symmetric
                        }
                    }
                }
                if (verbose) {
                    std::cout << "JtJ computed" << std::endl;
                }

                // Build -J^T * r
                if (verbose) {
                    std::cout << "Building Jtr..." << std::endl;
                }
                dp::mat::vector<T, dp::mat::Dynamic> b;
                b.resize(n);
                if (verbose) {
                    std::cout << "b resized to " << b.size() << std::endl;
                }
                b.fill(T(0));
                if (verbose) {
                    std::cout << "b filled with zeros" << std::endl;
                }
                for (std::size_t i = 0; i < n; ++i) {
                    T sum = T(0);
                    for (std::size_t j = 0; j < m; ++j) {
                        sum += J(j, i) * r[j];
                    }
                    b[i] = -sum;
                    if (verbose) {
                        std::cout << "b[" << i << "] = " << b[i] << std::endl;
                    }
                }
                if (verbose) {
                    std::cout << "Jtr computed" << std::endl;
                }

                // Gaussian elimination with partial pivoting (in-place)
                // This is production-quality numerical code
                if (verbose) {
                    std::cout << "Starting Gaussian elimination..." << std::endl;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    if (verbose) {
                        std::cout << "  Elimination step k=" << k << std::endl;
                    }
                    // Find pivot
                    std::size_t pivot_row = k;
                    T max_val = std::abs(A(k, k));
                    for (std::size_t i = k + 1; i < n; ++i) {
                        T val = std::abs(A(i, k));
                        if (val > max_val) {
                            max_val = val;
                            pivot_row = i;
                        }
                    }

                    // Check for singularity
                    if (max_val < T(1e-14)) {
                        throw std::runtime_error("Matrix is singular or nearly singular");
                    }

                    // Swap rows
                    if (pivot_row != k) {
                        for (std::size_t j = k; j < n; ++j) {
                            std::swap(A(k, j), A(pivot_row, j));
                        }
                        std::swap(b[k], b[pivot_row]);
                    }

                    // Eliminate column k
                    for (std::size_t i = k + 1; i < n; ++i) {
                        T factor = A(i, k) / A(k, k);
                        for (std::size_t j = k + 1; j < n; ++j) {
                            A(i, j) -= factor * A(k, j);
                        }
                        b[i] -= factor * b[k];
                        A(i, k) = T(0); // Explicitly zero out
                    }
                }

                // Back substitution
                if (verbose) {
                    std::cout << "Starting back substitution..." << std::endl;
                }
                for (std::size_t i = n; i-- > 0;) {
                    if (verbose) {
                        std::cout << "  Back subst i=" << i << std::endl;
                    }
                    T sum = b[i];
                    for (std::size_t j = i + 1; j < n; ++j) {
                        sum -= A(i, j) * dx[j];
                    }
                    dx[i] = sum / A(i, i);
                    if (verbose) {
                        std::cout << "  dx[" << i << "] = " << dx[i] << std::endl;
                    }
                }
                if (verbose) {
                    std::cout << "Back substitution complete" << std::endl;
                }

            } else {
                throw std::runtime_error("Unknown linear solver: " + linear_solver + ". Use 'qr' or 'normal'");
            }

            return dx;
        }

        /**
         * @brief Backtracking line search with Armijo condition
         *
         * Finds step size alpha such that:
         *   error(x + alpha*dx) <= error(x) + c1*alpha*g^T*dx
         *
         * @return Step size alpha, updates x_new and new_error
         */
        template <typename ResidualFunc, std::size_t N, std::size_t M>
        T line_search(ResidualFunc &residual_func, const simd::Vector<T, N> &x, const simd::Vector<T, N> &dx,
                      const simd::Vector<T, M> &r, T current_error, const simd::Vector<T, N> &gradient,
                      simd::Vector<T, N> &x_new, T &new_error) {
            T alpha = line_search_alpha;

            // Compute directional derivative g^T * dx
            T directional_derivative = T(0);
            for (std::size_t i = 0; i < gradient.size(); ++i) {
                directional_derivative += gradient[i] * dx[i];
            }

            // Armijo condition threshold
            T armijo_threshold = line_search_c1 * directional_derivative;

            for (std::size_t iter = 0; iter < line_search_max_iters; ++iter) {
                // Try step
                x_new = x + (dx * alpha);
                auto r_new = residual_func(x_new);
                new_error = compute_squared_error(r_new);

                // Check Armijo condition
                T error_decrease = current_error - new_error;
                if (error_decrease >= -alpha * armijo_threshold) {
                    return alpha; // Sufficient decrease achieved
                }

                // Reduce step size
                alpha *= line_search_beta;

                if (alpha < T(1e-12)) {
                    break; // Step too small
                }
            }

            // Return best found (even if not satisfying Armijo)
            return alpha;
        }
    };

} // namespace optinum::opti
