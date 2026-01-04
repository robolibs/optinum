#pragma once

// =============================================================================
// optinum/opti/quasi_newton/levenberg_marquardt.hpp
// Levenberg-Marquardt optimizer for nonlinear least squares
// =============================================================================

#include <datapod/datapod.hpp>
#include <optinum/lina/basic/jacobian.hpp>
#include <optinum/lina/decompose/lu.hpp>
#include <optinum/opti/core/callbacks.hpp>
#include <optinum/opti/core/types.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/bridge.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>

namespace optinum::opti {

    /**
     * @brief Levenberg-Marquardt optimizer for nonlinear least squares
     *
     * Solves: min_x ||f(x)||^2 where f: R^n -> R^m
     *
     * LM improves on Gauss-Newton by adding a damping term:
     *   (J^T * J + λ*I) * dx = -J^T * f(x)
     *
     * The damping parameter λ is adaptively adjusted:
     * - Small λ → behaves like Gauss-Newton (fast quadratic convergence)
     * - Large λ → behaves like gradient descent (robust but slow)
     *
     * **Advantages over Gauss-Newton:**
     * - More robust to poor initialization
     * - Better handling of ill-conditioned problems
     * - Guaranteed descent with proper lambda scheduling
     *
     * **When to use:**
     * - Poor initial guess
     * - Ill-conditioned Jacobian
     * - Need guaranteed descent
     * - Robotics/SLAM with uncertain initialization
     *
     * @tparam T Scalar type (float, double)
     *
     * @example
     * auto residual = [&data](const dp::mat::Vector<double, 3>& params) {
     *     // ... compute residuals ...
     * };
     *
     * LevenbergMarquardt<double> lm;
     * lm.max_iterations = 100;
     * lm.initial_lambda = 1e-3;
     * lm.lambda_factor = 10.0;
     *
     * dp::mat::Vector<double, 3> x0{1.0, 1.0, 0.0};
     * auto result = lm.optimize(residual, x0);
     */
    template <typename T = double> class LevenbergMarquardt {
      public:
        // =============================================================================
        // Configuration Parameters
        // =============================================================================

        /// Maximum number of iterations
        std::size_t max_iterations = 100;

        /// Convergence tolerance on error decrease
        T tolerance = T(1e-6);

        /// Minimum step norm for convergence
        T min_step_norm = T(1e-9);

        /// Minimum gradient norm for convergence
        T min_gradient_norm = T(1e-6);

        /// Initial damping parameter
        T initial_lambda = T(1e-3);

        /// Factor for increasing/decreasing lambda
        T lambda_factor = T(10.0);

        /// Minimum lambda value
        T min_lambda = T(1e-7);

        /// Maximum lambda value
        T max_lambda = T(1e7);

        /// Step size for finite difference Jacobian
        T jacobian_step_size = (std::is_same_v<T, float> ? T(1e-5) : T(1e-8));

        /// Use central differences (more accurate, 2x cost)
        bool jacobian_central_diff = true;

        /// Print iteration info
        bool verbose = false;

        // =============================================================================
        // Public Methods
        // =============================================================================

        /// Default constructor
        LevenbergMarquardt() = default;

        /**
         * @brief Optimize a residual function starting from initial point
         *
         * @param residual_func Residual function f: R^n -> R^m
         * @param x_init Initial parameter vector
         * @param callback Optional callback for monitoring
         * @return OptimizationResult with solution and diagnostics
         */
        template <typename ResidualFunc, std::size_t N, typename CallbackType = NoCallback>
        OptimizationResult<T, N> optimize(ResidualFunc &residual_func, const dp::mat::Vector<T, N> &x_init,
                                          CallbackType callback = NoCallback{}) {
            using vector_type = dp::mat::Vector<T, N>;

            // Working variables
            vector_type x = x_init;
            const std::size_t n = x.size();

            // Evaluate initial residual and error
            auto r = residual_func(x);
            const std::size_t m = r.size();
            T current_error = compute_squared_error(r);
            T initial_error = current_error;
            T lambda = initial_lambda;

            if (verbose) {
                std::cout << "=== Levenberg-Marquardt Optimization ===" << std::endl;
                std::cout << "Variables: " << n << ", Residuals: " << m << std::endl;
                std::cout << "Initial error: " << current_error << std::endl;
                std::cout << "Initial lambda: " << lambda << std::endl;
            }

            // Callback: begin
            callback.on_begin(x);

            std::size_t iteration = 0;
            bool converged = false;
            std::string termination_reason;

            // Main LM loop
            for (; iteration < max_iterations; ++iteration) {
                // Step 1: Compute Jacobian at current point
                auto J = compute_jacobian(residual_func, x);

                // Step 2: Compute gradient g = J^T * r
                auto gradient = compute_gradient<N>(J, r);
                T grad_norm = simd::view(gradient).norm();

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
                              << ", λ = " << lambda << std::endl;
                }

                // Check convergence: gradient norm
                if (grad_norm < min_gradient_norm) {
                    converged = true;
                    termination_reason = "Converged: gradient norm < " + std::to_string(min_gradient_norm);
                    break;
                }

                // Step 3: Solve augmented system (J^T*J + λ*I)*dx = -J^T*r
                auto dx_opt = solve_damped_system<N>(J, r, lambda);
                if (!dx_opt.has_value()) {
                    // Solver failed - increase lambda and retry
                    if (verbose) {
                        std::cout << "Solver failed: Damped system is singular, increasing lambda" << std::endl;
                    }
                    lambda = std::min(lambda * lambda_factor, max_lambda);
                    if (lambda >= max_lambda) {
                        termination_reason = "Damped system is singular (increase lambda)";
                        converged = false;
                        break;
                    }
                    continue; // Retry with larger lambda
                }
                vector_type dx = std::move(dx_opt.value());

                // Check step norm
                T step_norm = simd::view(dx).norm();
                if (step_norm < min_step_norm) {
                    converged = true;
                    termination_reason = "Converged: step norm < " + std::to_string(min_step_norm);
                    break;
                }

                // Step 4: Try the step and evaluate new error (SIMD-optimized)
                vector_type x_new;
                if constexpr (N == dp::mat::Dynamic) {
                    x_new.resize(n);
                }
                // x_new = x + 1.0 * dx
                simd::backend::axpy_runtime<T>(x_new.data(), x.data(), T(1), dx.data(), n);
                auto r_new = residual_func(x_new);
                T new_error = compute_squared_error(r_new);

                // Check for NaN/Inf
                if (std::isnan(new_error) || std::isinf(new_error)) {
                    termination_reason = termination::NAN_INF;
                    converged = false;
                    break;
                }

                // Step 5: Adaptive lambda adjustment
                T error_decrease = current_error - new_error;

                if (new_error < current_error) {
                    // ACCEPT step - error improved
                    if (verbose) {
                        std::cout << "  Step accepted, error decrease = " << error_decrease << std::endl;
                        std::cout << "  Decreasing λ: " << lambda << " → "
                                  << std::max(lambda / lambda_factor, min_lambda) << std::endl;
                    }

                    // Update state
                    x = x_new;
                    current_error = new_error;
                    r = r_new;

                    // Decrease lambda (approach Gauss-Newton)
                    lambda = std::max(lambda / lambda_factor, min_lambda);

                    // Check convergence on error decrease
                    if (std::abs(error_decrease) < tolerance) {
                        converged = true;
                        termination_reason = "Converged: error decrease < " + std::to_string(tolerance);
                        break;
                    }
                } else {
                    // REJECT step - error got worse
                    if (verbose) {
                        std::cout << "  Step rejected, error increase = " << -error_decrease << std::endl;
                        std::cout << "  Increasing λ: " << lambda << " → "
                                  << std::min(lambda * lambda_factor, max_lambda) << std::endl;
                    }

                    // Increase lambda (approach gradient descent)
                    lambda = std::min(lambda * lambda_factor, max_lambda);

                    // Check if lambda hit maximum
                    if (lambda >= max_lambda) {
                        termination_reason = "Failed: lambda reached maximum";
                        converged = false;
                        break;
                    }

                    // Don't update x, current_error, or r - retry with larger lambda
                }
            }

            // Check max iterations
            if (iteration >= max_iterations && !converged) {
                termination_reason = termination::MAX_ITERATIONS;
                converged = false;
            }

            // Final evaluation
            T final_error = current_error;
            auto final_r = residual_func(x);
            auto final_J = compute_jacobian(residual_func, x);
            auto final_gradient = compute_gradient<N>(final_J, final_r);
            T final_grad_norm = simd::view(final_gradient).norm();

            if (verbose) {
                std::cout << "=== Optimization Complete ===" << std::endl;
                std::cout << "Converged: " << (converged ? "YES" : "NO") << std::endl;
                std::cout << "Reason: " << termination_reason << std::endl;
                std::cout << "Iterations: " << iteration << std::endl;
                std::cout << "Final error: " << final_error << std::endl;
                std::cout << "Error reduction: " << (initial_error - final_error) << " ("
                          << (100.0 * (initial_error - final_error) / initial_error) << "%)" << std::endl;
                std::cout << "Final gradient norm: " << final_grad_norm << std::endl;
                std::cout << "Final lambda: " << lambda << std::endl;
            }

            // Callback: end
            OptimizationResult<T, N> result(x, final_error, final_grad_norm, iteration, converged, termination_reason);
            callback.on_end(result);

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
        dp::mat::Matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> compute_jacobian(ResidualFunc &residual_func,
                                                                                const dp::mat::Vector<T, N> &x) {
            if constexpr (requires { residual_func.jacobian(x); }) {
                return residual_func.jacobian(x);
            } else {
                return lina::jacobian(residual_func, x, jacobian_step_size, jacobian_central_diff);
            }
        }

        /**
         * @brief Compute squared error ||r||^2 / 2 (SIMD-optimized)
         */
        template <std::size_t M> T compute_squared_error(const dp::mat::Vector<T, M> &r) {
            // Use SIMD dot product: ||r||^2 = r · r
            T sum = simd::backend::dot_runtime<T>(r.data(), r.data(), r.size());
            return sum / T(2);
        }

        /**
         * @brief Compute gradient g = J^T * r (SIMD-optimized for column-major J)
         */
        template <std::size_t N, std::size_t M>
        dp::mat::Vector<T, N> compute_gradient(const dp::mat::Matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> &J,
                                               const dp::mat::Vector<T, M> &r) {
            const std::size_t m = J.rows();
            const std::size_t n = J.cols();

            dp::mat::Vector<T, N> g;
            if constexpr (N == dp::mat::Dynamic) {
                g.resize(n);
            }

            // For column-major matrix, column i is contiguous at J.data() + i*m
            const T *r_ptr = r.data();
            for (std::size_t i = 0; i < n; ++i) {
                const T *col_i = J.data() + i * m;
                g[i] = simd::backend::dot_runtime<T>(col_i, r_ptr, m);
            }

            return g;
        }

        /**
         * @brief Solve the damped normal equations: (J^T*J + λ*I)*dx = -J^T*r
         *
         * This is THE key difference from Gauss-Newton!
         * @return Solution vector, or std::nullopt if system is singular
         */
        template <std::size_t N, std::size_t M>
        std::optional<dp::mat::Vector<T, N>>
        solve_damped_system(const dp::mat::Matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> &J,
                            const dp::mat::Vector<T, M> &r, T lambda) {
            const std::size_t m = J.rows();
            const std::size_t n = J.cols();

            dp::mat::Vector<T, N> dx;
            if constexpr (N == dp::mat::Dynamic) {
                dx.resize(n);
            }
            simd::view(dx).fill(T(0));

            // Build J^T * J + λ*I using SIMD dot products
            dp::mat::Matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> A(n, n);
            simd::view(A).fill(T(0));

            // Compute upper triangle (symmetric) - columns are contiguous in column-major
            for (std::size_t i = 0; i < n; ++i) {
                const T *col_i = J.data() + i * m;
                for (std::size_t j = i; j < n; ++j) {
                    const T *col_j = J.data() + j * m;
                    T sum = simd::backend::dot_runtime<T>(col_i, col_j, m);
                    A(i, j) = sum;
                    if (i != j) {
                        A(j, i) = sum; // Symmetric
                    }
                }
            }

            // Add damping to diagonal: λ*I
            for (std::size_t i = 0; i < n; ++i) {
                A(i, i) += lambda;
            }

            // Build -J^T * r using SIMD dot products
            dp::mat::Vector<T, dp::mat::Dynamic> b;
            b.resize(n);
            const T *r_ptr = r.data();
            for (std::size_t i = 0; i < n; ++i) {
                const T *col_i = J.data() + i * m;
                b[i] = -simd::backend::dot_runtime<T>(col_i, r_ptr, m);
            }

            // Gaussian elimination with partial pivoting
            for (std::size_t k = 0; k < n; ++k) {
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

                // Check singularity - return nullopt instead of throwing
                if (max_val < T(1e-14)) {
                    return std::nullopt;
                }

                // Swap rows
                if (pivot_row != k) {
                    for (std::size_t j = k; j < n; ++j) {
                        std::swap(A(k, j), A(pivot_row, j));
                    }
                    std::swap(b[k], b[pivot_row]);
                }

                // Eliminate
                for (std::size_t i = k + 1; i < n; ++i) {
                    T factor = A(i, k) / A(k, k);
                    for (std::size_t j = k + 1; j < n; ++j) {
                        A(i, j) -= factor * A(k, j);
                    }
                    b[i] -= factor * b[k];
                    A(i, k) = T(0);
                }
            }

            // Back substitution
            for (std::size_t i = n; i-- > 0;) {
                T sum = b[i];
                for (std::size_t j = i + 1; j < n; ++j) {
                    sum -= A(i, j) * dx[j];
                }
                dx[i] = sum / A(i, i);
            }

            return dx;
        }
    };

} // namespace optinum::opti
