#pragma once

#include <cmath>
#include <limits>

#include <datapod/datapod.hpp>
#include <optinum/opti/core/callbacks.hpp>
#include <optinum/opti/core/function.hpp>
#include <optinum/opti/core/types.hpp>
#include <optinum/opti/decay/no_decay.hpp>
#include <optinum/opti/gradient/update_policies/vanilla_update.hpp>
#include <optinum/simd/bridge.hpp>

namespace optinum::opti {

    /**
     * Gradient Descent optimizer
     *
     * Minimizes a function using gradient descent with configurable:
     * - Update policy (how to apply gradients)
     * - Decay policy (how to adjust step size)
     * - Callbacks (monitoring, early stopping)
     *
     * Basic usage:
     * @code
     * // Define objective function
     * Sphere<double, 3> func;
     *
     * // Create optimizer
     * GradientDescent<> gd;
     * gd.step_size = 0.01;
     * gd.max_iterations = 1000;
     * gd.tolerance = 1e-6;
     *
     * // Initial point
     * dp::mat::Vector<double, 3> x0{1.0, 2.0, 3.0};
     *
     * // Optimize
     * auto result = gd.optimize(func, x0);
     * @endcode
     *
     * @tparam UpdatePolicy Policy for applying gradient (default: VanillaUpdate)
     * @tparam DecayPolicy Policy for adjusting step size (default: NoDecay)
     */
    template <typename UpdatePolicy = VanillaUpdate, typename DecayPolicy = NoDecay> class GradientDescent {
      public:
        // Configuration parameters
        double step_size = 0.01;             ///< Learning rate (α)
        std::size_t max_iterations = 100000; ///< Maximum number of iterations
        double tolerance = 1e-5;             ///< Convergence tolerance
        bool reset_policy = true;            ///< Reset update policy on each optimize() call

        /// Default constructor
        GradientDescent() = default;

        /// Constructor with parameters
        GradientDescent(double step, std::size_t max_iters, double tol, bool reset = true)
            : step_size(step), max_iterations(max_iters), tolerance(tol), reset_policy(reset) {}

        /**
         * Optimize a function starting from initial point
         *
         * @param function Objective function with evaluate/gradient methods
         * @param x_init Initial point (will be modified to contain solution)
         * @param callback Optional callback for monitoring
         * @return OptimizationResult with solution and convergence info
         */
        template <typename FunctionType, typename T, std::size_t N, typename CallbackType = NoCallback>
        OptimizationResult<T, N> optimize(FunctionType &function, dp::mat::Vector<T, N> &x_init,
                                          CallbackType callback = NoCallback{}) {
            using vector_type = dp::mat::Vector<T, N>;

            // Working variables
            vector_type x = x_init; // Current iterate
            vector_type gradient;   // Gradient storage

            // For Dynamic vectors, allocate gradient with same size as x
            if constexpr (N == dp::mat::Dynamic) {
                gradient.resize(x.size());
            }

            T current_step = T(step_size); // Current step size
            T last_objective = std::numeric_limits<T>::max();
            T current_objective{};

            // Initialize policies
            if (reset_policy) {
                update_policy.reset();
                decay_policy.reset();
            }
            // Use runtime size for initialization (handles both fixed and dynamic)
            update_policy.template initialize<T, N>(x.size());
            decay_policy.initialize();

            // Callback: begin optimization
            callback.on_begin(x);

            std::size_t iteration = 0;
            bool converged = false;
            std::string termination_reason;

            // Main optimization loop
            for (; iteration < max_iterations; ++iteration) {
                // Evaluate function and gradient
                current_objective = function.evaluate_with_gradient(x, gradient);

                // Compute gradient norm for monitoring
                T grad_norm = simd::view(gradient).norm();

                // Create iteration info for callback
                IterationInfo<T> info(iteration, current_objective, grad_norm, current_step);

                // Callback: check if should stop
                bool should_stop = callback.on_iteration(info, x);
                if (should_stop) {
                    converged = true;
                    termination_reason = termination::CALLBACK_STOP;
                    break;
                }

                // Check for NaN/Inf
                if (std::isnan(current_objective) || std::isinf(current_objective)) {
                    converged = false;
                    termination_reason = termination::NAN_INF;
                    break;
                }

                // Check convergence: change in objective
                if (std::abs(last_objective - current_objective) < tolerance) {
                    converged = true;
                    termination_reason = termination::CONVERGED;
                    break;
                }

                // Update iterate using policy
                update_policy.update(x, current_step, gradient);

                // Update step size using decay policy
                decay_policy.update(current_step, iteration);

                // Update for next iteration
                last_objective = current_objective;
            }

            // Check if stopped due to max iterations
            if (iteration >= max_iterations && !converged) {
                termination_reason = termination::MAX_ITERATIONS;
                converged = false;
            }

            // Compute final gradient for result
            T final_grad_norm = (iteration > 0) ? simd::view(gradient).norm() : std::numeric_limits<T>::max();

            // Create result
            OptimizationResult<T, N> result(x, current_objective, final_grad_norm, iteration, converged,
                                            termination_reason);

            // Callback: end optimization
            callback.on_end(result);

            // Update input with solution
            x_init = x;

            return result;
        }

        /**
         * Optimize with default-constructed function (for stateless functions)
         */
        template <typename FunctionType, typename T, std::size_t N, typename CallbackType = NoCallback>
        OptimizationResult<T, N> optimize(dp::mat::Vector<T, N> &x_init, CallbackType callback = NoCallback{}) {
            FunctionType function;
            return optimize(function, x_init, callback);
        }

        /// Get reference to update policy
        UpdatePolicy &get_update_policy() { return update_policy; }
        const UpdatePolicy &get_update_policy() const { return update_policy; }

        /// Get reference to decay policy
        DecayPolicy &get_decay_policy() { return decay_policy; }
        const DecayPolicy &get_decay_policy() const { return decay_policy; }

      private:
        UpdatePolicy update_policy; ///< Policy for applying gradient updates
        DecayPolicy decay_policy;   ///< Policy for adjusting step size
    };

} // namespace optinum::opti
