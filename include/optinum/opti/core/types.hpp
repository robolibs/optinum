#pragma once

#include <cstddef>
#include <string>

#include <optinum/simd/vector.hpp>

namespace optinum::opti {

    /**
     * Result of an optimization procedure
     *
     * Contains the final solution, objective value, iteration count,
     * convergence status, and termination reason.
     */
    template <typename T, std::size_t N> struct OptimizationResult {
        using vector_type = simd::Vector<T, N>;

        vector_type x;                  ///< Final solution vector
        T final_cost;                   ///< Final objective function value
        T gradient_norm;                ///< Final gradient norm
        std::size_t iterations;         ///< Number of iterations performed
        bool converged;                 ///< Whether convergence criteria were met
        std::string termination_reason; ///< Reason for termination

        /// Default constructor
        OptimizationResult()
            : x{}, final_cost{}, gradient_norm{}, iterations{0}, converged{false}, termination_reason{} {}

        /// Construct with solution, cost, and gradient norm
        OptimizationResult(const vector_type &x_final, T cost, T grad_norm, std::size_t iters, bool conv,
                           const std::string &reason)
            : x{x_final}, final_cost{cost}, gradient_norm{grad_norm}, iterations{iters}, converged{conv},
              termination_reason{reason} {}
    };

    /**
     * Information about a single optimization iteration
     *
     * Used for callbacks, monitoring, and debugging.
     */
    template <typename T> struct IterationInfo {
        std::size_t iteration; ///< Current iteration number (0-based)
        T objective;           ///< Current objective function value
        T gradient_norm;       ///< L2 norm of gradient
        T step_size;           ///< Current step size (learning rate)

        /// Default constructor
        IterationInfo() : iteration{0}, objective{}, gradient_norm{}, step_size{} {}

        /// Construct with all fields
        IterationInfo(std::size_t iter, T obj, T grad_norm, T step)
            : iteration{iter}, objective{obj}, gradient_norm{grad_norm}, step_size{step} {}
    };

    /**
     * Termination reasons for optimization
     */
    namespace termination {
        inline const char *CONVERGED = "Converged: change in objective below tolerance";
        inline const char *MAX_ITERATIONS = "Maximum iterations reached";
        inline const char *NAN_INF = "Encountered NaN or Inf in objective";
        inline const char *GRADIENT_CONVERGED = "Gradient norm below tolerance";
        inline const char *CALLBACK_STOP = "Stopped by callback";
        inline const char *USER_STOP = "Stopped by user";
    } // namespace termination

} // namespace optinum::opti
