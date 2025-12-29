#pragma once

// =============================================================================
// optinum/opti/line_search/line_search.hpp
// Line search algorithms: Armijo (backtracking) and Wolfe conditions
// =============================================================================

#include <datapod/matrix/vector.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/bridge.hpp>

#include <cmath>
#include <limits>
#include <string>

namespace optinum::opti {

    namespace dp = ::datapod;

    /**
     * @brief Result of a line search operation
     *
     * Contains the step size found, number of function evaluations,
     * and whether the search was successful.
     */
    template <typename T> struct LineSearchResult {
        T alpha;                        ///< Step size found
        T function_value;               ///< Function value at x + alpha * direction
        std::size_t function_evals;     ///< Number of function evaluations
        std::size_t gradient_evals;     ///< Number of gradient evaluations
        bool success;                   ///< Whether line search succeeded
        std::string termination_reason; ///< Reason for termination

        /// Default constructor
        LineSearchResult()
            : alpha{T(1)}, function_value{}, function_evals{0}, gradient_evals{0}, success{false},
              termination_reason{} {}

        /// Construct with all fields
        LineSearchResult(T a, T fval, std::size_t fevals, std::size_t gevals, bool succ, const std::string &reason)
            : alpha{a}, function_value{fval}, function_evals{fevals}, gradient_evals{gevals}, success{succ},
              termination_reason{reason} {}
    };

    /**
     * @brief Armijo (backtracking) line search
     *
     * Finds a step size alpha satisfying the Armijo (sufficient decrease) condition:
     *   f(x + alpha * d) <= f(x) + c1 * alpha * grad(f)^T * d
     *
     * This is a simple backtracking line search that starts with an initial step
     * and reduces it by a factor until the Armijo condition is satisfied.
     *
     * **Algorithm:**
     * 1. Start with alpha = alpha_init
     * 2. While Armijo condition not satisfied:
     *    - alpha = rho * alpha (reduce step)
     * 3. Return alpha
     *
     * **Parameters:**
     * - c1: Armijo parameter (typically 1e-4), controls sufficient decrease
     * - rho: Backtracking factor (typically 0.5), step reduction rate
     * - alpha_init: Initial step size (typically 1.0)
     * - max_iters: Maximum backtracking iterations
     *
     * **Use Cases:**
     * - Simple and robust for most optimization problems
     * - Good default choice when Wolfe conditions are not needed
     * - Works well with quasi-Newton methods (L-BFGS, BFGS)
     *
     * Reference:
     *   Nocedal & Wright (2006), "Numerical Optimization", Algorithm 3.1
     */
    template <typename T = double> class ArmijoLineSearch {
      public:
        // =============================================================================
        // Configuration Parameters
        // =============================================================================

        /// Armijo parameter c1 (sufficient decrease), typically 1e-4
        T c1 = T(1e-4);

        /// Backtracking factor rho, typically 0.5
        T rho = T(0.5);

        /// Initial step size
        T alpha_init = T(1.0);

        /// Minimum step size (below this, search fails)
        T alpha_min = T(1e-12);

        /// Maximum step size
        T alpha_max = T(1e6);

        /// Maximum number of backtracking iterations
        std::size_t max_iters = 50;

        // =============================================================================
        // Public Methods
        // =============================================================================

        /// Default constructor
        ArmijoLineSearch() = default;

        /// Constructor with parameters
        ArmijoLineSearch(T c1_val, T rho_val, T alpha_init_val = T(1.0))
            : c1(c1_val), rho(rho_val), alpha_init(alpha_init_val) {}

        /**
         * @brief Perform Armijo line search
         *
         * @param func Function to minimize (must have evaluate() method)
         * @param x Current point
         * @param direction Search direction (typically -gradient or Newton direction)
         * @param f0 Function value at x (f(x))
         * @param grad0 Gradient at x
         * @return LineSearchResult with step size and diagnostics
         *
         * @note The direction should be a descent direction (grad0^T * direction < 0)
         */
        template <typename FunctionType, std::size_t N>
        LineSearchResult<T> search(FunctionType &func, const dp::mat::vector<T, N> &x,
                                   const dp::mat::vector<T, N> &direction, T f0, const dp::mat::vector<T, N> &grad0) {
            const std::size_t n = x.size();

            // Compute directional derivative: phi'(0) = grad0^T * direction
            T directional_derivative = compute_dot(grad0, direction);

            // Check if direction is a descent direction
            if (directional_derivative >= T(0)) {
                return LineSearchResult<T>(T(0), f0, 0, 0, false, "Not a descent direction");
            }

            // Armijo threshold: f(x + alpha*d) <= f(x) + c1*alpha*phi'(0)
            T armijo_slope = c1 * directional_derivative;

            T alpha = alpha_init;
            std::size_t func_evals = 0;

            // Allocate x_new once
            dp::mat::vector<T, N> x_new;
            if constexpr (N == dp::mat::Dynamic) {
                x_new.resize(n);
            }

            for (std::size_t iter = 0; iter < max_iters; ++iter) {
                // Compute x_new = x + alpha * direction
                compute_step(x, direction, alpha, x_new);

                // Evaluate function at new point
                T f_new = func.evaluate(x_new);
                ++func_evals;

                // Check Armijo condition: f(x + alpha*d) <= f(x) + c1*alpha*phi'(0)
                if (f_new <= f0 + alpha * armijo_slope) {
                    return LineSearchResult<T>(alpha, f_new, func_evals, 0, true, "Armijo condition satisfied");
                }

                // Check for NaN/Inf
                if (std::isnan(f_new) || std::isinf(f_new)) {
                    alpha *= rho; // Reduce step and try again
                    if (alpha < alpha_min) {
                        return LineSearchResult<T>(alpha, f_new, func_evals, 0, false, "Step size too small (NaN/Inf)");
                    }
                    continue;
                }

                // Reduce step size
                alpha *= rho;

                // Check minimum step size
                if (alpha < alpha_min) {
                    return LineSearchResult<T>(alpha, f_new, func_evals, 0, false, "Step size below minimum");
                }
            }

            return LineSearchResult<T>(alpha, f0, func_evals, 0, false, "Maximum iterations reached");
        }

        /**
         * @brief Perform Armijo line search with function that returns value and gradient
         *
         * More efficient when gradient is needed anyway (avoids recomputation).
         */
        template <typename FunctionType, std::size_t N>
        LineSearchResult<T> search_with_gradient(FunctionType &func, const dp::mat::vector<T, N> &x,
                                                 const dp::mat::vector<T, N> &direction, T f0,
                                                 const dp::mat::vector<T, N> &grad0, dp::mat::vector<T, N> &grad_new) {
            const std::size_t n = x.size();

            // Compute directional derivative
            T directional_derivative = compute_dot(grad0, direction);

            if (directional_derivative >= T(0)) {
                return LineSearchResult<T>(T(0), f0, 0, 0, false, "Not a descent direction");
            }

            T armijo_slope = c1 * directional_derivative;
            T alpha = alpha_init;
            std::size_t func_evals = 0;

            dp::mat::vector<T, N> x_new;
            if constexpr (N == dp::mat::Dynamic) {
                x_new.resize(n);
                grad_new.resize(n);
            }

            for (std::size_t iter = 0; iter < max_iters; ++iter) {
                compute_step(x, direction, alpha, x_new);

                T f_new = func.evaluate_with_gradient(x_new, grad_new);
                ++func_evals;

                if (f_new <= f0 + alpha * armijo_slope) {
                    return LineSearchResult<T>(alpha, f_new, func_evals, func_evals, true,
                                               "Armijo condition satisfied");
                }

                if (std::isnan(f_new) || std::isinf(f_new)) {
                    alpha *= rho;
                    if (alpha < alpha_min) {
                        return LineSearchResult<T>(alpha, f_new, func_evals, func_evals, false,
                                                   "Step size too small (NaN/Inf)");
                    }
                    continue;
                }

                alpha *= rho;

                if (alpha < alpha_min) {
                    return LineSearchResult<T>(alpha, f_new, func_evals, func_evals, false, "Step size below minimum");
                }
            }

            return LineSearchResult<T>(alpha, f0, func_evals, func_evals, false, "Maximum iterations reached");
        }

      private:
        /// Compute dot product using SIMD
        template <std::size_t N> T compute_dot(const dp::mat::vector<T, N> &a, const dp::mat::vector<T, N> &b) const {
            return simd::view(a).dot(simd::view(b));
        }

        /// Compute x_new = x + alpha * direction (SIMD optimized)
        template <std::size_t N>
        void compute_step(const dp::mat::vector<T, N> &x, const dp::mat::vector<T, N> &direction, T alpha,
                          dp::mat::vector<T, N> &x_new) const {
            simd::backend::axpy_runtime<T>(x_new.data(), x.data(), alpha, direction.data(), x.size());
        }
    };

    /**
     * @brief Strong Wolfe line search
     *
     * Finds a step size alpha satisfying the strong Wolfe conditions:
     *   1. Armijo (sufficient decrease): f(x + alpha*d) <= f(x) + c1*alpha*grad^T*d
     *   2. Curvature: |grad(x + alpha*d)^T * d| <= c2 * |grad(x)^T * d|
     *
     * The strong Wolfe conditions ensure:
     * - Sufficient decrease in function value (Armijo)
     * - Sufficient decrease in gradient magnitude (curvature)
     *
     * **Algorithm:**
     * Uses a bracketing and zoom approach:
     * 1. Bracket: Find an interval [alpha_lo, alpha_hi] containing a point satisfying Wolfe
     * 2. Zoom: Bisection/interpolation to find the exact point
     *
     * **Parameters:**
     * - c1: Armijo parameter (typically 1e-4)
     * - c2: Curvature parameter (typically 0.9 for quasi-Newton, 0.1 for CG)
     *
     * **Use Cases:**
     * - Required for L-BFGS to maintain positive definiteness
     * - Conjugate gradient methods
     * - When more precise step sizes are needed
     *
     * Reference:
     *   Nocedal & Wright (2006), "Numerical Optimization", Algorithm 3.5 & 3.6
     */
    template <typename T = double> class WolfeLineSearch {
      public:
        // =============================================================================
        // Configuration Parameters
        // =============================================================================

        /// Armijo parameter c1 (sufficient decrease), typically 1e-4
        T c1 = T(1e-4);

        /// Curvature parameter c2, typically 0.9 for quasi-Newton, 0.1 for CG
        T c2 = T(0.9);

        /// Initial step size
        T alpha_init = T(1.0);

        /// Minimum step size
        T alpha_min = T(1e-12);

        /// Maximum step size
        T alpha_max = T(1e6);

        /// Maximum number of iterations in bracketing phase
        std::size_t max_bracket_iters = 20;

        /// Maximum number of iterations in zoom phase
        std::size_t max_zoom_iters = 20;

        // =============================================================================
        // Public Methods
        // =============================================================================

        /// Default constructor
        WolfeLineSearch() = default;

        /// Constructor with parameters
        WolfeLineSearch(T c1_val, T c2_val, T alpha_init_val = T(1.0))
            : c1(c1_val), c2(c2_val), alpha_init(alpha_init_val) {}

        /**
         * @brief Perform strong Wolfe line search
         *
         * @param func Function to minimize (must have evaluate() and gradient() methods)
         * @param x Current point
         * @param direction Search direction
         * @param f0 Function value at x
         * @param grad0 Gradient at x
         * @param grad_new Output: gradient at x + alpha * direction
         * @return LineSearchResult with step size and diagnostics
         */
        template <typename FunctionType, std::size_t N>
        LineSearchResult<T> search(FunctionType &func, const dp::mat::vector<T, N> &x,
                                   const dp::mat::vector<T, N> &direction, T f0, const dp::mat::vector<T, N> &grad0,
                                   dp::mat::vector<T, N> &grad_new) {
            const std::size_t n = x.size();

            // Compute initial directional derivative: phi'(0) = grad0^T * direction
            T dphi0 = compute_dot(grad0, direction);

            // Check descent direction
            if (dphi0 >= T(0)) {
                return LineSearchResult<T>(T(0), f0, 0, 0, false, "Not a descent direction");
            }

            // Wolfe thresholds
            T armijo_slope = c1 * dphi0;
            T curvature_threshold = c2 * std::abs(dphi0);

            // Allocate working vectors
            dp::mat::vector<T, N> x_new;
            if constexpr (N == dp::mat::Dynamic) {
                x_new.resize(n);
                grad_new.resize(n);
            }

            // Bracketing phase: find interval [alpha_lo, alpha_hi] containing a Wolfe point
            T alpha_prev = T(0);
            T f_prev = f0;
            T dphi_prev = dphi0;

            T alpha = alpha_init;
            std::size_t func_evals = 0;
            std::size_t grad_evals = 0;

            for (std::size_t iter = 0; iter < max_bracket_iters; ++iter) {
                // Evaluate at current alpha
                compute_step(x, direction, alpha, x_new);
                T f_alpha = func.evaluate(x_new);
                ++func_evals;

                // Check for NaN/Inf
                if (std::isnan(f_alpha) || std::isinf(f_alpha)) {
                    // Reduce alpha and try again
                    alpha = (alpha_prev + alpha) / T(2);
                    if (alpha < alpha_min) {
                        return LineSearchResult<T>(alpha_prev, f_prev, func_evals, grad_evals, false,
                                                   "Step size too small (NaN/Inf)");
                    }
                    continue;
                }

                // Check Armijo condition
                bool armijo_satisfied = (f_alpha <= f0 + alpha * armijo_slope);

                // If Armijo violated or function increased, zoom in [alpha_prev, alpha]
                if (!armijo_satisfied || (iter > 0 && f_alpha >= f_prev)) {
                    return zoom(func, x, direction, f0, dphi0, armijo_slope, curvature_threshold, alpha_prev, alpha,
                                f_prev, f_alpha, x_new, grad_new, func_evals, grad_evals);
                }

                // Evaluate gradient at alpha
                func.gradient(x_new, grad_new);
                ++grad_evals;
                T dphi_alpha = compute_dot(grad_new, direction);

                // Check strong Wolfe curvature condition
                if (std::abs(dphi_alpha) <= curvature_threshold) {
                    // Both conditions satisfied!
                    return LineSearchResult<T>(alpha, f_alpha, func_evals, grad_evals, true,
                                               "Strong Wolfe conditions satisfied");
                }

                // If gradient is positive, zoom in [alpha, alpha_prev]
                if (dphi_alpha >= T(0)) {
                    return zoom(func, x, direction, f0, dphi0, armijo_slope, curvature_threshold, alpha, alpha_prev,
                                f_alpha, f_prev, x_new, grad_new, func_evals, grad_evals);
                }

                // Expand interval
                alpha_prev = alpha;
                f_prev = f_alpha;
                dphi_prev = dphi_alpha;

                // Increase alpha (expand search)
                alpha = std::min(alpha * T(2), alpha_max);

                if (alpha >= alpha_max) {
                    // Accept current point if Armijo is satisfied
                    if (armijo_satisfied) {
                        return LineSearchResult<T>(alpha_prev, f_prev, func_evals, grad_evals, true,
                                                   "Armijo satisfied at alpha_max");
                    }
                    return LineSearchResult<T>(alpha_prev, f_prev, func_evals, grad_evals, false,
                                               "Reached alpha_max without satisfying Wolfe");
                }
            }

            return LineSearchResult<T>(alpha, f0, func_evals, grad_evals, false, "Bracketing phase exceeded max iters");
        }

      private:
        /**
         * @brief Zoom phase: find a point satisfying Wolfe conditions in [alpha_lo, alpha_hi]
         *
         * Uses bisection with safeguards to find the optimal step size.
         */
        template <typename FunctionType, std::size_t N>
        LineSearchResult<T> zoom(FunctionType &func, const dp::mat::vector<T, N> &x,
                                 const dp::mat::vector<T, N> &direction, T f0, T dphi0, T armijo_slope,
                                 T curvature_threshold, T alpha_lo, T alpha_hi, T f_lo, [[maybe_unused]] T f_hi,
                                 dp::mat::vector<T, N> &x_new, dp::mat::vector<T, N> &grad_new, std::size_t &func_evals,
                                 std::size_t &grad_evals) {
            for (std::size_t iter = 0; iter < max_zoom_iters; ++iter) {
                // Bisection (could use quadratic interpolation for faster convergence)
                T alpha = (alpha_lo + alpha_hi) / T(2);

                // Check for convergence
                if (std::abs(alpha_hi - alpha_lo) < alpha_min) {
                    return LineSearchResult<T>(alpha_lo, f_lo, func_evals, grad_evals, true,
                                               "Zoom converged (interval too small)");
                }

                // Evaluate at alpha
                compute_step(x, direction, alpha, x_new);
                T f_alpha = func.evaluate(x_new);
                ++func_evals;

                // Check for NaN/Inf
                if (std::isnan(f_alpha) || std::isinf(f_alpha)) {
                    alpha_hi = alpha;
                    f_hi = f_alpha;
                    continue;
                }

                // Check Armijo condition
                bool armijo_satisfied = (f_alpha <= f0 + alpha * armijo_slope);

                if (!armijo_satisfied || f_alpha >= f_lo) {
                    // Armijo violated or no improvement: shrink from high side
                    alpha_hi = alpha;
                    f_hi = f_alpha;
                } else {
                    // Armijo satisfied, check curvature
                    func.gradient(x_new, grad_new);
                    ++grad_evals;
                    T dphi_alpha = compute_dot(grad_new, direction);

                    // Check strong Wolfe curvature condition
                    if (std::abs(dphi_alpha) <= curvature_threshold) {
                        return LineSearchResult<T>(alpha, f_alpha, func_evals, grad_evals, true,
                                                   "Strong Wolfe conditions satisfied (zoom)");
                    }

                    // Adjust interval based on gradient sign
                    if (dphi_alpha * (alpha_hi - alpha_lo) >= T(0)) {
                        alpha_hi = alpha_lo;
                        f_hi = f_lo;
                    }

                    alpha_lo = alpha;
                    f_lo = f_alpha;
                }
            }

            return LineSearchResult<T>(alpha_lo, f_lo, func_evals, grad_evals, true,
                                       "Zoom phase exceeded max iters (returning best)");
        }

        /// Compute dot product using SIMD
        template <std::size_t N> T compute_dot(const dp::mat::vector<T, N> &a, const dp::mat::vector<T, N> &b) const {
            return simd::view(a).dot(simd::view(b));
        }

        /// Compute x_new = x + alpha * direction (SIMD optimized)
        template <std::size_t N>
        void compute_step(const dp::mat::vector<T, N> &x, const dp::mat::vector<T, N> &direction, T alpha,
                          dp::mat::vector<T, N> &x_new) const {
            simd::backend::axpy_runtime<T>(x_new.data(), x.data(), alpha, direction.data(), x.size());
        }
    };

    /**
     * @brief Weak Wolfe line search (simpler curvature condition)
     *
     * Uses the weak Wolfe curvature condition:
     *   grad(x + alpha*d)^T * d >= c2 * grad(x)^T * d
     *
     * This is less restrictive than strong Wolfe and may be sufficient
     * for some applications.
     */
    template <typename T = double> class WeakWolfeLineSearch {
      public:
        T c1 = T(1e-4);
        T c2 = T(0.9);
        T alpha_init = T(1.0);
        T alpha_min = T(1e-12);
        T alpha_max = T(1e6);
        std::size_t max_iters = 50;

        WeakWolfeLineSearch() = default;
        WeakWolfeLineSearch(T c1_val, T c2_val) : c1(c1_val), c2(c2_val) {}

        template <typename FunctionType, std::size_t N>
        LineSearchResult<T> search(FunctionType &func, const dp::mat::vector<T, N> &x,
                                   const dp::mat::vector<T, N> &direction, T f0, const dp::mat::vector<T, N> &grad0,
                                   dp::mat::vector<T, N> &grad_new) {
            const std::size_t n = x.size();

            T dphi0 = compute_dot(grad0, direction);

            if (dphi0 >= T(0)) {
                return LineSearchResult<T>(T(0), f0, 0, 0, false, "Not a descent direction");
            }

            T armijo_slope = c1 * dphi0;
            T curvature_threshold = c2 * dphi0; // Note: dphi0 is negative

            dp::mat::vector<T, N> x_new;
            if constexpr (N == dp::mat::Dynamic) {
                x_new.resize(n);
                grad_new.resize(n);
            }

            T alpha_lo = T(0);
            T alpha_hi = alpha_max;
            T alpha = alpha_init;
            std::size_t func_evals = 0;
            std::size_t grad_evals = 0;

            for (std::size_t iter = 0; iter < max_iters; ++iter) {
                compute_step(x, direction, alpha, x_new);
                T f_alpha = func.evaluate_with_gradient(x_new, grad_new);
                ++func_evals;
                ++grad_evals;

                if (std::isnan(f_alpha) || std::isinf(f_alpha)) {
                    alpha_hi = alpha;
                    alpha = (alpha_lo + alpha_hi) / T(2);
                    if (alpha < alpha_min) {
                        return LineSearchResult<T>(alpha_lo, f0, func_evals, grad_evals, false,
                                                   "Step size too small (NaN/Inf)");
                    }
                    continue;
                }

                T dphi_alpha = compute_dot(grad_new, direction);

                // Check Armijo condition
                if (f_alpha > f0 + alpha * armijo_slope) {
                    // Armijo violated: reduce alpha
                    alpha_hi = alpha;
                    alpha = (alpha_lo + alpha_hi) / T(2);
                }
                // Check weak Wolfe curvature: dphi_alpha >= c2 * dphi0
                else if (dphi_alpha < curvature_threshold) {
                    // Curvature not satisfied: increase alpha
                    alpha_lo = alpha;
                    if (alpha_hi >= alpha_max) {
                        alpha = T(2) * alpha;
                    } else {
                        alpha = (alpha_lo + alpha_hi) / T(2);
                    }
                } else {
                    // Both conditions satisfied
                    return LineSearchResult<T>(alpha, f_alpha, func_evals, grad_evals, true,
                                               "Weak Wolfe conditions satisfied");
                }

                if (alpha < alpha_min || std::abs(alpha_hi - alpha_lo) < alpha_min) {
                    return LineSearchResult<T>(alpha, f_alpha, func_evals, grad_evals, true,
                                               "Converged (interval too small)");
                }
            }

            return LineSearchResult<T>(alpha, f0, func_evals, grad_evals, false, "Maximum iterations reached");
        }

      private:
        template <std::size_t N> T compute_dot(const dp::mat::vector<T, N> &a, const dp::mat::vector<T, N> &b) const {
            return simd::view(a).dot(simd::view(b));
        }

        /// Compute x_new = x + alpha * direction (SIMD optimized)
        template <std::size_t N>
        void compute_step(const dp::mat::vector<T, N> &x, const dp::mat::vector<T, N> &direction, T alpha,
                          dp::mat::vector<T, N> &x_new) const {
            simd::backend::axpy_runtime<T>(x_new.data(), x.data(), alpha, direction.data(), x.size());
        }
    };

    /**
     * @brief Goldstein line search
     *
     * Finds a step size satisfying the Goldstein conditions:
     *   f(x) + (1 - c) * alpha * grad^T * d <= f(x + alpha * d) <= f(x) + c * alpha * grad^T * d
     *
     * This provides both upper and lower bounds on the function decrease.
     * Less commonly used than Wolfe conditions but can be useful in some cases.
     *
     * @param c Goldstein parameter, typically 0.25 (must be in (0, 0.5))
     */
    template <typename T = double> class GoldsteinLineSearch {
      public:
        T c = T(0.25);
        T alpha_init = T(1.0);
        T alpha_min = T(1e-12);
        T alpha_max = T(1e6);
        std::size_t max_iters = 50;

        GoldsteinLineSearch() = default;
        explicit GoldsteinLineSearch(T c_val) : c(c_val) {}

        template <typename FunctionType, std::size_t N>
        LineSearchResult<T> search(FunctionType &func, const dp::mat::vector<T, N> &x,
                                   const dp::mat::vector<T, N> &direction, T f0, const dp::mat::vector<T, N> &grad0) {
            const std::size_t n = x.size();

            T dphi0 = compute_dot(grad0, direction);

            if (dphi0 >= T(0)) {
                return LineSearchResult<T>(T(0), f0, 0, 0, false, "Not a descent direction");
            }

            // Goldstein bounds
            T lower_slope = (T(1) - c) * dphi0;
            T upper_slope = c * dphi0;

            dp::mat::vector<T, N> x_new;
            if constexpr (N == dp::mat::Dynamic) {
                x_new.resize(n);
            }

            T alpha_lo = T(0);
            T alpha_hi = alpha_max;
            T alpha = alpha_init;
            std::size_t func_evals = 0;

            for (std::size_t iter = 0; iter < max_iters; ++iter) {
                compute_step(x, direction, alpha, x_new);
                T f_alpha = func.evaluate(x_new);
                ++func_evals;

                if (std::isnan(f_alpha) || std::isinf(f_alpha)) {
                    alpha_hi = alpha;
                    alpha = (alpha_lo + alpha_hi) / T(2);
                    if (alpha < alpha_min) {
                        return LineSearchResult<T>(alpha_lo, f0, func_evals, 0, false, "Step size too small (NaN/Inf)");
                    }
                    continue;
                }

                // Check upper bound (Armijo-like)
                if (f_alpha > f0 + alpha * upper_slope) {
                    alpha_hi = alpha;
                    alpha = (alpha_lo + alpha_hi) / T(2);
                }
                // Check lower bound
                else if (f_alpha < f0 + alpha * lower_slope) {
                    alpha_lo = alpha;
                    if (alpha_hi >= alpha_max) {
                        alpha = T(2) * alpha;
                    } else {
                        alpha = (alpha_lo + alpha_hi) / T(2);
                    }
                } else {
                    // Both conditions satisfied
                    return LineSearchResult<T>(alpha, f_alpha, func_evals, 0, true, "Goldstein conditions satisfied");
                }

                if (alpha < alpha_min || std::abs(alpha_hi - alpha_lo) < alpha_min) {
                    return LineSearchResult<T>(alpha, f_alpha, func_evals, 0, true, "Converged (interval too small)");
                }
            }

            return LineSearchResult<T>(alpha, f0, func_evals, 0, false, "Maximum iterations reached");
        }

      private:
        template <std::size_t N> T compute_dot(const dp::mat::vector<T, N> &a, const dp::mat::vector<T, N> &b) const {
            return simd::view(a).dot(simd::view(b));
        }

        /// Compute x_new = x + alpha * direction (SIMD optimized)
        template <std::size_t N>
        void compute_step(const dp::mat::vector<T, N> &x, const dp::mat::vector<T, N> &direction, T alpha,
                          dp::mat::vector<T, N> &x_new) const {
            simd::backend::axpy_runtime<T>(x_new.data(), x.data(), alpha, direction.data(), x.size());
        }
    };

    // =============================================================================
    // Type aliases for convenience
    // =============================================================================

    using ArmijoLineSearchf = ArmijoLineSearch<float>;
    using ArmijoLineSearchd = ArmijoLineSearch<double>;

    using WolfeLineSearchf = WolfeLineSearch<float>;
    using WolfeLineSearchd = WolfeLineSearch<double>;

    using WeakWolfeLineSearchf = WeakWolfeLineSearch<float>;
    using WeakWolfeLineSearchd = WeakWolfeLineSearch<double>;

    using GoldsteinLineSearchf = GoldsteinLineSearch<float>;
    using GoldsteinLineSearchd = GoldsteinLineSearch<double>;

} // namespace optinum::opti
