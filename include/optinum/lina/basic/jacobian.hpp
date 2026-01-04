#pragma once

// =============================================================================
// optinum/lina/basic/jacobian.hpp
// Finite-difference Jacobian and gradient computation
// =============================================================================

#include <datapod/datapod.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>
#include <type_traits>

namespace optinum::lina {

    /**
     * @brief Compute Jacobian matrix using finite differences
     *
     * For function f: R^n -> R^m, computes the m×n Jacobian matrix:
     *   J[i,j] = ∂f_i/∂x_j
     *
     * Uses either forward differences:
     *   J[i,j] ≈ (f_i(x + h·e_j) - f_i(x)) / h
     *
     * Or central differences (more accurate, 2x cost):
     *   J[i,j] ≈ (f_i(x + h·e_j) - f_i(x - h·e_j)) / (2h)
     *
     * @param f Function to differentiate (f: R^n -> R^m)
     *          Must accept simd::Vector<T,N> and return simd::Vector<T,M>
     * @param x Point at which to evaluate Jacobian
     * @param h Step size (default: 1e-8 for double, 1e-5 for float)
     * @param central If true, use central differences (more accurate but 2x cost)
     * @return Jacobian matrix (m × n)
     *
     * @example
     * // Function f(x,y) = [x^2 + y, x*y]
     * auto f = [](const auto& x) {
     *     simd::Vector<double, 2> result;
     *     result[0] = x[0]*x[0] + x[1];
     *     result[1] = x[0]*x[1];
     *     return result;
     * };
     * simd::Vector<double, 2> x = {1.0, 2.0};
     * auto J = lina::jacobian(f, x);  // J = [[2x, 1], [y, x]] at (1,2)
     */
    template <typename Function, typename T, std::size_t N>
    dp::mat::Matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>
    jacobian(const Function &f, const simd::Vector<T, N> &x, T h = (std::is_same_v<T, float> ? T(1e-5) : T(1e-8)),
             bool central = true) {
        const std::size_t n = x.size();

        // Copy input to owning storage
        dp::mat::Vector<T, N> x_base;
        for (std::size_t i = 0; i < n; ++i) {
            x_base[i] = x[i];
        }

        // Evaluate at x to get output dimension
        auto fx = f(simd::Vector<T, N>(x_base));
        const std::size_t m = fx.size();

        // Allocate Jacobian matrix (always fully Dynamic, owning)
        dp::mat::Matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> J(m, n);

        // Temporary vectors for perturbations (owning types)
        dp::mat::Vector<T, N> x_plus = x_base;
        dp::mat::Vector<T, N> x_minus = x_base;

        // Compute each column of Jacobian (one per variable)
        for (std::size_t j = 0; j < n; ++j) {
            if (central) {
                // Central difference: (f(x+h) - f(x-h)) / (2h)
                x_plus[j] = x_base[j] + h;
                x_minus[j] = x_base[j] - h;

                auto f_plus = f(simd::Vector<T, N>(x_plus));
                auto f_minus = f(simd::Vector<T, N>(x_minus));

                for (std::size_t i = 0; i < m; ++i) {
                    J(i, j) = (f_plus[i] - f_minus[i]) / (T(2) * h);
                }

                // Reset for next iteration
                x_plus[j] = x_base[j];
                x_minus[j] = x_base[j];
            } else {
                // Forward difference: (f(x+h) - f(x)) / h
                x_plus[j] = x_base[j] + h;

                auto f_plus = f(simd::Vector<T, N>(x_plus));

                for (std::size_t i = 0; i < m; ++i) {
                    J(i, j) = (f_plus[i] - fx[i]) / h;
                }

                // Reset for next iteration
                x_plus[j] = x_base[j];
            }
        }

        return J;
    }

    /**
     * @brief Compute gradient using finite differences (optimized for scalar functions)
     *
     * For scalar function f: R^n -> R, computes the gradient vector:
     *   ∇f = [∂f/∂x_1, ∂f/∂x_2, ..., ∂f/∂x_n]
     *
     * This is more efficient than jacobian() for scalar functions since it
     * returns a vector instead of a 1×n matrix and avoids the extra dimension.
     *
     * @param f Scalar function to differentiate (f: R^n -> R)
     *          Must accept simd::Vector<T,N> and return T
     * @param x Point at which to evaluate gradient
     * @param h Step size (default: 1e-8 for double, 1e-5 for float)
     * @param central If true, use central differences (more accurate but 2x cost)
     * @return Gradient vector (n-dimensional)
     *
     * @example
     * // Function f(x,y) = x^2 + y^2 (sphere)
     * auto f = [](const auto& x) {
     *     return x[0]*x[0] + x[1]*x[1];
     * };
     * simd::Vector<double, 2> x = {3.0, 4.0};
     * auto grad = lina::gradient(f, x);  // grad = [2x, 2y] = [6.0, 8.0]
     */
    template <typename Function, typename T, std::size_t N>
    dp::mat::Vector<T, N> gradient(const Function &f, const simd::Vector<T, N> &x,
                                   T h = (std::is_same_v<T, float> ? T(1e-5) : T(1e-8)), bool central = true) {
        const std::size_t n = x.size();

        // Copy input to owning storage
        dp::mat::Vector<T, N> x_base;
        for (std::size_t i = 0; i < n; ++i) {
            x_base[i] = x[i];
        }

        dp::mat::Vector<T, N> grad;

        T fx = f(simd::Vector<T, N>(x_base));

        // Temporary vectors for perturbations (owning types)
        dp::mat::Vector<T, N> x_plus = x_base;
        dp::mat::Vector<T, N> x_minus = x_base;

        // Compute each component of gradient
        for (std::size_t j = 0; j < n; ++j) {
            if (central) {
                // Central difference: (f(x+h) - f(x-h)) / (2h)
                x_plus[j] = x_base[j] + h;
                x_minus[j] = x_base[j] - h;

                grad[j] = (f(simd::Vector<T, N>(x_plus)) - f(simd::Vector<T, N>(x_minus))) / (T(2) * h);

                // Reset for next iteration
                x_plus[j] = x_base[j];
                x_minus[j] = x_base[j];
            } else {
                // Forward difference: (f(x+h) - f(x)) / h
                x_plus[j] = x_base[j] + h;

                grad[j] = (f(simd::Vector<T, N>(x_plus)) - fx) / h;

                // Reset for next iteration
                x_plus[j] = x_base[j];
            }
        }

        return grad;
    }

    /**
     * @brief Check if Jacobian computation is accurate
     *
     * Compares finite-difference Jacobian against analytical Jacobian.
     * Returns maximum relative error.
     *
     * @param J_numerical Numerically computed Jacobian
     * @param J_analytical Analytically computed Jacobian
     * @param tol Tolerance for comparison (default: 1e-6)
     * @return Maximum relative error
     *
     * @example
     * auto J_num = lina::jacobian(f, x);
     * auto J_ana = analytical_jacobian(x);
     * double error = lina::jacobian_error(J_num, J_ana);
     * if (error < 1e-6) { // Jacobian is accurate
     * }
     */
    template <typename T, std::size_t R, std::size_t C>
    T jacobian_error(const simd::Matrix<T, R, C> &J_numerical, const simd::Matrix<T, R, C> &J_analytical,
                     T tol = T(1e-10)) {
        T max_error = T(0);
        const std::size_t m = J_numerical.rows();
        const std::size_t n = J_numerical.cols();

        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                T num = J_numerical(i, j);
                T ana = J_analytical(i, j);
                T abs_error = std::abs(num - ana);

                // Relative error (avoid division by zero)
                T rel_error = abs_error;
                if (std::abs(ana) > tol) {
                    rel_error = abs_error / std::abs(ana);
                }

                max_error = std::max(max_error, rel_error);
            }
        }

        return max_error;
    }

    // =============================================================================
    // Overloads for dp::mat::vector (used by opti module)
    // =============================================================================

    /**
     * @brief Compute Jacobian matrix using finite differences (dp::mat::vector version)
     */
    template <typename Function, typename T, std::size_t N>
    dp::mat::Matrix<T, dp::mat::Dynamic, dp::mat::Dynamic>
    jacobian(const Function &f, const dp::mat::Vector<T, N> &x, T h = (std::is_same_v<T, float> ? T(1e-5) : T(1e-8)),
             bool central = true) {
        const std::size_t n = x.size();

        // Evaluate at x to get output dimension
        auto fx = f(x);
        const std::size_t m = fx.size();

        // Allocate Jacobian matrix
        dp::mat::Matrix<T, dp::mat::Dynamic, dp::mat::Dynamic> J(m, n);

        // Temporary vectors for perturbations
        dp::mat::Vector<T, N> x_plus = x;
        dp::mat::Vector<T, N> x_minus = x;

        // Compute each column of Jacobian (one per variable)
        for (std::size_t j = 0; j < n; ++j) {
            if (central) {
                // Central difference: (f(x+h) - f(x-h)) / (2h)
                x_plus[j] = x[j] + h;
                x_minus[j] = x[j] - h;

                auto f_plus = f(x_plus);
                auto f_minus = f(x_minus);

                for (std::size_t i = 0; i < m; ++i) {
                    J(i, j) = (f_plus[i] - f_minus[i]) / (T(2) * h);
                }

                // Reset for next iteration
                x_plus[j] = x[j];
                x_minus[j] = x[j];
            } else {
                // Forward difference: (f(x+h) - f(x)) / h
                x_plus[j] = x[j] + h;

                auto f_plus = f(x_plus);

                for (std::size_t i = 0; i < m; ++i) {
                    J(i, j) = (f_plus[i] - fx[i]) / h;
                }

                // Reset for next iteration
                x_plus[j] = x[j];
            }
        }

        return J;
    }

    /**
     * @brief Compute gradient using finite differences (dp::mat::vector version)
     */
    template <typename Function, typename T, std::size_t N>
    dp::mat::Vector<T, N> gradient(const Function &f, const dp::mat::Vector<T, N> &x,
                                   T h = (std::is_same_v<T, float> ? T(1e-5) : T(1e-8)), bool central = true) {
        const std::size_t n = x.size();

        dp::mat::Vector<T, N> grad;

        T fx = f(x);
        dp::mat::Vector<T, N> x_plus = x;
        dp::mat::Vector<T, N> x_minus = x;

        // Compute each component of gradient
        for (std::size_t j = 0; j < n; ++j) {
            if (central) {
                // Central difference: (f(x+h) - f(x-h)) / (2h)
                x_plus[j] = x[j] + h;
                x_minus[j] = x[j] - h;

                grad[j] = (f(x_plus) - f(x_minus)) / (T(2) * h);

                // Reset for next iteration
                x_plus[j] = x[j];
                x_minus[j] = x[j];
            } else {
                // Forward difference: (f(x+h) - f(x)) / h
                x_plus[j] = x[j] + h;

                grad[j] = (f(x_plus) - fx) / h;

                // Reset for next iteration
                x_plus[j] = x[j];
            }
        }

        return grad;
    }

} // namespace optinum::lina
