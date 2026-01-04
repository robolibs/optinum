#pragma once

#include <datapod/datapod.hpp>

namespace optinum::opti {

    /**
     * Rosenbrock function - classic "banana" benchmark for optimization
     *
     * f(x) = sum_{i=0}^{n-2} [ 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]
     *
     * Global minimum: f(1, 1, ..., 1) = 0
     *
     * Properties:
     * - Non-convex with a narrow curved valley
     * - Easy to find the valley, hard to converge to the minimum
     * - Tests optimizer's ability to navigate ill-conditioned landscapes
     *
     * Typical search domain: x_i in [-5, 10] or [-2.048, 2.048]
     *
     * Reference: Rosenbrock, H.H. (1960)
     *   "An automatic method for finding the greatest or least value of a function"
     *
     * Supports both fixed-size and dynamic vectors:
     *   Rosenbrock<double, 10>      - Fixed size
     *   Rosenbrock<double, Dynamic> - Dynamic size
     */
    template <typename T, std::size_t N> struct Rosenbrock {
        using tensor_type = dp::mat::Vector<T, N>;

        static constexpr T A = T{100}; ///< Scaling factor for the quadratic term

        /// Evaluate f(x) = sum [ 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]
        T evaluate(const tensor_type &x) const noexcept {
            const std::size_t n = x.size();
            if (n < 2)
                return T{0};

            T sum{0};
            for (std::size_t i = 0; i < n - 1; ++i) {
                T xi = x[i];
                T xi1 = x[i + 1];
                T t1 = xi1 - xi * xi; // (x_{i+1} - x_i^2)
                T t2 = T{1} - xi;     // (1 - x_i)
                sum += A * t1 * t1 + t2 * t2;
            }
            return sum;
        }

        /// Compute gradient
        /// g_0 = -400*x_0*(x_1 - x_0^2) - 2*(1 - x_0)
        /// g_i = 200*(x_i - x_{i-1}^2) - 400*x_i*(x_{i+1} - x_i^2) - 2*(1 - x_i)  for 0 < i < n-1
        /// g_{n-1} = 200*(x_{n-1} - x_{n-2}^2)
        void gradient(const tensor_type &x, tensor_type &g) const noexcept {
            const std::size_t n = x.size();
            if (n < 2) {
                for (std::size_t i = 0; i < n; ++i)
                    g[i] = T{0};
                return;
            }

            // Initialize gradient to zero
            for (std::size_t i = 0; i < n; ++i) {
                g[i] = T{0};
            }

            for (std::size_t i = 0; i < n - 1; ++i) {
                T xi = x[i];
                T xi1 = x[i + 1];
                T t1 = xi1 - xi * xi; // (x_{i+1} - x_i^2)

                // Contribution to g[i]: d/dx_i of [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
                // = 100 * 2 * (x_{i+1} - x_i^2) * (-2*x_i) + 2*(1 - x_i)*(-1)
                // = -400 * x_i * (x_{i+1} - x_i^2) - 2*(1 - x_i)
                g[i] += T{-400} * xi * t1 - T{2} * (T{1} - xi);

                // Contribution to g[i+1]: d/dx_{i+1} of [100*(x_{i+1} - x_i^2)^2]
                // = 100 * 2 * (x_{i+1} - x_i^2) * 1
                // = 200 * (x_{i+1} - x_i^2)
                g[i + 1] += T{200} * t1;
            }
        }

        /// Evaluate and compute gradient in one pass
        T evaluate_with_gradient(const tensor_type &x, tensor_type &g) const noexcept {
            const std::size_t n = x.size();
            if (n < 2) {
                for (std::size_t i = 0; i < n; ++i)
                    g[i] = T{0};
                return T{0};
            }

            // Initialize gradient to zero
            for (std::size_t i = 0; i < n; ++i) {
                g[i] = T{0};
            }

            T sum{0};
            for (std::size_t i = 0; i < n - 1; ++i) {
                T xi = x[i];
                T xi1 = x[i + 1];
                T t1 = xi1 - xi * xi;
                T t2 = T{1} - xi;

                // Function value
                sum += A * t1 * t1 + t2 * t2;

                // Gradient contributions
                g[i] += T{-400} * xi * t1 - T{2} * t2;
                g[i + 1] += T{200} * t1;
            }
            return sum;
        }

        /// Get the global minimum location (all ones)
        static tensor_type minimum_location(std::size_t dim = N) {
            tensor_type x;
            if constexpr (N == dp::mat::Dynamic) {
                x.resize(dim);
            }
            for (std::size_t i = 0; i < x.size(); ++i) {
                x[i] = T{1};
            }
            return x;
        }

        /// Get the global minimum value (zero)
        static constexpr T minimum_value() noexcept { return T{0}; }
    };

} // namespace optinum::opti
