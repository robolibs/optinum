#pragma once

#include <cmath>

#include <datapod/datapod.hpp>

namespace optinum::opti {

    /**
     * Ackley function - multimodal benchmark with a nearly flat outer region
     *
     * f(x) = -a * exp(-b * sqrt(1/n * sum(x_i^2)))
     *        - exp(1/n * sum(cos(c * x_i)))
     *        + a + exp(1)
     *
     * where a = 20, b = 0.2, c = 2*pi (default parameters)
     *
     * Global minimum: f(0, 0, ..., 0) = 0
     *
     * Properties:
     * - Nearly flat outer region with a large hole at the center
     * - Many local minima surrounding the global minimum
     * - Tests optimizer's ability to escape flat regions and find the global basin
     * - Gradient becomes very small far from the origin
     *
     * Typical search domain: x_i in [-32.768, 32.768] or [-5, 5]
     *
     * Reference: Ackley, D.H. (1987)
     *   "A connectionist machine for genetic hillclimbing"
     *
     * Supports both fixed-size and dynamic vectors:
     *   Ackley<double, 10>      - Fixed size
     *   Ackley<double, Dynamic> - Dynamic size
     */
    template <typename T, std::size_t N> struct Ackley {
        using tensor_type = dp::mat::Vector<T, N>;

        static constexpr T A = T{20};                         ///< First exponential scaling
        static constexpr T B = T{0.2};                        ///< Decay rate for quadratic term
        static constexpr T C = T{2} * T{3.14159265358979323}; ///< Frequency of cosine term (2*pi)
        static constexpr T E = T{2.718281828459045};          ///< Euler's number

        /// Evaluate f(x)
        T evaluate(const tensor_type &x) const noexcept {
            const std::size_t n = x.size();
            if (n == 0)
                return T{0};

            T sum_sq{0};
            T sum_cos{0};
            for (std::size_t i = 0; i < n; ++i) {
                T xi = x[i];
                sum_sq += xi * xi;
                sum_cos += std::cos(C * xi);
            }

            T inv_n = T{1} / T(n);
            T term1 = -A * std::exp(-B * std::sqrt(inv_n * sum_sq));
            T term2 = -std::exp(inv_n * sum_cos);

            return term1 + term2 + A + E;
        }

        /// Compute gradient
        /// g_i = (a*b*x_i / sqrt(n * sum_sq)) * exp(-b * sqrt(sum_sq/n))
        ///     + (c/n) * sin(c*x_i) * exp(sum_cos/n)
        void gradient(const tensor_type &x, tensor_type &g) const noexcept {
            const std::size_t n = x.size();
            if (n == 0)
                return;

            T sum_sq{0};
            T sum_cos{0};
            for (std::size_t i = 0; i < n; ++i) {
                T xi = x[i];
                sum_sq += xi * xi;
                sum_cos += std::cos(C * xi);
            }

            T inv_n = T{1} / T(n);
            T sqrt_mean_sq = std::sqrt(inv_n * sum_sq);
            T exp1 = std::exp(-B * sqrt_mean_sq);
            T exp2 = std::exp(inv_n * sum_cos);

            // Avoid division by zero when at origin
            T factor1 = (sqrt_mean_sq > T{1e-12}) ? (A * B * exp1 / (T(n) * sqrt_mean_sq)) : T{0};
            T factor2 = C * inv_n * exp2;

            for (std::size_t i = 0; i < n; ++i) {
                T xi = x[i];
                g[i] = factor1 * xi + factor2 * std::sin(C * xi);
            }
        }

        /// Evaluate and compute gradient in one pass
        T evaluate_with_gradient(const tensor_type &x, tensor_type &g) const noexcept {
            const std::size_t n = x.size();
            if (n == 0) {
                return T{0};
            }

            T sum_sq{0};
            T sum_cos{0};
            for (std::size_t i = 0; i < n; ++i) {
                T xi = x[i];
                sum_sq += xi * xi;
                sum_cos += std::cos(C * xi);
            }

            T inv_n = T{1} / T(n);
            T sqrt_mean_sq = std::sqrt(inv_n * sum_sq);
            T exp1 = std::exp(-B * sqrt_mean_sq);
            T exp2 = std::exp(inv_n * sum_cos);

            // Function value
            T f = -A * exp1 - exp2 + A + E;

            // Gradient factors
            T factor1 = (sqrt_mean_sq > T{1e-12}) ? (A * B * exp1 / (T(n) * sqrt_mean_sq)) : T{0};
            T factor2 = C * inv_n * exp2;

            for (std::size_t i = 0; i < n; ++i) {
                T xi = x[i];
                g[i] = factor1 * xi + factor2 * std::sin(C * xi);
            }

            return f;
        }

        /// Get the global minimum location (all zeros)
        static tensor_type minimum_location(std::size_t dim = N) {
            tensor_type x;
            if constexpr (N == dp::mat::Dynamic) {
                x.resize(dim);
            }
            for (std::size_t i = 0; i < x.size(); ++i) {
                x[i] = T{0};
            }
            return x;
        }

        /// Get the global minimum value (zero)
        static constexpr T minimum_value() noexcept { return T{0}; }
    };

} // namespace optinum::opti
