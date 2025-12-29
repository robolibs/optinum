#pragma once

#include <cmath>

#include <datapod/matrix/vector.hpp>

namespace optinum::opti {

    namespace dp = ::datapod;

    /**
     * Rastrigin function - highly multimodal benchmark for optimization
     *
     * f(x) = A*n + sum_{i=0}^{n-1} [ x_i^2 - A*cos(2*pi*x_i) ]
     *
     * where A = 10 (default)
     *
     * Global minimum: f(0, 0, ..., 0) = 0
     *
     * Properties:
     * - Highly multimodal with many local minima arranged in a regular lattice
     * - Local minima are located at integer coordinates
     * - Tests optimizer's ability to escape local minima (global optimization)
     * - Number of local minima grows exponentially with dimension
     *
     * Typical search domain: x_i in [-5.12, 5.12]
     *
     * Reference: Rastrigin, L.A. (1974)
     *   "Systems of extremal control"
     *
     * Supports both fixed-size and dynamic vectors:
     *   Rastrigin<double, 10>      - Fixed size
     *   Rastrigin<double, Dynamic> - Dynamic size
     */
    template <typename T, std::size_t N> struct Rastrigin {
        using tensor_type = dp::mat::vector<T, N>;

        static constexpr T A = T{10};                              ///< Amplitude of cosine modulation
        static constexpr T TWO_PI = T{2} * T{3.14159265358979323}; ///< 2*pi constant

        /// Evaluate f(x) = A*n + sum [ x_i^2 - A*cos(2*pi*x_i) ]
        T evaluate(const tensor_type &x) const noexcept {
            const std::size_t n = x.size();
            T sum = A * T(n);
            for (std::size_t i = 0; i < n; ++i) {
                T xi = x[i];
                sum += xi * xi - A * std::cos(TWO_PI * xi);
            }
            return sum;
        }

        /// Compute gradient: g_i = 2*x_i + 2*pi*A*sin(2*pi*x_i)
        void gradient(const tensor_type &x, tensor_type &g) const noexcept {
            const std::size_t n = x.size();
            for (std::size_t i = 0; i < n; ++i) {
                T xi = x[i];
                g[i] = T{2} * xi + TWO_PI * A * std::sin(TWO_PI * xi);
            }
        }

        /// Evaluate and compute gradient in one pass
        T evaluate_with_gradient(const tensor_type &x, tensor_type &g) const noexcept {
            const std::size_t n = x.size();
            T sum = A * T(n);
            for (std::size_t i = 0; i < n; ++i) {
                T xi = x[i];
                T angle = TWO_PI * xi;
                sum += xi * xi - A * std::cos(angle);
                g[i] = T{2} * xi + TWO_PI * A * std::sin(angle);
            }
            return sum;
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
