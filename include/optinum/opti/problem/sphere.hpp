#pragma once

#include <datapod/matrix/vector.hpp>

namespace optinum::opti {

    namespace dp = ::datapod;

    /**
     * Sphere function - simplest benchmark for optimization
     *
     * f(x) = sum(x_i^2)
     * gradient: g_i = 2 * x_i
     *
     * Global minimum: f(0, 0, ..., 0) = 0
     *
     * Supports both fixed-size and dynamic vectors:
     *   Sphere<double, 10>      - Fixed size
     *   Sphere<double, Dynamic> - Dynamic size
     */
    template <typename T, std::size_t N> struct Sphere {
        using tensor_type = dp::mat::vector<T, N>;

        /// Evaluate f(x) = sum(x_i^2)
        T evaluate(const tensor_type &x) const noexcept {
            T sum{};
            const std::size_t n = x.size();
            for (std::size_t i = 0; i < n; ++i) {
                sum += x[i] * x[i];
            }
            return sum;
        }

        /// Compute gradient: g_i = 2 * x_i
        void gradient(const tensor_type &x, tensor_type &g) const noexcept {
            const std::size_t n = x.size();
            for (std::size_t i = 0; i < n; ++i) {
                g[i] = T{2} * x[i];
            }
        }

        /// Evaluate and compute gradient in one pass
        T evaluate_with_gradient(const tensor_type &x, tensor_type &g) const noexcept {
            T sum{};
            const std::size_t n = x.size();
            for (std::size_t i = 0; i < n; ++i) {
                sum += x[i] * x[i];
                g[i] = T{2} * x[i];
            }
            return sum;
        }
    };

} // namespace optinum::opti
