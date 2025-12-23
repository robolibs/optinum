#pragma once

#include <optinum/simd/tensor.hpp>

namespace optinum::opti {

    /**
     * Sphere function - simplest benchmark for optimization
     *
     * f(x) = sum(x_i^2)
     * gradient: g_i = 2 * x_i
     *
     * Global minimum: f(0, 0, ..., 0) = 0
     */
    template <typename T, std::size_t N> struct Sphere {
        using tensor_type = simd::Tensor<T, N>;

        /// Evaluate f(x) = sum(x_i^2)
        T evaluate(const tensor_type &x) const noexcept {
            T sum{};
            for (std::size_t i = 0; i < N; ++i) {
                sum += x[i] * x[i];
            }
            return sum;
        }

        /// Compute gradient: g_i = 2 * x_i
        void gradient(const tensor_type &x, tensor_type &g) const noexcept {
            for (std::size_t i = 0; i < N; ++i) {
                g[i] = T{2} * x[i];
            }
        }

        /// Evaluate and compute gradient in one pass
        T evaluate_with_gradient(const tensor_type &x, tensor_type &g) const noexcept {
            T sum{};
            for (std::size_t i = 0; i < N; ++i) {
                sum += x[i] * x[i];
                g[i] = T{2} * x[i];
            }
            return sum;
        }
    };

} // namespace optinum::opti
