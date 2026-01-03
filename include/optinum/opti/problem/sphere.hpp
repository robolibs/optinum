#pragma once

#include <datapod/matrix/vector.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>

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
        using tensor_type = dp::mat::Vector<T, N>;

        /// Evaluate f(x) = sum(x_i^2) using SIMD dot product
        T evaluate(const tensor_type &x) const noexcept {
            const std::size_t n = x.size();
            return simd::backend::dot_runtime<T>(x.data(), x.data(), n);
        }

        /// Compute gradient: g_i = 2 * x_i using SIMD scalar multiply
        void gradient(const tensor_type &x, tensor_type &g) const noexcept {
            const std::size_t n = x.size();
            simd::backend::mul_scalar_runtime<T>(g.data(), x.data(), T{2}, n);
        }

        /// Evaluate and compute gradient in one pass using SIMD
        T evaluate_with_gradient(const tensor_type &x, tensor_type &g) const noexcept {
            const std::size_t n = x.size();
            // Compute gradient: g = 2 * x
            simd::backend::mul_scalar_runtime<T>(g.data(), x.data(), T{2}, n);
            // Compute value: sum(x^2) = dot(x, x)
            return simd::backend::dot_runtime<T>(x.data(), x.data(), n);
        }
    };

} // namespace optinum::opti
