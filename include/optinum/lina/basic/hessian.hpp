#pragma once

// =============================================================================
// optinum/lina/basic/hessian.hpp
// Finite-difference Hessian computation
// =============================================================================

#include <datapod/matrix/matrix.hpp>
#include <datapod/matrix/vector.hpp>
#include <optinum/lina/basic/jacobian.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>
#include <type_traits>

namespace optinum::lina {

    namespace dp = ::datapod;

    /**
     * @brief Compute Hessian matrix using finite differences
     *
     * For scalar function f: R^n -> R, computes the n×n Hessian matrix:
     *   H[i,j] = ∂²f/∂x_i∂x_j
     *
     * Uses central differences for second derivatives:
     *   Diagonal: H[i,i] ≈ (f(x+h·e_i) - 2f(x) + f(x-h·e_i)) / h²
     *   Off-diag: H[i,j] ≈ (f(x+h·e_i+h·e_j) - f(x+h·e_i-h·e_j)
     *                       - f(x-h·e_i+h·e_j) + f(x-h·e_i-h·e_j)) / (4h²)
     *
     * The Hessian is symmetric for smooth functions (Schwarz's theorem).
     *
     * @param f Scalar function to differentiate (f: R^n -> R)
     *          Must accept simd::Vector<T,N> and return T
     * @param x Point at which to evaluate Hessian
     * @param h Step size (default: 1e-5 for double, 1e-4 for float)
     * @return Hessian matrix (n × n, symmetric)
     *
     * @example
     * // Function f(x,y) = x^2 + 2*x*y + 3*y^2
     * auto f = [](const auto& x) {
     *     return x[0]*x[0] + 2*x[0]*x[1] + 3*x[1]*x[1];
     * };
     * simd::Vector<double, 2> x = {1.0, 2.0};
     * auto H = lina::hessian(f, x);  // H = [[2, 2], [2, 6]]
     */
    template <typename Function, typename T, std::size_t N>
    dp::mat::Matrix<T, N, N> hessian(const Function &f, const simd::Vector<T, N> &x,
                                     T h = (std::is_same_v<T, float> ? T(1e-4) : T(1e-5))) {
        static_assert(N != simd::Dynamic, "hessian() currently requires fixed-size vectors");
        const std::size_t n = x.size();

        // Copy input to owning storage
        dp::mat::Vector<T, N> x_base;
        for (std::size_t i = 0; i < n; ++i) {
            x_base[i] = x[i];
        }

        // Allocate Hessian matrix (owning)
        dp::mat::Matrix<T, N, N> H;

        // Evaluate f at x (needed for diagonal elements)
        T fx = f(simd::Vector<T, N>(x_base));

        // Temporary vectors for perturbations (owning types)
        dp::mat::Vector<T, N> x_pp = x_base; // x + h*e_i + h*e_j
        dp::mat::Vector<T, N> x_pm = x_base; // x + h*e_i - h*e_j
        dp::mat::Vector<T, N> x_mp = x_base; // x - h*e_i + h*e_j
        dp::mat::Vector<T, N> x_mm = x_base; // x - h*e_i - h*e_j
        dp::mat::Vector<T, N> x_p = x_base;  // x + h*e_i
        dp::mat::Vector<T, N> x_m = x_base;  // x - h*e_i

        T h_sq = h * h;
        T four_h_sq = T(4) * h_sq;

        // Compute Hessian elements
        for (std::size_t i = 0; i < n; ++i) {
            // Diagonal element: H[i,i] = (f(x+h*e_i) - 2*f(x) + f(x-h*e_i)) / h^2
            x_p[i] = x_base[i] + h;
            x_m[i] = x_base[i] - h;

            T f_p = f(simd::Vector<T, N>(x_p));
            T f_m = f(simd::Vector<T, N>(x_m));

            H(i, i) = (f_p - T(2) * fx + f_m) / h_sq;

            // Reset
            x_p[i] = x_base[i];
            x_m[i] = x_base[i];

            // Off-diagonal elements: H[i,j] for j > i
            // Use symmetry: H[j,i] = H[i,j]
            for (std::size_t j = i + 1; j < n; ++j) {
                // Four-point stencil for mixed partial derivative
                x_pp[i] = x_base[i] + h;
                x_pp[j] = x_base[j] + h;
                x_pm[i] = x_base[i] + h;
                x_pm[j] = x_base[j] - h;
                x_mp[i] = x_base[i] - h;
                x_mp[j] = x_base[j] + h;
                x_mm[i] = x_base[i] - h;
                x_mm[j] = x_base[j] - h;

                T f_pp = f(simd::Vector<T, N>(x_pp));
                T f_pm = f(simd::Vector<T, N>(x_pm));
                T f_mp = f(simd::Vector<T, N>(x_mp));
                T f_mm = f(simd::Vector<T, N>(x_mm));

                T hij = (f_pp - f_pm - f_mp + f_mm) / four_h_sq;
                H(i, j) = hij;
                H(j, i) = hij; // Symmetry

                // Reset
                x_pp[i] = x_base[i];
                x_pp[j] = x_base[j];
                x_pm[i] = x_base[i];
                x_pm[j] = x_base[j];
                x_mp[i] = x_base[i];
                x_mp[j] = x_base[j];
                x_mm[i] = x_base[i];
                x_mm[j] = x_base[j];
            }
        }

        return H;
    }

    /**
     * @brief Compute Hessian-vector product using finite differences
     *
     * Computes H*v where H is the Hessian of f at x, without forming H explicitly.
     * This is more efficient for large n when only the product is needed.
     *
     * Uses the identity:
     *   H*v ≈ (∇f(x + h*v) - ∇f(x - h*v)) / (2h)
     *
     * Requires 2n+2 function evaluations (vs n² for full Hessian).
     *
     * @param f Scalar function (f: R^n -> R)
     * @param x Point at which to evaluate
     * @param v Direction vector
     * @param h Step size
     * @return Hessian-vector product H*v
     *
     * @example
     * auto f = [](const auto& x) { return x[0]*x[0] + x[1]*x[1]; };
     * simd::Vector<double, 2> x = {1.0, 2.0};
     * simd::Vector<double, 2> v = {1.0, 0.0};
     * auto Hv = lina::hessian_vector_product(f, x, v);  // Hv = [2, 0]
     */
    template <typename Function, typename T, std::size_t N>
    dp::mat::Vector<T, N> hessian_vector_product(const Function &f, const simd::Vector<T, N> &x,
                                                 const simd::Vector<T, N> &v,
                                                 T h = (std::is_same_v<T, float> ? T(1e-4) : T(1e-5))) {
        const std::size_t n = x.size();

        // Copy inputs to owning storage
        dp::mat::Vector<T, N> x_base;
        dp::mat::Vector<T, N> v_base;
        for (std::size_t i = 0; i < n; ++i) {
            x_base[i] = x[i];
            v_base[i] = v[i];
        }

        // Compute x + h*v and x - h*v (owning types)
        dp::mat::Vector<T, N> x_plus;
        dp::mat::Vector<T, N> x_minus;
        for (std::size_t i = 0; i < n; ++i) {
            x_plus[i] = x_base[i] + h * v_base[i];
            x_minus[i] = x_base[i] - h * v_base[i];
        }

        // Compute gradients at perturbed points
        auto grad_plus = gradient(f, simd::Vector<T, N>(x_plus), h);
        auto grad_minus = gradient(f, simd::Vector<T, N>(x_minus), h);

        // H*v ≈ (∇f(x+hv) - ∇f(x-hv)) / (2h)
        dp::mat::Vector<T, N> Hv;

        T two_h = T(2) * h;
        for (std::size_t i = 0; i < n; ++i) {
            Hv[i] = (grad_plus[i] - grad_minus[i]) / two_h;
        }

        return Hv;
    }

    // Note: is_positive_definite is available in optinum/lina/basic/properties.hpp

    /**
     * @brief Check if Hessian computation is accurate
     *
     * Compares finite-difference Hessian against analytical Hessian.
     * Returns maximum relative error.
     *
     * @param H_numerical Numerically computed Hessian
     * @param H_analytical Analytically computed Hessian
     * @param tol Tolerance for comparison (default: 1e-10)
     * @return Maximum relative error
     */
    template <typename T, std::size_t N>
    T hessian_error(const simd::Matrix<T, N, N> &H_numerical, const simd::Matrix<T, N, N> &H_analytical,
                    T tol = T(1e-10)) {
        T max_error = T(0);
        const std::size_t n = H_numerical.rows();

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                T num = H_numerical(i, j);
                T ana = H_analytical(i, j);
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

    /**
     * @brief Compute the trace of the Hessian (Laplacian)
     *
     * The trace of the Hessian is the Laplacian: ∇²f = Σ ∂²f/∂x_i²
     *
     * This is more efficient than computing the full Hessian when only
     * the Laplacian is needed (n+1 function evaluations vs n²).
     *
     * @param f Scalar function (f: R^n -> R)
     * @param x Point at which to evaluate
     * @param h Step size
     * @return Laplacian (trace of Hessian)
     */
    template <typename Function, typename T, std::size_t N>
    T laplacian(const Function &f, const simd::Vector<T, N> &x, T h = (std::is_same_v<T, float> ? T(1e-4) : T(1e-5))) {
        const std::size_t n = x.size();

        // Copy input to owning storage
        dp::mat::Vector<T, N> x_base;
        for (std::size_t i = 0; i < n; ++i) {
            x_base[i] = x[i];
        }

        T fx = f(simd::Vector<T, N>(x_base));
        T trace = T(0);
        T h_sq = h * h;

        dp::mat::Vector<T, N> x_p = x_base;
        dp::mat::Vector<T, N> x_m = x_base;

        for (std::size_t i = 0; i < n; ++i) {
            x_p[i] = x_base[i] + h;
            x_m[i] = x_base[i] - h;

            T f_p = f(simd::Vector<T, N>(x_p));
            T f_m = f(simd::Vector<T, N>(x_m));

            trace += (f_p - T(2) * fx + f_m) / h_sq;

            // Reset
            x_p[i] = x_base[i];
            x_m[i] = x_base[i];
        }

        return trace;
    }

    // =============================================================================
    // Overloads for dp::mat::vector (used by opti module)
    // =============================================================================

    /**
     * @brief Compute Hessian matrix using finite differences (dp::mat::vector version)
     */
    template <typename Function, typename T, std::size_t N>
    dp::mat::Matrix<T, N, N> hessian(const Function &f, const dp::mat::Vector<T, N> &x,
                                     T h = (std::is_same_v<T, float> ? T(1e-4) : T(1e-5))) {
        static_assert(N != dp::mat::Dynamic, "hessian() currently requires fixed-size vectors");
        const std::size_t n = x.size();

        // Allocate Hessian matrix (owning)
        dp::mat::Matrix<T, N, N> H;

        // Evaluate f at x (needed for diagonal elements)
        T fx = f(x);

        // Temporary vectors for perturbations (owning types)
        dp::mat::Vector<T, N> x_pp = x; // x + h*e_i + h*e_j
        dp::mat::Vector<T, N> x_pm = x; // x + h*e_i - h*e_j
        dp::mat::Vector<T, N> x_mp = x; // x - h*e_i + h*e_j
        dp::mat::Vector<T, N> x_mm = x; // x - h*e_i - h*e_j
        dp::mat::Vector<T, N> x_p = x;  // x + h*e_i
        dp::mat::Vector<T, N> x_m = x;  // x - h*e_i

        T h_sq = h * h;
        T four_h_sq = T(4) * h_sq;

        // Compute Hessian elements
        for (std::size_t i = 0; i < n; ++i) {
            // Diagonal element: H[i,i] = (f(x+h*e_i) - 2*f(x) + f(x-h*e_i)) / h^2
            x_p[i] = x[i] + h;
            x_m[i] = x[i] - h;

            T f_p = f(x_p);
            T f_m = f(x_m);

            H(i, i) = (f_p - T(2) * fx + f_m) / h_sq;

            // Reset
            x_p[i] = x[i];
            x_m[i] = x[i];

            // Off-diagonal elements: H[i,j] for j > i
            // Use symmetry: H[j,i] = H[i,j]
            for (std::size_t j = i + 1; j < n; ++j) {
                // Four-point stencil for mixed partial derivative
                x_pp[i] = x[i] + h;
                x_pp[j] = x[j] + h;
                x_pm[i] = x[i] + h;
                x_pm[j] = x[j] - h;
                x_mp[i] = x[i] - h;
                x_mp[j] = x[j] + h;
                x_mm[i] = x[i] - h;
                x_mm[j] = x[j] - h;

                T f_pp = f(x_pp);
                T f_pm = f(x_pm);
                T f_mp = f(x_mp);
                T f_mm = f(x_mm);

                T hij = (f_pp - f_pm - f_mp + f_mm) / four_h_sq;
                H(i, j) = hij;
                H(j, i) = hij; // Symmetry

                // Reset
                x_pp[i] = x[i];
                x_pp[j] = x[j];
                x_pm[i] = x[i];
                x_pm[j] = x[j];
                x_mp[i] = x[i];
                x_mp[j] = x[j];
                x_mm[i] = x[i];
                x_mm[j] = x[j];
            }
        }

        return H;
    }

    /**
     * @brief Compute Hessian-vector product using finite differences (dp::mat::vector version)
     */
    template <typename Function, typename T, std::size_t N>
    dp::mat::Vector<T, N> hessian_vector_product(const Function &f, const dp::mat::Vector<T, N> &x,
                                                 const dp::mat::Vector<T, N> &v,
                                                 T h = (std::is_same_v<T, float> ? T(1e-4) : T(1e-5))) {
        const std::size_t n = x.size();

        // Compute x + h*v and x - h*v
        dp::mat::Vector<T, N> x_plus;
        dp::mat::Vector<T, N> x_minus;
        for (std::size_t i = 0; i < n; ++i) {
            x_plus[i] = x[i] + h * v[i];
            x_minus[i] = x[i] - h * v[i];
        }

        // Compute gradients at perturbed points
        auto grad_plus = gradient(f, x_plus, h);
        auto grad_minus = gradient(f, x_minus, h);

        // H*v ≈ (∇f(x+hv) - ∇f(x-hv)) / (2h)
        dp::mat::Vector<T, N> Hv;

        T two_h = T(2) * h;
        for (std::size_t i = 0; i < n; ++i) {
            Hv[i] = (grad_plus[i] - grad_minus[i]) / two_h;
        }

        return Hv;
    }

    /**
     * @brief Compute the trace of the Hessian (Laplacian) - dp::mat::vector version
     */
    template <typename Function, typename T, std::size_t N>
    T laplacian(const Function &f, const dp::mat::Vector<T, N> &x,
                T h = (std::is_same_v<T, float> ? T(1e-4) : T(1e-5))) {
        const std::size_t n = x.size();

        T fx = f(x);
        T trace = T(0);
        T h_sq = h * h;

        dp::mat::Vector<T, N> x_p = x;
        dp::mat::Vector<T, N> x_m = x;

        for (std::size_t i = 0; i < n; ++i) {
            x_p[i] = x[i] + h;
            x_m[i] = x[i] - h;

            T f_p = f(x_p);
            T f_m = f(x_m);

            trace += (f_p - T(2) * fx + f_m) / h_sq;

            // Reset
            x_p[i] = x[i];
            x_m[i] = x[i];
        }

        return trace;
    }

} // namespace optinum::lina
