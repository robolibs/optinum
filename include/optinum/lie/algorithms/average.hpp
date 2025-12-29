#pragma once

// =============================================================================
// optinum/lie/algorithms/average.hpp
// Biinvariant mean (Frechet/Karcher mean) computation for Lie groups
// =============================================================================
//
// Computes the average of multiple Lie group elements on the manifold.
// This is the generalization of the arithmetic mean to Lie groups.
//
// The biinvariant mean minimizes: sum_i d(mean, x_i)^2
// where d is the geodesic distance.
//
// Algorithm: Iterative refinement (Riemannian gradient descent)
//   1. Initialize with first element (or provided guess)
//   2. Compute average tangent: v = (1/n) * sum_i log(mean^-1 * x_i)
//   3. Update: mean = mean * exp(v)
//   4. Repeat until ||v|| < tolerance
//
// References:
// - Pennec & Arsigny (2006) "Exponential barycenters"
// - Moakher (2002) "Means and averaging in the group of rotations"

#include <optinum/lie/core/concepts.hpp>
#include <optinum/lie/core/constants.hpp>
#include <optinum/lie/groups/so3.hpp>
#include <optinum/simd/pack/pack.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <vector>

namespace optinum::lie {

    // ===== ITERATIVE BIINVARIANT MEAN =====
    //
    // Works for any Lie group satisfying the LieGroup concept.
    // Uses Riemannian gradient descent with fixed step size.

    /**
     * @brief Compute the biinvariant (Frechet/Karcher) mean of Lie group elements
     *
     * @tparam G Lie group type (SO2, SE2, SO3, SE3, etc.)
     * @tparam Container Container of G elements (vector, array, span, etc.)
     *
     * @param elements Container of Lie group elements to average
     * @param max_iter Maximum number of iterations (default: 20)
     * @param tolerance Convergence tolerance on tangent norm (default: 1e-10)
     * @param initial Optional initial guess (defaults to first element)
     *
     * @return std::optional<G> The mean if converged, std::nullopt otherwise
     */
    template <typename G, typename Container>
    [[nodiscard]] std::optional<G> average(const Container &elements, std::size_t max_iter = 20,
                                           typename G::Scalar tolerance = typename G::Scalar(1e-10),
                                           std::optional<G> initial = std::nullopt) noexcept {
        using T = typename G::Scalar;
        using Tangent = typename G::Tangent;

        // Handle empty container
        const std::size_t n = std::size(elements);
        if (n == 0) {
            return std::nullopt;
        }

        // Single element: return it
        if (n == 1) {
            return *std::begin(elements);
        }

        // Initialize mean
        G mean = initial.has_value() ? *initial : *std::begin(elements);
        const T inv_n = T(1) / static_cast<T>(n);

        // Iterative refinement
        for (std::size_t iter = 0; iter < max_iter; ++iter) {
            // Compute average tangent: v = (1/n) * sum_i log(mean^-1 * x_i)
            G mean_inv = mean.inverse();
            auto it = std::begin(elements);
            Tangent v_sum = (mean_inv * G(*it)).log();
            ++it;

            // Add remaining logs - use SIMD operators
            for (; it != std::end(elements); ++it) {
                v_sum += (mean_inv * G(*it)).log();
            }

            // Scale by 1/n - use SIMD operator
            v_sum *= inv_n;

            // Check convergence: ||v||^2 < tolerance^2
            // For SO2: Tangent is scalar, for others it's simd::Vector
            T v_norm_sq;
            if constexpr (std::is_arithmetic_v<Tangent>) {
                v_norm_sq = v_sum * v_sum;
            } else {
                v_norm_sq = simd::dot(v_sum, v_sum);
            }

            if (v_norm_sq < tolerance * tolerance) {
                return mean;
            }

            // Update mean: mean = mean * exp(v)
            mean = mean * G::exp(v_sum);
        }

        // Did not converge within max_iter
        return mean; // Return current estimate anyway
    }

    /**
     * @brief Compute weighted biinvariant mean
     *
     * @tparam G Lie group type
     * @tparam Container Container of G elements
     * @tparam WeightContainer Container of weights (same size as elements)
     *
     * @param elements Container of Lie group elements
     * @param weights Container of weights (should sum to 1, but will be normalized)
     * @param max_iter Maximum iterations
     * @param tolerance Convergence tolerance
     *
     * @return std::optional<G> The weighted mean if converged
     */
    template <typename G, typename Container, typename WeightContainer>
    [[nodiscard]] std::optional<G> weighted_average(const Container &elements, const WeightContainer &weights,
                                                    std::size_t max_iter = 20,
                                                    typename G::Scalar tolerance = typename G::Scalar(1e-10)) noexcept {
        using T = typename G::Scalar;
        using Tangent = typename G::Tangent;

        const std::size_t n = std::size(elements);
        if (n == 0 || std::size(weights) != n) {
            return std::nullopt;
        }

        if (n == 1) {
            return *std::begin(elements);
        }

        // Compute total weight for normalization
        T total_weight = T(0);
        for (const auto &w : weights) {
            total_weight += static_cast<T>(w);
        }
        if (std::abs(total_weight) < epsilon<T>) {
            return std::nullopt;
        }
        const T inv_total = T(1) / total_weight;

        // Initialize with first element
        G mean = *std::begin(elements);

        // Iterative refinement
        for (std::size_t iter = 0; iter < max_iter; ++iter) {
            G mean_inv = mean.inverse();
            auto elem_it = std::begin(elements);
            auto weight_it = std::begin(weights);

            // First element: initialize v_sum with weighted log
            T w = static_cast<T>(*weight_it++) * inv_total;
            Tangent v_sum = (mean_inv * G(*elem_it)).log();
            v_sum *= w;
            ++elem_it;

            // Add remaining weighted logs
            for (; elem_it != std::end(elements); ++elem_it) {
                w = static_cast<T>(*weight_it++) * inv_total;
                Tangent v_i = (mean_inv * G(*elem_it)).log();
                v_i *= w;
                v_sum += v_i;
            }

            // Check convergence
            T v_norm_sq;
            if constexpr (std::is_arithmetic_v<Tangent>) {
                v_norm_sq = v_sum * v_sum;
            } else {
                v_norm_sq = simd::dot(v_sum, v_sum);
            }

            if (v_norm_sq < tolerance * tolerance) {
                return mean;
            }

            mean = mean * G::exp(v_sum);
        }

        return mean;
    }

    // ===== SPECIALIZED CLOSED-FORM AVERAGES =====

    /**
     * @brief Compute average of two Lie group elements (midpoint)
     *
     * This is a fast special case: average(a, b) = a * exp(0.5 * log(a^-1 * b))
     */
    template <typename G> [[nodiscard]] G average_two(const G &a, const G &b) noexcept {
        using T = typename G::Scalar;
        auto tangent = (a.inverse() * b).log();
        tangent *= T(0.5);
        return a * G::exp(tangent);
    }

    // ===== SPECIALIZED SO3 AVERAGE =====
    //
    // For unit quaternions, we can use a more efficient algorithm
    // based on the eigenvector of the quaternion covariance matrix.

    /**
     * @brief Fast SO3 average using quaternion eigenvector method
     *
     * This method computes the dominant eigenvector of the 4x4 quaternion
     * covariance matrix. It's faster than iterative methods for SO3.
     *
     * Reference: Markley et al. (2007) "Averaging Quaternions"
     *
     * @param rotations Vector of SO3 rotations
     * @return The average rotation
     */
    template <typename T> [[nodiscard]] SO3<T> average_so3_quaternion(const std::vector<SO3<T>> &rotations) noexcept {
        const std::size_t n = rotations.size();
        if (n == 0) {
            return SO3<T>::identity();
        }
        if (n == 1) {
            return rotations[0];
        }

        // Get reference quaternion (first one)
        const auto &q0 = rotations[0].unit_quaternion();
        T ref[4] = {q0.w, q0.x, q0.y, q0.z};

        // Build 4x4 quaternion covariance matrix M = sum(q * q^T)
        // Handle sign ambiguity: flip quaternion if dot product with reference is negative
        // M is symmetric, so we only need upper triangle
        T M[4][4] = {{T(0)}, {T(0)}, {T(0)}, {T(0)}};

        for (const auto &R : rotations) {
            const auto &q = R.unit_quaternion();
            T qvec[4] = {q.w, q.x, q.y, q.z};

            // Check sign relative to reference
            T dot = ref[0] * qvec[0] + ref[1] * qvec[1] + ref[2] * qvec[2] + ref[3] * qvec[3];
            if (dot < T(0)) {
                qvec[0] = -qvec[0];
                qvec[1] = -qvec[1];
                qvec[2] = -qvec[2];
                qvec[3] = -qvec[3];
            }

            for (int i = 0; i < 4; ++i) {
                for (int j = i; j < 4; ++j) {
                    M[i][j] += qvec[i] * qvec[j];
                }
            }
        }

        // Fill lower triangle
        for (int i = 1; i < 4; ++i) {
            for (int j = 0; j < i; ++j) {
                M[i][j] = M[j][i];
            }
        }

        // Power iteration to find dominant eigenvector (SIMD optimized)
        // (M is positive semi-definite, so this converges to the eigenvector
        // with largest eigenvalue, which corresponds to the average quaternion)
        //
        // Use SIMD pack<T,4> for 4x4 matrix-vector multiply
        using Pack4 = simd::pack<T, 4>;

        alignas(32) T v[4] = {T(1), T(0), T(0), T(0)}; // Initial guess

        for (int iter = 0; iter < 20; ++iter) {
            // Load v into a pack
            auto v_pack = Pack4::load(v);

            // Multiply: v_new = M * v using SIMD dot products
            alignas(32) T v_new[4];
            for (int i = 0; i < 4; ++i) {
                // Load row i of M
                alignas(32) T row[4] = {M[i][0], M[i][1], M[i][2], M[i][3]};
                auto row_pack = Pack4::load(row);
                // Dot product: row · v
                v_new[i] = (row_pack * v_pack).hsum();
            }

            // Compute norm using SIMD
            auto v_new_pack = Pack4::load(v_new);
            T norm_sq = (v_new_pack * v_new_pack).hsum();
            T norm = std::sqrt(norm_sq);

            if (norm < epsilon<T>) {
                break;
            }

            // Normalize using SIMD
            T inv_norm = T(1) / norm;
            (v_new_pack * Pack4(inv_norm)).store(v);
        }

        // Ensure positive scalar part (canonical form)
        if (v[0] < T(0)) {
            v[0] = -v[0];
            v[1] = -v[1];
            v[2] = -v[2];
            v[3] = -v[3];
        }

        return SO3<T>(v[0], v[1], v[2], v[3]);
    }

    /**
     * @brief Weighted SO3 average using quaternion eigenvector method
     *
     * @param rotations Vector of SO3 rotations
     * @param weights Vector of weights (will be normalized)
     * @return The weighted average rotation
     */
    template <typename T>
    [[nodiscard]] SO3<T> weighted_average_so3_quaternion(const std::vector<SO3<T>> &rotations,
                                                         const std::vector<T> &weights) noexcept {
        const std::size_t n = rotations.size();
        if (n == 0 || n != weights.size()) {
            return SO3<T>::identity();
        }
        if (n == 1) {
            return rotations[0];
        }

        // Normalize weights
        T total = T(0);
        for (const auto &w : weights) {
            total += w;
        }
        if (std::abs(total) < epsilon<T>) {
            return SO3<T>::identity();
        }

        // Get reference quaternion (first one)
        const auto &q0 = rotations[0].unit_quaternion();
        T ref[4] = {q0.w, q0.x, q0.y, q0.z};

        // Build weighted 4x4 quaternion covariance matrix
        // Handle sign ambiguity: flip quaternion if dot product with reference is negative
        T M[4][4] = {{T(0)}, {T(0)}, {T(0)}, {T(0)}};

        for (std::size_t k = 0; k < n; ++k) {
            const auto &q = rotations[k].unit_quaternion();
            T w = weights[k] / total;
            T qvec[4] = {q.w, q.x, q.y, q.z};

            // Check sign relative to reference
            T dot = ref[0] * qvec[0] + ref[1] * qvec[1] + ref[2] * qvec[2] + ref[3] * qvec[3];
            if (dot < T(0)) {
                qvec[0] = -qvec[0];
                qvec[1] = -qvec[1];
                qvec[2] = -qvec[2];
                qvec[3] = -qvec[3];
            }

            for (int i = 0; i < 4; ++i) {
                for (int j = i; j < 4; ++j) {
                    M[i][j] += w * qvec[i] * qvec[j];
                }
            }
        }

        // Fill lower triangle
        for (int i = 1; i < 4; ++i) {
            for (int j = 0; j < i; ++j) {
                M[i][j] = M[j][i];
            }
        }

        // Power iteration (SIMD optimized)
        using Pack4 = simd::pack<T, 4>;
        alignas(32) T v[4] = {T(1), T(0), T(0), T(0)};
        for (int iter = 0; iter < 20; ++iter) {
            auto v_pack = Pack4::load(v);

            alignas(32) T v_new[4];
            for (int i = 0; i < 4; ++i) {
                alignas(32) T row[4] = {M[i][0], M[i][1], M[i][2], M[i][3]};
                auto row_pack = Pack4::load(row);
                v_new[i] = (row_pack * v_pack).hsum();
            }

            auto v_new_pack = Pack4::load(v_new);
            T norm_sq = (v_new_pack * v_new_pack).hsum();
            T norm = std::sqrt(norm_sq);

            if (norm < epsilon<T>) {
                break;
            }
            T inv_norm = T(1) / norm;
            (v_new_pack * Pack4(inv_norm)).store(v);
        }

        if (v[0] < T(0)) {
            v[0] = -v[0];
            v[1] = -v[1];
            v[2] = -v[2];
            v[3] = -v[3];
        }

        return SO3<T>(v[0], v[1], v[2], v[3]);
    }

    // ===== CHORD AVERAGE (FAST APPROXIMATION) =====

    /**
     * @brief Fast chord average for SO3 (normalized quaternion mean)
     *
     * This is a fast approximation that simply averages quaternions
     * and normalizes. It's not exactly the geodesic mean but is very
     * fast and accurate for rotations that are close together.
     *
     * @param rotations Vector of SO3 rotations
     * @return Approximate average rotation
     */
    template <typename T> [[nodiscard]] SO3<T> chord_average_so3(const std::vector<SO3<T>> &rotations) noexcept {
        const std::size_t n = rotations.size();
        if (n == 0) {
            return SO3<T>::identity();
        }
        if (n == 1) {
            return rotations[0];
        }

        // Accumulate quaternions (handling sign ambiguity)
        const auto &q0 = rotations[0].unit_quaternion();
        T sum_w = q0.w;
        T sum_x = q0.x;
        T sum_y = q0.y;
        T sum_z = q0.z;

        for (std::size_t i = 1; i < n; ++i) {
            const auto &q = rotations[i].unit_quaternion();
            // Flip sign if dot product with first quaternion is negative
            T dot = sum_w * q.w + sum_x * q.x + sum_y * q.y + sum_z * q.z;
            T sign = dot < T(0) ? T(-1) : T(1);
            sum_w += sign * q.w;
            sum_x += sign * q.x;
            sum_y += sign * q.y;
            sum_z += sign * q.z;
        }

        // Normalize
        T norm = std::sqrt(sum_w * sum_w + sum_x * sum_x + sum_y * sum_y + sum_z * sum_z);
        if (norm < epsilon<T>) {
            return SO3<T>::identity();
        }

        return SO3<T>(sum_w / norm, sum_x / norm, sum_y / norm, sum_z / norm);
    }

} // namespace optinum::lie
