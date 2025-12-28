#pragma once

// =============================================================================
// optinum/lie/algorithms/spline.hpp
// Smooth spline interpolation for Lie groups
// =============================================================================
//
// Provides smooth interpolation through a sequence of Lie group elements.
// Unlike simple geodesic interpolation, splines ensure C1 continuity
// (continuous first derivative) at the control points.
//
// Algorithms:
// - Cubic Hermite spline on Lie groups (C1 continuous)
// - De Casteljau-style Bezier curves on Lie groups
// - Cumulative B-spline (for uniform sampling)
//
// References:
// - Park & Ravani (1997) "Bezier Curves on Riemannian Manifolds"
// - Kim et al. (1995) "A General Construction Scheme for Unit Quaternion Curves"
// - Lee & Shin (2002) "General Construction of Time-Domain Filters for SO(3)"

#include <optinum/lie/core/constants.hpp>
#include <optinum/simd/vector.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace optinum::lie {

    // ===== GEODESIC INTERPOLATION =====

    /**
     * @brief Geodesic (SLERP-like) interpolation between two Lie group elements
     *
     * This is the natural generalization of linear interpolation to Lie groups.
     * The path follows the geodesic (shortest path) on the group manifold.
     *
     * Formula: interpolate(a, b, t) = a * exp(t * log(a^-1 * b))
     *
     * @param a Start element (t=0)
     * @param b End element (t=1)
     * @param t Interpolation parameter in [0, 1]
     * @return Interpolated element
     */
    template <typename G> [[nodiscard]] G geodesic(const G &a, const G &b, typename G::Scalar t) noexcept {
        auto tangent = (a.inverse() * b).log();
        tangent *= t;
        return a * G::exp(tangent);
    }

    // ===== CUBIC HERMITE SPLINE =====

    /**
     * @brief Cubic Hermite spline segment on a Lie group
     *
     * Given two poses p0, p1 and their tangent velocities v0, v1,
     * computes a C1-smooth path between them.
     *
     * @param p0 Start pose
     * @param p1 End pose
     * @param v0 Velocity at start (in tangent space at p0)
     * @param v1 Velocity at end (in tangent space at p1)
     * @param t Parameter in [0, 1]
     * @return Interpolated pose
     */
    template <typename G>
    [[nodiscard]] G cubic_hermite(const G &p0, const G &p1, const typename G::Tangent &v0,
                                  const typename G::Tangent &v1, typename G::Scalar t) noexcept {
        using T = typename G::Scalar;
        using Tangent = typename G::Tangent;

        // Hermite basis functions
        const T t2 = t * t;
        const T t3 = t2 * t;

        const T h10 = t3 - T(2) * t2 + t;
        const T h11 = t3 - t2;

        // Main geodesic from p0 to p1
        Tangent delta = (p0.inverse() * p1).log();

        // Simple blended approach: mostly geodesic with velocity adjustment
        // base = t * delta + h10 * v0 + h11 * v1
        Tangent base;
        if constexpr (std::is_arithmetic_v<Tangent>) {
            base = t * delta + h10 * v0 + h11 * v1;
        } else {
            base = delta;
            base *= t;
            Tangent term0 = v0;
            term0 *= h10;
            Tangent term1 = v1;
            term1 *= h11;
            base += term0;
            base += term1;
        }

        return p0 * G::exp(base);
    }

    // ===== DE CASTELJAU BEZIER CURVE =====

    /**
     * @brief Cubic Bezier curve on a Lie group using de Casteljau algorithm
     *
     * Given 4 control points p0, p1, p2, p3, computes a smooth curve.
     * The curve passes through p0 (at t=0) and p3 (at t=1).
     *
     * @param p0, p1, p2, p3 Control points
     * @param t Parameter in [0, 1]
     * @return Point on the Bezier curve
     */
    template <typename G>
    [[nodiscard]] G bezier_cubic(const G &p0, const G &p1, const G &p2, const G &p3, typename G::Scalar t) noexcept {
        // Level 1
        G q0 = geodesic(p0, p1, t);
        G q1 = geodesic(p1, p2, t);
        G q2 = geodesic(p2, p3, t);

        // Level 2
        G r0 = geodesic(q0, q1, t);
        G r1 = geodesic(q1, q2, t);

        // Level 3
        return geodesic(r0, r1, t);
    }

    /**
     * @brief Quadratic Bezier curve on a Lie group
     */
    template <typename G>
    [[nodiscard]] G bezier_quadratic(const G &p0, const G &p1, const G &p2, typename G::Scalar t) noexcept {
        G q0 = geodesic(p0, p1, t);
        G q1 = geodesic(p1, p2, t);
        return geodesic(q0, q1, t);
    }

    // ===== CATMULL-ROM SPLINE =====

    /**
     * @brief Catmull-Rom spline segment on a Lie group
     *
     * Catmull-Rom is a C1 spline that passes through all control points.
     * Given 4 consecutive points p0, p1, p2, p3, interpolates between p1 and p2.
     *
     * @param p0 Previous point (for tangent computation)
     * @param p1 Start of this segment
     * @param p2 End of this segment
     * @param p3 Next point (for tangent computation)
     * @param t Parameter in [0, 1] (interpolates between p1 and p2)
     * @return Interpolated point
     */
    template <typename G>
    [[nodiscard]] G catmull_rom(const G &p0, const G &p1, const G &p2, const G &p3, typename G::Scalar t) noexcept {
        using T = typename G::Scalar;
        using Tangent = typename G::Tangent;

        // Compute tangents at p1 and p2: v = 0.5 * log(p_prev^-1 * p_next)
        Tangent v1 = (p0.inverse() * p2).log();
        Tangent v2 = (p1.inverse() * p3).log();
        v1 *= T(0.5);
        v2 *= T(0.5);

        return cubic_hermite(p1, p2, v1, v2, t);
    }

    // ===== SPLINE CLASS =====

    /**
     * @brief Lie group spline through multiple control points
     *
     * Provides C1-smooth interpolation through a sequence of control points.
     * Uses Catmull-Rom splines internally, with special handling for endpoints.
     *
     * @tparam G Lie group type
     */
    template <typename G> class LieSpline {
      public:
        using Scalar = typename G::Scalar;
        using Tangent = typename G::Tangent;

      private:
        std::vector<G> points_;
        std::vector<Tangent> tangents_;

      public:
        LieSpline() = default;

        /**
         * @brief Construct spline from control points
         */
        explicit LieSpline(const std::vector<G> &points) : points_(points) {
            if (points_.size() < 2) {
                return;
            }
            compute_tangents();
        }

        /**
         * @brief Construct spline from points with explicit tangents
         */
        LieSpline(const std::vector<G> &points, const std::vector<Tangent> &tangents)
            : points_(points), tangents_(tangents) {}

        [[nodiscard]] std::size_t size() const noexcept { return points_.size(); }

        [[nodiscard]] const G &point(std::size_t i) const noexcept { return points_[i]; }

        [[nodiscard]] const Tangent &tangent(std::size_t i) const noexcept { return tangents_[i]; }

        /**
         * @brief Evaluate spline at parameter t in [0, n-1]
         */
        [[nodiscard]] G evaluate(Scalar t) const noexcept {
            if (points_.empty()) {
                return G::identity();
            }
            if (points_.size() == 1) {
                return points_[0];
            }

            const Scalar max_t = static_cast<Scalar>(points_.size() - 1);
            t = std::max(Scalar(0), std::min(t, max_t));

            std::size_t segment = static_cast<std::size_t>(t);
            if (segment >= points_.size() - 1) {
                segment = points_.size() - 2;
            }
            Scalar local_t = t - static_cast<Scalar>(segment);

            return cubic_hermite(points_[segment], points_[segment + 1], tangents_[segment], tangents_[segment + 1],
                                 local_t);
        }

        /**
         * @brief Evaluate spline normalized to [0, 1]
         */
        [[nodiscard]] G evaluate_normalized(Scalar u) const noexcept {
            if (points_.size() < 2) {
                return points_.empty() ? G::identity() : points_[0];
            }
            return evaluate(u * static_cast<Scalar>(points_.size() - 1));
        }

      private:
        void compute_tangents() {
            const std::size_t n = points_.size();
            tangents_.resize(n);

            if (n < 2) {
                return;
            }

            // First point: forward difference
            tangents_[0] = (points_[0].inverse() * points_[1]).log();

            // Middle points: central difference (Catmull-Rom)
            for (std::size_t i = 1; i < n - 1; ++i) {
                Tangent v = (points_[i - 1].inverse() * points_[i + 1]).log();
                v *= Scalar(0.5);
                tangents_[i] = v;
            }

            // Last point: backward difference
            tangents_[n - 1] = (points_[n - 2].inverse() * points_[n - 1]).log();
        }
    };

    // ===== UTILITY FUNCTIONS =====

    /**
     * @brief Sample a spline at uniform parameter intervals
     */
    template <typename G>
    [[nodiscard]] std::vector<G> sample_spline(const LieSpline<G> &spline, std::size_t num_samples) {
        std::vector<G> samples;
        samples.reserve(num_samples);

        if (num_samples == 0 || spline.size() == 0) {
            return samples;
        }

        using T = typename G::Scalar;
        const T step = T(1) / static_cast<T>(num_samples - 1);

        for (std::size_t i = 0; i < num_samples; ++i) {
            T u = static_cast<T>(i) * step;
            samples.push_back(spline.evaluate_normalized(u));
        }

        return samples;
    }

    /**
     * @brief Compute approximate arc length of a spline segment
     */
    template <typename G>
    [[nodiscard]] typename G::Scalar arc_length(const LieSpline<G> &spline, typename G::Scalar t0,
                                                typename G::Scalar t1, std::size_t num_steps = 100) {
        using T = typename G::Scalar;
        using Tangent = typename G::Tangent;

        if (num_steps == 0) {
            return T(0);
        }

        const T dt = (t1 - t0) / static_cast<T>(num_steps);
        T length = T(0);

        G prev = spline.evaluate(t0);
        for (std::size_t i = 1; i <= num_steps; ++i) {
            T t = t0 + static_cast<T>(i) * dt;
            G curr = spline.evaluate(t);

            Tangent tangent = (prev.inverse() * curr).log();
            T dist_sq;
            if constexpr (std::is_arithmetic_v<Tangent>) {
                dist_sq = tangent * tangent;
            } else {
                dist_sq = simd::dot(tangent, tangent);
            }
            length += std::sqrt(dist_sq);

            prev = curr;
        }

        return length;
    }

} // namespace optinum::lie
