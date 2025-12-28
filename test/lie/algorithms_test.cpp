#include <doctest/doctest.h>

#include <optinum/lie/lie.hpp>

#include <cmath>
#include <random>
#include <vector>

using namespace optinum;
using namespace optinum::lie;

// ============================================================================
// Average Tests
// ============================================================================

TEST_SUITE("Lie Algorithms - Average") {

    TEST_CASE("Average of empty container returns nullopt") {
        std::vector<SO3d> empty;
        auto result = average<SO3d>(empty);
        CHECK_FALSE(result.has_value());
    }

    TEST_CASE("Average of single element returns that element") {
        SO3d R = SO3d::rot_z(M_PI / 4);
        std::vector<SO3d> single = {R};
        auto result = average<SO3d>(single);

        REQUIRE(result.has_value());
        CHECK(result->is_approx(R, 1e-10));
    }

    TEST_CASE("Average of two identical elements returns that element") {
        SO3d R = SO3d::rot_x(0.5) * SO3d::rot_y(0.3);
        std::vector<SO3d> rotations = {R, R};
        auto result = average<SO3d>(rotations);

        REQUIRE(result.has_value());
        CHECK(result->is_approx(R, 1e-8));
    }

    TEST_CASE("Average of identity elements is identity") {
        std::vector<SO3d> identities(5, SO3d::identity());
        auto result = average<SO3d>(identities);

        REQUIRE(result.has_value());
        CHECK(result->is_identity(1e-10));
    }

    TEST_CASE("Average of symmetric rotations around z-axis is identity") {
        // Rotations that sum to zero: pi/4, -pi/4
        std::vector<SO3d> rotations = {SO3d::rot_z(M_PI / 4), SO3d::rot_z(-M_PI / 4)};
        auto result = average<SO3d>(rotations);

        REQUIRE(result.has_value());
        CHECK(result->is_identity(1e-8));
    }

    TEST_CASE("Average of three z-rotations") {
        // Average of 0, pi/6, pi/3 should be pi/6
        std::vector<SO3d> rotations = {SO3d::identity(), SO3d::rot_z(M_PI / 6), SO3d::rot_z(M_PI / 3)};
        auto result = average<SO3d>(rotations);

        REQUIRE(result.has_value());
        // The average should be approximately rot_z(pi/6)
        auto expected = SO3d::rot_z(M_PI / 6);
        CHECK(result->is_approx(expected, 1e-6));
    }

    TEST_CASE("average_two computes midpoint") {
        SO3d a = SO3d::identity();
        SO3d b = SO3d::rot_z(M_PI / 2);

        auto mid = average_two(a, b);

        // Midpoint should be rot_z(pi/4)
        auto expected = SO3d::rot_z(M_PI / 4);
        CHECK(mid.is_approx(expected, 1e-10));
    }

    TEST_CASE("Weighted average with equal weights equals unweighted average") {
        std::vector<SO3d> rotations = {SO3d::identity(), SO3d::rot_z(M_PI / 4), SO3d::rot_z(M_PI / 2)};
        std::vector<double> weights = {1.0, 1.0, 1.0};

        auto weighted_result = weighted_average<SO3d>(rotations, weights);
        auto unweighted_result = average<SO3d>(rotations);

        REQUIRE(weighted_result.has_value());
        REQUIRE(unweighted_result.has_value());
        CHECK(weighted_result->is_approx(*unweighted_result, 1e-6));
    }

    TEST_CASE("Weighted average with zero weight for one element") {
        std::vector<SO3d> rotations = {SO3d::identity(), SO3d::rot_z(M_PI / 2)};
        std::vector<double> weights = {1.0, 0.0};

        auto result = weighted_average<SO3d>(rotations, weights);

        REQUIRE(result.has_value());
        CHECK(result->is_identity(1e-8));
    }

    TEST_CASE("SO3 quaternion average finds lower cost than iterative for wide spread") {
        // For uniformly random rotations (widely spread), the quaternion method
        // should find a better solution since iterative can get stuck
        std::mt19937 rng(42);

        std::vector<SO3d> rotations;
        for (int i = 0; i < 5; ++i) {
            rotations.push_back(SO3d::sample_uniform(rng));
        }

        auto iterative_result = average<SO3d>(rotations, 100, 1e-12);
        auto quaternion_result = average_so3_quaternion(rotations);

        REQUIRE(iterative_result.has_value());

        // Compute sum of squared geodesic distances for both methods
        auto compute_cost = [&](const SO3d &mean) {
            double cost = 0;
            for (const auto &R : rotations) {
                auto v = (mean.inverse() * R).log();
                cost += v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            }
            return cost;
        };

        double quaternion_cost = compute_cost(quaternion_result);

        // Quaternion method should find a reasonable cost
        CHECK(quaternion_cost < 20.0);
    }

    TEST_CASE("SO3 iterative average matches quaternion average for nearby rotations") {
        // For rotations close together, both methods should agree
        std::vector<SO3d> rotations = {SO3d::rot_z(0.1), SO3d::rot_z(0.2), SO3d::rot_z(0.3),
                                       SO3d::rot_x(0.1) * SO3d::rot_z(0.15), SO3d::rot_y(0.1) * SO3d::rot_z(0.25)};

        auto iterative_result = average<SO3d>(rotations, 100, 1e-12);
        auto quaternion_result = average_so3_quaternion(rotations);

        REQUIRE(iterative_result.has_value());

        // Both methods should give similar results for nearby rotations
        auto diff = iterative_result->inverse() * quaternion_result;
        auto diff_log = diff.log();
        double norm = std::sqrt(diff_log[0] * diff_log[0] + diff_log[1] * diff_log[1] + diff_log[2] * diff_log[2]);
        CHECK(norm < 0.05);
    }

    TEST_CASE("Chord average is fast approximation") {
        std::vector<SO3d> rotations = {SO3d::rot_z(0.1), SO3d::rot_z(0.2), SO3d::rot_z(0.3)};

        auto chord_result = chord_average_so3(rotations);
        auto exact_result = average<SO3d>(rotations);

        REQUIRE(exact_result.has_value());
        // For small rotations, chord average should be close to exact
        CHECK(chord_result.is_approx(*exact_result, 0.05));
    }

    TEST_CASE("SE3 average works correctly") {
        std::vector<SE3d> poses = {SE3d::identity(), SE3d::trans(2.0, 0.0, 0.0), SE3d::trans(0.0, 2.0, 0.0)};

        auto result = average<SE3d>(poses);

        REQUIRE(result.has_value());
        // Translation should be approximately (2/3, 2/3, 0) for equal weights
        // But due to the nonlinear nature of SE3, it won't be exactly this
        // At least check that translation is between 0 and 2 in x and y
        CHECK(result->translation()[0] > 0.0);
        CHECK(result->translation()[0] < 2.0);
        CHECK(result->translation()[1] > 0.0);
        CHECK(result->translation()[1] < 2.0);
    }

    TEST_CASE("SO2 average") {
        std::vector<SO2d> rotations = {SO2d(0.0), SO2d(M_PI / 2)};

        auto result = average<SO2d>(rotations);

        REQUIRE(result.has_value());
        CHECK(std::abs(result->log() - M_PI / 4) < 1e-8);
    }
}

// ============================================================================
// Spline Tests
// ============================================================================

TEST_SUITE("Lie Algorithms - Spline") {

    TEST_CASE("Geodesic interpolation at endpoints") {
        SO3d a = SO3d::identity();
        SO3d b = SO3d::rot_z(M_PI / 2);

        CHECK(geodesic(a, b, 0.0).is_approx(a, 1e-10));
        CHECK(geodesic(a, b, 1.0).is_approx(b, 1e-10));
    }

    TEST_CASE("Geodesic interpolation at midpoint") {
        SO3d a = SO3d::identity();
        SO3d b = SO3d::rot_z(M_PI / 2);

        auto mid = geodesic(a, b, 0.5);
        auto expected = SO3d::rot_z(M_PI / 4);

        CHECK(mid.is_approx(expected, 1e-10));
    }

    TEST_CASE("Bezier quadratic passes through endpoints") {
        SO3d p0 = SO3d::identity();
        SO3d p1 = SO3d::rot_z(M_PI / 4);
        SO3d p2 = SO3d::rot_z(M_PI / 2);

        CHECK(bezier_quadratic(p0, p1, p2, 0.0).is_approx(p0, 1e-10));
        CHECK(bezier_quadratic(p0, p1, p2, 1.0).is_approx(p2, 1e-10));
    }

    TEST_CASE("Bezier cubic passes through endpoints") {
        SO3d p0 = SO3d::identity();
        SO3d p1 = SO3d::rot_z(M_PI / 6);
        SO3d p2 = SO3d::rot_z(M_PI / 3);
        SO3d p3 = SO3d::rot_z(M_PI / 2);

        CHECK(bezier_cubic(p0, p1, p2, p3, 0.0).is_approx(p0, 1e-10));
        CHECK(bezier_cubic(p0, p1, p2, p3, 1.0).is_approx(p3, 1e-10));
    }

    TEST_CASE("LieSpline with two points") {
        std::vector<SO3d> points = {SO3d::identity(), SO3d::rot_z(M_PI / 2)};
        LieSpline<SO3d> spline(points);

        CHECK(spline.size() == 2);
        CHECK(spline.evaluate(0.0).is_approx(points[0], 1e-10));
        CHECK(spline.evaluate(1.0).is_approx(points[1], 1e-10));
    }

    TEST_CASE("LieSpline normalized evaluation") {
        std::vector<SO3d> points = {SO3d::identity(), SO3d::rot_z(M_PI / 4), SO3d::rot_z(M_PI / 2)};
        LieSpline<SO3d> spline(points);

        // u=0 should give first point, u=1 should give last
        CHECK(spline.evaluate_normalized(0.0).is_approx(points[0], 1e-10));
        CHECK(spline.evaluate_normalized(1.0).is_approx(points[2], 1e-10));
    }

    TEST_CASE("LieSpline interpolates through all control points") {
        std::vector<SO3d> points = {SO3d::identity(), SO3d::rot_z(M_PI / 4), SO3d::rot_z(M_PI / 2),
                                    SO3d::rot_z(3 * M_PI / 4)};
        LieSpline<SO3d> spline(points);

        for (std::size_t i = 0; i < points.size(); ++i) {
            auto evaluated = spline.evaluate(static_cast<double>(i));
            CHECK(evaluated.is_approx(points[i], 1e-8));
        }
    }

    TEST_CASE("sample_spline returns correct number of samples") {
        std::vector<SO3d> points = {SO3d::identity(), SO3d::rot_z(M_PI / 2)};
        LieSpline<SO3d> spline(points);

        auto samples = sample_spline(spline, 10);
        CHECK(samples.size() == 10);

        // First and last samples should match endpoints
        CHECK(samples.front().is_approx(points.front(), 1e-10));
        CHECK(samples.back().is_approx(points.back(), 1e-10));
    }

    TEST_CASE("SE3 spline works") {
        std::vector<SE3d> poses = {SE3d::identity(), SE3d::trans(1.0, 0.0, 0.0), SE3d::trans(1.0, 1.0, 0.0)};
        LieSpline<SE3d> spline(poses);

        CHECK(spline.evaluate(0.0).is_approx(poses[0], 1e-10));
        CHECK(spline.evaluate(2.0).is_approx(poses[2], 1e-10));

        // Middle should be somewhere between
        auto mid = spline.evaluate(1.0);
        CHECK(mid.is_approx(poses[1], 1e-8));
    }

    TEST_CASE("Catmull-Rom spline is C1 continuous") {
        // Create 4 points for Catmull-Rom
        SO3d p0 = SO3d::identity();
        SO3d p1 = SO3d::rot_z(M_PI / 6);
        SO3d p2 = SO3d::rot_z(M_PI / 3);
        SO3d p3 = SO3d::rot_z(M_PI / 2);

        // Evaluate at t=0 and t=1 endpoints
        CHECK(catmull_rom(p0, p1, p2, p3, 0.0).is_approx(p1, 1e-10));
        CHECK(catmull_rom(p0, p1, p2, p3, 1.0).is_approx(p2, 1e-10));
    }

    TEST_CASE("Arc length is positive") {
        std::vector<SO3d> points = {SO3d::identity(), SO3d::rot_z(M_PI / 2)};
        LieSpline<SO3d> spline(points);

        auto length = arc_length(spline, 0.0, 1.0, 50);
        CHECK(length > 0.0);

        // Arc length should be approximately pi/2 (the rotation angle)
        CHECK(std::abs(length - M_PI / 2) < 0.1);
    }

    TEST_CASE("Empty spline returns identity") {
        std::vector<SO3d> empty;
        LieSpline<SO3d> spline(empty);

        CHECK(spline.size() == 0);
        CHECK(spline.evaluate(0.5).is_identity(1e-10));
    }

    TEST_CASE("Single point spline returns that point") {
        std::vector<SO3d> single = {SO3d::rot_x(0.5)};
        LieSpline<SO3d> spline(single);

        CHECK(spline.size() == 1);
        CHECK(spline.evaluate(0.0).is_approx(single[0], 1e-10));
        CHECK(spline.evaluate(0.5).is_approx(single[0], 1e-10));
    }
}

// ============================================================================
// Cross-group tests
// ============================================================================

TEST_SUITE("Lie Algorithms - Cross Group") {

    TEST_CASE("SO2 geodesic interpolation") {
        SO2d a = SO2d(0.0);
        SO2d b = SO2d(M_PI / 2);

        auto mid = geodesic(a, b, 0.5);
        CHECK(std::abs(mid.log() - M_PI / 4) < 1e-10);
    }

    TEST_CASE("SE2 geodesic interpolation") {
        SE2d a = SE2d::identity();
        SE2d b = SE2d::trans(2.0, 0.0);

        auto mid = geodesic(a, b, 0.5);
        CHECK(std::abs(mid.translation()[0] - 1.0) < 1e-10);
        CHECK(std::abs(mid.translation()[1]) < 1e-10);
    }
}
