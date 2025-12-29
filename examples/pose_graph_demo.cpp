// =============================================================================
// pose_graph_demo.cpp
// Demonstrates SE3 pose graph optimization using Levenberg-Marquardt
// =============================================================================
//
// This example shows how to optimize poses on the SE3 manifold:
// 1. Simple pose chain optimization
// 2. Loop closure (pose graph with cycles)
// 3. Multi-robot localization

#include <cmath>
#include <iomanip>
#include <iostream>
#include <optinum/lie/lie.hpp>
#include <optinum/optinum.hpp>
#include <random>
#include <vector>

using namespace optinum;

// =============================================================================
// Example 1: Simple Pose Chain
// =============================================================================
// Optimize a chain of poses connected by relative measurements.
// This is the simplest form of pose graph (no loops).

void example_pose_chain() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Example 1: Simple Pose Chain\n";
    std::cout << "========================================\n\n";

    // Ground truth poses: robot moves forward with slight rotations
    std::vector<lie::SE3d> poses_true;
    poses_true.push_back(lie::SE3d::identity());                                  // Pose 0: origin
    poses_true.push_back(lie::SE3d::trans(1.0, 0.0, 0.0));                        // Pose 1
    poses_true.push_back(lie::SE3d(lie::SO3d::rot_z(M_PI / 8), {2.0, 0.2, 0.0})); // Pose 2
    poses_true.push_back(lie::SE3d(lie::SO3d::rot_z(M_PI / 6), {2.8, 0.8, 0.0})); // Pose 3
    poses_true.push_back(lie::SE3d(lie::SO3d::rot_z(M_PI / 4), {3.2, 1.5, 0.0})); // Pose 4

    const int num_poses = static_cast<int>(poses_true.size());
    std::cout << "True pose chain with " << num_poses << " poses\n\n";

    // Generate noisy relative measurements
    std::mt19937 rng(42);
    std::normal_distribution<double> noise_trans(0.0, 0.02);
    std::normal_distribution<double> noise_rot(0.0, 0.01);

    struct RelativeMeasurement {
        int from, to;
        lie::SE3d T_relative;
    };
    std::vector<RelativeMeasurement> measurements;

    std::cout << "Generating noisy odometry measurements...\n\n";

    for (int i = 0; i < num_poses - 1; ++i) {
        // True relative transform
        lie::SE3d T_rel_true = poses_true[i].inverse() * poses_true[i + 1];

        // Add noise
        dp::mat::vector<double, 6> noise_twist;
        noise_twist[0] = noise_trans(rng);
        noise_twist[1] = noise_trans(rng);
        noise_twist[2] = noise_trans(rng);
        noise_twist[3] = noise_rot(rng);
        noise_twist[4] = noise_rot(rng);
        noise_twist[5] = noise_rot(rng);

        lie::SE3d T_rel_noisy = T_rel_true * lie::SE3d::exp(noise_twist);
        measurements.push_back({i, i + 1, T_rel_noisy});
    }

    // Initialize poses from odometry (accumulate relative measurements)
    std::vector<lie::SE3d> poses_estimate(num_poses);
    poses_estimate[0] = lie::SE3d::identity(); // First pose fixed
    for (int i = 1; i < num_poses; ++i) {
        poses_estimate[i] = poses_estimate[i - 1] * measurements[i - 1].T_relative;
    }

    // Define residual function for pose graph optimization
    // Residual for edge (i, j): log(T_i^{-1} * T_j * T_measured^{-1})
    auto make_residual = [&]() {
        return [&](const dp::mat::vector<double, Dynamic> &delta) {
            // Apply delta to poses (except first which is fixed)
            std::vector<lie::SE3d> poses_updated(num_poses);
            poses_updated[0] = poses_estimate[0];
            for (int i = 1; i < num_poses; ++i) {
                dp::mat::vector<double, 6> delta_i;
                for (int j = 0; j < 6; ++j) {
                    delta_i[j] = delta[(i - 1) * 6 + j];
                }
                poses_updated[i] = poses_estimate[i] * lie::SE3d::exp(delta_i);
            }

            // Compute residuals for all edges
            dp::mat::vector<double, Dynamic> residuals;
            residuals.resize(measurements.size() * 6);

            for (std::size_t k = 0; k < measurements.size(); ++k) {
                int i = measurements[k].from;
                int j = measurements[k].to;
                lie::SE3d error = poses_updated[i].inverse() * poses_updated[j] * measurements[k].T_relative.inverse();
                auto twist = error.log();
                for (int d = 0; d < 6; ++d) {
                    residuals[k * 6 + d] = twist[d];
                }
            }
            return residuals;
        };
    };

    // Compute initial error
    auto compute_error = [&](const std::vector<lie::SE3d> &poses) {
        double total_error = 0.0;
        for (int i = 1; i < num_poses; ++i) {
            auto twist = (poses_true[i].inverse() * poses[i]).log();
            total_error += simd::dot(twist, twist);
        }
        return std::sqrt(total_error / (num_poses - 1));
    };

    std::cout << "Initial pose error (RMS): " << compute_error(poses_estimate) << "\n\n";

    // Optimize using Gauss-Newton
    opti::GaussNewton<double> gn;
    gn.max_iterations = 10;
    gn.tolerance = 1e-10;
    gn.verbose = false;

    std::cout << "Optimizing pose graph...\n\n";

    for (int outer = 0; outer < 5; ++outer) {
        auto residual = make_residual();
        dp::mat::vector<double, Dynamic> delta_init;
        delta_init.resize((num_poses - 1) * 6);
        for (std::size_t i = 0; i < delta_init.size(); ++i) {
            delta_init[i] = 0.0;
        }

        auto result = gn.optimize(residual, delta_init);

        // Update poses
        for (int i = 1; i < num_poses; ++i) {
            dp::mat::vector<double, 6> delta_i;
            for (int j = 0; j < 6; ++j) {
                delta_i[j] = result.x[(i - 1) * 6 + j];
            }
            poses_estimate[i] = poses_estimate[i] * lie::SE3d::exp(delta_i);
        }

        double error = compute_error(poses_estimate);
        std::cout << "  Iteration " << outer + 1 << ": RMS error = " << error << "\n";

        if (result.final_cost < 1e-12) {
            break;
        }
    }

    std::cout << "\nFinal poses vs ground truth:\n";
    for (int i = 0; i < num_poses; ++i) {
        auto t_est = poses_estimate[i].translation();
        auto t_true = poses_true[i].translation();
        std::cout << "  Pose " << i << ": est=[" << t_est[0] << ", " << t_est[1] << ", " << t_est[2] << "] "
                  << "true=[" << t_true[0] << ", " << t_true[1] << ", " << t_true[2] << "]\n";
    }
}

// =============================================================================
// Example 2: Loop Closure
// =============================================================================
// Pose graph with a loop - the robot returns to a previously visited location.
// Loop closure constraints help correct accumulated drift.

void example_loop_closure() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Example 2: Loop Closure\n";
    std::cout << "========================================\n\n";

    // Ground truth: square trajectory returning to origin
    std::vector<lie::SE3d> poses_true;
    poses_true.push_back(lie::SE3d::identity());                                   // 0: origin
    poses_true.push_back(lie::SE3d::trans(2.0, 0.0, 0.0));                         // 1: move right
    poses_true.push_back(lie::SE3d(lie::SO3d::rot_z(M_PI / 2), {2.0, 2.0, 0.0}));  // 2: move up, turn left
    poses_true.push_back(lie::SE3d(lie::SO3d::rot_z(M_PI), {0.0, 2.0, 0.0}));      // 3: move left, turn left
    poses_true.push_back(lie::SE3d(lie::SO3d::rot_z(-M_PI / 2), {0.0, 0.0, 0.0})); // 4: back to origin

    const int num_poses = static_cast<int>(poses_true.size());
    std::cout << "Square trajectory with " << num_poses << " poses (returns to origin)\n\n";

    // Generate noisy odometry
    std::mt19937 rng(123);
    std::normal_distribution<double> noise_trans(0.0, 0.05);
    std::normal_distribution<double> noise_rot(0.0, 0.02);

    struct Edge {
        int from, to;
        lie::SE3d T_relative;
        double weight; // Lower weight for loop closure (less certain)
    };
    std::vector<Edge> edges;

    // Odometry edges (sequential)
    for (int i = 0; i < num_poses - 1; ++i) {
        lie::SE3d T_rel_true = poses_true[i].inverse() * poses_true[i + 1];
        dp::mat::vector<double, 6> noise_twist;
        noise_twist[0] = noise_trans(rng);
        noise_twist[1] = noise_trans(rng);
        noise_twist[2] = noise_trans(rng);
        noise_twist[3] = noise_rot(rng);
        noise_twist[4] = noise_rot(rng);
        noise_twist[5] = noise_rot(rng);
        lie::SE3d T_rel_noisy = T_rel_true * lie::SE3d::exp(noise_twist);
        edges.push_back({i, i + 1, T_rel_noisy, 1.0});
    }

    // Loop closure edge: pose 4 sees pose 0 again
    {
        lie::SE3d T_loop_true = poses_true[num_poses - 1].inverse() * poses_true[0];
        dp::mat::vector<double, 6> noise_twist;
        noise_twist[0] = noise_trans(rng) * 0.5; // Loop closure often more accurate
        noise_twist[1] = noise_trans(rng) * 0.5;
        noise_twist[2] = noise_trans(rng) * 0.5;
        noise_twist[3] = noise_rot(rng) * 0.5;
        noise_twist[4] = noise_rot(rng) * 0.5;
        noise_twist[5] = noise_rot(rng) * 0.5;
        lie::SE3d T_loop_noisy = T_loop_true * lie::SE3d::exp(noise_twist);
        edges.push_back({num_poses - 1, 0, T_loop_noisy, 2.0}); // Higher weight for loop closure
    }

    std::cout << "Generated " << edges.size() << " edges (including loop closure)\n\n";

    // Initialize from odometry (will have drift)
    std::vector<lie::SE3d> poses_estimate(num_poses);
    poses_estimate[0] = lie::SE3d::identity();
    for (int i = 1; i < num_poses; ++i) {
        poses_estimate[i] = poses_estimate[i - 1] * edges[i - 1].T_relative;
    }

    // Show drift before optimization
    auto t_final = poses_estimate[num_poses - 1].translation();
    std::cout << "Before loop closure optimization:\n";
    std::cout << "  Final pose (should be near origin): [" << t_final[0] << ", " << t_final[1] << ", " << t_final[2]
              << "]\n\n";

    // Define weighted residual
    auto make_residual = [&]() {
        return [&](const dp::mat::vector<double, Dynamic> &delta) {
            std::vector<lie::SE3d> poses_updated(num_poses);
            poses_updated[0] = poses_estimate[0];
            for (int i = 1; i < num_poses; ++i) {
                dp::mat::vector<double, 6> delta_i;
                for (int j = 0; j < 6; ++j) {
                    delta_i[j] = delta[(i - 1) * 6 + j];
                }
                poses_updated[i] = poses_estimate[i] * lie::SE3d::exp(delta_i);
            }

            dp::mat::vector<double, Dynamic> residuals;
            residuals.resize(edges.size() * 6);

            for (std::size_t k = 0; k < edges.size(); ++k) {
                int i = edges[k].from;
                int j = edges[k].to;
                lie::SE3d error = poses_updated[i].inverse() * poses_updated[j] * edges[k].T_relative.inverse();
                auto twist = error.log();
                double w = std::sqrt(edges[k].weight);
                for (int d = 0; d < 6; ++d) {
                    residuals[k * 6 + d] = w * twist[d];
                }
            }
            return residuals;
        };
    };

    // Optimize
    opti::LevenbergMarquardt<double> lm;
    lm.max_iterations = 20;
    lm.tolerance = 1e-12;
    lm.verbose = false;

    std::cout << "Optimizing with loop closure constraint...\n\n";

    for (int outer = 0; outer < 10; ++outer) {
        auto residual = make_residual();
        dp::mat::vector<double, Dynamic> delta_init;
        delta_init.resize((num_poses - 1) * 6);
        for (std::size_t i = 0; i < delta_init.size(); ++i) {
            delta_init[i] = 0.0;
        }

        auto result = lm.optimize(residual, delta_init);

        for (int i = 1; i < num_poses; ++i) {
            dp::mat::vector<double, 6> delta_i;
            for (int j = 0; j < 6; ++j) {
                delta_i[j] = result.x[(i - 1) * 6 + j];
            }
            poses_estimate[i] = poses_estimate[i] * lie::SE3d::exp(delta_i);
        }

        if (result.final_cost < 1e-10) {
            std::cout << "  Converged at iteration " << outer + 1 << "\n";
            break;
        }
    }

    // Show result
    t_final = poses_estimate[num_poses - 1].translation();
    std::cout << "\nAfter loop closure optimization:\n";
    std::cout << "  Final pose: [" << t_final[0] << ", " << t_final[1] << ", " << t_final[2] << "]\n";

    std::cout << "\nAll poses:\n";
    for (int i = 0; i < num_poses; ++i) {
        auto t = poses_estimate[i].translation();
        auto t_true = poses_true[i].translation();
        double error = std::sqrt((t[0] - t_true[0]) * (t[0] - t_true[0]) + (t[1] - t_true[1]) * (t[1] - t_true[1]) +
                                 (t[2] - t_true[2]) * (t[2] - t_true[2]));
        std::cout << "  Pose " << i << ": [" << t[0] << ", " << t[1] << ", " << t[2] << "] error=" << error << "\n";
    }
}

// =============================================================================
// Example 3: Trajectory Interpolation with Splines
// =============================================================================
// Use Lie group splines for smooth trajectory interpolation.

void example_trajectory_spline() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Example 3: Trajectory Spline\n";
    std::cout << "========================================\n\n";

    // Create waypoints for a trajectory
    std::vector<lie::SE3d> waypoints;
    waypoints.push_back(lie::SE3d::identity());
    waypoints.push_back(lie::SE3d::trans(1.0, 0.0, 0.0));
    waypoints.push_back(lie::SE3d(lie::SO3d::rot_z(M_PI / 4), {2.0, 0.5, 0.0}));
    waypoints.push_back(lie::SE3d(lie::SO3d::rot_z(M_PI / 2), {2.0, 2.0, 0.0}));
    waypoints.push_back(lie::SE3d(lie::SO3d::rot_z(M_PI), {0.0, 2.0, 0.0}));

    std::cout << "Created trajectory with " << waypoints.size() << " waypoints\n\n";

    // Create spline
    lie::LieSpline<lie::SE3d> spline(waypoints);

    // Sample the spline at various points
    std::cout << "Sampling spline at uniform intervals:\n";
    const int num_samples = 9;
    for (int i = 0; i <= num_samples; ++i) {
        double u = static_cast<double>(i) / num_samples;
        lie::SE3d pose = spline.evaluate_normalized(u);
        auto t = pose.translation();
        auto omega = pose.so3().log();
        double angle = std::sqrt(omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]);
        std::cout << "  u=" << u << ": pos=[" << t[0] << ", " << t[1] << ", " << t[2]
                  << "] rot=" << angle * 180.0 / M_PI << " deg\n";
    }

    // Compute arc length
    double arc_len = lie::arc_length(spline, 0.0, static_cast<double>(waypoints.size() - 1), 100);
    std::cout << "\nApproximate arc length: " << arc_len << "\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "=============================================================================\n";
    std::cout << "                    SE3 POSE GRAPH OPTIMIZATION DEMO\n";
    std::cout << "=============================================================================\n";
    std::cout << "Demonstrating pose graph optimization, loop closure, and splines\n";

    example_pose_chain();
    example_loop_closure();
    example_trajectory_spline();

    std::cout << "\n";
    std::cout << "=============================================================================\n";
    std::cout << "                          DEMO COMPLETE\n";
    std::cout << "=============================================================================\n";

    return 0;
}
