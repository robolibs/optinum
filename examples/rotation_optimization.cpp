// =============================================================================
// rotation_optimization.cpp
// Demonstrates SO3 rotation optimization using Gauss-Newton / Levenberg-Marquardt
// =============================================================================
//
// This example shows how to optimize rotations on the SO3 manifold:
// 1. Point cloud alignment (3D registration)
// 2. Camera calibration (extrinsic estimation)
// 3. Rotation averaging from multiple measurements

#include <cmath>
#include <iomanip>
#include <iostream>
#include <optinum/lie/lie.hpp>
#include <optinum/optinum.hpp>
#include <random>

using namespace optinum;

// =============================================================================
// Example 1: Point Cloud Alignment
// =============================================================================
// Given two point clouds related by unknown rotation R:
//   p_target = R * p_source
// Find the optimal rotation R that minimizes the sum of squared errors.

void example_point_cloud_alignment() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Example 1: Point Cloud Alignment\n";
    std::cout << "========================================\n\n";

    // Ground truth rotation: 30 deg around axis [1, 1, 1] normalized
    dp::mat::vector<double, 3> axis{1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
    double angle = M_PI / 6; // 30 degrees
    dp::mat::vector<double, 3> omega_true;
    omega_true[0] = axis[0] * angle;
    omega_true[1] = axis[1] * angle;
    omega_true[2] = axis[2] * angle;
    lie::SO3d R_true = lie::SO3d::exp(omega_true);

    std::cout << "Ground truth rotation: 30 deg around [1,1,1]\n\n";

    // Generate source points and transform to target
    std::vector<dp::mat::vector<double, 3>> source_points = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
                                                          {1.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0},
                                                          {1.0, 1.0, 1.0}, {-1.0, 0.5, 0.3}};

    std::vector<dp::mat::vector<double, 3>> target_points;
    for (const auto &p : source_points) {
        target_points.push_back(R_true * p);
    }

    std::cout << "Generated " << source_points.size() << " point correspondences\n\n";

    // Define residual function: r_i = R * p_source_i - p_target_i
    // We parameterize R = R_current * exp(delta) and optimize over delta
    auto make_residual = [&](const lie::SO3d &R_current) {
        return [&, R_current](const dp::mat::vector<double, 3> &delta) {
            lie::SO3d R = R_current * lie::SO3d::exp(delta);
            dp::mat::vector<double, Dynamic> residuals;
            residuals.resize(source_points.size() * 3);

            for (std::size_t i = 0; i < source_points.size(); ++i) {
                auto p_transformed = R * source_points[i];
                residuals[i * 3 + 0] = p_transformed[0] - target_points[i][0];
                residuals[i * 3 + 1] = p_transformed[1] - target_points[i][1];
                residuals[i * 3 + 2] = p_transformed[2] - target_points[i][2];
            }
            return residuals;
        };
    };

    // Iterative optimization on the manifold
    lie::SO3d R_estimate = lie::SO3d::identity(); // Start from identity
    opti::GaussNewton<double> gn;
    gn.max_iterations = 10;
    gn.tolerance = 1e-10;
    gn.verbose = false;

    std::cout << "Optimizing with Gauss-Newton on SO3 manifold...\n\n";

    for (int outer = 0; outer < 5; ++outer) {
        auto residual = make_residual(R_estimate);
        dp::mat::vector<double, 3> delta_init{0.0, 0.0, 0.0};

        auto result = gn.optimize(residual, delta_init);

        // Update on manifold: R = R * exp(delta)
        R_estimate = R_estimate * lie::SO3d::exp(result.x);

        // Compute error
        auto omega_error = (R_true.inverse() * R_estimate).log();
        double error_angle = std::sqrt(omega_error[0] * omega_error[0] + omega_error[1] * omega_error[1] +
                                       omega_error[2] * omega_error[2]);

        std::cout << "  Iteration " << outer + 1 << ": rotation error = " << error_angle * 180.0 / M_PI
                  << " degrees, cost = " << result.final_cost << "\n";

        if (error_angle < 1e-8) {
            break;
        }
    }

    std::cout << "\nFinal rotation quaternion:\n";
    auto q_est = R_estimate.unit_quaternion();
    auto q_true = R_true.unit_quaternion();
    std::cout << "  Estimated: [w=" << q_est.w << ", x=" << q_est.x << ", y=" << q_est.y << ", z=" << q_est.z << "]\n";
    std::cout << "  True:      [w=" << q_true.w << ", x=" << q_true.x << ", y=" << q_true.y << ", z=" << q_true.z
              << "]\n";
}

// =============================================================================
// Example 2: Rotation Averaging
// =============================================================================
// Given noisy rotation measurements R_1, R_2, ..., R_n, find the mean rotation.
// This is useful for sensor fusion, calibration, etc.

void example_rotation_averaging() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Example 2: Rotation Averaging\n";
    std::cout << "========================================\n\n";

    // True rotation
    lie::SO3d R_true = lie::SO3d::rot_z(M_PI / 4) * lie::SO3d::rot_x(M_PI / 6);

    // Generate noisy measurements
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.05); // ~3 degrees std dev

    std::vector<lie::SO3d> measurements;
    const int num_measurements = 10;

    std::cout << "Generating " << num_measurements << " noisy rotation measurements...\n";
    std::cout << "  (noise std dev: ~3 degrees)\n\n";

    for (int i = 0; i < num_measurements; ++i) {
        dp::mat::vector<double, 3> noise_omega{noise(rng), noise(rng), noise(rng)};
        lie::SO3d R_noisy = R_true * lie::SO3d::exp(noise_omega);
        measurements.push_back(R_noisy);
    }

    // Method 1: Use built-in average function
    auto R_avg = lie::average<lie::SO3d>(measurements);
    if (R_avg) {
        auto omega_error = (R_true.inverse() * (*R_avg)).log();
        double error_angle = std::sqrt(omega_error[0] * omega_error[0] + omega_error[1] * omega_error[1] +
                                       omega_error[2] * omega_error[2]);
        std::cout << "Method 1 (iterative mean):\n";
        std::cout << "  Angular error: " << error_angle * 180.0 / M_PI << " degrees\n\n";
    }

    // Method 2: Quaternion averaging (faster, closed-form)
    lie::SO3d R_quat_avg = lie::average_so3_quaternion(measurements);
    {
        auto omega_error = (R_true.inverse() * R_quat_avg).log();
        double error_angle = std::sqrt(omega_error[0] * omega_error[0] + omega_error[1] * omega_error[1] +
                                       omega_error[2] * omega_error[2]);
        std::cout << "Method 2 (quaternion eigenvector):\n";
        std::cout << "  Angular error: " << error_angle * 180.0 / M_PI << " degrees\n\n";
    }

    // Method 3: Chord average (fast approximation)
    lie::SO3d R_chord_avg = lie::chord_average_so3(measurements);
    {
        auto omega_error = (R_true.inverse() * R_chord_avg).log();
        double error_angle = std::sqrt(omega_error[0] * omega_error[0] + omega_error[1] * omega_error[1] +
                                       omega_error[2] * omega_error[2]);
        std::cout << "Method 3 (chord average):\n";
        std::cout << "  Angular error: " << error_angle * 180.0 / M_PI << " degrees\n";
    }
}

// =============================================================================
// Example 3: Camera Calibration (Extrinsic)
// =============================================================================
// Find the rotation between camera and IMU frames from correspondences.

void example_camera_imu_calibration() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Example 3: Camera-IMU Calibration\n";
    std::cout << "========================================\n\n";

    // True camera-to-IMU rotation
    lie::SO3d R_cam_imu_true = lie::SO3d::rot_y(M_PI / 8) * lie::SO3d::rot_z(M_PI / 12);

    std::cout << "True camera-to-IMU rotation:\n";
    auto q_true = R_cam_imu_true.unit_quaternion();
    std::cout << "  [w=" << q_true.w << ", x=" << q_true.x << ", y=" << q_true.y << ", z=" << q_true.z << "]\n\n";

    // Generate paired rotation measurements
    // R_imu = R_cam_imu * R_cam * R_cam_imu^{-1}  (for same physical rotation)
    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, 0.02);

    struct RotationPair {
        lie::SO3d R_cam;
        lie::SO3d R_imu;
    };
    std::vector<RotationPair> pairs;

    std::cout << "Generating 8 rotation pairs with noise...\n\n";

    for (int i = 0; i < 8; ++i) {
        // Random camera rotation
        lie::SO3d R_cam = lie::SO3d::sample_uniform(rng);

        // Corresponding IMU rotation (with noise)
        lie::SO3d R_imu_true = R_cam_imu_true * R_cam * R_cam_imu_true.inverse();
        dp::mat::vector<double, 3> noise_omega{noise(rng), noise(rng), noise(rng)};
        lie::SO3d R_imu = R_imu_true * lie::SO3d::exp(noise_omega);

        pairs.push_back({R_cam, R_imu});
    }

    // Hand-eye calibration residual:
    // R_imu * R_cam_imu = R_cam_imu * R_cam
    // Residual: log(R_imu * R_cam_imu * R_cam^{-1} * R_cam_imu^{-1})
    auto make_calibration_residual = [&](const lie::SO3d &R_current) {
        return [&, R_current](const dp::mat::vector<double, 3> &delta) {
            lie::SO3d R = R_current * lie::SO3d::exp(delta);
            dp::mat::vector<double, Dynamic> residuals;
            residuals.resize(pairs.size() * 3);

            for (std::size_t i = 0; i < pairs.size(); ++i) {
                lie::SO3d error_rot = pairs[i].R_imu * R * pairs[i].R_cam.inverse() * R.inverse();
                auto omega = error_rot.log();
                residuals[i * 3 + 0] = omega[0];
                residuals[i * 3 + 1] = omega[1];
                residuals[i * 3 + 2] = omega[2];
            }
            return residuals;
        };
    };

    // Optimize
    lie::SO3d R_estimate = lie::SO3d::identity();
    opti::LevenbergMarquardt<double> lm;
    lm.max_iterations = 10;
    lm.tolerance = 1e-12;
    lm.verbose = false;

    std::cout << "Optimizing with Levenberg-Marquardt...\n\n";

    for (int outer = 0; outer < 10; ++outer) {
        auto residual = make_calibration_residual(R_estimate);
        dp::mat::vector<double, 3> delta_init{0.0, 0.0, 0.0};

        auto result = lm.optimize(residual, delta_init);
        R_estimate = R_estimate * lie::SO3d::exp(result.x);

        auto omega_error = (R_cam_imu_true.inverse() * R_estimate).log();
        double error_angle = std::sqrt(omega_error[0] * omega_error[0] + omega_error[1] * omega_error[1] +
                                       omega_error[2] * omega_error[2]);

        if (outer < 3 || error_angle < 0.01) {
            std::cout << "  Iteration " << outer + 1 << ": error = " << error_angle * 180.0 / M_PI << " degrees\n";
        }

        if (error_angle < 1e-8) {
            break;
        }
    }

    std::cout << "\nEstimated camera-to-IMU rotation:\n";
    auto q_est = R_estimate.unit_quaternion();
    std::cout << "  [w=" << q_est.w << ", x=" << q_est.x << ", y=" << q_est.y << ", z=" << q_est.z << "]\n";

    auto omega_final = (R_cam_imu_true.inverse() * R_estimate).log();
    double final_error =
        std::sqrt(omega_final[0] * omega_final[0] + omega_final[1] * omega_final[1] + omega_final[2] * omega_final[2]);
    std::cout << "\nFinal angular error: " << final_error * 180.0 / M_PI << " degrees\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "=============================================================================\n";
    std::cout << "                   SO3 ROTATION OPTIMIZATION DEMO\n";
    std::cout << "=============================================================================\n";
    std::cout << "Demonstrating manifold optimization on SO3 using Gauss-Newton and LM\n";

    example_point_cloud_alignment();
    example_rotation_averaging();
    example_camera_imu_calibration();

    std::cout << "\n";
    std::cout << "=============================================================================\n";
    std::cout << "                          DEMO COMPLETE\n";
    std::cout << "=============================================================================\n";

    return 0;
}
