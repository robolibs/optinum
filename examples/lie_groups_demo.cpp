// =============================================================================
// lie_groups_demo.cpp
// Demonstrates basic Lie group operations: SO2, SE2, SO3, SE3, and similarity groups
// =============================================================================

#include <cmath>
#include <iomanip>
#include <iostream>
#include <optinum/lie/lie.hpp>
#include <optinum/simd/simd.hpp>

namespace dp = datapod;
namespace lie = optinum::lie;
namespace simd = optinum::simd;

// =============================================================================
// Example 1: SO2 - 2D Rotations
// =============================================================================

void example_so2() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Example 1: SO2 - 2D Rotations\n";
    std::cout << "========================================\n\n";

    // Create rotations from angles
    lie::SO2d R1(M_PI / 4); // 45 degrees
    lie::SO2d R2(M_PI / 6); // 30 degrees

    std::cout << "R1 = rotation by 45 degrees\n";
    std::cout << "R2 = rotation by 30 degrees\n\n";

    // Composition
    lie::SO2d R3 = R1 * R2;
    std::cout << "R1 * R2 angle: " << R3.angle() * 180.0 / M_PI << " degrees\n";
    std::cout << "  (expected: 75 degrees)\n\n";

    // Inverse
    lie::SO2d R1_inv = R1.inverse();
    lie::SO2d identity = R1 * R1_inv;
    std::cout << "R1 * R1.inverse() angle: " << identity.angle() * 180.0 / M_PI << " degrees\n";
    std::cout << "  (expected: 0 degrees)\n\n";

    // Exp/Log maps
    double theta = M_PI / 3; // 60 degrees
    lie::SO2d R_exp = lie::SO2d::exp(theta);
    double theta_back = R_exp.log();
    std::cout << "exp(60 deg) -> log() = " << theta_back * 180.0 / M_PI << " degrees\n\n";

    // Rotate a point using operator*
    dp::mat::Vector<double, 2> p{1.0, 0.0};
    dp::mat::Vector<double, 2> p_rotated = R1 * p;
    std::cout << "Rotate [1, 0] by 45 degrees: [" << p_rotated[0] << ", " << p_rotated[1] << "]\n";
    std::cout << "  (expected: [0.707, 0.707])\n";
}

// =============================================================================
// Example 2: SE2 - 2D Rigid Transforms
// =============================================================================

void example_se2() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Example 2: SE2 - 2D Rigid Transforms\n";
    std::cout << "========================================\n\n";

    // Create poses
    lie::SE2d T1 = lie::SE2d::trans(1.0, 2.0);                 // Pure translation
    lie::SE2d T2 = lie::SE2d(lie::SO2d(M_PI / 2), {0.0, 0.0}); // Pure rotation (90 deg)
    lie::SE2d T3 = lie::SE2d(lie::SO2d(M_PI / 4), {1.0, 0.0}); // Rotation + translation

    std::cout << "T1 = translation by [1, 2]\n";
    std::cout << "T2 = rotation by 90 degrees\n";
    std::cout << "T3 = rotation by 45 degrees + translation [1, 0]\n\n";

    // Transform a point using operator*
    dp::mat::Vector<double, 2> p{1.0, 0.0};

    auto p1 = T1 * p;
    std::cout << "T1 * [1, 0] = [" << p1[0] << ", " << p1[1] << "]\n";
    std::cout << "  (expected: [2, 2])\n\n";

    auto p2 = T2 * p;
    std::cout << "T2 * [1, 0] = [" << p2[0] << ", " << p2[1] << "]\n";
    std::cout << "  (expected: [0, 1])\n\n";

    // Composition
    lie::SE2d T_composed = T2 * T1; // First translate, then rotate
    auto p_composed = T_composed * dp::mat::Vector<double, 2>{0.0, 0.0};
    std::cout << "T2 * T1 applied to origin: [" << p_composed[0] << ", " << p_composed[1] << "]\n";
    std::cout << "  (T1 moves to [1,2], T2 rotates to [-2, 1])\n\n";

    // Log map (tangent vector)
    auto twist = T3.log();
    std::cout << "T3.log() = [" << twist[0] << ", " << twist[1] << ", " << twist[2] << "]\n";
    std::cout << "  (twist: [vx, vy, theta])\n";
}

// =============================================================================
// Example 3: SO3 - 3D Rotations
// =============================================================================

void example_so3() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Example 3: SO3 - 3D Rotations\n";
    std::cout << "========================================\n\n";

    // Create rotations around axes
    lie::SO3d Rx = lie::SO3d::rot_x(M_PI / 4); // 45 deg around X
    lie::SO3d Ry = lie::SO3d::rot_y(M_PI / 4); // 45 deg around Y
    lie::SO3d Rz = lie::SO3d::rot_z(M_PI / 2); // 90 deg around Z

    std::cout << "Rx = 45 deg rotation around X axis\n";
    std::cout << "Ry = 45 deg rotation around Y axis\n";
    std::cout << "Rz = 90 deg rotation around Z axis\n\n";

    // Rotate a point using operator*
    dp::mat::Vector<double, 3> p{1.0, 0.0, 0.0};

    auto p_rz = Rz * p;
    std::cout << "Rz * [1, 0, 0] = [" << p_rz[0] << ", " << p_rz[1] << ", " << p_rz[2] << "]\n";
    std::cout << "  (expected: [0, 1, 0])\n\n";

    // Composition (Euler angles: Z-Y-X)
    lie::SO3d R_euler = Rz * Ry * Rx;
    std::cout << "Combined rotation Rz * Ry * Rx created\n\n";

    // Exp/Log maps (axis-angle)
    dp::mat::Vector<double, 3> omega{0.0, 0.0, M_PI / 2}; // 90 deg around Z
    lie::SO3d R_from_omega = lie::SO3d::exp(omega);
    auto omega_back = R_from_omega.log();
    std::cout << "exp([0, 0, pi/2]) -> log() = [" << omega_back[0] << ", " << omega_back[1] << ", " << omega_back[2]
              << "]\n\n";

    // Quaternion access
    auto q = Rz.unit_quaternion();
    std::cout << "Rz quaternion: [w=" << q.w << ", x=" << q.x << ", y=" << q.y << ", z=" << q.z << "]\n";
    std::cout << "  (expected: w=0.707, z=0.707)\n";
}

// =============================================================================
// Example 4: SE3 - 3D Rigid Transforms
// =============================================================================

void example_se3() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Example 4: SE3 - 3D Rigid Transforms\n";
    std::cout << "========================================\n\n";

    // Create poses
    lie::SE3d T1 = lie::SE3d::trans(1.0, 2.0, 3.0);                        // Pure translation
    lie::SE3d T2 = lie::SE3d(lie::SO3d::rot_z(M_PI / 2), {0.0, 0.0, 0.0}); // Pure rotation
    lie::SE3d T3 = lie::SE3d(lie::SO3d::rot_x(M_PI / 4), {1.0, 0.0, 0.0}); // Camera-like pose

    std::cout << "T1 = translation by [1, 2, 3]\n";
    std::cout << "T2 = 90 deg rotation around Z\n";
    std::cout << "T3 = 45 deg rotation around X + translation [1, 0, 0]\n\n";

    // Transform a point using operator*
    dp::mat::Vector<double, 3> p{1.0, 0.0, 0.0};

    auto p1 = T1 * p;
    std::cout << "T1 * [1, 0, 0] = [" << p1[0] << ", " << p1[1] << ", " << p1[2] << "]\n";
    std::cout << "  (expected: [2, 2, 3])\n\n";

    auto p2 = T2 * p;
    std::cout << "T2 * [1, 0, 0] = [" << p2[0] << ", " << p2[1] << ", " << p2[2] << "]\n";
    std::cout << "  (expected: [0, 1, 0])\n\n";

    // Inverse
    lie::SE3d T1_inv = T1.inverse();
    auto origin_back = T1_inv * p1;
    std::cout << "T1.inverse() * T1 * p = [" << origin_back[0] << ", " << origin_back[1] << ", " << origin_back[2]
              << "]\n";
    std::cout << "  (expected: [1, 0, 0])\n\n";

    // Log map (6D twist)
    auto twist = T3.log();
    std::cout << "T3.log() = [" << twist[0] << ", " << twist[1] << ", " << twist[2] << ", " << twist[3] << ", "
              << twist[4] << ", " << twist[5] << "]\n";
    std::cout << "  (twist: [vx, vy, vz, wx, wy, wz])\n";
}

// =============================================================================
// Example 5: Similarity Groups (Sim3)
// =============================================================================

void example_similarity() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Example 5: Sim3 - Similarity Transforms\n";
    std::cout << "========================================\n\n";

    // Create similarity transform: scale + rotation + translation
    double scale = 2.0;
    lie::SO3d R = lie::SO3d::rot_z(M_PI / 4); // 45 deg rotation
    dp::mat::Vector<double, 3> t{1.0, 0.0, 0.0};

    lie::Sim3d S = lie::Sim3d(lie::RxSO3d(scale, R), t);

    std::cout << "Sim3 with scale=2, rotation=45 deg around Z, translation=[1, 0, 0]\n\n";

    // Transform a point using operator*
    dp::mat::Vector<double, 3> p{1.0, 0.0, 0.0};
    auto p_transformed = S * p;
    std::cout << "S * [1, 0, 0] = [" << p_transformed[0] << ", " << p_transformed[1] << ", " << p_transformed[2]
              << "]\n";
    std::cout << "  (scale * rotate * p + t = 2 * [0.707, 0.707, 0] + [1, 0, 0])\n\n";

    // Extract components
    std::cout << "Scale: " << S.scale() << "\n";
    std::cout << "Translation: [" << S.translation()[0] << ", " << S.translation()[1] << ", " << S.translation()[2]
              << "]\n";
}

// =============================================================================
// Example 6: Interpolation and Averaging
// =============================================================================

void example_interpolation() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Example 6: Interpolation & Averaging\n";
    std::cout << "========================================\n\n";

    // Geodesic interpolation on SO3
    lie::SO3d R1 = lie::SO3d::identity();
    lie::SO3d R2 = lie::SO3d::rot_z(M_PI / 2); // 90 degrees

    std::cout << "Interpolating between identity and 90 deg Z rotation:\n";
    for (double t = 0.0; t <= 1.0; t += 0.25) {
        lie::SO3d R_interp = lie::geodesic(R1, R2, t);
        auto omega = R_interp.log();
        double angle = std::sqrt(omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]);
        std::cout << "  t=" << t << ": angle = " << angle * 180.0 / M_PI << " degrees\n";
    }
    std::cout << "\n";

    // Average of rotations
    std::vector<lie::SO3d> rotations = {lie::SO3d::rot_z(0.1), lie::SO3d::rot_z(0.2), lie::SO3d::rot_z(0.3)};

    auto avg = lie::average<lie::SO3d>(rotations);
    if (avg) {
        auto omega = avg->log();
        double angle = std::sqrt(omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]);
        std::cout << "Average of rotations [0.1, 0.2, 0.3] rad around Z:\n";
        std::cout << "  Result angle: " << angle << " rad (expected: ~0.2)\n";
    }
}

// =============================================================================
// Example 7: Batched Operations (SIMD)
// =============================================================================

void example_batched() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Example 7: Batched SIMD Operations\n";
    std::cout << "========================================\n\n";

    // Create batch of 4 rotations
    lie::SO3Batch<double, 4> rotations;
    rotations.set(0, lie::SO3d::rot_z(0.0));
    rotations.set(1, lie::SO3d::rot_z(M_PI / 4));
    rotations.set(2, lie::SO3d::rot_z(M_PI / 2));
    rotations.set(3, lie::SO3d::rot_z(M_PI));

    std::cout << "Batch of 4 rotations: [0, 45, 90, 180] degrees around Z\n\n";

    // Rotate 4 points in parallel (SIMD)
    double vx[4] = {1.0, 1.0, 1.0, 1.0};
    double vy[4] = {0.0, 0.0, 0.0, 0.0};
    double vz[4] = {0.0, 0.0, 0.0, 0.0};

    rotations.rotate(vx, vy, vz);

    std::cout << "Rotating [1, 0, 0] by each rotation:\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "  R" << i << " * [1,0,0] = [" << vx[i] << ", " << vy[i] << ", " << vz[i] << "]\n";
    }
    std::cout << "\n";

    // Batch inverse
    auto inv_rotations = rotations.inverse();
    std::cout << "Inverse rotations computed in parallel (SIMD)\n";

    // Batch composition
    auto composed = rotations * inv_rotations;
    std::cout << "R * R.inverse() = identity (all 4 computed in parallel)\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "=============================================================================\n";
    std::cout << "                     OPTINUM LIE GROUPS DEMO\n";
    std::cout << "=============================================================================\n";
    std::cout << "Demonstrating SO2, SE2, SO3, SE3, Sim3, interpolation, and batched SIMD ops\n";

    example_so2();
    example_se2();
    example_so3();
    example_se3();
    example_similarity();
    example_interpolation();
    example_batched();

    std::cout << "\n";
    std::cout << "=============================================================================\n";
    std::cout << "                          DEMO COMPLETE\n";
    std::cout << "=============================================================================\n";

    return 0;
}
