#pragma once

// ===== OPTINUM LIE GROUPS MODULE =====
//
// Lie groups for geometry and robotics:
// - SO2: 2D rotations (unit complex number)
// - SE2: 2D rigid transforms (rotation + translation)
// - SO3: 3D rotations (unit quaternion)
// - SE3: 3D rigid transforms (rotation + translation)
//
// Batched operations (SIMD-accelerated):
// - SO3Batch<T, N>: N rotations processed in parallel
// - SE3Batch<T, N>: N rigid transforms processed in parallel
//
// Algorithms:
// - average: Biinvariant (Frechet/Karcher) mean on Lie groups
// - LieSpline: C1-smooth spline interpolation through control points
// - geodesic, bezier, catmull_rom: Interpolation functions
//
// All groups provide:
// - exp/log maps (tangent space <-> group)
// - Group composition (operator*)
// - Inverse
// - Adjoint representation
// - Jacobians for optimization
//
// SIMD-accelerated via optinum::simd infrastructure.

// Core
#include <optinum/lie/core/concepts.hpp>
#include <optinum/lie/core/constants.hpp>

// Groups (order matters: SO before SE, RxSO before Sim)
// clang-format off
#include <optinum/lie/groups/so2.hpp>
#include <optinum/lie/groups/so3.hpp>
#include <optinum/lie/groups/euler_angles.hpp>
#include <optinum/lie/groups/angular_velocity.hpp>
#include <optinum/lie/groups/se2.hpp>
#include <optinum/lie/groups/se3.hpp>

// Similarity groups (rotation + scale)
#include <optinum/lie/groups/rxso2.hpp>
#include <optinum/lie/groups/rxso3.hpp>
#include <optinum/lie/groups/sim2.hpp>
#include <optinum/lie/groups/sim3.hpp>

// Batched operations (SIMD-accelerated)
#include <optinum/lie/batch/so3_batch.hpp>
#include <optinum/lie/batch/se3_batch.hpp>

// Algorithms
#include <optinum/lie/algorithms/average.hpp>
#include <optinum/lie/algorithms/spline.hpp>
// clang-format on

namespace optinum::lie {

    // Re-export for convenience
    // (types are already in optinum::lie namespace)

} // namespace optinum::lie
