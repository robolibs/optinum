#pragma once

// ===== OPTINUM LIE GROUPS MODULE =====
//
// Lie groups for geometry and robotics:
// - SO2: 2D rotations (unit complex number)
// - SE2: 2D rigid transforms (rotation + translation) [TODO]
// - SO3: 3D rotations (unit quaternion) [TODO]
// - SE3: 3D rigid transforms (rotation + translation) [TODO]
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

// Groups
#include <optinum/lie/groups/so2.hpp>
// #include <optinum/lie/groups/se2.hpp>  // TODO
// #include <optinum/lie/groups/so3.hpp>  // TODO
// #include <optinum/lie/groups/se3.hpp>  // TODO

namespace optinum::lie {

    // Re-export for convenience
    // (types are already in optinum::lie namespace)

} // namespace optinum::lie
