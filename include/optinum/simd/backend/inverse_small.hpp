#pragma once

// =============================================================================
// optinum/simd/backend/inverse_small.hpp
// Specialized inverse kernels for small matrices (2x2, 3x3, 4x4)
// Direct formulas using adjugate matrix - much faster than general LU solve
// =============================================================================

#include <cmath>
#include <cstddef>
#include <optinum/simd/arch/macros.hpp>
#include <optinum/simd/backend/det_small.hpp>

namespace optinum::simd::backend {

    // =========================================================================
    // 2x2 Inverse
    // =========================================================================
    // Formula: A^-1 = (1/det(A)) * adj(A)
    // where adj(A) = | a11  -a01 |
    //                | -a10  a00 |
    //
    // Matrix layout (column-major):
    //   | a00  a01 |
    //   | a10  a11 |

    template <typename T>
    [[nodiscard]] OPTINUM_INLINE constexpr bool inverse_2x2(const T *data, T *result, T epsilon = T(1e-10)) noexcept {
        // Column-major: [a00, a10, a01, a11]
        const T a00 = data[0];
        const T a10 = data[1];
        const T a01 = data[2];
        const T a11 = data[3];

        // Compute determinant
        const T det = a00 * a11 - a01 * a10;

        // Check for singularity
        if (std::abs(det) < epsilon) {
            return false;
        }

        const T inv_det = T(1) / det;

        // Compute adjugate and scale by 1/det
        // Column 0
        result[0] = a11 * inv_det;  // inv_00
        result[1] = -a10 * inv_det; // inv_10

        // Column 1
        result[2] = -a01 * inv_det; // inv_01
        result[3] = a00 * inv_det;  // inv_11

        return true;
    }

    // =========================================================================
    // 3x3 Inverse
    // =========================================================================
    // Formula: A^-1 = (1/det(A)) * adj(A)
    // where adj(A) is the transpose of the cofactor matrix
    //
    // Matrix layout (column-major):
    //   | a00  a01  a02 |
    //   | a10  a11  a12 |
    //   | a20  a21  a22 |

    template <typename T>
    [[nodiscard]] OPTINUM_INLINE constexpr bool inverse_3x3(const T *data, T *result, T epsilon = T(1e-10)) noexcept {
        // Column-major: [a00, a10, a20, a01, a11, a21, a02, a12, a22]
        const T a00 = data[0];
        const T a10 = data[1];
        const T a20 = data[2];
        const T a01 = data[3];
        const T a11 = data[4];
        const T a21 = data[5];
        const T a02 = data[6];
        const T a12 = data[7];
        const T a22 = data[8];

        // Compute cofactors (2x2 minors with alternating signs)
        const T c00 = a11 * a22 - a12 * a21;    // + minor(1,1,2,2)
        const T c01 = -(a10 * a22 - a12 * a20); // - minor(1,0,2,2)
        const T c02 = a10 * a21 - a11 * a20;    // + minor(1,0,2,1)

        // Compute determinant using first row
        const T det = a00 * c00 + a01 * c01 + a02 * c02;

        // Check for singularity
        if (std::abs(det) < epsilon) {
            return false;
        }

        const T inv_det = T(1) / det;

        // Compute remaining cofactors
        const T c10 = -(a01 * a22 - a02 * a21);
        const T c11 = a00 * a22 - a02 * a20;
        const T c12 = -(a00 * a21 - a01 * a20);

        const T c20 = a01 * a12 - a02 * a11;
        const T c21 = -(a00 * a12 - a02 * a10);
        const T c22 = a00 * a11 - a01 * a10;

        // Transpose of cofactor matrix scaled by 1/det (adjugate scaled)
        // Column 0
        result[0] = c00 * inv_det;
        result[1] = c01 * inv_det;
        result[2] = c02 * inv_det;

        // Column 1
        result[3] = c10 * inv_det;
        result[4] = c11 * inv_det;
        result[5] = c12 * inv_det;

        // Column 2
        result[6] = c20 * inv_det;
        result[7] = c21 * inv_det;
        result[8] = c22 * inv_det;

        return true;
    }

    // =========================================================================
    // 4x4 Inverse
    // =========================================================================
    // Formula: A^-1 = (1/det(A)) * adj(A)
    // Uses optimized algorithm with 2x2 determinant pairs
    //
    // Matrix layout (column-major):
    //   | a00  a01  a02  a03 |
    //   | a10  a11  a12  a13 |
    //   | a20  a21  a22  a23 |
    //   | a30  a31  a32  a33 |

    template <typename T>
    [[nodiscard]] OPTINUM_INLINE constexpr bool inverse_4x4(const T *data, T *result, T epsilon = T(1e-10)) noexcept {
        // Column-major layout
        const T a00 = data[0];
        const T a10 = data[1];
        const T a20 = data[2];
        const T a30 = data[3];
        const T a01 = data[4];
        const T a11 = data[5];
        const T a21 = data[6];
        const T a31 = data[7];
        const T a02 = data[8];
        const T a12 = data[9];
        const T a22 = data[10];
        const T a32 = data[11];
        const T a03 = data[12];
        const T a13 = data[13];
        const T a23 = data[14];
        const T a33 = data[15];

        // Compute pairs of 2x2 sub-determinants
        const T s0 = a00 * a11 - a01 * a10;
        const T s1 = a00 * a12 - a02 * a10;
        const T s2 = a00 * a13 - a03 * a10;
        const T s3 = a01 * a12 - a02 * a11;
        const T s4 = a01 * a13 - a03 * a11;
        const T s5 = a02 * a13 - a03 * a12;

        const T c0 = a20 * a31 - a21 * a30;
        const T c1 = a20 * a32 - a22 * a30;
        const T c2 = a20 * a33 - a23 * a30;
        const T c3 = a21 * a32 - a22 * a31;
        const T c4 = a21 * a33 - a23 * a31;
        const T c5 = a22 * a33 - a23 * a32;

        // Compute determinant
        const T det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;

        // Check for singularity
        if (std::abs(det) < epsilon) {
            return false;
        }

        const T inv_det = T(1) / det;

        // Compute inverse using cofactor expansion
        // This is the adjugate matrix scaled by 1/det

        // Row 0
        result[0] = (a11 * c5 - a12 * c4 + a13 * c3) * inv_det;
        result[4] = (-a01 * c5 + a02 * c4 - a03 * c3) * inv_det;
        result[8] = (a31 * s5 - a32 * s4 + a33 * s3) * inv_det;
        result[12] = (-a21 * s5 + a22 * s4 - a23 * s3) * inv_det;

        // Row 1
        result[1] = (-a10 * c5 + a12 * c2 - a13 * c1) * inv_det;
        result[5] = (a00 * c5 - a02 * c2 + a03 * c1) * inv_det;
        result[9] = (-a30 * s5 + a32 * s2 - a33 * s1) * inv_det;
        result[13] = (a20 * s5 - a22 * s2 + a23 * s1) * inv_det;

        // Row 2
        result[2] = (a10 * c4 - a11 * c2 + a13 * c0) * inv_det;
        result[6] = (-a00 * c4 + a01 * c2 - a03 * c0) * inv_det;
        result[10] = (a30 * s4 - a31 * s2 + a33 * s0) * inv_det;
        result[14] = (-a20 * s4 + a21 * s2 - a23 * s0) * inv_det;

        // Row 3
        result[3] = (-a10 * c3 + a11 * c1 - a12 * c0) * inv_det;
        result[7] = (a00 * c3 - a01 * c1 + a02 * c0) * inv_det;
        result[11] = (-a30 * s3 + a31 * s1 - a32 * s0) * inv_det;
        result[15] = (a20 * s3 - a21 * s1 + a22 * s0) * inv_det;

        return true;
    }

} // namespace optinum::simd::backend
