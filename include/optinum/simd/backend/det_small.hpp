#pragma once

// =============================================================================
// optinum/simd/backend/det_small.hpp
// Specialized determinant kernels for small matrices (2x2, 3x3, 4x4)
// Direct formulas - much faster than general LU decomposition
// =============================================================================

#include <cstddef>
#include <optinum/simd/arch/macros.hpp>

namespace optinum::simd::backend {

    // =========================================================================
    // 2x2 Determinant
    // =========================================================================
    // Formula: det(A) = a00*a11 - a01*a10
    // Matrix layout (column-major):
    //   | a00  a01 |
    //   | a10  a11 |

    template <typename T> [[nodiscard]] OPTINUM_INLINE constexpr T det_2x2(const T *data) noexcept {
        // Column-major: [a00, a10, a01, a11]
        const T a00 = data[0];
        const T a10 = data[1];
        const T a01 = data[2];
        const T a11 = data[3];

        return a00 * a11 - a01 * a10;
    }

    // =========================================================================
    // 3x3 Determinant
    // =========================================================================
    // Formula (Sarrus' rule / cofactor expansion):
    // det(A) = a00*(a11*a22 - a12*a21) - a01*(a10*a22 - a12*a20) + a02*(a10*a21 - a11*a20)
    //
    // Matrix layout (column-major):
    //   | a00  a01  a02 |
    //   | a10  a11  a12 |
    //   | a20  a21  a22 |

    template <typename T> [[nodiscard]] OPTINUM_INLINE constexpr T det_3x3(const T *data) noexcept {
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

        // Compute 2x2 minors
        const T minor00 = a11 * a22 - a12 * a21; // det of [[a11,a12],[a21,a22]]
        const T minor01 = a10 * a22 - a12 * a20; // det of [[a10,a12],[a20,a22]]
        const T minor02 = a10 * a21 - a11 * a20; // det of [[a10,a11],[a20,a21]]

        // Cofactor expansion along first row
        return a00 * minor00 - a01 * minor01 + a02 * minor02;
    }

    // =========================================================================
    // 4x4 Determinant
    // =========================================================================
    // Formula: Cofactor expansion along first row
    // det(A) = a00*M00 - a01*M01 + a02*M02 - a03*M03
    // where M_ij are 3x3 minors
    //
    // Matrix layout (column-major):
    //   | a00  a01  a02  a03 |
    //   | a10  a11  a12  a13 |
    //   | a20  a21  a22  a23 |
    //   | a30  a31  a32  a33 |

    template <typename T> [[nodiscard]] OPTINUM_INLINE constexpr T det_4x4(const T *data) noexcept {
        // Column-major: [a00,a10,a20,a30, a01,a11,a21,a31, a02,a12,a22,a32, a03,a13,a23,a33]
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

        // Compute 2x2 sub-determinants (used multiple times)
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

        // Determinant = dot product of 2x2 determinants
        // det = s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0
        return s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    }

} // namespace optinum::simd::backend
