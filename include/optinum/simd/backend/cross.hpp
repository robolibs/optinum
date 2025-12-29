#pragma once

// =============================================================================
// optinum/simd/backend/cross.hpp
// 3D cross product operations for Lie groups (SO3, SE3, Sim3)
// =============================================================================

#include <optinum/simd/backend/backend.hpp>

namespace optinum::simd::backend {

    /**
     * Compute 3D cross product: out = a × b
     *
     * Formula: a × b = [a1*b2 - a2*b1, a2*b0 - a0*b2, a0*b1 - a1*b0]
     *
     * This is a fundamental operation in Lie group computations:
     * - SO3::rotate uses cross product for Rodrigues formula
     * - SE3::exp/log use cross products for twist operations
     * - Angular velocity computations
     *
     * @param a First 3D vector
     * @param b Second 3D vector
     * @param out Result vector (a × b)
     */
    template <typename T>
    OPTINUM_INLINE void cross(const T *OPTINUM_RESTRICT a, const T *OPTINUM_RESTRICT b,
                              T *OPTINUM_RESTRICT out) noexcept {
        // a × b = [a1*b2 - a2*b1, a2*b0 - a0*b2, a0*b1 - a1*b0]
        out[0] = a[1] * b[2] - a[2] * b[1];
        out[1] = a[2] * b[0] - a[0] * b[2];
        out[2] = a[0] * b[1] - a[1] * b[0];
    }

    /**
     * Compute 3D cross product and add to existing vector: out += a × b
     *
     * Useful for accumulating cross products in iterative algorithms.
     *
     * @param a First 3D vector
     * @param b Second 3D vector
     * @param out Result vector (out += a × b)
     */
    template <typename T>
    OPTINUM_INLINE void cross_add(const T *OPTINUM_RESTRICT a, const T *OPTINUM_RESTRICT b,
                                  T *OPTINUM_RESTRICT out) noexcept {
        out[0] += a[1] * b[2] - a[2] * b[1];
        out[1] += a[2] * b[0] - a[0] * b[2];
        out[2] += a[0] * b[1] - a[1] * b[0];
    }

    /**
     * Compute 3D cross product with scalar: out = s * (a × b)
     *
     * Common in Rodrigues formula: sin(θ) * (ω × v)
     *
     * @param a First 3D vector
     * @param b Second 3D vector
     * @param s Scalar multiplier
     * @param out Result vector (s * (a × b))
     */
    template <typename T>
    OPTINUM_INLINE void cross_scale(const T *OPTINUM_RESTRICT a, const T *OPTINUM_RESTRICT b, T s,
                                    T *OPTINUM_RESTRICT out) noexcept {
        out[0] = s * (a[1] * b[2] - a[2] * b[1]);
        out[1] = s * (a[2] * b[0] - a[0] * b[2]);
        out[2] = s * (a[0] * b[1] - a[1] * b[0]);
    }

    /**
     * Compute 3D cross product with scalar and add: out += s * (a × b)
     *
     * Common in Rodrigues formula accumulation.
     *
     * @param a First 3D vector
     * @param b Second 3D vector
     * @param s Scalar multiplier
     * @param out Result vector (out += s * (a × b))
     */
    template <typename T>
    OPTINUM_INLINE void cross_scale_add(const T *OPTINUM_RESTRICT a, const T *OPTINUM_RESTRICT b, T s,
                                        T *OPTINUM_RESTRICT out) noexcept {
        out[0] += s * (a[1] * b[2] - a[2] * b[1]);
        out[1] += s * (a[2] * b[0] - a[0] * b[2]);
        out[2] += s * (a[0] * b[1] - a[1] * b[0]);
    }

    /**
     * Batched cross product: compute W cross products in parallel using SIMD
     *
     * For W vectors a_i and b_i, compute out_i = a_i × b_i simultaneously.
     * This is useful when processing multiple rotations or transformations.
     *
     * Data layout: SoA (Structure of Arrays)
     * - ax, ay, az: x, y, z components of W vectors a
     * - bx, by, bz: x, y, z components of W vectors b
     * - ox, oy, oz: x, y, z components of W output vectors
     *
     * @tparam T Scalar type (float or double)
     * @tparam W SIMD width (number of vectors to process in parallel)
     */
    template <typename T, std::size_t W>
    OPTINUM_INLINE void cross_batch(const pack<T, W> &ax, const pack<T, W> &ay, const pack<T, W> &az,
                                    const pack<T, W> &bx, const pack<T, W> &by, const pack<T, W> &bz, pack<T, W> &ox,
                                    pack<T, W> &oy, pack<T, W> &oz) noexcept {
        // ox = ay*bz - az*by
        // oy = az*bx - ax*bz
        // oz = ax*by - ay*bx
        ox = ay * bz - az * by;
        oy = az * bx - ax * bz;
        oz = ax * by - ay * bx;
    }

    /**
     * Batched cross product with FMA: compute W cross products using fused multiply-add
     *
     * Uses FMA for better performance and accuracy:
     * ox = fma(ay, bz, -az*by) = ay*bz - az*by
     *
     * @tparam T Scalar type (float or double)
     * @tparam W SIMD width
     */
    template <typename T, std::size_t W>
    OPTINUM_INLINE void cross_batch_fma(const pack<T, W> &ax, const pack<T, W> &ay, const pack<T, W> &az,
                                        const pack<T, W> &bx, const pack<T, W> &by, const pack<T, W> &bz,
                                        pack<T, W> &ox, pack<T, W> &oy, pack<T, W> &oz) noexcept {
        // ox = ay*bz - az*by = fms(ay, bz, az*by)
        // Using: fma(a, b, c) = a*b + c, so we need fma(ay, bz, -(az*by))
        ox = pack<T, W>::fma(ay, bz, -(az * by));
        oy = pack<T, W>::fma(az, bx, -(ax * bz));
        oz = pack<T, W>::fma(ax, by, -(ay * bx));
    }

    /**
     * Compute skew-symmetric matrix from 3D vector (hat operator)
     *
     * For ω = [ω0, ω1, ω2], the skew-symmetric matrix is:
     *   [ω]× = [ 0   -ω2   ω1]
     *          [ ω2   0   -ω0]
     *          [-ω1   ω0   0 ]
     *
     * This is used in SO3::exp and SE3::exp for the Rodrigues formula.
     * The matrix is stored in column-major order.
     *
     * @param omega 3D vector
     * @param out 3x3 skew-symmetric matrix (column-major, 9 elements)
     */
    template <typename T> OPTINUM_INLINE void skew(const T *OPTINUM_RESTRICT omega, T *OPTINUM_RESTRICT out) noexcept {
        // Column-major layout:
        // out[0] = (0,0), out[1] = (1,0), out[2] = (2,0)
        // out[3] = (0,1), out[4] = (1,1), out[5] = (2,1)
        // out[6] = (0,2), out[7] = (1,2), out[8] = (2,2)
        out[0] = T{0};
        out[1] = omega[2];
        out[2] = -omega[1];
        out[3] = -omega[2];
        out[4] = T{0};
        out[5] = omega[0];
        out[6] = omega[1];
        out[7] = -omega[0];
        out[8] = T{0};
    }

    /**
     * Extract vector from skew-symmetric matrix (vee operator)
     *
     * Inverse of skew(): extracts ω from [ω]×
     *
     * @param mat 3x3 skew-symmetric matrix (column-major)
     * @param omega Output 3D vector
     */
    template <typename T> OPTINUM_INLINE void vee(const T *OPTINUM_RESTRICT mat, T *OPTINUM_RESTRICT omega) noexcept {
        // Extract from: mat[5] = ω0, mat[6] = ω1, mat[1] = ω2
        omega[0] = mat[5]; // (2,1) element
        omega[1] = mat[6]; // (0,2) element
        omega[2] = mat[1]; // (1,0) element
    }

} // namespace optinum::simd::backend
