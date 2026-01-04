#pragma once

// =============================================================================
// optinum/simd/voigt.hpp
// Voigt notation conversion for mechanics (stress/strain tensors)
// =============================================================================

#include <datapod/datapod.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

namespace optinum::simd {

    /**
     * @brief Convert 3x3 symmetric tensor to Voigt notation (6-vector)
     *
     * Voigt notation stores symmetric 3x3 tensor as 6-component vector:
     *
     * Tensor:              Voigt vector:
     * [σ11  σ12  σ13]      [σ11]   (0)
     * [σ12  σ22  σ23]  →   [σ22]   (1)
     * [σ13  σ23  σ33]      [σ33]   (2)
     *                      [σ23]   (3)
     *                      [σ13]   (4)
     *                      [σ12]   (5)
     *
     * Used in continuum mechanics for stress/strain tensors.
     */
    template <typename T>
    [[nodiscard]] constexpr dp::mat::Vector<T, 6> to_voigt(const Matrix<T, 3, 3> &tensor) noexcept {
        dp::mat::Vector<T, 6> voigt{};
        voigt[0] = tensor(0, 0); // σ11
        voigt[1] = tensor(1, 1); // σ22
        voigt[2] = tensor(2, 2); // σ33
        voigt[3] = tensor(1, 2); // σ23
        voigt[4] = tensor(0, 2); // σ13
        voigt[5] = tensor(0, 1); // σ12
        return voigt;
    }

    /// Overload for dp::mat::matrix input
    template <typename T>
    [[nodiscard]] constexpr dp::mat::Vector<T, 6> to_voigt(const dp::mat::Matrix<T, 3, 3> &tensor) noexcept {
        Matrix<T, 3, 3> view(tensor);
        return to_voigt(view);
    }

    /**
     * @brief Convert Voigt notation (6-vector) to 3x3 symmetric tensor
     *
     * Reconstructs symmetric tensor from Voigt vector.
     */
    template <typename T>
    [[nodiscard]] constexpr dp::mat::Matrix<T, 3, 3> from_voigt(const Vector<T, 6> &voigt) noexcept {
        dp::mat::Matrix<T, 3, 3> tensor{};
        tensor(0, 0) = voigt[0]; // σ11
        tensor(1, 1) = voigt[1]; // σ22
        tensor(2, 2) = voigt[2]; // σ33
        tensor(1, 2) = voigt[3]; // σ23
        tensor(2, 1) = voigt[3]; // Symmetric
        tensor(0, 2) = voigt[4]; // σ13
        tensor(2, 0) = voigt[4]; // Symmetric
        tensor(0, 1) = voigt[5]; // σ12
        tensor(1, 0) = voigt[5]; // Symmetric
        return tensor;
    }

    /// Overload for dp::mat::vector input
    template <typename T>
    [[nodiscard]] constexpr dp::mat::Matrix<T, 3, 3> from_voigt(const dp::mat::Vector<T, 6> &voigt) noexcept {
        Vector<T, 6> view(voigt);
        return from_voigt(view);
    }

    /**
     * @brief Convert 4th-order elasticity tensor (3x3x3x3) to Voigt matrix (6x6)
     *
     * The 4th-order stiffness tensor C[i][j][k][l] is reduced to 6x6 matrix
     * using Voigt notation for both pairs of indices.
     *
     * Voigt mapping: (i,j) → I, (k,l) → J
     *   (0,0)→0  (1,1)→1  (2,2)→2  (1,2)→3  (0,2)→4  (0,1)→5
     *
     * Used for elasticity matrices in FEM.
     */
    template <typename T>
    [[nodiscard]] constexpr dp::mat::Matrix<T, 6, 6> elasticity_to_voigt(const T C[3][3][3][3]) noexcept {
        // Voigt index mapping
        constexpr int voigt_map[3][3] = {
            {0, 5, 4}, // (0,0)→0, (0,1)→5, (0,2)→4
            {5, 1, 3}, // (1,0)→5, (1,1)→1, (1,2)→3
            {4, 3, 2}  // (2,0)→4, (2,1)→3, (2,2)→2
        };

        dp::mat::Matrix<T, 6, 6> voigt_matrix{};
        // Initialize to zero
        for (std::size_t i = 0; i < 36; ++i) {
            voigt_matrix[i] = T{};
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        int I = voigt_map[i][j];
                        int J = voigt_map[k][l];
                        voigt_matrix(I, J) = C[i][j][k][l];
                    }
                }
            }
        }

        return voigt_matrix;
    }

    /**
     * @brief Convert 4th-order elasticity tensor to 6x6 Voigt matrix (reference version)
     */
    template <typename T>
    [[nodiscard]] dp::mat::Matrix<T, 6, 6> elasticity_to_voigt(const T (&C)[3][3][3][3]) noexcept {
        // Voigt index mapping
        constexpr int voigt_map[3][3] = {
            {0, 5, 4}, // i=0: (0,0)->0, (0,1)->5, (0,2)->4
            {5, 1, 3}, // i=1: (1,0)->5, (1,1)->1, (1,2)->3
            {4, 3, 2}  // i=2: (2,0)->4, (2,1)->3, (2,2)->2
        };

        dp::mat::Matrix<T, 6, 6> voigt_C{};
        // Initialize to zero
        for (std::size_t i = 0; i < 36; ++i) {
            voigt_C[i] = T{};
        }

        // Direct mapping from 4th-order tensor to Voigt notation
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                for (std::size_t k = 0; k < 3; ++k) {
                    for (std::size_t l = 0; l < 3; ++l) {
                        int p = voigt_map[i][j];
                        int q = voigt_map[k][l];
                        voigt_C(p, q) = C[i][j][k][l];
                    }
                }
            }
        }

        return voigt_C;
    }

    /**
     * @brief Convert strain tensor to Voigt notation with engineering shear strains
     *
     * For strain tensors, shear components are often doubled (engineering convention):
     *
     * Strain tensor:       Engineering Voigt:
     * [ε11  ε12  ε13]      [ε11]
     * [ε12  ε22  ε23]  →   [ε22]
     * [ε13  ε23  ε33]      [ε33]
     *                      [2*ε23]  (engineering shear strain γ23)
     *                      [2*ε13]  (engineering shear strain γ13)
     *                      [2*ε12]  (engineering shear strain γ12)
     */
    template <typename T>
    [[nodiscard]] constexpr dp::mat::Vector<T, 6> strain_to_voigt_engineering(const Matrix<T, 3, 3> &strain) noexcept {
        dp::mat::Vector<T, 6> voigt{};
        voigt[0] = strain(0, 0);        // ε11
        voigt[1] = strain(1, 1);        // ε22
        voigt[2] = strain(2, 2);        // ε33
        voigt[3] = T{2} * strain(1, 2); // γ23 = 2*ε23
        voigt[4] = T{2} * strain(0, 2); // γ13 = 2*ε13
        voigt[5] = T{2} * strain(0, 1); // γ12 = 2*ε12
        return voigt;
    }

    /// Overload for dp::mat::matrix input
    template <typename T>
    [[nodiscard]] constexpr dp::mat::Vector<T, 6>
    strain_to_voigt_engineering(const dp::mat::Matrix<T, 3, 3> &strain) noexcept {
        Matrix<T, 3, 3> view(strain);
        return strain_to_voigt_engineering(view);
    }

    /**
     * @brief Convert engineering Voigt strain to strain tensor
     *
     * Converts engineering shear strains (γ) back to tensor shear components (ε).
     */
    template <typename T>
    [[nodiscard]] constexpr dp::mat::Matrix<T, 3, 3> strain_from_voigt_engineering(const Vector<T, 6> &voigt) noexcept {
        dp::mat::Matrix<T, 3, 3> strain{};
        strain(0, 0) = voigt[0];        // ε11
        strain(1, 1) = voigt[1];        // ε22
        strain(2, 2) = voigt[2];        // ε33
        strain(1, 2) = voigt[3] / T{2}; // ε23 = γ23/2
        strain(2, 1) = voigt[3] / T{2}; // Symmetric
        strain(0, 2) = voigt[4] / T{2}; // ε13 = γ13/2
        strain(2, 0) = voigt[4] / T{2}; // Symmetric
        strain(0, 1) = voigt[5] / T{2}; // ε12 = γ12/2
        strain(1, 0) = voigt[5] / T{2}; // Symmetric
        return strain;
    }

    /// Overload for dp::mat::vector input
    template <typename T>
    [[nodiscard]] constexpr dp::mat::Matrix<T, 3, 3>
    strain_from_voigt_engineering(const dp::mat::Vector<T, 6> &voigt) noexcept {
        Vector<T, 6> view(voigt);
        return strain_from_voigt_engineering(view);
    }

} // namespace optinum::simd
