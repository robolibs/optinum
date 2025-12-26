#pragma once

// =============================================================================
// optinum/simd/permute.hpp
// Tensor dimension permutation
// =============================================================================

#include <optinum/simd/tensor.hpp>

#include <array>
#include <cstddef>

namespace optinum::simd {

    /**
     * Permute tensor dimensions
     *
     * Reorders the dimensions of a tensor according to a permutation.
     * For example, permute(T, {2, 0, 1}) swaps dimensions: new_dim[i] = old_dim[perm[i]]
     *
     * This is a generalization of transpose to arbitrary-rank tensors.
     *
     * @param t Input tensor
     * @param perm Permutation array (must be a valid permutation of 0..N-1)
     * @return Tensor with permuted dimensions
     */
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2>
    [[nodiscard]] Tensor<T, D2, D0, D1> permute_012_to_201(const Tensor<T, D0, D1, D2> &t) noexcept {
        // Permutation: (0,1,2) -> (2,0,1)
        Tensor<T, D2, D0, D1> result{};

        for (std::size_t i = 0; i < D0; ++i) {
            for (std::size_t j = 0; j < D1; ++j) {
                for (std::size_t k = 0; k < D2; ++k) {
                    result(k, i, j) = t(i, j, k);
                }
            }
        }

        return result;
    }

    /**
     * Transpose a 3D tensor (swap first two dimensions)
     *
     * Equivalent to permute with perm = {1, 0, 2}
     */
    template <typename T, std::size_t D0, std::size_t D1, std::size_t D2>
    [[nodiscard]] Tensor<T, D1, D0, D2> transpose_3d(const Tensor<T, D0, D1, D2> &t) noexcept {
        Tensor<T, D1, D0, D2> result{};

        for (std::size_t i = 0; i < D0; ++i) {
            for (std::size_t j = 0; j < D1; ++j) {
                for (std::size_t k = 0; k < D2; ++k) {
                    result(j, i, k) = t(i, j, k);
                }
            }
        }

        return result;
    }

} // namespace optinum::simd
