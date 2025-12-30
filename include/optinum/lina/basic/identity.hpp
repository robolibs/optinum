#pragma once

// =============================================================================
// optinum/lina/basic/identity.hpp
// =============================================================================

#include <datapod/matrix/matrix.hpp>
#include <optinum/simd/backend/elementwise.hpp>

#include <cstddef>

namespace optinum::lina {

    // Returns an identity matrix (owning type) - SIMD accelerated fill
    template <typename T, std::size_t N> [[nodiscard]] datapod::mat::matrix<T, N, N> identity() noexcept {
        datapod::mat::matrix<T, N, N> result;
        // Use SIMD fill for zeroing the matrix
        simd::backend::fill<T, N * N>(result.data(), T{});
        // Set diagonal elements to 1
        for (std::size_t i = 0; i < N; ++i)
            result(i, i) = T{1};
        return result;
    }

} // namespace optinum::lina
