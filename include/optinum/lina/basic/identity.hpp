#pragma once

// =============================================================================
// optinum/lina/basic/identity.hpp
// =============================================================================

#include <datapod/matrix/matrix.hpp>

#include <cstddef>

namespace optinum::lina {

    // Returns an identity matrix (owning type)
    template <typename T, std::size_t N> [[nodiscard]] constexpr datapod::mat::matrix<T, N, N> identity() noexcept {
        datapod::mat::matrix<T, N, N> result;
        result.fill(T{});
        for (std::size_t i = 0; i < N; ++i)
            result(i, i) = T{1};
        return result;
    }

} // namespace optinum::lina
