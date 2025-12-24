#pragma once

// =============================================================================
// optinum/simd/backend/dot.hpp
// Dot products with SIMD when available
// =============================================================================

#include <optinum/simd/backend/backend.hpp>

namespace optinum::simd::backend {

    template <typename T, std::size_t N>
    [[nodiscard]] OPTINUM_INLINE T dot(const T *OPTINUM_RESTRICT a, const T *OPTINUM_RESTRICT b) noexcept {
        constexpr std::size_t W = preferred_simd_lanes<T, N>();
        constexpr std::size_t main = main_loop_count<N, W>();

        pack<T, W> acc(T{});
        for (std::size_t i = 0; i < main; i += W) {
            const auto va = pack<T, W>::loadu(a + i);
            const auto vb = pack<T, W>::loadu(b + i);
            acc = pack<T, W>::fma(va, vb, acc);
        }

        T result = acc.hsum();
        for (std::size_t i = main; i < N; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

} // namespace optinum::simd::backend
