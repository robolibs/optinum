#pragma once

// =============================================================================
// optinum/simd/backend/backend.hpp
// Common SIMD backend utilities
// =============================================================================

#include <optinum/simd/arch/arch.hpp>
// Pull in pack primary template + available specialisations
#include <optinum/simd/pack/avx.hpp>
#include <optinum/simd/pack/avx512.hpp>
#include <optinum/simd/pack/neon.hpp>
#include <optinum/simd/pack/pack.hpp>
#include <optinum/simd/pack/sse.hpp>

#include <cstddef>
#include <type_traits>

namespace optinum::simd::backend {

    template <typename T, std::size_t N> [[nodiscard]] consteval std::size_t preferred_simd_lanes() noexcept {
        static_assert(N > 0);
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double>) {
            return 1;
        } else {
            // Pick the widest supported width that still fits N.
            if constexpr (arch::simd_level() >= 512) {
                constexpr std::size_t w = 64 / sizeof(T);
                if constexpr (N >= w)
                    return w;
            }
            if constexpr (arch::simd_level() >= 256) {
                constexpr std::size_t w = 32 / sizeof(T);
                if constexpr (N >= w)
                    return w;
            }
            if constexpr (arch::simd_level() >= 128) {
                constexpr std::size_t w = 16 / sizeof(T);
                if constexpr (N >= w)
                    return w;
            }
            return 1;
        }
    }

    template <std::size_t N, std::size_t W> [[nodiscard]] consteval std::size_t main_loop_count() noexcept {
        static_assert(W > 0);
        return (N / W) * W;
    }

} // namespace optinum::simd::backend
