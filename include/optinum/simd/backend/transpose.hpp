#pragma once

// =============================================================================
// optinum/simd/backend/transpose.hpp
// Column-major matrix transpose
// =============================================================================

#include <optinum/simd/backend/backend.hpp>

namespace optinum::simd::backend {

    // Column-major transpose: out = transpose(in)
    // in: (R x C), out: (C x R)
    template <typename T, std::size_t R, std::size_t C>
    OPTINUM_INLINE void transpose(T *OPTINUM_RESTRICT out, const T *OPTINUM_RESTRICT in) noexcept {
        // in(row, col)  = in[col * R + row]
        // out(row',col') = out[col' * C + row'] with row'=col, col'=row
        for (std::size_t col = 0; col < C; ++col) {
            for (std::size_t row = 0; row < R; ++row) {
                out[row * C + col] = in[col * R + row];
            }
        }
    }

} // namespace optinum::simd::backend
