#pragma once

// =============================================================================
// optinum/simd/pack/neon.hpp
// ARM NEON specializations for pack<float,4>, pack<double,2>, pack<int32_t,4>, pack<int64_t,2>
// =============================================================================

#include <optinum/simd/pack/pack.hpp>

#ifdef OPTINUM_HAS_NEON

namespace optinum::simd {

    // TODO: Implement ARM NEON specializations
    // pack<float, 4>     - float32x4_t
    // pack<double, 2>    - float64x2_t (ARMv8+)
    // pack<int32_t, 4>   - int32x4_t
    // pack<int64_t, 2>   - int64x2_t
    //
    // For now, the scalar fallback in pack.hpp will be used
    // Port code from intrinsic/neon.hpp when ARM support is needed

} // namespace optinum::simd

#endif // OPTINUM_HAS_NEON
