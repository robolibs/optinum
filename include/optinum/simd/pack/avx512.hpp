#pragma once

// =============================================================================
// optinum/simd/pack/avx512.hpp
// AVX-512 specializations for pack<float,16>, pack<double,8>, pack<int32_t,16>, pack<int64_t,8>
// =============================================================================

#include <optinum/simd/pack/pack.hpp>

#ifdef OPTINUM_HAS_AVX512F

namespace optinum::simd {

    // TODO: Implement AVX-512 specializations
    // pack<float, 16>    - __m512
    // pack<double, 8>    - __m512d
    // pack<int32_t, 16>  - __m512i
    // pack<int64_t, 8>   - __m512i
    //
    // For now, the scalar fallback in pack.hpp will be used
    // Port code from intrinsic/avx512.hpp when AVX-512 support is needed

} // namespace optinum::simd

#endif // OPTINUM_HAS_AVX512F
