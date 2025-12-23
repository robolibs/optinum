#include <doctest/doctest.h>
#include <optinum/simd/math/simd_math.hpp>

TEST_CASE("simd math header compiles") {
    using optinum::simd::SIMDVec;
    SIMDVec<float, 4> v(1.0f);
    auto w = optinum::simd::exp(v);
    float out[4]{};
    w.storeu(out);
    CHECK(out[0] == doctest::Approx(std::exp(1.0f)));
}

