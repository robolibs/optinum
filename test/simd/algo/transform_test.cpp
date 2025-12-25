// =============================================================================
// test/simd/algo/transform_test.cpp
// Tests for SIMD transformation algorithms (exp, log, sin, cos, etc.)
// =============================================================================

#include <doctest/doctest.h>

#include <cmath>
#include <datapod/matrix.hpp>
#include <optinum/simd/algo/transform.hpp>
#include <optinum/simd/bridge.hpp>

namespace on = optinum;

// =============================================================================
// exp tests
// =============================================================================

TEST_CASE("algo::exp - y = exp(x)") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 3.0f}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::exp(vx, vy);

    // Verify results (allow some error for fast approximation)
    CHECK(y[0] == doctest::Approx(std::exp(0.0f)).epsilon(0.01));  // e^0 = 1
    CHECK(y[1] == doctest::Approx(std::exp(1.0f)).epsilon(0.01));  // e^1 ≈ 2.718
    CHECK(y[2] == doctest::Approx(std::exp(-1.0f)).epsilon(0.01)); // e^-1 ≈ 0.368
    CHECK(y[3] == doctest::Approx(std::exp(2.0f)).epsilon(0.01));  // e^2 ≈ 7.389
    CHECK(y[4] == doctest::Approx(std::exp(-2.0f)).epsilon(0.01)); // e^-2 ≈ 0.135
    CHECK(y[7] == doctest::Approx(std::exp(3.0f)).epsilon(0.01));  // e^3 ≈ 20.086
}

TEST_CASE("algo::exp - in-place x = exp(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{0.0f, 1.0f, -1.0f, 2.0f}};

    auto vx = on::simd::view<4>(x);
    on::simd::exp(vx);

    CHECK(x[0] == doctest::Approx(std::exp(0.0f)).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(std::exp(1.0f)).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(std::exp(-1.0f)).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(std::exp(2.0f)).epsilon(0.01));
}

TEST_CASE("algo::exp - tail handling") {
    using vec_t = datapod::mat::vector<float, 10>;

    alignas(32) vec_t x, y;
    for (std::size_t i = 0; i < 10; ++i) {
        x[i] = static_cast<float>(i) * 0.1f;
    }

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::exp(vx, vy);

    for (std::size_t i = 0; i < 10; ++i) {
        CHECK(y[i] == doctest::Approx(std::exp(x[i])).epsilon(0.01));
    }
}

TEST_CASE("algo::exp - extreme values") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{-100.0f, 100.0f, 10.0f, -10.0f}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::exp(vx, vy);

    // Large negative should be close to 0
    CHECK(y[0] < 0.0001f);
    CHECK(y[0] >= 0.0f);

    // Large positive (clamped at 88) should be very large
    CHECK(y[1] > 1e30f);

    // Moderate values
    CHECK(y[2] == doctest::Approx(std::exp(10.0f)).epsilon(0.01));
    CHECK(y[3] == doctest::Approx(std::exp(-10.0f)).epsilon(0.01));
}

TEST_CASE("algo::exp - accuracy test") {
    using vec_t = datapod::mat::vector<float, 16>;

    alignas(32) vec_t x, y;

    // Test range [-5, 5]
    for (std::size_t i = 0; i < 16; ++i) {
        x[i] = -5.0f + static_cast<float>(i) * (10.0f / 15.0f);
    }

    auto vx = on::simd::view<8>(x);
    auto vy = on::simd::view<8>(y);

    on::simd::exp(vx, vy);

    // Check that all results are within 1% of std::exp
    for (std::size_t i = 0; i < 16; ++i) {
        float expected = std::exp(x[i]);
        float relative_error = std::abs((y[i] - expected) / expected);
        CHECK(relative_error < 0.01f); // < 1% error
    }
}

TEST_CASE("algo::exp - double precision") {
    using vec_t = datapod::mat::vector<double, 4>;

    alignas(32) vec_t x{{0.0, 1.0, -1.0, 2.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<2>(x);
    auto vy = on::simd::view<2>(y);

    on::simd::exp(vx, vy);

    CHECK(y[0] == doctest::Approx(std::exp(0.0)).epsilon(0.001));
    CHECK(y[1] == doctest::Approx(std::exp(1.0)).epsilon(0.001));
    CHECK(y[2] == doctest::Approx(std::exp(-1.0)).epsilon(0.001));
    CHECK(y[3] == doctest::Approx(std::exp(2.0)).epsilon(0.001));
}

// =============================================================================
// log tests
// =============================================================================

TEST_CASE("algo::log - y = log(x)") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{1.0f, 2.0f, 0.5f, 10.0f, 0.1f, std::exp(1.0f), std::exp(2.0f), std::exp(-1.0f)}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::log(vx, vy);

    CHECK(y[0] == doctest::Approx(std::log(1.0f)).epsilon(0.01));  // log(1) = 0
    CHECK(y[1] == doctest::Approx(std::log(2.0f)).epsilon(0.01));  // log(2) ≈ 0.693
    CHECK(y[2] == doctest::Approx(std::log(0.5f)).epsilon(0.01));  // log(0.5) ≈ -0.693
    CHECK(y[3] == doctest::Approx(std::log(10.0f)).epsilon(0.01)); // log(10) ≈ 2.303
    CHECK(y[5] == doctest::Approx(1.0f).epsilon(0.01));            // log(e) = 1
}

TEST_CASE("algo::log - in-place") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{1.0f, 2.0f, 10.0f, std::exp(1.0f)}};

    auto vx = on::simd::view<4>(x);
    on::simd::log(vx);

    CHECK(x[0] == doctest::Approx(std::log(1.0f)).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(std::log(2.0f)).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(std::log(10.0f)).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(1.0f).epsilon(0.01));
}

// =============================================================================
// sin/cos tests - SKIPPED
// =============================================================================
// NOTE: sin() and cos() were ported from fast_trig.hpp, but the original
// implementation has bugs in the quadrant handling. The functions compile and
// have correct polynomial approximations, but produce incorrect output due to
// range reduction/quadrant selection errors.
//
// This was discovered during testing - the OLD fast_sin() also produces wrong
// output, meaning the original code was never properly validated.
//
// These functions need a correct algorithm implementation (separate task).
// For now, we focus on the 4 working functions: exp, log, tanh, sqrt.
// =============================================================================

TEST_CASE("algo::sin - in-place") {
    using vec_t = datapod::mat::vector<float, 4>;

    constexpr float PI = 3.14159265358979323846f;
    alignas(32) vec_t x{{0.0f, PI / 2.0f, PI, 3.0f * PI / 2.0f}};

    auto vx = on::simd::view<4>(x);
    on::simd::sin(vx);

    CHECK(x[0] == doctest::Approx(0.0f).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(1.0f).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(0.0f).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(-1.0f).epsilon(0.01));
}

// =============================================================================
// cos tests
// =============================================================================

TEST_CASE("algo::cos - y = cos(x)") {
    using vec_t = datapod::mat::vector<float, 8>;

    constexpr float PI = 3.14159265358979323846f;
    alignas(32) vec_t x{{0.0f, PI / 6.0f, PI / 4.0f, PI / 3.0f, PI / 2.0f, PI, -PI / 2.0f, 2.0f * PI}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::cos(vx, vy);

    CHECK(y[0] == doctest::Approx(1.0f).epsilon(0.01));       // cos(0) = 1
    CHECK(y[1] == doctest::Approx(0.8660254f).epsilon(0.01)); // cos(π/6) ≈ √3/2
    CHECK(y[2] == doctest::Approx(0.7071067f).epsilon(0.01)); // cos(π/4) ≈ √2/2
    CHECK(y[3] == doctest::Approx(0.5f).epsilon(0.01));       // cos(π/3) = 0.5
    CHECK(y[4] == doctest::Approx(0.0f).epsilon(0.01));       // cos(π/2) ≈ 0
    CHECK(y[5] == doctest::Approx(-1.0f).epsilon(0.01));      // cos(π) = -1
    CHECK(y[6] == doctest::Approx(0.0f).epsilon(0.01));       // cos(-π/2) ≈ 0
    CHECK(y[7] == doctest::Approx(1.0f).epsilon(0.01));       // cos(2π) = 1
}

TEST_CASE("algo::cos - in-place") {
    using vec_t = datapod::mat::vector<float, 4>;

    constexpr float PI = 3.14159265358979323846f;
    alignas(32) vec_t x{{0.0f, PI / 2.0f, PI, 3.0f * PI / 2.0f}};

    auto vx = on::simd::view<4>(x);
    on::simd::cos(vx);

    CHECK(x[0] == doctest::Approx(1.0f).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(0.0f).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(-1.0f).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(0.0f).epsilon(0.01));
}

// =============================================================================
// tanh tests
// =============================================================================

TEST_CASE("algo::tanh - y = tanh(x)") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 5.0f, -5.0f, 0.5f}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::tanh(vx, vy);

    CHECK(y[0] == doctest::Approx(0.0f).epsilon(0.01));             // tanh(0) = 0
    CHECK(y[1] == doctest::Approx(std::tanh(1.0f)).epsilon(0.01));  // tanh(1) ≈ 0.762
    CHECK(y[2] == doctest::Approx(std::tanh(-1.0f)).epsilon(0.01)); // tanh(-1) ≈ -0.762
    CHECK(y[3] == doctest::Approx(std::tanh(2.0f)).epsilon(0.01));  // tanh(2) ≈ 0.964
    CHECK(y[4] == doctest::Approx(std::tanh(-2.0f)).epsilon(0.01)); // tanh(-2) ≈ -0.964
    CHECK(y[5] == doctest::Approx(1.0f).epsilon(0.01));             // tanh(5) ≈ 1
    CHECK(y[6] == doctest::Approx(-1.0f).epsilon(0.01));            // tanh(-5) ≈ -1
}

TEST_CASE("algo::tanh - in-place") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{0.0f, 1.0f, -1.0f, 2.0f}};

    auto vx = on::simd::view<4>(x);
    on::simd::tanh(vx);

    CHECK(x[0] == doctest::Approx(0.0f).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(std::tanh(1.0f)).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(std::tanh(-1.0f)).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(std::tanh(2.0f)).epsilon(0.01));
}

// =============================================================================
// sqrt tests
// =============================================================================

TEST_CASE("algo::sqrt - y = sqrt(x)") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{0.0f, 1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 100.0f, 2.0f}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::sqrt(vx, vy);

    CHECK(y[0] == doctest::Approx(0.0f).epsilon(0.01));            // √0 = 0
    CHECK(y[1] == doctest::Approx(1.0f).epsilon(0.01));            // √1 = 1
    CHECK(y[2] == doctest::Approx(2.0f).epsilon(0.01));            // √4 = 2
    CHECK(y[3] == doctest::Approx(3.0f).epsilon(0.01));            // √9 = 3
    CHECK(y[4] == doctest::Approx(4.0f).epsilon(0.01));            // √16 = 4
    CHECK(y[5] == doctest::Approx(5.0f).epsilon(0.01));            // √25 = 5
    CHECK(y[6] == doctest::Approx(10.0f).epsilon(0.01));           // √100 = 10
    CHECK(y[7] == doctest::Approx(std::sqrt(2.0f)).epsilon(0.01)); // √2 ≈ 1.414
}

TEST_CASE("algo::sqrt - in-place") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{1.0f, 4.0f, 9.0f, 16.0f}};

    auto vx = on::simd::view<4>(x);
    on::simd::sqrt(vx);

    CHECK(x[0] == doctest::Approx(1.0f).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(2.0f).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(3.0f).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(4.0f).epsilon(0.01));
}

// =============================================================================
// Matrix tests - verify transforms work with matrix_view
// =============================================================================

TEST_CASE("algo::exp - matrix_view") {
    using mat_t = datapod::mat::matrix<float, 4, 3>;

    alignas(32) mat_t x;
    alignas(32) mat_t y;

    // Fill with values from -2 to 2
    for (std::size_t i = 0; i < 12; ++i) {
        x[i] = -2.0f + static_cast<float>(i) * (4.0f / 11.0f);
    }

    auto mx = on::simd::view<4>(x);
    auto my = on::simd::view<4>(y);

    on::simd::exp(mx, my);

    // Verify all elements
    for (std::size_t i = 0; i < 12; ++i) {
        CHECK(y[i] == doctest::Approx(std::exp(x[i])).epsilon(0.01));
    }
}

TEST_CASE("algo::exp - matrix_view in-place") {
    using mat_t = datapod::mat::matrix<float, 4, 2>;

    alignas(32) mat_t x;
    std::array<float, 8> original;

    for (std::size_t i = 0; i < 8; ++i) {
        x[i] = static_cast<float>(i) * 0.5f;
        original[i] = x[i];
    }

    auto mx = on::simd::view<4>(x);
    on::simd::exp(mx);

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(x[i] == doctest::Approx(std::exp(original[i])).epsilon(0.01));
    }
}

TEST_CASE("algo::log - matrix_view") {
    using mat_t = datapod::mat::matrix<float, 3, 3>;

    alignas(32) mat_t x;
    alignas(32) mat_t y;

    for (std::size_t i = 0; i < 9; ++i) {
        x[i] = 0.5f + static_cast<float>(i); // 0.5, 1.5, 2.5, ...
    }

    auto mx = on::simd::view<4>(x);
    auto my = on::simd::view<4>(y);

    on::simd::log(mx, my);

    for (std::size_t i = 0; i < 9; ++i) {
        CHECK(y[i] == doctest::Approx(std::log(x[i])).epsilon(0.01));
    }
}

TEST_CASE("algo::tanh - matrix_view") {
    using mat_t = datapod::mat::matrix<float, 2, 4>;

    alignas(32) mat_t x;
    alignas(32) mat_t y;

    for (std::size_t i = 0; i < 8; ++i) {
        x[i] = -2.0f + static_cast<float>(i) * 0.5f;
    }

    auto mx = on::simd::view<4>(x);
    auto my = on::simd::view<4>(y);

    on::simd::tanh(mx, my);

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(y[i] == doctest::Approx(std::tanh(x[i])).epsilon(0.01));
    }
}

TEST_CASE("algo::sqrt - matrix_view") {
    using mat_t = datapod::mat::matrix<float, 4, 4>;

    alignas(32) mat_t x;
    alignas(32) mat_t y;

    for (std::size_t i = 0; i < 16; ++i) {
        x[i] = static_cast<float>(i + 1);
    }

    auto mx = on::simd::view<4>(x);
    auto my = on::simd::view<4>(y);

    on::simd::sqrt(mx, my);

    for (std::size_t i = 0; i < 16; ++i) {
        CHECK(y[i] == doctest::Approx(std::sqrt(x[i])).epsilon(0.01));
    }
}

// =============================================================================
// Tensor tests - verify transforms work with tensor_view
// =============================================================================

TEST_CASE("algo::exp - tensor_view") {
    using tensor_t = datapod::mat::tensor<float, 2, 3, 4>;

    alignas(32) tensor_t x;
    alignas(32) tensor_t y;

    // Fill with values
    for (std::size_t i = 0; i < 24; ++i) {
        x[i] = -3.0f + static_cast<float>(i) * (6.0f / 23.0f);
    }

    auto tx = on::simd::view<4>(x);
    auto ty = on::simd::view<4>(y);

    on::simd::exp(tx, ty);

    for (std::size_t i = 0; i < 24; ++i) {
        CHECK(y[i] == doctest::Approx(std::exp(x[i])).epsilon(0.01));
    }
}

TEST_CASE("algo::exp - tensor_view in-place") {
    using tensor_t = datapod::mat::tensor<float, 2, 2, 4>;

    alignas(32) tensor_t x;
    std::array<float, 16> original;

    for (std::size_t i = 0; i < 16; ++i) {
        x[i] = static_cast<float>(i) * 0.25f - 2.0f;
        original[i] = x[i];
    }

    auto tx = on::simd::view<4>(x);
    on::simd::exp(tx);

    for (std::size_t i = 0; i < 16; ++i) {
        CHECK(x[i] == doctest::Approx(std::exp(original[i])).epsilon(0.01));
    }
}

TEST_CASE("algo::log - tensor_view") {
    using tensor_t = datapod::mat::tensor<float, 3, 2, 2>;

    alignas(32) tensor_t x;
    alignas(32) tensor_t y;

    for (std::size_t i = 0; i < 12; ++i) {
        x[i] = 1.0f + static_cast<float>(i) * 0.5f;
    }

    auto tx = on::simd::view<4>(x);
    auto ty = on::simd::view<4>(y);

    on::simd::log(tx, ty);

    for (std::size_t i = 0; i < 12; ++i) {
        CHECK(y[i] == doctest::Approx(std::log(x[i])).epsilon(0.01));
    }
}

TEST_CASE("algo::tanh - tensor_view") {
    using tensor_t = datapod::mat::tensor<float, 2, 2, 3>;

    alignas(32) tensor_t x;
    alignas(32) tensor_t y;

    for (std::size_t i = 0; i < 12; ++i) {
        x[i] = -3.0f + static_cast<float>(i) * 0.5f;
    }

    auto tx = on::simd::view<4>(x);
    auto ty = on::simd::view<4>(y);

    on::simd::tanh(tx, ty);

    for (std::size_t i = 0; i < 12; ++i) {
        CHECK(y[i] == doctest::Approx(std::tanh(x[i])).epsilon(0.01));
    }
}

TEST_CASE("algo::sqrt - tensor_view") {
    using tensor_t = datapod::mat::tensor<float, 2, 3, 2>;

    alignas(32) tensor_t x;
    alignas(32) tensor_t y;

    for (std::size_t i = 0; i < 12; ++i) {
        x[i] = static_cast<float>(i + 1);
    }

    auto tx = on::simd::view<4>(x);
    auto ty = on::simd::view<4>(y);

    on::simd::sqrt(tx, ty);

    for (std::size_t i = 0; i < 12; ++i) {
        CHECK(y[i] == doctest::Approx(std::sqrt(x[i])).epsilon(0.01));
    }
}

// =============================================================================
// Cross-view tests - ensure different view types can work together
// =============================================================================

TEST_CASE("algo - matrix and vector same underlying data") {
    // A 4x2 matrix has same storage as an 8-element vector
    using mat_t = datapod::mat::matrix<float, 4, 2>;
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) mat_t m;
    alignas(32) vec_t v;

    for (std::size_t i = 0; i < 8; ++i) {
        m[i] = static_cast<float>(i) * 0.5f;
        v[i] = static_cast<float>(i) * 0.5f;
    }

    auto mm = on::simd::view<4>(m);
    auto vv = on::simd::view<4>(v);

    on::simd::exp(mm);
    on::simd::exp(vv);

    // Both should produce identical results
    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(m[i] == doctest::Approx(v[i]).epsilon(0.0001));
    }
}

// =============================================================================
// Double precision tests for all functions
// =============================================================================

TEST_CASE("algo::log - double precision") {
    using vec_t = datapod::mat::vector<double, 4>;

    alignas(32) vec_t x{{1.0, 2.0, 0.5, std::exp(1.0)}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<2>(x);
    auto vy = on::simd::view<2>(y);

    on::simd::log(vx, vy);

    CHECK(y[0] == doctest::Approx(std::log(1.0)).epsilon(0.001));           // log(1) = 0
    CHECK(y[1] == doctest::Approx(std::log(2.0)).epsilon(0.001));           // log(2) ≈ 0.693
    CHECK(y[2] == doctest::Approx(std::log(0.5)).epsilon(0.001));           // log(0.5) ≈ -0.693
    CHECK(y[3] == doctest::Approx(std::log(std::exp(1.0))).epsilon(0.001)); // log(e) = 1
}

TEST_CASE("algo::sin - double precision") {
    using vec_t = datapod::mat::vector<double, 8>;

    constexpr double PI = 3.141592653589793;
    alignas(32) vec_t x{{0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0, PI, -PI / 2.0, 2.0 * PI}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::sin(vx, vy);

    CHECK(y[0] == doctest::Approx(std::sin(0.0)).epsilon(0.001));
    CHECK(y[1] == doctest::Approx(std::sin(PI / 6.0)).epsilon(0.001));  // sin(π/6) = 0.5
    CHECK(y[2] == doctest::Approx(std::sin(PI / 4.0)).epsilon(0.001));  // sin(π/4) ≈ 0.707
    CHECK(y[3] == doctest::Approx(std::sin(PI / 3.0)).epsilon(0.001));  // sin(π/3) ≈ 0.866
    CHECK(y[4] == doctest::Approx(std::sin(PI / 2.0)).epsilon(0.001));  // sin(π/2) = 1
    CHECK(y[5] == doctest::Approx(std::sin(PI)).epsilon(0.001));        // sin(π) ≈ 0
    CHECK(y[6] == doctest::Approx(std::sin(-PI / 2.0)).epsilon(0.001)); // sin(-π/2) = -1
    CHECK(y[7] == doctest::Approx(std::sin(2.0 * PI)).epsilon(0.001));  // sin(2π) ≈ 0
}

TEST_CASE("algo::cos - double precision") {
    using vec_t = datapod::mat::vector<double, 8>;

    constexpr double PI = 3.141592653589793;
    alignas(32) vec_t x{{0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0, PI, -PI / 2.0, 2.0 * PI}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::cos(vx, vy);

    CHECK(y[0] == doctest::Approx(std::cos(0.0)).epsilon(0.001));       // cos(0) = 1
    CHECK(y[1] == doctest::Approx(std::cos(PI / 6.0)).epsilon(0.001));  // cos(π/6) ≈ 0.866
    CHECK(y[2] == doctest::Approx(std::cos(PI / 4.0)).epsilon(0.001));  // cos(π/4) ≈ 0.707
    CHECK(y[3] == doctest::Approx(std::cos(PI / 3.0)).epsilon(0.001));  // cos(π/3) = 0.5
    CHECK(y[4] == doctest::Approx(std::cos(PI / 2.0)).epsilon(0.001));  // cos(π/2) ≈ 0
    CHECK(y[5] == doctest::Approx(std::cos(PI)).epsilon(0.001));        // cos(π) = -1
    CHECK(y[6] == doctest::Approx(std::cos(-PI / 2.0)).epsilon(0.001)); // cos(-π/2) ≈ 0
    CHECK(y[7] == doctest::Approx(std::cos(2.0 * PI)).epsilon(0.001));  // cos(2π) = 1
}

TEST_CASE("algo::tanh - double precision") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::tanh(vx, vy);

    CHECK(y[0] == doctest::Approx(std::tanh(0.0)).epsilon(0.001));  // tanh(0) = 0
    CHECK(y[1] == doctest::Approx(std::tanh(0.5)).epsilon(0.001));  // tanh(0.5) ≈ 0.462
    CHECK(y[2] == doctest::Approx(std::tanh(-0.5)).epsilon(0.001)); // tanh(-0.5) ≈ -0.462
    CHECK(y[3] == doctest::Approx(std::tanh(1.0)).epsilon(0.001));  // tanh(1) ≈ 0.762
    CHECK(y[4] == doctest::Approx(std::tanh(-1.0)).epsilon(0.001)); // tanh(-1) ≈ -0.762
    CHECK(y[5] == doctest::Approx(std::tanh(2.0)).epsilon(0.001));  // tanh(2) ≈ 0.964
    CHECK(y[6] == doctest::Approx(std::tanh(-2.0)).epsilon(0.001)); // tanh(-2) ≈ -0.964
    CHECK(y[7] == doctest::Approx(std::tanh(5.0)).epsilon(0.001));  // tanh(5) ≈ 0.9999
}

TEST_CASE("algo::sqrt - double precision") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 2.0, 0.25}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::sqrt(vx, vy);

    CHECK(y[0] == doctest::Approx(std::sqrt(0.0)).epsilon(0.001));  // sqrt(0) = 0
    CHECK(y[1] == doctest::Approx(std::sqrt(1.0)).epsilon(0.001));  // sqrt(1) = 1
    CHECK(y[2] == doctest::Approx(std::sqrt(4.0)).epsilon(0.001));  // sqrt(4) = 2
    CHECK(y[3] == doctest::Approx(std::sqrt(9.0)).epsilon(0.001));  // sqrt(9) = 3
    CHECK(y[4] == doctest::Approx(std::sqrt(16.0)).epsilon(0.001)); // sqrt(16) = 4
    CHECK(y[5] == doctest::Approx(std::sqrt(25.0)).epsilon(0.001)); // sqrt(25) = 5
    CHECK(y[6] == doctest::Approx(std::sqrt(2.0)).epsilon(0.001));  // sqrt(2) ≈ 1.414
    CHECK(y[7] == doctest::Approx(std::sqrt(0.25)).epsilon(0.001)); // sqrt(0.25) = 0.5
}

// =============================================================================
// tan tests
// =============================================================================

TEST_CASE("algo::tan - y = tan(x)") {
    using vec_t = datapod::mat::vector<float, 8>;

    constexpr float PI = 3.14159265f;
    alignas(32) vec_t x{{0.0f, PI / 6.0f, PI / 4.0f, PI / 3.0f, -PI / 6.0f, -PI / 4.0f, -PI / 3.0f, 0.5f}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::tan(vx, vy);

    CHECK(y[0] == doctest::Approx(std::tan(0.0f)).epsilon(0.01));       // tan(0) = 0
    CHECK(y[1] == doctest::Approx(std::tan(PI / 6.0f)).epsilon(0.01));  // tan(π/6) ≈ 0.577
    CHECK(y[2] == doctest::Approx(std::tan(PI / 4.0f)).epsilon(0.01));  // tan(π/4) = 1
    CHECK(y[3] == doctest::Approx(std::tan(PI / 3.0f)).epsilon(0.01));  // tan(π/3) ≈ 1.732
    CHECK(y[4] == doctest::Approx(std::tan(-PI / 6.0f)).epsilon(0.01)); // tan(-π/6) ≈ -0.577
    CHECK(y[5] == doctest::Approx(std::tan(-PI / 4.0f)).epsilon(0.01)); // tan(-π/4) = -1
    CHECK(y[6] == doctest::Approx(std::tan(-PI / 3.0f)).epsilon(0.01)); // tan(-π/3) ≈ -1.732
    CHECK(y[7] == doctest::Approx(std::tan(0.5f)).epsilon(0.01));       // tan(0.5) ≈ 0.546
}

TEST_CASE("algo::tan - in-place x = tan(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    constexpr float PI = 3.14159265f;
    alignas(32) vec_t x{{0.0f, PI / 6.0f, PI / 4.0f, -PI / 4.0f}};

    auto vx = on::simd::view<4>(x);
    on::simd::tan(vx);

    CHECK(x[0] == doctest::Approx(std::tan(0.0f)).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(std::tan(PI / 6.0f)).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(std::tan(PI / 4.0f)).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(std::tan(-PI / 4.0f)).epsilon(0.01));
}

TEST_CASE("algo::tan - double precision") {
    using vec_t = datapod::mat::vector<double, 8>;

    constexpr double PI = 3.141592653589793;
    alignas(32) vec_t x{{0.0, PI / 6.0, PI / 4.0, PI / 3.0, -PI / 6.0, -PI / 4.0, -PI / 3.0, 1.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::tan(vx, vy);

    CHECK(y[0] == doctest::Approx(std::tan(0.0)).epsilon(0.001));       // tan(0) = 0
    CHECK(y[1] == doctest::Approx(std::tan(PI / 6.0)).epsilon(0.001));  // tan(π/6) ≈ 0.577
    CHECK(y[2] == doctest::Approx(std::tan(PI / 4.0)).epsilon(0.001));  // tan(π/4) = 1
    CHECK(y[3] == doctest::Approx(std::tan(PI / 3.0)).epsilon(0.001));  // tan(π/3) ≈ 1.732
    CHECK(y[4] == doctest::Approx(std::tan(-PI / 6.0)).epsilon(0.001)); // tan(-π/6) ≈ -0.577
    CHECK(y[5] == doctest::Approx(std::tan(-PI / 4.0)).epsilon(0.001)); // tan(-π/4) = -1
    CHECK(y[6] == doctest::Approx(std::tan(-PI / 3.0)).epsilon(0.001)); // tan(-π/3) ≈ -1.732
    CHECK(y[7] == doctest::Approx(std::tan(1.0)).epsilon(0.001));       // tan(1) ≈ 1.557
}

// =============================================================================
// ceil tests
// =============================================================================

TEST_CASE("algo::ceil - y = ceil(x)") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{1.5f, -1.5f, 2.9f, -2.9f, 0.1f, -0.1f, 3.0f, -3.0f}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::ceil(vx, vy);

    CHECK(y[0] == 2.0f);  // ceil(1.5) = 2
    CHECK(y[1] == -1.0f); // ceil(-1.5) = -1
    CHECK(y[2] == 3.0f);  // ceil(2.9) = 3
    CHECK(y[3] == -2.0f); // ceil(-2.9) = -2
    CHECK(y[4] == 1.0f);  // ceil(0.1) = 1
    CHECK(y[5] == -0.0f); // ceil(-0.1) = -0
    CHECK(y[6] == 3.0f);  // ceil(3.0) = 3
    CHECK(y[7] == -3.0f); // ceil(-3.0) = -3
}

TEST_CASE("algo::ceil - in-place x = ceil(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{1.5f, -1.5f, 2.9f, -2.9f}};

    auto vx = on::simd::view<4>(x);
    on::simd::ceil(vx);

    CHECK(x[0] == 2.0f);
    CHECK(x[1] == -1.0f);
    CHECK(x[2] == 3.0f);
    CHECK(x[3] == -2.0f);
}

TEST_CASE("algo::ceil - double precision") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{1.5, -1.5, 2.9, -2.9, 0.1, -0.1, 3.0, -3.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::ceil(vx, vy);

    CHECK(y[0] == 2.0);
    CHECK(y[1] == -1.0);
    CHECK(y[2] == 3.0);
    CHECK(y[3] == -2.0);
    CHECK(y[4] == 1.0);
    CHECK(y[5] == -0.0);
    CHECK(y[6] == 3.0);
    CHECK(y[7] == -3.0);
}

// =============================================================================
// floor tests
// =============================================================================

TEST_CASE("algo::floor - y = floor(x)") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{1.5f, -1.5f, 2.1f, -2.1f, 0.9f, -0.9f, 3.0f, -3.0f}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::floor(vx, vy);

    CHECK(y[0] == 1.0f);  // floor(1.5) = 1
    CHECK(y[1] == -2.0f); // floor(-1.5) = -2
    CHECK(y[2] == 2.0f);  // floor(2.1) = 2
    CHECK(y[3] == -3.0f); // floor(-2.1) = -3
    CHECK(y[4] == 0.0f);  // floor(0.9) = 0
    CHECK(y[5] == -1.0f); // floor(-0.9) = -1
    CHECK(y[6] == 3.0f);  // floor(3.0) = 3
    CHECK(y[7] == -3.0f); // floor(-3.0) = -3
}

TEST_CASE("algo::floor - in-place x = floor(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{1.5f, -1.5f, 2.1f, -2.1f}};

    auto vx = on::simd::view<4>(x);
    on::simd::floor(vx);

    CHECK(x[0] == 1.0f);
    CHECK(x[1] == -2.0f);
    CHECK(x[2] == 2.0f);
    CHECK(x[3] == -3.0f);
}

TEST_CASE("algo::floor - double precision") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{1.5, -1.5, 2.1, -2.1, 0.9, -0.9, 3.0, -3.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::floor(vx, vy);

    CHECK(y[0] == 1.0);
    CHECK(y[1] == -2.0);
    CHECK(y[2] == 2.0);
    CHECK(y[3] == -3.0);
    CHECK(y[4] == 0.0);
    CHECK(y[5] == -1.0);
    CHECK(y[6] == 3.0);
    CHECK(y[7] == -3.0);
}

// =============================================================================
// round tests
// =============================================================================

TEST_CASE("algo::round - y = round(x)") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{1.5f, 2.5f, -1.5f, -2.5f, 1.4f, 1.6f, -1.4f, -1.6f}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::round(vx, vy);

    // Note: uses banker's rounding (round to nearest even)
    CHECK(y[0] == 2.0f);  // round(1.5) = 2 (even)
    CHECK(y[1] == 2.0f);  // round(2.5) = 2 (even)
    CHECK(y[2] == -2.0f); // round(-1.5) = -2 (even)
    CHECK(y[3] == -2.0f); // round(-2.5) = -2 (even)
    CHECK(y[4] == 1.0f);  // round(1.4) = 1
    CHECK(y[5] == 2.0f);  // round(1.6) = 2
    CHECK(y[6] == -1.0f); // round(-1.4) = -1
    CHECK(y[7] == -2.0f); // round(-1.6) = -2
}

TEST_CASE("algo::round - in-place x = round(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{1.5f, 2.5f, -1.5f, -2.5f}};

    auto vx = on::simd::view<4>(x);
    on::simd::round(vx);

    CHECK(x[0] == 2.0f);
    CHECK(x[1] == 2.0f);
    CHECK(x[2] == -2.0f);
    CHECK(x[3] == -2.0f);
}

TEST_CASE("algo::round - double precision") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{1.5, 2.5, -1.5, -2.5, 1.4, 1.6, -1.4, -1.6}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::round(vx, vy);

    CHECK(y[0] == 2.0);
    CHECK(y[1] == 2.0);
    CHECK(y[2] == -2.0);
    CHECK(y[3] == -2.0);
    CHECK(y[4] == 1.0);
    CHECK(y[5] == 2.0);
    CHECK(y[6] == -1.0);
    CHECK(y[7] == -2.0);
}

// =============================================================================
// trunc tests
// =============================================================================

TEST_CASE("algo::trunc - y = trunc(x)") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{1.9f, -1.9f, 2.1f, -2.1f, 0.9f, -0.9f, 3.0f, -3.0f}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::trunc(vx, vy);

    CHECK(y[0] == 1.0f);  // trunc(1.9) = 1
    CHECK(y[1] == -1.0f); // trunc(-1.9) = -1
    CHECK(y[2] == 2.0f);  // trunc(2.1) = 2
    CHECK(y[3] == -2.0f); // trunc(-2.1) = -2
    CHECK(y[4] == 0.0f);  // trunc(0.9) = 0
    CHECK(y[5] == -0.0f); // trunc(-0.9) = -0
    CHECK(y[6] == 3.0f);  // trunc(3.0) = 3
    CHECK(y[7] == -3.0f); // trunc(-3.0) = -3
}

TEST_CASE("algo::trunc - in-place x = trunc(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{1.9f, -1.9f, 2.1f, -2.1f}};

    auto vx = on::simd::view<4>(x);
    on::simd::trunc(vx);

    CHECK(x[0] == 1.0f);
    CHECK(x[1] == -1.0f);
    CHECK(x[2] == 2.0f);
    CHECK(x[3] == -2.0f);
}

TEST_CASE("algo::trunc - double precision") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{1.9, -1.9, 2.1, -2.1, 0.9, -0.9, 3.0, -3.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::trunc(vx, vy);

    CHECK(y[0] == 1.0);
    CHECK(y[1] == -1.0);
    CHECK(y[2] == 2.0);
    CHECK(y[3] == -2.0);
    CHECK(y[4] == 0.0);
    CHECK(y[5] == -0.0);
    CHECK(y[6] == 3.0);
    CHECK(y[7] == -3.0);
}

// =============================================================================
// pow tests
// =============================================================================

TEST_CASE("algo::pow - z = pow(x, y)") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 10.0f, 1.5f, 2.5f}};
    alignas(32) vec_t y{{3.0f, 2.0f, 0.5f, 0.0f, -1.0f, 2.0f, 3.0f, 2.0f}};
    alignas(32) vec_t z;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);
    auto vz = on::simd::view<4>(z);

    on::simd::pow(vx, vy, vz);

    CHECK(z[0] == doctest::Approx(8.0f).epsilon(0.01));   // 2^3 = 8
    CHECK(z[1] == doctest::Approx(9.0f).epsilon(0.01));   // 3^2 = 9
    CHECK(z[2] == doctest::Approx(2.0f).epsilon(0.01));   // 4^0.5 = 2
    CHECK(z[3] == doctest::Approx(1.0f).epsilon(0.01));   // 5^0 = 1
    CHECK(z[4] == doctest::Approx(0.5f).epsilon(0.01));   // 2^-1 = 0.5
    CHECK(z[5] == doctest::Approx(100.0f).epsilon(0.01)); // 10^2 = 100
    CHECK(z[6] == doctest::Approx(3.375f).epsilon(0.01)); // 1.5^3 = 3.375
    CHECK(z[7] == doctest::Approx(6.25f).epsilon(0.01));  // 2.5^2 = 6.25
}

TEST_CASE("algo::pow - double precision") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{2.0, 3.0, 4.0, 5.0, 2.0, 10.0, 1.5, 2.5}};
    alignas(32) vec_t y{{3.0, 2.0, 0.5, 0.0, -1.0, 2.0, 3.0, 2.0}};
    alignas(32) vec_t z;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);
    auto vz = on::simd::view<4>(z);

    on::simd::pow(vx, vy, vz);

    CHECK(z[0] == doctest::Approx(8.0).epsilon(0.001));   // 2^3 = 8
    CHECK(z[1] == doctest::Approx(9.0).epsilon(0.001));   // 3^2 = 9
    CHECK(z[2] == doctest::Approx(2.0).epsilon(0.001));   // 4^0.5 = 2
    CHECK(z[3] == doctest::Approx(1.0).epsilon(0.001));   // 5^0 = 1
    CHECK(z[4] == doctest::Approx(0.5).epsilon(0.001));   // 2^-1 = 0.5
    CHECK(z[5] == doctest::Approx(100.0).epsilon(0.001)); // 10^2 = 100
    CHECK(z[6] == doctest::Approx(3.375).epsilon(0.001)); // 1.5^3 = 3.375
    CHECK(z[7] == doctest::Approx(6.25).epsilon(0.001));  // 2.5^2 = 6.25
}

TEST_CASE("algo::pow - special cases") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{0.0f, 1.0f, 2.0f, 16.0f}};
    alignas(32) vec_t y{{5.0f, 100.0f, 10.0f, 0.25f}};
    alignas(32) vec_t z;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);
    auto vz = on::simd::view<4>(z);

    on::simd::pow(vx, vy, vz);

    CHECK(z[0] == doctest::Approx(0.0f).epsilon(0.01));    // 0^5 = 0
    CHECK(z[1] == doctest::Approx(1.0f).epsilon(0.01));    // 1^100 = 1
    CHECK(z[2] == doctest::Approx(1024.0f).epsilon(0.01)); // 2^10 = 1024
    CHECK(z[3] == doctest::Approx(2.0f).epsilon(0.01));    // 16^0.25 = 2 (4th root)
}

// =============================================================================
// sinh tests
// =============================================================================

TEST_CASE("algo::sinh - y = sinh(x)") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::sinh(vx, vy);

    CHECK(y[0] == doctest::Approx(std::sinh(-2.0)).epsilon(0.001));
    CHECK(y[1] == doctest::Approx(std::sinh(-1.0)).epsilon(0.001));
    CHECK(y[2] == doctest::Approx(std::sinh(-0.5)).epsilon(0.001));
    CHECK(y[3] == doctest::Approx(std::sinh(0.0)).epsilon(0.001));
    CHECK(y[4] == doctest::Approx(std::sinh(0.5)).epsilon(0.001));
    CHECK(y[5] == doctest::Approx(std::sinh(1.0)).epsilon(0.001));
    CHECK(y[6] == doctest::Approx(std::sinh(2.0)).epsilon(0.001));
    CHECK(y[7] == doctest::Approx(std::sinh(3.0)).epsilon(0.001));
}

TEST_CASE("algo::sinh - in-place x = sinh(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{-1.0f, 0.0f, 0.5f, 1.0f}};

    auto vx = on::simd::view<4>(x);
    on::simd::sinh(vx);

    CHECK(x[0] == doctest::Approx(std::sinh(-1.0f)).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(std::sinh(0.0f)).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(std::sinh(0.5f)).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(std::sinh(1.0f)).epsilon(0.01));
}

// =============================================================================
// cosh tests
// =============================================================================

TEST_CASE("algo::cosh - y = cosh(x)") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::cosh(vx, vy);

    CHECK(y[0] == doctest::Approx(std::cosh(-2.0)).epsilon(0.001));
    CHECK(y[1] == doctest::Approx(std::cosh(-1.0)).epsilon(0.001));
    CHECK(y[2] == doctest::Approx(std::cosh(-0.5)).epsilon(0.001));
    CHECK(y[3] == doctest::Approx(std::cosh(0.0)).epsilon(0.001)); // cosh(0) = 1
    CHECK(y[4] == doctest::Approx(std::cosh(0.5)).epsilon(0.001));
    CHECK(y[5] == doctest::Approx(std::cosh(1.0)).epsilon(0.001));
    CHECK(y[6] == doctest::Approx(std::cosh(2.0)).epsilon(0.001));
    CHECK(y[7] == doctest::Approx(std::cosh(3.0)).epsilon(0.001));
}

TEST_CASE("algo::cosh - in-place x = cosh(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{-1.0f, 0.0f, 0.5f, 1.0f}};

    auto vx = on::simd::view<4>(x);
    on::simd::cosh(vx);

    CHECK(x[0] == doctest::Approx(std::cosh(-1.0f)).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(1.0f).epsilon(0.01)); // cosh(0) = 1
    CHECK(x[2] == doctest::Approx(std::cosh(0.5f)).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(std::cosh(1.0f)).epsilon(0.01));
}

// =============================================================================
// exp2 tests
// =============================================================================

TEST_CASE("algo::exp2 - y = exp2(x)") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 10.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::exp2(vx, vy);

    CHECK(y[0] == doctest::Approx(0.25).epsilon(0.001));   // 2^-2 = 0.25
    CHECK(y[1] == doctest::Approx(0.5).epsilon(0.001));    // 2^-1 = 0.5
    CHECK(y[2] == doctest::Approx(1.0).epsilon(0.001));    // 2^0 = 1
    CHECK(y[3] == doctest::Approx(2.0).epsilon(0.001));    // 2^1 = 2
    CHECK(y[4] == doctest::Approx(4.0).epsilon(0.001));    // 2^2 = 4
    CHECK(y[5] == doctest::Approx(8.0).epsilon(0.001));    // 2^3 = 8
    CHECK(y[6] == doctest::Approx(16.0).epsilon(0.001));   // 2^4 = 16
    CHECK(y[7] == doctest::Approx(1024.0).epsilon(0.001)); // 2^10 = 1024
}

TEST_CASE("algo::exp2 - in-place x = exp2(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{0.0f, 1.0f, 2.0f, 3.0f}};

    auto vx = on::simd::view<4>(x);
    on::simd::exp2(vx);

    CHECK(x[0] == doctest::Approx(1.0f).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(2.0f).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(4.0f).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(8.0f).epsilon(0.01));
}

// =============================================================================
// log2 tests
// =============================================================================

TEST_CASE("algo::log2 - y = log2(x)") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 1024.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::log2(vx, vy);

    CHECK(y[0] == doctest::Approx(-2.0).epsilon(0.001)); // log2(0.25) = -2
    CHECK(y[1] == doctest::Approx(-1.0).epsilon(0.001)); // log2(0.5) = -1
    CHECK(y[2] == doctest::Approx(0.0).epsilon(0.001));  // log2(1) = 0
    CHECK(y[3] == doctest::Approx(1.0).epsilon(0.001));  // log2(2) = 1
    CHECK(y[4] == doctest::Approx(2.0).epsilon(0.001));  // log2(4) = 2
    CHECK(y[5] == doctest::Approx(3.0).epsilon(0.001));  // log2(8) = 3
    CHECK(y[6] == doctest::Approx(4.0).epsilon(0.001));  // log2(16) = 4
    CHECK(y[7] == doctest::Approx(10.0).epsilon(0.001)); // log2(1024) = 10
}

TEST_CASE("algo::log2 - in-place x = log2(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{1.0f, 2.0f, 4.0f, 8.0f}};

    auto vx = on::simd::view<4>(x);
    on::simd::log2(vx);

    CHECK(x[0] == doctest::Approx(0.0f).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(1.0f).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(2.0f).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(3.0f).epsilon(0.01));
}

// =============================================================================
// log10 tests
// =============================================================================

TEST_CASE("algo::log10 - y = log10(x)") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 2.0, 5.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::log10(vx, vy);

    CHECK(y[0] == doctest::Approx(-1.0).epsilon(0.001)); // log10(0.1) = -1
    CHECK(y[1] == doctest::Approx(0.0).epsilon(0.001));  // log10(1) = 0
    CHECK(y[2] == doctest::Approx(1.0).epsilon(0.001));  // log10(10) = 1
    CHECK(y[3] == doctest::Approx(2.0).epsilon(0.001));  // log10(100) = 2
    CHECK(y[4] == doctest::Approx(3.0).epsilon(0.001));  // log10(1000) = 3
    CHECK(y[5] == doctest::Approx(4.0).epsilon(0.001));  // log10(10000) = 4
    CHECK(y[6] == doctest::Approx(std::log10(2.0)).epsilon(0.001));
    CHECK(y[7] == doctest::Approx(std::log10(5.0)).epsilon(0.001));
}

TEST_CASE("algo::log10 - in-place x = log10(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{1.0f, 10.0f, 100.0f, 1000.0f}};

    auto vx = on::simd::view<4>(x);
    on::simd::log10(vx);

    CHECK(x[0] == doctest::Approx(0.0f).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(1.0f).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(2.0f).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(3.0f).epsilon(0.01));
}

// =============================================================================
// atan tests
// =============================================================================

TEST_CASE("algo::atan - y = atan(x)") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 10.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::atan(vx, vy);

    CHECK(y[0] == doctest::Approx(std::atan(-2.0)).epsilon(0.001));
    CHECK(y[1] == doctest::Approx(std::atan(-1.0)).epsilon(0.001)); // -π/4
    CHECK(y[2] == doctest::Approx(std::atan(-0.5)).epsilon(0.001));
    CHECK(y[3] == doctest::Approx(std::atan(0.0)).epsilon(0.001)); // 0
    CHECK(y[4] == doctest::Approx(std::atan(0.5)).epsilon(0.001));
    CHECK(y[5] == doctest::Approx(std::atan(1.0)).epsilon(0.001)); // π/4
    CHECK(y[6] == doctest::Approx(std::atan(2.0)).epsilon(0.001));
    CHECK(y[7] == doctest::Approx(std::atan(10.0)).epsilon(0.001));
}

TEST_CASE("algo::atan - in-place x = atan(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{-1.0f, 0.0f, 1.0f, 5.0f}};

    auto vx = on::simd::view<4>(x);
    on::simd::atan(vx);

    CHECK(x[0] == doctest::Approx(std::atan(-1.0f)).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(std::atan(0.0f)).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(std::atan(1.0f)).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(std::atan(5.0f)).epsilon(0.01));
}

TEST_CASE("algo::atan - double precision") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{-100.0, -10.0, -1.0, -0.1, 0.1, 1.0, 10.0, 100.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::atan(vx, vy);

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(y[i] == doctest::Approx(std::atan(x[i])).epsilon(0.001));
    }
}

// =============================================================================
// atan2 tests
// =============================================================================

TEST_CASE("algo::atan2 - z = atan2(y, x)") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t y_vals{{1.0, 1.0, -1.0, -1.0, 0.0, 1.0, -1.0, 2.0}};
    alignas(32) vec_t x_vals{{1.0, -1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 2.0}};
    alignas(32) vec_t z;

    auto vy = on::simd::view<4>(y_vals);
    auto vx = on::simd::view<4>(x_vals);
    auto vz = on::simd::view<4>(z);

    on::simd::atan2(vy, vx, vz);

    // Quadrant I: y=1, x=1 → π/4
    CHECK(z[0] == doctest::Approx(std::atan2(1.0, 1.0)).epsilon(0.001));
    // Quadrant II: y=1, x=-1 → 3π/4
    CHECK(z[1] == doctest::Approx(std::atan2(1.0, -1.0)).epsilon(0.001));
    // Quadrant III: y=-1, x=-1 → -3π/4
    CHECK(z[2] == doctest::Approx(std::atan2(-1.0, -1.0)).epsilon(0.001));
    // Quadrant IV: y=-1, x=1 → -π/4
    CHECK(z[3] == doctest::Approx(std::atan2(-1.0, 1.0)).epsilon(0.001));
    // Special: y=0, x=1 → 0
    CHECK(z[4] == doctest::Approx(std::atan2(0.0, 1.0)).epsilon(0.001));
    // Special: y=1, x=0 → π/2
    CHECK(z[5] == doctest::Approx(std::atan2(1.0, 0.0)).epsilon(0.001));
    // Special: y=-1, x=0 → -π/2
    CHECK(z[6] == doctest::Approx(std::atan2(-1.0, 0.0)).epsilon(0.001));
    // 45 degree angle
    CHECK(z[7] == doctest::Approx(std::atan2(2.0, 2.0)).epsilon(0.001));
}

TEST_CASE("algo::atan2 - float precision") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t y_vals{{1.0f, -1.0f, 1.0f, 0.0f}};
    alignas(32) vec_t x_vals{{1.0f, 1.0f, 0.0f, -1.0f}};
    alignas(32) vec_t z;

    auto vy = on::simd::view<4>(y_vals);
    auto vx = on::simd::view<4>(x_vals);
    auto vz = on::simd::view<4>(z);

    on::simd::atan2(vy, vx, vz);

    CHECK(z[0] == doctest::Approx(std::atan2(1.0f, 1.0f)).epsilon(0.01));
    CHECK(z[1] == doctest::Approx(std::atan2(-1.0f, 1.0f)).epsilon(0.01));
    CHECK(z[2] == doctest::Approx(std::atan2(1.0f, 0.0f)).epsilon(0.01));
    CHECK(z[3] == doctest::Approx(std::atan2(0.0f, -1.0f)).epsilon(0.01));
}

TEST_CASE("algo::atan2 - robotics use case (Cartesian to polar)") {
    using vec_t = datapod::mat::vector<double, 8>;

    // Robot positions in 2D space
    alignas(32) vec_t x_pos{{3.0, 0.0, -4.0, -5.0, 1.0, 1.732, -1.0, 0.0}};
    alignas(32) vec_t y_pos{{4.0, 5.0, 3.0, -12.0, 1.0, 1.0, -1.732, 0.0}};
    alignas(32) vec_t angles;

    auto vx = on::simd::view<4>(x_pos);
    auto vy = on::simd::view<4>(y_pos);
    auto va = on::simd::view<4>(angles);

    on::simd::atan2(vy, vx, va);

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(angles[i] == doctest::Approx(std::atan2(y_pos[i], x_pos[i])).epsilon(0.001));
    }
}

// =============================================================================
// asin tests
// =============================================================================

TEST_CASE("algo::asin - y = asin(x)") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{-1.0, -0.866, -0.5, 0.0, 0.5, 0.866, 1.0, 0.707}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::asin(vx, vy);

    CHECK(y[0] == doctest::Approx(std::asin(-1.0)).epsilon(0.001));   // -π/2
    CHECK(y[1] == doctest::Approx(std::asin(-0.866)).epsilon(0.001)); // -π/3
    CHECK(y[2] == doctest::Approx(std::asin(-0.5)).epsilon(0.001));   // -π/6
    CHECK(y[3] == doctest::Approx(std::asin(0.0)).epsilon(0.001));    // 0
    CHECK(y[4] == doctest::Approx(std::asin(0.5)).epsilon(0.001));    // π/6
    CHECK(y[5] == doctest::Approx(std::asin(0.866)).epsilon(0.001));  // π/3
    CHECK(y[6] == doctest::Approx(std::asin(1.0)).epsilon(0.001));    // π/2
    CHECK(y[7] == doctest::Approx(std::asin(0.707)).epsilon(0.001));  // π/4
}

TEST_CASE("algo::asin - in-place x = asin(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{-1.0f, 0.0f, 0.5f, 1.0f}};

    auto vx = on::simd::view<4>(x);
    on::simd::asin(vx);

    CHECK(x[0] == doctest::Approx(std::asin(-1.0f)).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(std::asin(0.0f)).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(std::asin(0.5f)).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(std::asin(1.0f)).epsilon(0.01));
}

TEST_CASE("algo::asin - double precision") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x;
    alignas(32) vec_t y;

    // Test range [-1, 1]
    for (std::size_t i = 0; i < 8; ++i) {
        x[i] = -1.0 + (i / 7.0) * 2.0;
    }

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::asin(vx, vy);

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(y[i] == doctest::Approx(std::asin(x[i])).epsilon(0.001));
    }
}

// =============================================================================
// acos tests
// =============================================================================

TEST_CASE("algo::acos - y = acos(x)") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{-1.0, -0.866, -0.5, 0.0, 0.5, 0.866, 1.0, 0.707}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::acos(vx, vy);

    CHECK(y[0] == doctest::Approx(std::acos(-1.0)).epsilon(0.001));   // π
    CHECK(y[1] == doctest::Approx(std::acos(-0.866)).epsilon(0.001)); // 5π/6
    CHECK(y[2] == doctest::Approx(std::acos(-0.5)).epsilon(0.001));   // 2π/3
    CHECK(y[3] == doctest::Approx(std::acos(0.0)).epsilon(0.001));    // π/2
    CHECK(y[4] == doctest::Approx(std::acos(0.5)).epsilon(0.001));    // π/3
    CHECK(y[5] == doctest::Approx(std::acos(0.866)).epsilon(0.001));  // π/6
    CHECK(y[6] == doctest::Approx(std::acos(1.0)).epsilon(0.001));    // 0
    CHECK(y[7] == doctest::Approx(std::acos(0.707)).epsilon(0.001));  // π/4
}

TEST_CASE("algo::acos - in-place x = acos(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{-1.0f, 0.0f, 0.5f, 1.0f}};

    auto vx = on::simd::view<4>(x);
    on::simd::acos(vx);

    CHECK(x[0] == doctest::Approx(std::acos(-1.0f)).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(std::acos(0.0f)).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(std::acos(0.5f)).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(std::acos(1.0f)).epsilon(0.01));
}

TEST_CASE("algo::acos - double precision") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x;
    alignas(32) vec_t y;

    // Test range [-1, 1]
    for (std::size_t i = 0; i < 8; ++i) {
        x[i] = -1.0 + (i / 7.0) * 2.0;
    }

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::acos(vx, vy);

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(y[i] == doctest::Approx(std::acos(x[i])).epsilon(0.001));
    }
}

TEST_CASE("algo::acos - asin/acos identity: asin(x) + acos(x) = π/2") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{-0.9, -0.7, -0.3, 0.0, 0.3, 0.7, 0.9, 0.5}};
    alignas(32) vec_t y_sin, y_cos;

    auto vx = on::simd::view<4>(x);
    auto vy_sin = on::simd::view<4>(y_sin);
    auto vy_cos = on::simd::view<4>(y_cos);

    on::simd::asin(vx, vy_sin);
    on::simd::acos(vx, vy_cos);

    const double pi_over_2 = 1.5707963267948966;

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK((y_sin[i] + y_cos[i]) == doctest::Approx(pi_over_2).epsilon(0.001));
    }
}

// =============================================================================
// asinh tests
// =============================================================================

TEST_CASE("algo::asinh - y = asinh(x)") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0, 100.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::asinh(vx, vy);

    CHECK(y[0] == doctest::Approx(std::asinh(-10.0)).epsilon(0.001));
    CHECK(y[1] == doctest::Approx(std::asinh(-2.0)).epsilon(0.001));
    CHECK(y[2] == doctest::Approx(std::asinh(-1.0)).epsilon(0.001));
    CHECK(y[3] == doctest::Approx(std::asinh(0.0)).epsilon(0.001)); // 0
    CHECK(y[4] == doctest::Approx(std::asinh(1.0)).epsilon(0.001));
    CHECK(y[5] == doctest::Approx(std::asinh(2.0)).epsilon(0.001));
    CHECK(y[6] == doctest::Approx(std::asinh(10.0)).epsilon(0.001));
    CHECK(y[7] == doctest::Approx(std::asinh(100.0)).epsilon(0.001));
}

TEST_CASE("algo::asinh - in-place x = asinh(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{-1.0f, 0.0f, 1.0f, 5.0f}};

    auto vx = on::simd::view<4>(x);
    on::simd::asinh(vx);

    CHECK(x[0] == doctest::Approx(std::asinh(-1.0f)).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(std::asinh(0.0f)).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(std::asinh(1.0f)).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(std::asinh(5.0f)).epsilon(0.01));
}

TEST_CASE("algo::asinh - double precision") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{-1000.0, -100.0, -10.0, -1.0, 1.0, 10.0, 100.0, 1000.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::asinh(vx, vy);

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(y[i] == doctest::Approx(std::asinh(x[i])).epsilon(0.001));
    }
}

// =============================================================================
// acosh tests
// =============================================================================

TEST_CASE("algo::acosh - y = acosh(x)") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::acosh(vx, vy);

    CHECK(y[0] == doctest::Approx(std::acosh(1.0)).epsilon(0.001)); // 0
    CHECK(y[1] == doctest::Approx(std::acosh(1.5)).epsilon(0.001));
    CHECK(y[2] == doctest::Approx(std::acosh(2.0)).epsilon(0.001));
    CHECK(y[3] == doctest::Approx(std::acosh(3.0)).epsilon(0.001));
    CHECK(y[4] == doctest::Approx(std::acosh(5.0)).epsilon(0.001));
    CHECK(y[5] == doctest::Approx(std::acosh(10.0)).epsilon(0.001));
    CHECK(y[6] == doctest::Approx(std::acosh(50.0)).epsilon(0.001));
    CHECK(y[7] == doctest::Approx(std::acosh(100.0)).epsilon(0.001));
}

TEST_CASE("algo::acosh - in-place x = acosh(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{1.0f, 2.0f, 5.0f, 10.0f}};

    auto vx = on::simd::view<4>(x);
    on::simd::acosh(vx);

    CHECK(x[0] == doctest::Approx(std::acosh(1.0f)).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(std::acosh(2.0f)).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(std::acosh(5.0f)).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(std::acosh(10.0f)).epsilon(0.01));
}

TEST_CASE("algo::acosh - double precision") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{1.0, 1.1, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::acosh(vx, vy);

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(y[i] == doctest::Approx(std::acosh(x[i])).epsilon(0.001));
    }
}

// =============================================================================
// atanh tests
// =============================================================================

TEST_CASE("algo::atanh - y = atanh(x)") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x{{-0.9, -0.7, -0.5, 0.0, 0.5, 0.7, 0.9, 0.99}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::atanh(vx, vy);

    CHECK(y[0] == doctest::Approx(std::atanh(-0.9)).epsilon(0.001));
    CHECK(y[1] == doctest::Approx(std::atanh(-0.7)).epsilon(0.001));
    CHECK(y[2] == doctest::Approx(std::atanh(-0.5)).epsilon(0.001));
    CHECK(y[3] == doctest::Approx(std::atanh(0.0)).epsilon(0.001)); // 0
    CHECK(y[4] == doctest::Approx(std::atanh(0.5)).epsilon(0.001));
    CHECK(y[5] == doctest::Approx(std::atanh(0.7)).epsilon(0.001));
    CHECK(y[6] == doctest::Approx(std::atanh(0.9)).epsilon(0.001));
    CHECK(y[7] == doctest::Approx(std::atanh(0.99)).epsilon(0.001));
}

TEST_CASE("algo::atanh - in-place x = atanh(x)") {
    using vec_t = datapod::mat::vector<float, 4>;

    alignas(32) vec_t x{{-0.5f, 0.0f, 0.5f, 0.9f}};

    auto vx = on::simd::view<4>(x);
    on::simd::atanh(vx);

    CHECK(x[0] == doctest::Approx(std::atanh(-0.5f)).epsilon(0.01));
    CHECK(x[1] == doctest::Approx(std::atanh(0.0f)).epsilon(0.01));
    CHECK(x[2] == doctest::Approx(std::atanh(0.5f)).epsilon(0.01));
    CHECK(x[3] == doctest::Approx(std::atanh(0.9f)).epsilon(0.01));
}

TEST_CASE("algo::atanh - double precision") {
    using vec_t = datapod::mat::vector<double, 8>;

    alignas(32) vec_t x;
    alignas(32) vec_t y;

    // Test range [-0.95, 0.95]
    for (std::size_t i = 0; i < 8; ++i) {
        x[i] = -0.95 + (i / 7.0) * 1.9;
    }

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::atanh(vx, vy);

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(y[i] == doctest::Approx(std::atanh(x[i])).epsilon(0.001));
    }
}
