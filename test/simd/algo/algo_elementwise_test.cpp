// =============================================================================
// test/simd/algo/elementwise_test.cpp
// Tests for SIMD elementwise algorithms
// =============================================================================

#include <doctest/doctest.h>

#include <datapod/matrix.hpp>
#include <optinum/simd/algo/elementwise.hpp>
#include <optinum/simd/bridge.hpp>

namespace on = optinum;

// =============================================================================
// axpy tests
// =============================================================================

TEST_CASE("algo::axpy - y = alpha*x + y") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{1, 2, 3, 4, 5, 6, 7, 8}};
    alignas(32) vec_t y{{10, 20, 30, 40, 50, 60, 70, 80}};

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::axpy(2.0f, vx, vy);

    // y = 2*x + y_old
    CHECK(y[0] == doctest::Approx(12.0f)); // 2*1 + 10
    CHECK(y[1] == doctest::Approx(24.0f)); // 2*2 + 20
    CHECK(y[2] == doctest::Approx(36.0f)); // 2*3 + 30
    CHECK(y[7] == doctest::Approx(96.0f)); // 2*8 + 80
}

TEST_CASE("algo::axpy - tail handling") {
    using vec_t = datapod::mat::vector<float, 10>;

    alignas(32) vec_t x;
    alignas(32) vec_t y;
    for (std::size_t i = 0; i < 10; ++i) {
        x[i] = static_cast<float>(i + 1);
        y[i] = static_cast<float>((i + 1) * 10);
    }

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::axpy(3.0f, vx, vy);

    CHECK(y[0] == doctest::Approx(13.0f));  // 3*1 + 10
    CHECK(y[8] == doctest::Approx(117.0f)); // 3*9 + 90
    CHECK(y[9] == doctest::Approx(130.0f)); // 3*10 + 100
}

// =============================================================================
// scale tests
// =============================================================================

TEST_CASE("algo::scale - x = alpha*x") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{1, 2, 3, 4, 5, 6, 7, 8}};

    auto vx = on::simd::view<4>(x);
    on::simd::scale(2.5f, vx);

    CHECK(x[0] == doctest::Approx(2.5f));
    CHECK(x[3] == doctest::Approx(10.0f));
    CHECK(x[7] == doctest::Approx(20.0f));
}

// =============================================================================
// add tests
// =============================================================================

TEST_CASE("algo::add - z = x + y") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{1, 2, 3, 4, 5, 6, 7, 8}};
    alignas(32) vec_t y{{10, 20, 30, 40, 50, 60, 70, 80}};
    alignas(32) vec_t z;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);
    auto vz = on::simd::view<4>(z);

    on::simd::add(vx, vy, vz);

    CHECK(z[0] == doctest::Approx(11.0f));
    CHECK(z[7] == doctest::Approx(88.0f));
}

TEST_CASE("algo::add - in-place x = x + y") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{1, 2, 3, 4, 5, 6, 7, 8}};
    alignas(32) vec_t y{{10, 20, 30, 40, 50, 60, 70, 80}};

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::add(vx, vy);

    CHECK(x[0] == doctest::Approx(11.0f));
    CHECK(x[7] == doctest::Approx(88.0f));
}

// =============================================================================
// sub tests
// =============================================================================

TEST_CASE("algo::sub - z = x - y") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{10, 20, 30, 40, 50, 60, 70, 80}};
    alignas(32) vec_t y{{1, 2, 3, 4, 5, 6, 7, 8}};
    alignas(32) vec_t z;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);
    auto vz = on::simd::view<4>(z);

    on::simd::sub(vx, vy, vz);

    CHECK(z[0] == doctest::Approx(9.0f));
    CHECK(z[7] == doctest::Approx(72.0f));
}

TEST_CASE("algo::sub - in-place x = x - y") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{10, 20, 30, 40, 50, 60, 70, 80}};
    alignas(32) vec_t y{{1, 2, 3, 4, 5, 6, 7, 8}};

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::sub(vx, vy);

    CHECK(x[0] == doctest::Approx(9.0f));
    CHECK(x[7] == doctest::Approx(72.0f));
}

// =============================================================================
// mul tests
// =============================================================================

TEST_CASE("algo::mul - z = x * y (Hadamard)") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{1, 2, 3, 4, 5, 6, 7, 8}};
    alignas(32) vec_t y{{2, 3, 4, 5, 6, 7, 8, 9}};
    alignas(32) vec_t z;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);
    auto vz = on::simd::view<4>(z);

    on::simd::mul(vx, vy, vz);

    CHECK(z[0] == doctest::Approx(2.0f));  // 1*2
    CHECK(z[3] == doctest::Approx(20.0f)); // 4*5
    CHECK(z[7] == doctest::Approx(72.0f)); // 8*9
}

TEST_CASE("algo::mul - in-place x = x * y") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{1, 2, 3, 4, 5, 6, 7, 8}};
    alignas(32) vec_t y{{2, 3, 4, 5, 6, 7, 8, 9}};

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::mul(vx, vy);

    CHECK(x[0] == doctest::Approx(2.0f));
    CHECK(x[7] == doctest::Approx(72.0f));
}

// =============================================================================
// div tests
// =============================================================================

TEST_CASE("algo::div - z = x / y") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{10, 20, 30, 40, 50, 60, 70, 80}};
    alignas(32) vec_t y{{2, 4, 5, 8, 10, 12, 14, 16}};
    alignas(32) vec_t z;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);
    auto vz = on::simd::view<4>(z);

    on::simd::div(vx, vy, vz);

    CHECK(z[0] == doctest::Approx(5.0f)); // 10/2
    CHECK(z[3] == doctest::Approx(5.0f)); // 40/8
    CHECK(z[7] == doctest::Approx(5.0f)); // 80/16
}

TEST_CASE("algo::div - in-place x = x / y") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{10, 20, 30, 40, 50, 60, 70, 80}};
    alignas(32) vec_t y{{2, 4, 5, 8, 10, 12, 14, 16}};

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::div(vx, vy);

    CHECK(x[0] == doctest::Approx(5.0f));
    CHECK(x[7] == doctest::Approx(5.0f));
}

// =============================================================================
// fill tests
// =============================================================================

TEST_CASE("algo::fill - x = alpha") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x;

    auto vx = on::simd::view<4>(x);
    on::simd::fill(vx, 42.0f);

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(x[i] == doctest::Approx(42.0f));
    }
}

TEST_CASE("algo::fill - tail handling") {
    using vec_t = datapod::mat::vector<float, 10>;

    alignas(32) vec_t x;

    auto vx = on::simd::view<4>(x);
    on::simd::fill(vx, 99.0f);

    for (std::size_t i = 0; i < 10; ++i) {
        CHECK(x[i] == doctest::Approx(99.0f));
    }
}

// =============================================================================
// copy tests
// =============================================================================

TEST_CASE("algo::copy - y = x") {
    using vec_t = datapod::mat::vector<float, 8>;

    alignas(32) vec_t x{{1, 2, 3, 4, 5, 6, 7, 8}};
    alignas(32) vec_t y;

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::copy(vx, vy);

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(y[i] == doctest::Approx(x[i]));
    }
}

TEST_CASE("algo::copy - tail handling") {
    using vec_t = datapod::mat::vector<float, 10>;

    alignas(32) vec_t x;
    alignas(32) vec_t y;

    for (std::size_t i = 0; i < 10; ++i) {
        x[i] = static_cast<float>(i + 1);
    }

    auto vx = on::simd::view<4>(x);
    auto vy = on::simd::view<4>(y);

    on::simd::copy(vx, vy);

    for (std::size_t i = 0; i < 10; ++i) {
        CHECK(y[i] == doctest::Approx(x[i]));
    }
}

// =============================================================================
// Combined operations tests
// =============================================================================

TEST_CASE("algo::combined - typical linear algebra workflow") {
    using vec_t = datapod::mat::vector<float, 16>;

    alignas(32) vec_t x, y, z;

    // Initialize
    auto vx = on::simd::view<8>(x);
    auto vy = on::simd::view<8>(y);
    auto vz = on::simd::view<8>(z);

    on::simd::fill(vx, 1.0f);
    on::simd::fill(vy, 2.0f);
    on::simd::fill(vz, 0.0f);

    // z = x + y
    on::simd::add(vx, vy, vz);
    CHECK(z[0] == doctest::Approx(3.0f));

    // z = 2*z
    on::simd::scale(2.0f, vz);
    CHECK(z[0] == doctest::Approx(6.0f));

    // z = 3*x + z
    on::simd::axpy(3.0f, vx, vz);
    CHECK(z[0] == doctest::Approx(9.0f)); // 6 + 3*1

    for (std::size_t i = 0; i < 16; ++i) {
        CHECK(z[i] == doctest::Approx(9.0f));
    }
}
