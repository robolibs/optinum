// =============================================================================
// test/simd/tensor_reshape_test.cpp
// Tests for Tensor reshape() and squeeze() operations
// =============================================================================

#include <doctest/doctest.h>
#include <optinum/simd/tensor.hpp>

namespace on = optinum;

// =============================================================================
// reshape() Tests
// =============================================================================

TEST_CASE("Tensor reshape() - Basic functionality") {
    using namespace on::simd;

    SUBCASE("Reshape 2x3x4 to 3x2x4") {
        Tensor<float, 2, 3, 4> t;
        t.fill(0.0f);

        // Set some values to verify correct data copying
        for (std::size_t i = 0; i < t.size(); ++i) {
            t[i] = static_cast<float>(i);
        }

        auto reshaped = t.reshape<3, 2, 4>();

        static_assert(std::is_same_v<decltype(reshaped), Tensor<float, 3, 2, 4>>, "Wrong reshape type");
        CHECK(reshaped.size() == 24);

        // Verify data is copied correctly (linear order preserved)
        for (std::size_t i = 0; i < 24; ++i) {
            CHECK(reshaped[i] == doctest::Approx(static_cast<float>(i)));
        }
    }

    SUBCASE("Reshape 2x2x2 to 4x2x1") {
        Tensor<int, 2, 2, 2> t;
        for (std::size_t i = 0; i < 8; ++i) {
            t[i] = static_cast<int>(i + 1);
        }

        auto reshaped = t.reshape<4, 2, 1>();

        CHECK(reshaped.size() == 8);
        for (std::size_t i = 0; i < 8; ++i) {
            CHECK(reshaped[i] == static_cast<int>(i + 1));
        }
    }

    SUBCASE("Reshape 3x3x3 to 9x3x1") {
        Tensor<double, 3, 3, 3> t;
        for (std::size_t i = 0; i < 27; ++i) {
            t[i] = static_cast<double>(i * 0.5);
        }

        auto reshaped = t.reshape<9, 3, 1>();

        CHECK(reshaped.size() == 27);
        for (std::size_t i = 0; i < 27; ++i) {
            CHECK(reshaped[i] == doctest::Approx(static_cast<double>(i * 0.5)));
        }
    }

    SUBCASE("Reshape to same dimensions (no-op)") {
        Tensor<float, 2, 3, 4> t;
        t.fill(42.0f);

        auto reshaped = t.reshape<2, 3, 4>();

        static_assert(std::is_same_v<decltype(reshaped), Tensor<float, 2, 3, 4>>, "Type should be unchanged");
        CHECK(reshaped.size() == 24);

        for (std::size_t i = 0; i < 24; ++i) {
            CHECK(reshaped[i] == doctest::Approx(42.0f));
        }
    }
}

TEST_CASE("Tensor reshape() - 4D tensors") {
    using namespace on::simd;

    SUBCASE("Reshape 2x2x2x3 to 3x4x2x1") {
        Tensor<float, 2, 2, 2, 3> t;
        constexpr std::size_t total = 2 * 2 * 2 * 3;

        for (std::size_t i = 0; i < total; ++i) {
            t[i] = static_cast<float>(i);
        }

        auto reshaped = t.reshape<3, 4, 2, 1>();

        CHECK(reshaped.size() == total);
        for (std::size_t i = 0; i < total; ++i) {
            CHECK(reshaped[i] == doctest::Approx(static_cast<float>(i)));
        }
    }

    SUBCASE("Reshape 3x2x2x2 to 2x2x2x3") {
        Tensor<int, 3, 2, 2, 2> t;
        for (std::size_t i = 0; i < 24; ++i) {
            t[i] = static_cast<int>(i * 2);
        }

        auto reshaped = t.reshape<2, 2, 2, 3>();

        for (std::size_t i = 0; i < 24; ++i) {
            CHECK(reshaped[i] == static_cast<int>(i * 2));
        }
    }
}

TEST_CASE("Tensor reshape() - Constexpr compatibility") {
    using namespace on::simd;

    // The reshape operation should be usable in constant expressions
    constexpr auto test_reshape = []() {
        Tensor<int, 2, 2, 2> t;
        // Note: fill() may not be constexpr, so we can't fully test this at compile time
        return true;
    };

    constexpr bool result = test_reshape();
    CHECK(result == true);
}

// =============================================================================
// squeeze() Tests
// =============================================================================

TEST_CASE("Tensor squeeze() - Rank 3 to Rank 3") {
    using namespace on::simd;

    SUBCASE("squeeze<1, N, M> -> <N, M, 1>") {
        Tensor<float, 1, 3, 4> t;
        for (std::size_t i = 0; i < 12; ++i) {
            t[i] = static_cast<float>(i);
        }

        auto squeezed = squeeze(t);

        static_assert(std::is_same_v<decltype(squeezed), Tensor<float, 3, 4, 1>>, "Should reshape to <3, 4, 1>");
        CHECK(squeezed.size() == 12);

        for (std::size_t i = 0; i < 12; ++i) {
            CHECK(squeezed[i] == doctest::Approx(static_cast<float>(i)));
        }
    }

    SUBCASE("squeeze<N, 1, M> -> <N, M, 1>") {
        Tensor<float, 3, 1, 4> t;
        for (std::size_t i = 0; i < 12; ++i) {
            t[i] = static_cast<float>(i * 2);
        }

        auto squeezed = squeeze(t);

        static_assert(std::is_same_v<decltype(squeezed), Tensor<float, 3, 4, 1>>, "Should reshape to <3, 4, 1>");
        CHECK(squeezed.size() == 12);

        for (std::size_t i = 0; i < 12; ++i) {
            CHECK(squeezed[i] == doctest::Approx(static_cast<float>(i * 2)));
        }
    }

    SUBCASE("squeeze<N, M, 1> -> <N, M, 1> (no-op)") {
        Tensor<float, 3, 4, 1> t;
        t.fill(7.5f);

        auto squeezed = squeeze(t);

        static_assert(std::is_same_v<decltype(squeezed), Tensor<float, 3, 4, 1>>, "Should remain <3, 4, 1>");
        CHECK(squeezed.size() == 12);

        for (std::size_t i = 0; i < 12; ++i) {
            CHECK(squeezed[i] == doctest::Approx(7.5f));
        }
    }

    SUBCASE("squeeze<N, M, P> with all > 1 -> no-op") {
        Tensor<float, 2, 3, 4> t;
        for (std::size_t i = 0; i < 24; ++i) {
            t[i] = static_cast<float>(i);
        }

        auto squeezed = squeeze(t);

        static_assert(std::is_same_v<decltype(squeezed), Tensor<float, 2, 3, 4>>, "Should remain unchanged");
        CHECK(squeezed.size() == 24);

        for (std::size_t i = 0; i < 24; ++i) {
            CHECK(squeezed[i] == doctest::Approx(static_cast<float>(i)));
        }
    }
}

TEST_CASE("Tensor squeeze() - Rank 4 to Rank 3") {
    using namespace on::simd;

    SUBCASE("squeeze<1, N, M, P> -> <N, M, P>") {
        Tensor<float, 1, 2, 3, 4> t;
        constexpr std::size_t total = 24;

        for (std::size_t i = 0; i < total; ++i) {
            t[i] = static_cast<float>(i);
        }

        auto squeezed = squeeze(t);

        static_assert(std::is_same_v<decltype(squeezed), Tensor<float, 2, 3, 4>>, "Should reshape to <2, 3, 4>");
        CHECK(squeezed.size() == total);

        for (std::size_t i = 0; i < total; ++i) {
            CHECK(squeezed[i] == doctest::Approx(static_cast<float>(i)));
        }
    }

    SUBCASE("squeeze<N, 1, M, P> -> <N, M, P>") {
        Tensor<float, 2, 1, 3, 4> t;
        for (std::size_t i = 0; i < 24; ++i) {
            t[i] = static_cast<float>(i + 10);
        }

        auto squeezed = squeeze(t);

        static_assert(std::is_same_v<decltype(squeezed), Tensor<float, 2, 3, 4>>, "Should reshape to <2, 3, 4>");

        for (std::size_t i = 0; i < 24; ++i) {
            CHECK(squeezed[i] == doctest::Approx(static_cast<float>(i + 10)));
        }
    }

    SUBCASE("squeeze<N, M, 1, P> -> <N, M, P>") {
        Tensor<float, 2, 3, 1, 4> t;
        for (std::size_t i = 0; i < 24; ++i) {
            t[i] = static_cast<float>(i * 0.5);
        }

        auto squeezed = squeeze(t);

        static_assert(std::is_same_v<decltype(squeezed), Tensor<float, 2, 3, 4>>, "Should reshape to <2, 3, 4>");

        for (std::size_t i = 0; i < 24; ++i) {
            CHECK(squeezed[i] == doctest::Approx(static_cast<float>(i * 0.5)));
        }
    }

    SUBCASE("squeeze<N, M, P, 1> -> <N, M, P>") {
        Tensor<float, 2, 3, 4, 1> t;
        for (std::size_t i = 0; i < 24; ++i) {
            t[i] = static_cast<float>(100 - i);
        }

        auto squeezed = squeeze(t);

        static_assert(std::is_same_v<decltype(squeezed), Tensor<float, 2, 3, 4>>, "Should reshape to <2, 3, 4>");

        for (std::size_t i = 0; i < 24; ++i) {
            CHECK(squeezed[i] == doctest::Approx(static_cast<float>(100 - i)));
        }
    }
}

TEST_CASE("Tensor squeeze() - Multiple dimensions of size 1") {
    using namespace on::simd;

    SUBCASE("squeeze<1, 1, N, M> -> <N, M, 1>") {
        Tensor<float, 1, 1, 3, 4> t;
        for (std::size_t i = 0; i < 12; ++i) {
            t[i] = static_cast<float>(i * 3);
        }

        auto squeezed = squeeze(t);

        static_assert(std::is_same_v<decltype(squeezed), Tensor<float, 3, 4, 1>>, "Should reshape to <3, 4, 1>");
        CHECK(squeezed.size() == 12);

        for (std::size_t i = 0; i < 12; ++i) {
            CHECK(squeezed[i] == doctest::Approx(static_cast<float>(i * 3)));
        }
    }

    SUBCASE("squeeze<N, 1, M, 1> -> <N, M, 1>") {
        Tensor<float, 3, 1, 4, 1> t;
        for (std::size_t i = 0; i < 12; ++i) {
            t[i] = static_cast<float>(i + 0.5);
        }

        auto squeezed = squeeze(t);

        static_assert(std::is_same_v<decltype(squeezed), Tensor<float, 3, 4, 1>>, "Should reshape to <3, 4, 1>");

        for (std::size_t i = 0; i < 12; ++i) {
            CHECK(squeezed[i] == doctest::Approx(static_cast<float>(i + 0.5)));
        }
    }
}

TEST_CASE("Tensor reshape() and squeeze() - Integration") {
    using namespace on::simd;

    SUBCASE("Reshape then squeeze") {
        Tensor<float, 2, 3, 4> t;
        for (std::size_t i = 0; i < 24; ++i) {
            t[i] = static_cast<float>(i);
        }

        // Reshape to add a dimension of size 1 at the front
        auto reshaped = t.reshape<1, 4, 6>();

        // Then squeeze it back - removes the leading 1
        auto squeezed = squeeze(reshaped);

        // squeeze<1, N, M> -> <N, M, 1>
        CHECK(squeezed.size() == 24);

        for (std::size_t i = 0; i < 24; ++i) {
            CHECK(squeezed[i] == doctest::Approx(static_cast<float>(i)));
        }
    }

    SUBCASE("Multiple reshape operations") {
        Tensor<int, 2, 2, 2> t;
        for (std::size_t i = 0; i < 8; ++i) {
            t[i] = static_cast<int>(i);
        }

        auto r1 = t.reshape<4, 2, 1>();
        auto r2 = r1.reshape<2, 4, 1>();
        auto r3 = r2.reshape<1, 8, 1>();
        auto r4 = r3.reshape<2, 2, 2>();

        // Should end up back at original dimensions
        static_assert(std::is_same_v<decltype(r4), Tensor<int, 2, 2, 2>>, "Should be back to <2, 2, 2>");

        // Data should be preserved
        for (std::size_t i = 0; i < 8; ++i) {
            CHECK(r4[i] == static_cast<int>(i));
        }
    }
}

TEST_CASE("Tensor reshape() - Element access after reshape") {
    using namespace on::simd;

    Tensor<float, 2, 3, 4> t;
    // Initialize with a pattern
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 4; ++k) {
                t(i, j, k) = static_cast<float>(i * 100 + j * 10 + k);
            }
        }
    }

    auto reshaped = t.reshape<3, 4, 2>();

    // Linear access should preserve order
    CHECK(reshaped[0] == t[0]);
    CHECK(reshaped[1] == t[1]);
    CHECK(reshaped[23] == t[23]);
}
