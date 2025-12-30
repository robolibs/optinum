// =============================================================================
// test/simd/factory_test.cpp
// Tests for factory functions and utility methods:
// fill(), iota(), reverse() on views, and dp::mat::* types directly
// =============================================================================

#include <doctest/doctest.h>
#include <optinum/simd/simd.hpp>
#include <random>

namespace dp = datapod;
namespace simd = optinum::simd;

TEST_CASE("Vector - fill()") {
    dp::mat::vector<float, 5> storage;
    simd::Vector<float, 5> v(storage);
    v.fill(3.14f);

    for (std::size_t i = 0; i < 5; ++i) {
        CHECK(v[i] == 3.14f);
    }
}

TEST_CASE("Vector - iota()") {
    SUBCASE("iota() - default (0, 1, 2, ...)") {
        dp::mat::vector<int, 6> storage;
        simd::Vector<int, 6> v(storage);
        v.iota();

        CHECK(v[0] == 0);
        CHECK(v[1] == 1);
        CHECK(v[2] == 2);
        CHECK(v[3] == 3);
        CHECK(v[4] == 4);
        CHECK(v[5] == 5);
    }

    SUBCASE("iota(start) - custom start") {
        dp::mat::vector<float, 5> storage;
        simd::Vector<float, 5> v(storage);
        v.iota(10.0f);

        CHECK(v[0] == 10.0f);
        CHECK(v[1] == 11.0f);
        CHECK(v[2] == 12.0f);
        CHECK(v[3] == 13.0f);
        CHECK(v[4] == 14.0f);
    }

    SUBCASE("iota(start, step) - custom start and step") {
        dp::mat::vector<double, 4> storage;
        simd::Vector<double, 4> v(storage);
        v.iota(5.0, 2.5);

        CHECK(v[0] == 5.0);
        CHECK(v[1] == 7.5);
        CHECK(v[2] == 10.0);
        CHECK(v[3] == 12.5);
    }
}

TEST_CASE("Vector - zeros() via pod_type") {
    // dp::mat::vector default constructs to zeros
    dp::mat::vector<double, 8> v{};

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(v[i] == 0.0);
    }
}

TEST_CASE("Vector - ones() via view fill") {
    dp::mat::vector<float, 6> storage;
    simd::Vector<float, 6> v(storage);
    v.fill(1.0f);

    for (std::size_t i = 0; i < 6; ++i) {
        CHECK(v[i] == 1.0f);
    }
}

TEST_CASE("Vector - arange() via iota") {
    SUBCASE("arange() - default") {
        dp::mat::vector<int, 5> storage;
        simd::Vector<int, 5> v(storage);
        v.iota();

        CHECK(v[0] == 0);
        CHECK(v[1] == 1);
        CHECK(v[2] == 2);
        CHECK(v[3] == 3);
        CHECK(v[4] == 4);
    }

    SUBCASE("arange(start)") {
        dp::mat::vector<float, 4> storage;
        simd::Vector<float, 4> v(storage);
        v.iota(10.0f);

        CHECK(v[0] == 10.0f);
        CHECK(v[1] == 11.0f);
        CHECK(v[2] == 12.0f);
        CHECK(v[3] == 13.0f);
    }

    SUBCASE("arange(start, step)") {
        dp::mat::vector<double, 3> storage;
        simd::Vector<double, 3> v(storage);
        v.iota(2.0, 0.5);

        CHECK(v[0] == 2.0);
        CHECK(v[1] == 2.5);
        CHECK(v[2] == 3.0);
    }
}

TEST_CASE("Vector - reverse()") {
    SUBCASE("Reverse integers") {
        dp::mat::vector<int, 5> storage;
        simd::Vector<int, 5> v(storage);
        v.iota(); // [0, 1, 2, 3, 4]
        v.reverse();

        CHECK(v[0] == 4);
        CHECK(v[1] == 3);
        CHECK(v[2] == 2);
        CHECK(v[3] == 1);
        CHECK(v[4] == 0);
    }

    SUBCASE("Reverse floats") {
        dp::mat::vector<float, 4> storage;
        simd::Vector<float, 4> v(storage);
        v.iota(10.0f); // [10, 11, 12, 13]
        v.reverse();

        CHECK(v[0] == 13.0f);
        CHECK(v[1] == 12.0f);
        CHECK(v[2] == 11.0f);
        CHECK(v[3] == 10.0f);
    }

    SUBCASE("Reverse even-sized vector") {
        dp::mat::vector<int, 6> storage;
        simd::Vector<int, 6> v(storage);
        v.iota(); // [0, 1, 2, 3, 4, 5]
        v.reverse();

        CHECK(v[0] == 5);
        CHECK(v[1] == 4);
        CHECK(v[2] == 3);
        CHECK(v[3] == 2);
        CHECK(v[4] == 1);
        CHECK(v[5] == 0);
    }
}

// =============================================================================
// Matrix Tests
// =============================================================================

TEST_CASE("Matrix - fill()") {
    dp::mat::matrix<double, 3, 4> storage;
    simd::Matrix<double, 3, 4> m(storage);
    m.fill(2.71);

    for (std::size_t i = 0; i < 12; ++i) {
        CHECK(m[i] == 2.71);
    }
}

TEST_CASE("Matrix - iota()") {
    SUBCASE("iota() - default (0, 1, 2, ...)") {
        dp::mat::matrix<int, 2, 3> storage;
        simd::Matrix<int, 2, 3> m(storage);
        m.iota();

        // Linear indexing
        for (int i = 0; i < 6; ++i) {
            CHECK(m[i] == i);
        }
    }

    SUBCASE("iota(start)") {
        dp::mat::matrix<float, 2, 2> storage;
        simd::Matrix<float, 2, 2> m(storage);
        m.iota(10.0f);

        CHECK(m[0] == 10.0f);
        CHECK(m[1] == 11.0f);
        CHECK(m[2] == 12.0f);
        CHECK(m[3] == 13.0f);
    }

    SUBCASE("iota(start, step)") {
        dp::mat::matrix<double, 3, 2> storage;
        simd::Matrix<double, 3, 2> m(storage);
        m.iota(0.0, 0.5);

        CHECK(m[0] == 0.0);
        CHECK(m[1] == 0.5);
        CHECK(m[2] == 1.0);
        CHECK(m[3] == 1.5);
        CHECK(m[4] == 2.0);
        CHECK(m[5] == 2.5);
    }
}

TEST_CASE("Matrix - zeros() via pod_type") {
    dp::mat::matrix<float, 3, 3> m{};

    for (std::size_t i = 0; i < 9; ++i) {
        CHECK(m[i] == 0.0f);
    }
}

TEST_CASE("Matrix - ones() via fill") {
    dp::mat::matrix<double, 2, 4> storage;
    simd::Matrix<double, 2, 4> m(storage);
    m.fill(1.0);

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(m[i] == 1.0);
    }
}

TEST_CASE("Matrix - arange() via iota") {
    SUBCASE("arange() - default") {
        dp::mat::matrix<int, 2, 3> storage;
        simd::Matrix<int, 2, 3> m(storage);
        m.iota();

        for (int i = 0; i < 6; ++i) {
            CHECK(m[i] == i);
        }
    }

    SUBCASE("arange(start)") {
        dp::mat::matrix<float, 2, 2> storage;
        simd::Matrix<float, 2, 2> m(storage);
        m.iota(5.0f);

        CHECK(m[0] == 5.0f);
        CHECK(m[1] == 6.0f);
        CHECK(m[2] == 7.0f);
        CHECK(m[3] == 8.0f);
    }

    SUBCASE("arange(start, step)") {
        dp::mat::matrix<double, 2, 2> storage;
        simd::Matrix<double, 2, 2> m(storage);
        m.iota(10.0, 2.0);

        CHECK(m[0] == 10.0);
        CHECK(m[1] == 12.0);
        CHECK(m[2] == 14.0);
        CHECK(m[3] == 16.0);
    }
}

TEST_CASE("Matrix - reverse()") {
    dp::mat::matrix<int, 2, 3> storage;
    simd::Matrix<int, 2, 3> m(storage);
    m.iota(); // [0, 1, 2, 3, 4, 5] in linear order
    m.reverse();

    CHECK(m[0] == 5);
    CHECK(m[1] == 4);
    CHECK(m[2] == 3);
    CHECK(m[3] == 2);
    CHECK(m[4] == 1);
    CHECK(m[5] == 0);
}

TEST_CASE("Matrix - random() manual") {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    dp::mat::matrix<float, 5, 5> storage;
    for (std::size_t i = 0; i < 25; ++i) {
        storage[i] = dis(gen);
    }

    // Check all values in [0, 1)
    for (std::size_t i = 0; i < 25; ++i) {
        CHECK(storage[i] >= 0.0f);
        CHECK(storage[i] < 1.0f);
    }

    // Check variance (not all same)
    bool has_different = false;
    for (std::size_t i = 1; i < 25; ++i) {
        if (storage[i] != storage[0]) {
            has_different = true;
            break;
        }
    }
    CHECK(has_different);
}

TEST_CASE("Matrix - randint() manual") {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dis(0, 100);

    dp::mat::matrix<int, 4, 4> storage;
    for (std::size_t i = 0; i < 16; ++i) {
        storage[i] = dis(gen);
    }

    for (std::size_t i = 0; i < 16; ++i) {
        CHECK(storage[i] >= 0);
        CHECK(storage[i] <= 100);
    }
}

TEST_CASE("Constexpr compatibility") {
    SUBCASE("Compile-time zeros") {
        constexpr dp::mat::vector<int, 3> v{};
        static_assert(v[0] == 0);
        static_assert(v[1] == 0);
        static_assert(v[2] == 0);
    }

    SUBCASE("Compile-time ones - manual") {
        constexpr auto make_ones = []() constexpr {
            dp::mat::vector<float, 2> v{};
            v[0] = 1.0f;
            v[1] = 1.0f;
            return v;
        };
        constexpr auto v = make_ones();
        static_assert(v[0] == 1.0f);
        static_assert(v[1] == 1.0f);
    }

    SUBCASE("Compile-time arange - manual") {
        constexpr auto make_arange = []() constexpr {
            dp::mat::vector<int, 4> v{};
            for (int i = 0; i < 4; ++i)
                v[i] = i;
            return v;
        };
        constexpr auto v = make_arange();
        static_assert(v[0] == 0);
        static_assert(v[1] == 1);
        static_assert(v[2] == 2);
        static_assert(v[3] == 3);
    }
}
