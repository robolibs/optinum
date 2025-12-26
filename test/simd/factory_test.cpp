// =============================================================================
// test/simd/factory_test.cpp
// Tests for factory functions and utility methods:
// fill(), iota(), zeros(), ones(), arange(), random(), randint(), reverse()
// =============================================================================

#include <doctest/doctest.h>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

using namespace optinum::simd;

TEST_CASE("Vector - fill()") {
    Vector<float, 5> v;
    v.fill(3.14f);

    for (std::size_t i = 0; i < 5; ++i) {
        CHECK(v[i] == 3.14f);
    }
}

TEST_CASE("Vector - iota()") {
    SUBCASE("iota() - default (0, 1, 2, ...)") {
        Vector<int, 6> v;
        v.iota();

        CHECK(v[0] == 0);
        CHECK(v[1] == 1);
        CHECK(v[2] == 2);
        CHECK(v[3] == 3);
        CHECK(v[4] == 4);
        CHECK(v[5] == 5);
    }

    SUBCASE("iota(start) - custom start") {
        Vector<float, 5> v;
        v.iota(10.0f);

        CHECK(v[0] == 10.0f);
        CHECK(v[1] == 11.0f);
        CHECK(v[2] == 12.0f);
        CHECK(v[3] == 13.0f);
        CHECK(v[4] == 14.0f);
    }

    SUBCASE("iota(start, step) - custom start and step") {
        Vector<double, 4> v;
        v.iota(5.0, 2.5);

        CHECK(v[0] == 5.0);
        CHECK(v[1] == 7.5);
        CHECK(v[2] == 10.0);
        CHECK(v[3] == 12.5);
    }
}

TEST_CASE("Vector - zeros()") {
    auto v = Vector<double, 8>::zeros();

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(v[i] == 0.0);
    }
}

TEST_CASE("Vector - ones()") {
    auto v = Vector<float, 6>::ones();

    for (std::size_t i = 0; i < 6; ++i) {
        CHECK(v[i] == 1.0f);
    }
}

TEST_CASE("Vector - arange()") {
    SUBCASE("arange() - default") {
        auto v = Vector<int, 5>::arange();

        CHECK(v[0] == 0);
        CHECK(v[1] == 1);
        CHECK(v[2] == 2);
        CHECK(v[3] == 3);
        CHECK(v[4] == 4);
    }

    SUBCASE("arange(start)") {
        auto v = Vector<float, 4>::arange(10.0f);

        CHECK(v[0] == 10.0f);
        CHECK(v[1] == 11.0f);
        CHECK(v[2] == 12.0f);
        CHECK(v[3] == 13.0f);
    }

    SUBCASE("arange(start, step)") {
        auto v = Vector<double, 3>::arange(2.0, 0.5);

        CHECK(v[0] == 2.0);
        CHECK(v[1] == 2.5);
        CHECK(v[2] == 3.0);
    }
}

TEST_CASE("Vector - reverse()") {
    SUBCASE("Reverse integers") {
        Vector<int, 5> v;
        v.iota(); // [0, 1, 2, 3, 4]
        v.reverse();

        CHECK(v[0] == 4);
        CHECK(v[1] == 3);
        CHECK(v[2] == 2);
        CHECK(v[3] == 1);
        CHECK(v[4] == 0);
    }

    SUBCASE("Reverse floats") {
        Vector<float, 4> v;
        v.iota(10.0f); // [10, 11, 12, 13]
        v.reverse();

        CHECK(v[0] == 13.0f);
        CHECK(v[1] == 12.0f);
        CHECK(v[2] == 11.0f);
        CHECK(v[3] == 10.0f);
    }

    SUBCASE("Reverse even-sized vector") {
        Vector<int, 6> v;
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

TEST_CASE("Vector - random()") {
    SUBCASE("Float random [0, 1)") {
        Vector<float, 100> v;
        v.random();

        // Check all values are in range [0, 1)
        for (std::size_t i = 0; i < 100; ++i) {
            CHECK(v[i] >= 0.0f);
            CHECK(v[i] < 1.0f);
        }

        // Check not all values are the same (very unlikely with random)
        bool has_different = false;
        for (std::size_t i = 1; i < 100; ++i) {
            if (v[i] != v[0]) {
                has_different = true;
                break;
            }
        }
        CHECK(has_different);
    }

    SUBCASE("Double random [0, 1)") {
        Vector<double, 50> v;
        v.random();

        for (std::size_t i = 0; i < 50; ++i) {
            CHECK(v[i] >= 0.0);
            CHECK(v[i] < 1.0);
        }
    }
}

TEST_CASE("Vector - randint()") {
    SUBCASE("Random integers in [0, 10]") {
        Vector<int, 50> v;
        v.randint(0, 10);

        for (std::size_t i = 0; i < 50; ++i) {
            CHECK(v[i] >= 0);
            CHECK(v[i] <= 10);
        }
    }

    SUBCASE("Random integers in [100, 200]") {
        Vector<int, 30> v;
        v.randint(100, 200);

        for (std::size_t i = 0; i < 30; ++i) {
            CHECK(v[i] >= 100);
            CHECK(v[i] <= 200);
        }
    }
}

// =============================================================================
// Matrix Tests
// =============================================================================

TEST_CASE("Matrix - fill()") {
    Matrix<double, 3, 4> m;
    m.fill(2.71);

    for (std::size_t i = 0; i < 12; ++i) {
        CHECK(m[i] == 2.71);
    }
}

TEST_CASE("Matrix - iota()") {
    SUBCASE("iota() - default (0, 1, 2, ...)") {
        Matrix<int, 2, 3> m;
        m.iota();

        // Linear indexing
        for (int i = 0; i < 6; ++i) {
            CHECK(m[i] == i);
        }
    }

    SUBCASE("iota(start)") {
        Matrix<float, 2, 2> m;
        m.iota(10.0f);

        CHECK(m[0] == 10.0f);
        CHECK(m[1] == 11.0f);
        CHECK(m[2] == 12.0f);
        CHECK(m[3] == 13.0f);
    }

    SUBCASE("iota(start, step)") {
        Matrix<double, 3, 2> m;
        m.iota(0.0, 0.5);

        CHECK(m[0] == 0.0);
        CHECK(m[1] == 0.5);
        CHECK(m[2] == 1.0);
        CHECK(m[3] == 1.5);
        CHECK(m[4] == 2.0);
        CHECK(m[5] == 2.5);
    }
}

TEST_CASE("Matrix - zeros()") {
    auto m = Matrix<float, 3, 3>::zeros();

    for (std::size_t i = 0; i < 9; ++i) {
        CHECK(m[i] == 0.0f);
    }
}

TEST_CASE("Matrix - ones()") {
    auto m = Matrix<double, 2, 4>::ones();

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(m[i] == 1.0);
    }
}

TEST_CASE("Matrix - arange()") {
    SUBCASE("arange() - default") {
        auto m = Matrix<int, 2, 3>::arange();

        for (int i = 0; i < 6; ++i) {
            CHECK(m[i] == i);
        }
    }

    SUBCASE("arange(start)") {
        auto m = Matrix<float, 2, 2>::arange(5.0f);

        CHECK(m[0] == 5.0f);
        CHECK(m[1] == 6.0f);
        CHECK(m[2] == 7.0f);
        CHECK(m[3] == 8.0f);
    }

    SUBCASE("arange(start, step)") {
        auto m = Matrix<double, 2, 2>::arange(10.0, 2.0);

        CHECK(m[0] == 10.0);
        CHECK(m[1] == 12.0);
        CHECK(m[2] == 14.0);
        CHECK(m[3] == 16.0);
    }
}

TEST_CASE("Matrix - reverse()") {
    Matrix<int, 2, 3> m;
    m.iota(); // [0, 1, 2, 3, 4, 5] in linear order
    m.reverse();

    CHECK(m[0] == 5);
    CHECK(m[1] == 4);
    CHECK(m[2] == 3);
    CHECK(m[3] == 2);
    CHECK(m[4] == 1);
    CHECK(m[5] == 0);
}

TEST_CASE("Matrix - random()") {
    Matrix<float, 5, 5> m;
    m.random();

    // Check all values in [0, 1)
    for (std::size_t i = 0; i < 25; ++i) {
        CHECK(m[i] >= 0.0f);
        CHECK(m[i] < 1.0f);
    }

    // Check variance (not all same)
    bool has_different = false;
    for (std::size_t i = 1; i < 25; ++i) {
        if (m[i] != m[0]) {
            has_different = true;
            break;
        }
    }
    CHECK(has_different);
}

TEST_CASE("Matrix - randint()") {
    Matrix<int, 4, 4> m;
    m.randint(0, 100);

    for (std::size_t i = 0; i < 16; ++i) {
        CHECK(m[i] >= 0);
        CHECK(m[i] <= 100);
    }
}

TEST_CASE("Constexpr compatibility") {
    SUBCASE("Compile-time zeros") {
        constexpr auto v = Vector<int, 3>::zeros();
        static_assert(v[0] == 0);
        static_assert(v[1] == 0);
        static_assert(v[2] == 0);
    }

    SUBCASE("Compile-time ones") {
        constexpr auto v = Vector<float, 2>::ones();
        static_assert(v[0] == 1.0f);
        static_assert(v[1] == 1.0f);
    }

    SUBCASE("Compile-time arange") {
        constexpr auto v = Vector<int, 4>::arange();
        static_assert(v[0] == 0);
        static_assert(v[1] == 1);
        static_assert(v[2] == 2);
        static_assert(v[3] == 3);
    }
}
